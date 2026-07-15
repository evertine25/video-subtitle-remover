import sys
from functools import cached_property

import cv2
from tqdm import tqdm

from .model_config import ModelConfig
from .hardware_accelerator import HardwareAccelerator
from .common_tools import get_readable_path
from .ocr import get_coordinates
from backend.config import config, tr
from backend.scenedetect import scene_detect
from backend.scenedetect.detectors import ContentDetector
from backend.tools.inpaint_tools import is_frame_number_in_ab_sections

class SubtitleDetect:
    """
    文本框检测类，用于检测视频帧中是否存在文本框
    """

    # 采样间隔，根据视频帧率在 _init_sample_step 中自适应设置
    SAMPLE_STEP = 3

    def __init__(
        self,
        video_path,
        sub_areas=[],
        *,
        det_limit_side_len=None,
        det_thresh=None,
        det_box_thresh=None,
        det_unclip_ratio=None,
        sample_step=None,
        max_frames=None,
        crop_before_ocr=False,
        crop_padding=0,
        crop_upscale=1.0,
        crop_dedup_iou=0.7,
    ):
        self.video_path = video_path
        self.sub_areas = sub_areas
        self.max_frames = int(max_frames) if max_frames is not None else None
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError("max_frames must be at least 1")
        self.crop_before_ocr = bool(crop_before_ocr)
        self.crop_padding = int(crop_padding)
        self.crop_upscale = float(crop_upscale)
        self.crop_dedup_iou = float(crop_dedup_iou)
        if self.crop_padding < 0:
            raise ValueError("crop_padding must be non-negative")
        if self.crop_upscale <= 0:
            raise ValueError("crop_upscale must be positive")
        if not 0.0 <= self.crop_dedup_iou <= 1.0:
            raise ValueError("crop_dedup_iou must be in [0, 1]")
        # PaddleOCR TextDetection.predict parameters.  None preserves the
        # defaults bundled with the selected detection model.
        self.det_predict_kwargs = {
            key: value
            for key, value in {
                "limit_side_len": det_limit_side_len,
                "thresh": det_thresh,
                "box_thresh": det_box_thresh,
                "unclip_ratio": det_unclip_ratio,
            }.items()
            if value is not None
        }
        self._init_sample_step()
        if sample_step is not None:
            if int(sample_step) < 1:
                raise ValueError("sample_step must be at least 1")
            self.SAMPLE_STEP = int(sample_step)

    def _init_sample_step(self):
        """根据视频帧率自适应设置采样间隔，保持每秒至少采样8帧"""
        cap = cv2.VideoCapture(get_readable_path(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps >= 60:
            self.SAMPLE_STEP = 4
        elif fps >= 30:
            self.SAMPLE_STEP = 3
        else:
            self.SAMPLE_STEP = 2

    @cached_property
    def text_detector(self):
        import paddle
        paddle.disable_signal_handler()
        from paddleocr import TextDetection
        hardware_accelerator = HardwareAccelerator.instance()
        onnx_providers = hardware_accelerator.onnx_providers
        model_config = ModelConfig()
        return TextDetection(
            model_name=model_config.DET_MODEL_NAME,
            model_dir=model_config.DET_MODEL_DIR,
            device="gpu:0" if hardware_accelerator.has_cuda() else "cpu",
            enable_hpi=len(onnx_providers) > 0,
        )

    def _predict_coordinates(self, img):
        """Run text detection once and return rectangular coordinates."""
        coordinates = []
        results = self.text_detector.predict(img, **self.det_predict_kwargs)
        for res in results:
            dt_polys = res['dt_polys']
            if dt_polys is None or len(dt_polys) == 0:
                continue
            coordinate_list = get_coordinates(dt_polys.tolist())
            if coordinate_list:
                coordinates.extend(coordinate_list)
        return coordinates

    @staticmethod
    def _map_box_from_crop(
        box,
        crop_xmin,
        crop_ymin,
        scale_x,
        scale_y,
        frame_width,
        frame_height,
    ):
        """Map an OCR box from an optionally resized crop to the source frame."""
        xmin, xmax, ymin, ymax = box
        mapped = (
            int(round(crop_xmin + xmin / scale_x)),
            int(round(crop_xmin + xmax / scale_x)),
            int(round(crop_ymin + ymin / scale_y)),
            int(round(crop_ymin + ymax / scale_y)),
        )
        return (
            max(0, min(frame_width - 1, mapped[0])),
            max(0, min(frame_width - 1, mapped[1])),
            max(0, min(frame_height - 1, mapped[2])),
            max(0, min(frame_height - 1, mapped[3])),
        )

    @staticmethod
    def _box_iou(first, second):
        ax1, ax2, ay1, ay2 = first
        bx1, bx2, by1, by2 = second
        intersection_width = max(0, min(ax2, bx2) - max(ax1, bx1) + 1)
        intersection_height = max(0, min(ay2, by2) - max(ay1, by1) + 1)
        intersection = intersection_width * intersection_height
        first_area = max(0, ax2 - ax1 + 1) * max(0, ay2 - ay1 + 1)
        second_area = max(0, bx2 - bx1 + 1) * max(0, by2 - by1 + 1)
        union = first_area + second_area - intersection
        return intersection / union if union else 0.0

    @classmethod
    def _deduplicate_boxes(cls, boxes, iou_threshold):
        """Merge duplicate detections produced by overlapping OCR crops."""
        if iou_threshold <= 0:
            return list(boxes)
        merged = []
        for box in boxes:
            for index, existing in enumerate(merged):
                if cls._box_iou(box, existing) >= iou_threshold:
                    merged[index] = (
                        min(box[0], existing[0]),
                        max(box[1], existing[1]),
                        min(box[2], existing[2]),
                        max(box[3], existing[3]),
                    )
                    break
            else:
                merged.append(box)
        return merged

    def _detect_in_crops(self, img, sub_areas):
        """Run OCR separately inside each configured area and restore coordinates."""
        frame_height, frame_width = img.shape[:2]
        boxes = []
        for s_ymin, s_ymax, s_xmin, s_xmax in sub_areas:
            area_xmin = max(0, min(frame_width, int(s_xmin)))
            area_xmax = max(0, min(frame_width, int(s_xmax)))
            area_ymin = max(0, min(frame_height, int(s_ymin)))
            area_ymax = max(0, min(frame_height, int(s_ymax)))
            if area_xmin >= area_xmax or area_ymin >= area_ymax:
                continue

            crop_xmin = max(0, area_xmin - self.crop_padding)
            crop_xmax = min(frame_width, area_xmax + self.crop_padding)
            crop_ymin = max(0, area_ymin - self.crop_padding)
            crop_ymax = min(frame_height, area_ymax + self.crop_padding)
            crop = img[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
            source_height, source_width = crop.shape[:2]
            if self.crop_upscale != 1.0:
                resized_width = max(1, int(round(source_width * self.crop_upscale)))
                resized_height = max(1, int(round(source_height * self.crop_upscale)))
                crop = cv2.resize(
                    crop,
                    (resized_width, resized_height),
                    interpolation=cv2.INTER_CUBIC,
                )
            scale_x = crop.shape[1] / source_width
            scale_y = crop.shape[0] / source_height
            for local_box in self._predict_coordinates(crop):
                box = self._map_box_from_crop(
                    local_box,
                    crop_xmin,
                    crop_ymin,
                    scale_x,
                    scale_y,
                    frame_width,
                    frame_height,
                )
                # Padding supplies context to OCR, but detections still belong
                # to the original area according to their centre point.
                center_x = (box[0] + box[1]) / 2
                center_y = (box[2] + box[3]) / 2
                if (
                    area_xmin <= center_x < area_xmax
                    and area_ymin <= center_y < area_ymax
                ):
                    boxes.append(box)
        return self._deduplicate_boxes(boxes, self.crop_dedup_iou)

    def detect_subtitle(self, img):
        sub_areas = self.sub_areas
        has_areas = sub_areas is not None and len(sub_areas) > 0
        if self.crop_before_ocr and has_areas:
            return self._detect_in_crops(img, sub_areas)

        coordinate_list = self._predict_coordinates(img)
        if not has_areas:
            return coordinate_list

        temp_list = []
        for xmin, xmax, ymin, ymax in coordinate_list:
            for s_ymin, s_ymax, s_xmin, s_xmax in sub_areas:
                if (
                    s_xmin <= xmin
                    and xmax <= s_xmax
                    and s_ymin <= ymin
                    and ymax <= s_ymax
                ):
                    temp_list.append((xmin, xmax, ymin, ymax))
                    break
        return temp_list

    def find_subtitle_frame_no(self, sub_remover=None):
        video_cap = cv2.VideoCapture(get_readable_path(self.video_path))
        frame_count = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if self.max_frames is not None:
            frame_count = min(frame_count, self.max_frames)
        # tqdm normally writes to stderr. Keep the detector on the same stream
        # as the mask-preview progress bar: some IDE consoles render carriage
        # returns correctly on stderr but turn sys.__stdout__ updates into many
        # separate lines.
        progress_output = sys.stderr
        # IDE run consoles and GUI-launched processes often do not support
        # carriage-return line updates. In those environments tqdm would print
        # one "Subtitle Finding" line per refresh instead of updating in place.
        show_terminal_progress = bool(
            progress_output
            and hasattr(progress_output, "isatty")
            and progress_output.isatty()
        )
        tbar = tqdm(
            total=int(frame_count),
            unit='frame',
            position=0,
            file=progress_output,
            desc='Subtitle Finding',
            mininterval=1.0,
            dynamic_ncols=True,
            disable=not show_terminal_progress,
        )
        current_frame_no = 0
        # 阶段1：采样检测，仅对每隔 sample_step 帧执行 OCR
        sampled_results = {}  # frame_no -> temp_list
        if sub_remover:
            sub_remover.append_output(tr['Main']['ProcessingStartFindingSubtitles'])
        while video_cap.isOpened():
            if self.max_frames is not None and current_frame_no >= self.max_frames:
                break
            ret, frame = video_cap.read()
            # 如果读取视频帧失败（视频读到最后一帧）
            if not ret:
                break
            # 读取视频帧成功
            current_frame_no += 1
            if not is_frame_number_in_ab_sections(current_frame_no - 1, sub_remover.ab_sections):
                tbar.update(1)
                continue
            # 仅对采样帧执行 OCR 推理
            if (current_frame_no - 1) % self.SAMPLE_STEP == 0 or self.SAMPLE_STEP <= 1:
                temp_list = self.detect_subtitle(frame)
                if len(temp_list) > 0:
                    sampled_results[current_frame_no] = temp_list
            tbar.update(1)
            if sub_remover:
                sub_remover.progress_total = (100 * float(current_frame_no) / float(frame_count)) // 2
        video_cap.release()
        tbar.close()
        # 阶段2：每个采样帧的 OCR 框覆盖其所属采样块。
        # 例如 SAMPLE_STEP=3 时，第1帧结果用于1~3帧，第4帧结果用于4~6帧。
        # 后续颜色精细化仍然逐帧执行，所以字幕提前消失时不会直接产生矩形 mask。
        subtitle_frame_no_box_dict = self.expand_sampled_results(
            sampled_results,
            self.SAMPLE_STEP,
            int(frame_count),
        )
        subtitle_frame_no_box_dict = self.unify_regions(subtitle_frame_no_box_dict)
        if sub_remover:
            sub_remover.append_output(tr['Main']['FinishedFindingSubtitles'])
        new_subtitle_frame_no_box_dict = dict()
        for key in subtitle_frame_no_box_dict.keys():
            if len(subtitle_frame_no_box_dict[key]) > 0:
                new_subtitle_frame_no_box_dict[key] = subtitle_frame_no_box_dict[key]
        return new_subtitle_frame_no_box_dict

    @staticmethod
    def expand_sampled_results(sampled_results, sample_step, max_frame_no):
        """Apply each successful OCR result to its complete sampling block."""
        expanded = {}
        for sample_frame_no in sorted(sampled_results):
            boxes = sampled_results[sample_frame_no]
            block_end = min(
                max_frame_no,
                sample_frame_no + max(1, sample_step) - 1,
            )
            for frame_no in range(sample_frame_no, block_end + 1):
                expanded[frame_no] = boxes
        return expanded

    @staticmethod
    def split_range_by_scene(intervals, points):
        # 确保离散值列表是有序的
        points.sort()
        # 用于存储结果区间的列表
        result_intervals = []
        # 遍历区间
        for start, end in intervals:
            # 在当前区间内的点
            current_points = [p for p in points if start <= p <= end]

            # 遍历当前区间内的离散点
            for p in current_points:
                # 如果当前离散点不是区间的起始点，添加从区间开始到离散点前一个数字的区间
                if start < p:
                    result_intervals.append((start, p - 1))
                # 更新区间开始为当前离散点
                start = p
            # 添加从最后一个离散点或区间开始到区间结束的区间
            result_intervals.append((start, end))
        # 输出结果
        return result_intervals

    @staticmethod
    def get_scene_div_frame_no(v_path):
        """
        获取发生场景切换的帧号
        """
        scene_div_frame_no_list = []
        scene_list = scene_detect(v_path, ContentDetector())
        for scene in scene_list:
            start, end = scene
            if start.frame_num == 0:
                pass
            else:
                scene_div_frame_no_list.append(start.frame_num + 1)
        return scene_div_frame_no_list

    @staticmethod
    def are_similar(region1, region2):
        """判断两个区域是否相似。"""
        xmin1, xmax1, ymin1, ymax1 = region1
        xmin2, xmax2, ymin2, ymax2 = region2

        return abs(xmin1 - xmin2) <= config.subtitleAreaPixelToleranceXPixel.value and abs(xmax1 - xmax2) <= config.subtitleAreaPixelToleranceXPixel.value and \
            abs(ymin1 - ymin2) <= config.subtitleAreaPixelToleranceYPixel.value and abs(ymax1 - ymax2) <= config.subtitleAreaPixelToleranceYPixel.value

    def unify_regions(self, raw_regions):
        """将连续相似的区域统一，保持列表结构。"""
        if len(raw_regions) > 0:
            keys = sorted(raw_regions.keys())  # 对键进行排序以确保它们是连续的
            unified_regions = {}

            # 初始化
            last_key = keys[0]
            unify_value_map = {last_key: raw_regions[last_key]}

            for key in keys[1:]:
                current_regions = raw_regions[key]

                # 新增一个列表来存放匹配过的标准区间
                new_unify_values = []

                for idx, region in enumerate(current_regions):
                    last_standard_region = unify_value_map[last_key][idx] if idx < len(unify_value_map[last_key]) else None

                    # 如果当前的区间与前一个键的对应区间相似，我们统一它们
                    if last_standard_region and self.are_similar(region, last_standard_region):
                        new_unify_values.append(last_standard_region)
                    else:
                        new_unify_values.append(region)

                # 更新unify_value_map为最新的区间值
                unify_value_map[key] = new_unify_values
                last_key = key

            # 将最终统一后的结果传递给unified_regions
            for key in keys:
                unified_regions[key] = unify_value_map[key]
            return unified_regions
        else:
            return raw_regions

    @staticmethod
    def find_continuous_ranges(subtitle_frame_no_box_dict):
        """
        获取字幕出现的起始帧号与结束帧号
        """
        numbers = sorted(list(subtitle_frame_no_box_dict.keys()))
        ranges = []
        start = numbers[0]  # 初始区间开始值

        for i in range(1, len(numbers)):
            # 如果当前数字与前一个数字间隔超过1，
            # 则上一个区间结束，记录当前区间的开始与结束
            if numbers[i] - numbers[i - 1] != 1:
                end = numbers[i - 1]  # 则该数字是当前连续区间的终点
                ranges.append((start, end))
                start = numbers[i]  # 开始下一个连续区间
        # 添加最后一个区间
        ranges.append((start, numbers[-1]))
        return ranges

    @staticmethod
    def find_continuous_ranges_with_same_mask(subtitle_frame_no_box_dict):
        numbers = sorted(list(subtitle_frame_no_box_dict.keys()))
        ranges = []
        start = numbers[0]  # 初始区间开始值
        for i in range(1, len(numbers)):
            # 如果当前帧号与前一个帧号间隔超过1，
            # 则上一个区间结束，记录当前区间的开始与结束
            if numbers[i] - numbers[i - 1] != 1:
                end = numbers[i - 1]  # 则该数字是当前连续区间的终点
                ranges.append((start, end))
                start = numbers[i]  # 开始下一个连续区间
            # 如果当前帧号与前一个帧号间隔为1，且当前帧号对应的坐标点与上一帧号对应的坐标点不一致
            # 记录当前区间的开始与结束
            if numbers[i] - numbers[i - 1] == 1:
                if subtitle_frame_no_box_dict[numbers[i]] != subtitle_frame_no_box_dict[numbers[i - 1]]:
                    end = numbers[i - 1]  # 则该数字是当前连续区间的终点
                    ranges.append((start, end))
                    start = numbers[i]  # 开始下一个连续区间
        # 添加最后一个区间
        ranges.append((start, numbers[-1]))
        return ranges

    @staticmethod
    def filter_and_merge_intervals(intervals, target_length):
        """
        合并传入的字幕起始区间，确保区间大小最低为STTN_REFERENCE_LENGTH
        复杂度 O(n log n)
        """
        if not intervals:
            return []
        intervals = sorted(intervals, key=lambda x: x[0])
        # 一次遍历：扩展单点区间，利用排序后的相邻关系 O(n)
        expanded = []
        for i, (start, end) in enumerate(intervals):
            if start == end:  # 单点区间
                prev_end = expanded[-1][1] if expanded else float('-inf')
                next_start = intervals[i + 1][0] if i + 1 < len(intervals) else float('inf')
                half = (target_length - 1) // 2
                new_start = max(start - half, prev_end + 1)
                new_end = min(start + half, next_start - 1)
                if new_end < new_start:
                    new_start, new_end = start, start
                expanded.append((new_start, new_end))
            else:
                expanded.append((start, end))
        # 一次遍历：合并重叠或相邻的短区间 O(n)
        merged = [expanded[0]]
        for start, end in expanded[1:]:
            last_start, last_end = merged[-1]
            last_len = last_end - last_start + 1
            cur_len = end - start + 1
            if (start <= last_end or start == last_end + 1) and (cur_len < target_length or last_len < target_length):
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        return merged
