import gc
import torch
import shutil
import traceback
import subprocess
import os
from pathlib import Path
import threading
import cv2
import sys
from functools import cached_property

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.config import *
from backend.tools.hardware_accelerator import HardwareAccelerator
from backend.tools.common_tools import is_video_or_image, is_image_file, get_readable_path, read_image
from backend.inpaint.sttn_auto_inpaint import STTNAutoInpaint
from backend.inpaint.sttn_det_inpaint import STTNDetInpaint
from backend.inpaint.lama_inpaint import LamaInpaint
from backend.inpaint.opencv_inpaint import OpenCVInpaint
from backend.inpaint.propainter_inpaint import PropainterInpaint
from backend.tools.inpaint_tools import create_mask, batch_generator, expand_frame_ranges
from backend.tools.model_config import ModelConfig
from backend.tools.ffmpeg_cli import FFmpegCLI
from backend.tools.subtitle_detect import SubtitleDetect
from backend.tools.subtitle_mask import SubtitleMaskGenerator
from backend.tools.refined_mask_runtime import (
    DEFAULT_CONFIG_PATH as REFINED_MASK_CONFIG_PATH,
    build_refined_mask_cache,
    load_refined_mask_runtime_config,
    resolve_mask_preview_output,
)
from backend.tools.video_io import FramePrefetcher, FFmpegVideoWriter
import tempfile
import multiprocessing
import time
from tqdm import tqdm
import numpy as np

class SubtitleRemover:
    def __init__(self, vd_path, gui_mode=False):
        # 线程锁
        self.lock = threading.RLock()
        # 用户指定的字幕区域位置
        self.sub_areas = []
        # 是否为gui运行，gui运行需要显示预览
        self.gui_mode = gui_mode
        self.hardware_accelerator = HardwareAccelerator.instance()
        # 是否使用硬件加速
        self.hardware_accelerator.set_enabled(config.hardwareAcceleration.value)
        self.model_config = ModelConfig()
        # 输入路径统一转换为绝对路径，避免 Windows 短路径转换失败后
        # VideoCapture 收到空字符串。
        self.video_path = os.path.abspath(os.path.expanduser(os.fspath(vd_path)))
        if not os.path.isfile(self.video_path):
            raise FileNotFoundError(f"input media file not found: {self.video_path}")
        # 判断是否为图片
        self.is_picture = is_image_file(self.video_path)
        self.video_cap = cv2.VideoCapture(get_readable_path(self.video_path))
        # 通过视频路径获取视频名称
        self.vd_name = Path(self.video_path).stem
        # 视频帧总数
        self.frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5)
        # 视频帧率
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        # 视频尺寸。部分容器无法直接返回元数据，此时读取首帧回退。
        self.frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.is_picture:
            original_image = read_image(self.video_path)
            if original_image is None:
                raise RuntimeError(f"cannot read input image: {self.video_path}")
            self.frame_height, self.frame_width = original_image.shape[:2]
            self.frame_count = 1
            self.fps = 1.0
        elif not self.video_cap.isOpened():
            raise RuntimeError(f"cannot open input video: {self.video_path}")
        elif self.frame_width <= 0 or self.frame_height <= 0:
            readable, first_frame = self.video_cap.read()
            if not readable or first_frame is None:
                raise RuntimeError(f"cannot decode input video: {self.video_path}")
            self.frame_height, self.frame_width = first_frame.shape[:2]
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if self.frame_width <= 0 or self.frame_height <= 0:
            raise RuntimeError(f"input media has invalid dimensions: {self.video_path}")
        if not self.is_picture and self.fps <= 0:
            raise RuntimeError(f"input video has invalid frame rate: {self.video_path}")
        self.size = (self.frame_width, self.frame_height)
        self.mask_size = (self.frame_height, self.frame_width)
        # 创建视频临时对象，windows下delete=True会有permission denied的报错
        self.video_temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        # 创建视频写对象（使用 FFmpeg libx264 编码，比 mp4v 质量更好、文件更小）
        try:
            self.video_writer = FFmpegVideoWriter(get_readable_path(self.video_temp_file.name), self.fps, self.size)
        except Exception:
            self.video_writer = cv2.VideoWriter(get_readable_path(self.video_temp_file.name), cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.size)
        self.video_out_path = os.path.abspath(os.path.join(os.path.dirname(self.video_path), f'{self.vd_name}_no_sub.mp4'))
        self.propainter_inpaint = None
        self.ext = os.path.splitext(self.video_path)[-1]
        if self.is_picture:
            pic_dir = os.path.join(os.path.dirname(self.video_path), 'no_sub')
            if not os.path.exists(pic_dir):
                os.makedirs(pic_dir)
            self.video_out_path = os.path.join(pic_dir, f'{self.vd_name}{self.ext}')

        # 总处理进度
        self.progress_total = 0
        self.progress_remover = 0
        self.isFinished = False
        # 是否将原音频嵌入到去除字幕后的视频
        self.is_successful_merged = False
        # 进度监听器列表
        self.progress_listeners = []
        # inpaint的frame_no区域列表, 默认为inpaint所有帧
        self.ab_sections = None

    @staticmethod
    def is_current_frame_no_start(frame_no, continuous_frame_no_list):
        """
        判断给定的帧号是否为开头，是的话返回结束帧号，不是的话返回-1
        """
        for start_no, end_no in continuous_frame_no_list:
            if start_no == frame_no:
                return True
        return False

    @staticmethod
    def find_frame_no_end(frame_no, continuous_frame_no_list):
        """
        判断给定的帧号是否为开头，是的话返回结束帧号，不是的话返回-1
        """
        for start_no, end_no in continuous_frame_no_list:
            if start_no <= frame_no <= end_no:
                return end_no
        return -1

    def update_progress(self, tbar, increment):
        tbar.update(increment)
        current_percentage = (tbar.n / tbar.total) * 100
        self.progress_remover = int(current_percentage)
        self.progress_total = self.progress_remover
        self.notify_progress_listeners()

    def append_output(self, *args):
        """输出信息到控制台
        Args:
            *args: 要输出的内容，多个参数将用空格连接
        """
        print(*args)
    
    def add_progress_listener(self, listener):
        """
        添加进度监听器
        
        Args:
            listener: 一个回调函数，接收参数 (progress_total, isFinished)
        """
        if listener not in self.progress_listeners:
            self.progress_listeners.append(listener)
    
    def remove_progress_listener(self, listener):
        """
        移除进度监听器
        
        Args:
            listener: 要移除的监听器函数
        """
        if listener in self.progress_listeners:
            self.progress_listeners.remove(listener)
            
    def notify_progress_listeners(self):
        """
        通知所有进度监听器当前进度
        """
        for listener in self.progress_listeners:
            try:
                listener(self.progress_total, self.isFinished)
            except Exception as e:
                traceback.print_exc()

    def update_preview_with_comp(self, frame_ori, frame_comp):
        """
        更新预览
        """
        pass

    @cached_property
    def refined_mask_config(self):
        return load_refined_mask_runtime_config(REFINED_MASK_CONFIG_PATH)

    def create_subtitle_detector(self):
        kwargs = (
            self.refined_mask_config.ocr_kwargs
            if self.refined_mask_config.enabled else {}
        )
        return SubtitleDetect(self.video_path, self.sub_areas, **(kwargs or {}))

    def build_refined_masks(self, subtitle_boxes):
        if not self.refined_mask_config.enabled:
            return None
        self.append_output(
            f"Generating frame-wise refined masks using {REFINED_MASK_CONFIG_PATH}"
        )
        preview_output = (
            resolve_mask_preview_output(self.video_path, self.refined_mask_config)
            if self.refined_mask_config.write_mask_preview else None
        )
        cache = build_refined_mask_cache(
            self.video_path,
            subtitle_boxes,
            self.refined_mask_config,
            self.frame_count,
            preview_output,
        )
        self.append_output(f"Refined masks generated for {len(cache)} frames")
        if preview_output:
            self.append_output(f"Mask preview video: {preview_output}")
        return cache

    def propainter_mode(self, tbar):
        sub_detector = self.create_subtitle_detector()
        sub_list = sub_detector.find_subtitle_frame_no(sub_remover=self)
        if len(sub_list) == 0:
            raise Exception(tr['Main']['NoSubtitleDetected'].format(self.video_path))
        refined_masks = self.build_refined_masks(sub_list)
        if refined_masks is not None:
            if len(refined_masks) == 0:
                raise Exception(tr['Main']['NoSubtitleDetected'].format(self.video_path))
            active_frames = {frame_no: True for frame_no in refined_masks.keys()}
            continuous_frame_no_list = sub_detector.find_continuous_ranges(active_frames)
        else:
            active_frames = sub_list
            continuous_frame_no_list = sub_detector.find_continuous_ranges_with_same_mask(sub_list)
        scene_div_points = sub_detector.get_scene_div_frame_no(self.video_path)
        continuous_frame_no_list = sub_detector.split_range_by_scene(continuous_frame_no_list,
                                                                          scene_div_points)
        del sub_detector
        gc.collect()        
        device = self.hardware_accelerator.device if self.hardware_accelerator.has_cuda() else torch.device("cpu")
        propainter_inpaint = PropainterInpaint(device, self.model_config.PROPAINTER_MODEL_DIR, config.propainterMaxLoadNum.value)
        self.append_output(tr['Main']['ProcessingStartRemovingSubtitles'])
        index = 0
        # 使用帧预读取，I/O 与推理重叠
        reader = FramePrefetcher(self.video_cap)
        while True:
            ret, frame = reader.read()
            if not ret:
                break
            index += 1
            # 如果当前帧没有水印/文本则直接写
            if index not in active_frames:
                self.video_writer.write(frame)
                # self.append_output(f'write frame: {index}')
                self.update_progress(tbar, increment=1)
                self.update_preview_with_comp(frame, frame)
                continue
            # 如果有水印，判断该帧是不是开头帧
            else:
                # 如果是开头帧，则批推理到尾帧
                if self.is_current_frame_no_start(index, continuous_frame_no_list):
                    # self.append_output(f'No 1 Current index: {index}')
                    start_frame_no = index
                    # self.append_output(f'find start: {start_frame_no}')
                    # 找到结束帧
                    end_frame_no = self.find_frame_no_end(index, continuous_frame_no_list)
                    # 判断当前帧号是不是字幕起始位置
                    # 如果获取的结束帧号不为-1则说明
                    if end_frame_no != -1:
                        # self.append_output(f'find end: {end_frame_no}')
                        # ************ 读取该区间所有帧 start ************
                        temp_frames = list()
                        temp_frame_nos = list()
                        # 将头帧加入处理列表
                        temp_frames.append(frame)
                        temp_frame_nos.append(index)
                        inner_index = 0
                        # 一直读取到尾帧
                        while index < end_frame_no:
                            ret, frame = reader.read()
                            if not ret:
                                break
                            index += 1
                            temp_frames.append(frame)
                            temp_frame_nos.append(index)
                        # ************ 读取该区间所有帧 end ************
                        if len(temp_frames) < 1:
                            # 没有待处理，直接跳过
                            continue
                        elif len(temp_frames) == 1:
                            inner_index += 1
                            single_mask = (
                                refined_masks.get(index)
                                if refined_masks is not None
                                else create_mask(self.mask_size, sub_list[index])
                            )
                            inpainted_frame = self.lama_inpaint.inpaint(frame, single_mask)
                            self.video_writer.write(inpainted_frame)
                            # self.append_output(f'write frame: {start_frame_no + inner_index} with mask {sub_list[start_frame_no]}')
                            self.update_progress(tbar, increment=1)
                            continue
                        else:
                            # 将读取的视频帧分批处理
                            # 1. 获取当前批次使用的mask
                            mask = (
                                None
                                if refined_masks is not None
                                else create_mask(self.mask_size, sub_list[start_frame_no])
                            )
                            batch_offset = 0
                            for batch in batch_generator(temp_frames, config.propainterMaxLoadNum.value):
                                if refined_masks is not None:
                                    batch_frame_nos = temp_frame_nos[
                                        batch_offset:batch_offset + len(batch)
                                    ]
                                    batch_mask = [
                                        refined_masks.get(frame_no)
                                        for frame_no in batch_frame_nos
                                    ]
                                else:
                                    batch_mask = mask
                                # 2. 调用批推理
                                if len(batch) == 1:
                                    single_mask = (
                                        batch_mask[0]
                                        if refined_masks is not None else mask
                                    )
                                    inpainted_frame = self.lama_inpaint.inpaint(
                                        batch[0], single_mask
                                    )
                                    self.video_writer.write(inpainted_frame)
                                    # self.append_output(f'write frame: {start_frame_no + inner_index} with mask {sub_list[start_frame_no]}')
                                    inner_index += 1
                                elif len(batch) > 1:
                                    inpainted_frames = propainter_inpaint(batch, batch_mask)
                                    for i, inpainted_frame in enumerate(inpainted_frames):
                                        self.video_writer.write(inpainted_frame)
                                        # self.append_output(f'write frame: {start_frame_no + inner_index} with mask {sub_list[index]}')
                                        inner_index += 1
                                        preview_mask = (
                                            batch_mask[i]
                                            if refined_masks is not None else mask
                                        )
                                        self.update_preview_with_comp(np.clip(batch[i]+preview_mask[:,:,np.newaxis]*0.3,0,255).astype(np.uint8), inpainted_frame)
                                self.update_progress(tbar, increment=len(batch))
                                batch_offset += len(batch)

    def sttn_auto_mode(self, tbar):
        """
        使用sttn对选中区域进行重绘，不进行字幕检测
        """
        self.append_output(tr['Main']['ProcessingStartRemovingSubtitles'])
        mask_area_coordinates = []
        for sub_area in self.sub_areas:
            ymin, ymax, xmin, xmax = sub_area
            mask_area_coordinates.append((xmin, xmax, ymin, ymax))
        mask = create_mask(self.mask_size, mask_area_coordinates)
        sttn_video_inpaint = STTNAutoInpaint(self.hardware_accelerator.device, self.model_config.STTN_AUTO_MODEL_PATH, self.video_path)
        sttn_video_inpaint(input_mask=mask, input_sub_remover=self, tbar=tbar)

    def video_inpaint(self, tbar, model):
        sub_detector = self.create_subtitle_detector()
        sub_list = sub_detector.find_subtitle_frame_no(sub_remover=self)
        if len(sub_list) == 0:
            raise Exception(tr['Main']['NoSubtitleDetected'].format(self.video_path))
        refined_masks = self.build_refined_masks(sub_list)
        if refined_masks is not None:
            if len(refined_masks) == 0:
                raise Exception(tr['Main']['NoSubtitleDetected'].format(self.video_path))
            active_frames = {frame_no: True for frame_no in refined_masks.keys()}
            continuous_frame_no_list = sub_detector.find_continuous_ranges(active_frames)
        else:
            continuous_frame_no_list = sub_detector.find_continuous_ranges_with_same_mask(sub_list)
        tbar.write(f"Subtitle detected: {continuous_frame_no_list}")
        continuous_frame_no_list = expand_frame_ranges(continuous_frame_no_list, config.subtitleTimelineBackwardFrameCount.value, config.subtitleTimelineForwardFrameCount.value)
        tbar.write(f"Subtitle timeline expand ({config.subtitleTimelineBackwardFrameCount.value} <- -> {config.subtitleTimelineForwardFrameCount.value}): {continuous_frame_no_list}")
        continuous_frame_no_list = sub_detector.filter_and_merge_intervals(continuous_frame_no_list, config.sttnReferenceLength.value)
        tbar.write(f'Subtitle filter_and_merge_intervals: {continuous_frame_no_list}')
        del sub_detector
        gc.collect()
        start_end_map = dict()
        for start, end in continuous_frame_no_list:
            # 确保区间不超出视频总帧数，否则会导致 FramePrefetcher 哨兵被内循环消费后外层死锁
            start_end_map[start] = min(end, self.frame_count)
        current_frame_index = 0
        self.append_output(tr['Main']['ProcessingStartRemovingSubtitles'])
        # 使用帧预读取，I/O 与推理重叠
        reader = FramePrefetcher(self.video_cap)
        while True:
            ret, frame = reader.read()
            # 如果读取到为，则结束
            if not ret:
                break
            current_frame_index += 1
            # 判断当前帧号是不是字幕区间开始, 如果不是，则直接写
            if current_frame_index not in start_end_map.keys():
                self.video_writer.write(frame)
                # self.append_output(f'write frame: {current_frame_index}')
                self.update_progress(tbar, increment=1)
                self.update_preview_with_comp(frame, frame)
            # 如果是区间开始，则找到尾巴
            else:
                start_frame_index = current_frame_index
                end_frame_index = start_end_map[current_frame_index]
                tbar.write(f'processing frame {start_frame_index} to {end_frame_index}')
                # 用于存储需要去字幕的视频帧
                frames_need_inpaint = list()
                frame_nos_need_inpaint = list()
                frames_need_inpaint.append(frame)
                frame_nos_need_inpaint.append(current_frame_index)
                inner_index = 0
                # 接着往下读，直到读取到尾巴
                for j in range(end_frame_index - start_frame_index):
                    ret, frame = reader.read()
                    if not ret:
                        break
                    current_frame_index += 1
                    frames_need_inpaint.append(frame)
                    frame_nos_need_inpaint.append(current_frame_index)
                if refined_masks is not None:
                    masks_need_inpaint = [
                        refined_masks.get(frame_no)
                        for frame_no in frame_nos_need_inpaint
                    ]
                    mask = None
                else:
                    mask_area_coordinates = []
                    # 1. 获取当前批次的mask坐标全集
                    for mask_index in range(start_frame_index, end_frame_index):
                        if mask_index in sub_list.keys():
                            for area in sub_list[mask_index]:
                                xmin, xmax, ymin, ymax = area
                                # 判断是不是非字幕区域(如果宽大于长，则认为是错误检测)
                                if (ymax - ymin) - (xmax - xmin) > config.subtitleYXAxisDifferencePixel.value:
                                    continue
                                if area not in mask_area_coordinates:
                                    mask_area_coordinates.append(area)
                    # 1. 获取当前批次使用的mask
                    mask = create_mask(self.mask_size, mask_area_coordinates)
                # self.append_output(f'inpaint with mask: {mask_area_coordinates}')
                batch_offset = 0
                for batch in batch_generator(frames_need_inpaint, config.getSttnMaxLoadNum()):
                    # 2. 调用批推理
                    if len(batch) >= 1:
                        batch_mask = (
                            masks_need_inpaint[
                                batch_offset:batch_offset + len(batch)
                            ]
                            if refined_masks is not None else mask
                        )
                        inpainted_frames = model(batch, batch_mask)
                        for i, inpainted_frame in enumerate(inpainted_frames):
                            self.video_writer.write(inpainted_frame)
                            # self.append_output(f'write frame: {start_frame_index + inner_index} with mask')
                            inner_index += 1
                            preview_mask = (
                                batch_mask[i]
                                if refined_masks is not None else mask
                            )
                            self.update_preview_with_comp(np.clip(batch[i]+preview_mask[:,:,np.newaxis]*0.3,0,255).astype(np.uint8), inpainted_frame)
                    self.update_progress(tbar, increment=len(batch))
                    batch_offset += len(batch)
        reader.stop()

    def run(self):
        # 记录开始时间
        start_time = time.time()
        if len(self.sub_areas) == 0:
            self.append_output(tr['Main']['FullScreenProcessingNote'])
            self.sub_areas.append((0, self.frame_height, 0, self.frame_width))
        self.append_output(tr['Main']['SubtitleArea'].format(self.sub_areas))
        self.append_output(tr['Main']['ABSection'].format(str(self.ab_sections).replace("range", "") if self.ab_sections is not None and len(self.ab_sections) > 0 else tr['Main']['ABSectionAll']))
        # 如果使用GPU加速，则打印GPU加速提示
        if self.hardware_accelerator.has_accelerator():
            accelerator_name = self.hardware_accelerator.accelerator_name
            if accelerator_name == 'DirectML' and config.inpaintMode.value not in [InpaintMode.STTN_AUTO, InpaintMode.STTN_DET]:
                self.append_output(tr['Main']['DirectMLWarning'])
        os.makedirs(os.path.dirname(self.video_out_path), exist_ok=True)
        # 重置进度条
        self.progress_total = 0
        tbar = tqdm(total=int(self.frame_count), unit='frame', position=0, file=sys.__stdout__,
                    desc='Subtitle Removing')
        if self.is_picture:
            original_frame = read_image(self.video_path)
            if original_frame is None:
                self.append_output(tr['Main']['ReadImageFailed'].format(self.video_path))
                return
            sub_detector = self.create_subtitle_detector()
            sub_list = sub_detector.detect_subtitle(original_frame)
            del sub_detector
            gc.collect()
            if len(sub_list):
                mask = (
                    SubtitleMaskGenerator(self.refined_mask_config.mask).generate(
                        original_frame, sub_list
                    )
                    if self.refined_mask_config.enabled
                    else create_mask(original_frame.shape[0:2], sub_list)
                )
                if np.any(mask):
                    inpainted_frame = self.lama_inpaint.inpaint(original_frame, mask)
                    self.update_preview_with_comp(np.clip(original_frame+mask[:,:,np.newaxis]*0.3,0,255).astype(np.uint8), inpainted_frame)
                else:
                    inpainted_frame = original_frame
                    self.update_preview_with_comp(original_frame, inpainted_frame)
            else:
                inpainted_frame = original_frame
                self.update_preview_with_comp(original_frame, inpainted_frame)
            cv2.imencode(self.ext, inpainted_frame)[1].tofile(self.video_out_path)
            tbar.update(1)
            self.progress_total = 100
        else:
            # 精准模式下，获取场景分割的帧号，进一步切割
            self.log_model()
            if config.inpaintMode.value == InpaintMode.PROPAINTER:
                self.propainter_mode(tbar)
            elif config.inpaintMode.value == InpaintMode.STTN_AUTO:
                self.sttn_auto_mode(tbar)
            elif config.inpaintMode.value == InpaintMode.STTN_DET:
                self.video_inpaint(tbar, self.sttn_det_inpaint)
            elif config.inpaintMode.value == InpaintMode.LAMA:
                self.video_inpaint(tbar, self.lama_inpaint)
            elif config.inpaintMode.value == InpaintMode.OPENCV:
                self.video_inpaint(tbar, OpenCVInpaint())
            else:
                raise Exception(f'inpaint mode: {config.inpaintMode.value} not implemented')

        self.video_cap.release()
        self.video_writer.release()
        if not self.is_picture:
            # 将原音频合并到新生成的视频文件中
            self.merge_audio_to_video()
        self.append_output(tr['Main']['FinishedProcessing'].format(self.video_out_path))
        self.append_output(tr['Main']['ProcessingTime'].format(round(time.time() - start_time)))
        self.isFinished = True
        self.progress_total = 100
        if os.path.exists(self.video_temp_file.name):
            try:
                os.remove(self.video_temp_file.name)
            except Exception:
                pass #ignore

    def log_model(self):
        model_friendly_name = list(tr['InpaintMode'].values())[list(InpaintMode).index(config.inpaintMode.value)]
        model_device = 'CPU'
        if config.inpaintMode.value != InpaintMode.OPENCV and self.hardware_accelerator.has_accelerator():
            accelerator_name = self.hardware_accelerator.accelerator_name
            if accelerator_name == 'DirectML' and config.inpaintMode.value in [InpaintMode.STTN_AUTO, InpaintMode.STTN_DET]:
                model_device = 'DirectML'
            if self.hardware_accelerator.has_cuda() or self.hardware_accelerator.has_mps():
                model_device = accelerator_name
        self.append_output(tr['Main']['SubtitleRemoverModel'].format(f"{model_friendly_name} ({model_device})"))
        providers = ", ".join(self.hardware_accelerator.onnx_providers)
        providers_str = f" ({providers})" if providers else ""
        detect_mode_name = list(tr['SubtitleDetectMode'].values())[list(SubtitleDetectMode).index(config.subtitleDetectMode.value)]
        self.append_output(tr['Main']['SubtitleDetectionModel'].format(f"{detect_mode_name}{providers_str}"))
        if config.inpaintMode.value == InpaintMode.STTN_AUTO:
            self.append_output("Mask mode: legacy selected-area mask (STTN_AUTO skips OCR)")
        elif self.refined_mask_config.enabled:
            self.append_output(f"Mask mode: frame-wise refined mask ({REFINED_MASK_CONFIG_PATH})")
        else:
            self.append_output("Mask mode: legacy OCR rectangle mask")

    def merge_audio_to_video(self):
        # 创建音频临时对象，windows下delete=True会有permission denied的报错
        temp = tempfile.NamedTemporaryFile(suffix='.aac', delete=False)
        audio_extract_command = [FFmpegCLI.instance().ffmpeg_path,
                                 "-y", "-i", self.video_path,
                                 "-acodec", "copy",
                                 "-vn", "-loglevel", "error", temp.name]
        use_shell = True if os.name == "nt" else False
        try:
            subprocess.check_output(audio_extract_command, stdin=open(os.devnull), shell=use_shell, timeout=600)
        except Exception as e:
            traceback.print_exc()
            self.append_output(tr['Main']['FailToExtractAudio'].format(str(e)))
            return
        else:
            if os.path.exists(self.video_temp_file.name):
                audio_merge_command = [FFmpegCLI.instance().ffmpeg_path,
                                       "-y", "-i", self.video_temp_file.name,
                                       "-i", temp.name,
                                       "-vcodec", "copy",
                                       "-acodec", "copy",
                                       "-loglevel", "error", self.video_out_path]
                try:
                    subprocess.check_output(audio_merge_command, stdin=open(os.devnull), shell=use_shell, timeout=600)
                except Exception as e:
                    traceback.print_exc()
                    self.append_output(tr['Main']['FailToMergeAudio'].format(str(e)))
                    return
            if os.path.exists(temp.name):
                try:
                    os.remove(temp.name)
                except Exception:
                    #ignore
                    pass
            self.is_successful_merged = True
        finally:
            temp.close()
            if not self.is_successful_merged:
                try:
                    shutil.copy2(self.video_temp_file.name, self.video_out_path)
                except IOError as e:
                    self.append_output(tr['Main']['CopyFileFailed'].format(self.video_temp_file.name, self.video_out_path, str(e)))
            self.video_temp_file.close()

    @cached_property
    def lama_inpaint(self):
        model_path = os.path.join(self.model_config.LAMA_MODEL_DIR, 'big-lama.pt')
        device = self.hardware_accelerator.device if self.hardware_accelerator.has_cuda() or self.hardware_accelerator.has_mps() else torch.device("cpu")
        return LamaInpaint(device, model_path)

    @cached_property
    def sttn_det_inpaint(self):
        return STTNDetInpaint(self.hardware_accelerator.device, self.model_config.STTN_DET_MODEL_PATH)


if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
    from backend.tools.args_handler import parse_args, subtitle_area_ratios_to_pixels
    args = parse_args()
    # force english
    config.set(config.interface, 'en')
    TRANSLATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'interface', f"{config.interface.value}.ini")
    tr.read(TRANSLATION_FILE, encoding='utf-8')
    if not is_video_or_image(args.input):
        print(f'Error: {args.input} is not a supported video or image file.')
        exit(-1)
    sr = SubtitleRemover(args.input)
    sr.sub_areas = (
        subtitle_area_ratios_to_pixels(
            args.subtitle_area_ratios,
            sr.frame_width,
            sr.frame_height,
        )
        if args.subtitle_area_ratios
        else args.subtitle_area_coords
    )
    if args.output:
        sr.video_out_path = os.path.abspath(os.path.expanduser(args.output))
    config.inpaintMode.value = args.inpaint_mode
    sr.run()
        
