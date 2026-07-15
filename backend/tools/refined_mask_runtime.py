"""Runtime integration for frame-wise colour-refined subtitle masks."""

from collections import deque
from dataclasses import dataclass
import configparser
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from backend.tools.common_tools import get_readable_path
from backend.tools.subtitle_mask import SubtitleMaskConfig, SubtitleMaskGenerator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "subtitle_mask_config.ini"


@dataclass(frozen=True)
class RefinedMaskRuntimeConfig:
    enabled: bool
    mask: SubtitleMaskConfig
    future_mask_frames: int = 0
    mask_stability_frames: int = 0
    mask_stability_iou: float = 0.995
    preserve_future_mask_continuity: bool = True
    ocr_kwargs: Optional[Dict[str, object]] = None
    write_mask_preview: bool = False
    mask_preview_output: Optional[str] = None
    draw_ocr_boxes: bool = True
    draw_expanded_boxes: bool = True


class EncodedMaskCache:
    """Sparse, PNG-compressed frame mask cache using one-based frame numbers."""

    def __init__(self, shape: Tuple[int, int]):
        self.shape = tuple(shape)
        self._encoded = {}

    def put(self, frame_no: int, mask: np.ndarray):
        if not np.any(mask):
            return
        ok, encoded = cv2.imencode(".png", mask)
        if not ok:
            raise RuntimeError(f"cannot encode refined mask for frame {frame_no}")
        self._encoded[int(frame_no)] = encoded

    def get(self, frame_no: int) -> np.ndarray:
        encoded = self._encoded.get(int(frame_no))
        if encoded is None:
            return np.zeros(self.shape, dtype=np.uint8)
        mask = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"cannot decode refined mask for frame {frame_no}")
        return mask

    def __contains__(self, frame_no):
        return int(frame_no) in self._encoded

    def __len__(self):
        return len(self._encoded)

    def keys(self):
        return self._encoded.keys()


def _parse_rgb(value: str):
    value = value.strip()
    if value.startswith("#"):
        if len(value) != 7:
            raise ValueError("colour must use #RRGGBB")
        channels = tuple(int(value[index:index + 2], 16) for index in (1, 3, 5))
    else:
        channels = tuple(int(part.strip()) for part in value.split(","))
    if len(channels) != 3 or any(not 0 <= channel <= 255 for channel in channels):
        raise ValueError("colour must contain three channels in [0, 255]")
    return channels


def _optional(section, option, converter):
    raw = section.get(option, fallback="").strip()
    if not raw or raw.lower() in {"none", "default"}:
        return None
    return converter(raw)


def load_refined_mask_runtime_config(
    path=DEFAULT_CONFIG_PATH,
) -> RefinedMaskRuntimeConfig:
    """Load the inpainting switch and mask parameters from the preview INI."""
    config_path = Path(path).expanduser().resolve()
    ini = configparser.ConfigParser(inline_comment_prefixes=(";",))
    if not config_path.is_file() or not ini.read(config_path, encoding="utf-8"):
        return RefinedMaskRuntimeConfig(False, SubtitleMaskConfig())

    integration = ini["integration"] if ini.has_section("integration") else None
    enabled = (
        integration.getboolean("use_refined_mask", fallback=False)
        if integration is not None else False
    )
    if not enabled:
        return RefinedMaskRuntimeConfig(False, SubtitleMaskConfig())
    if not ini.has_section("mask"):
        raise ValueError("refined mask is enabled but [mask] is missing")

    mask = ini["mask"]
    secondary = ini["secondary_mask"] if ini.has_section("secondary_mask") else None
    temporal = ini["temporal"] if ini.has_section("temporal") else None
    ocr = ini["ocr"] if ini.has_section("ocr") else None
    preview = ini["preview"] if ini.has_section("preview") else None
    red, green, blue = _parse_rgb(mask.get("color", "255,255,255"))
    secondary_red, secondary_green, secondary_blue = _parse_rgb(
        secondary.get("color", "75,75,75") if secondary is not None else "75,75,75"
    )

    mask_config = SubtitleMaskConfig(
        target_bgr=(blue, green, red),
        max_channel_spread=mask.getint("max_channel_spread", fallback=15),
        core_tolerance=mask.getfloat("core_tolerance", fallback=18.0),
        edge_tolerance=mask.getfloat("edge_tolerance", fallback=38.0),
        box_padding=mask.getint("box_padding", fallback=4),
        box_expand_enabled=mask.getboolean("box_expand_enabled", fallback=True),
        box_expand_step=mask.getint("box_expand_step", fallback=6),
        box_expand_max_x=mask.getint("box_expand_max_x", fallback=80),
        box_expand_max_y=mask.getint("box_expand_max_y", fallback=12),
        box_expand_min_core_pixels=mask.getint(
            "box_expand_min_core_pixels", fallback=2
        ),
        secondary_enabled=(
            secondary.getboolean("enabled", fallback=False)
            if secondary is not None else False
        ),
        secondary_left_ratio=(
            secondary.getfloat("left_ratio", fallback=0.25)
            if secondary is not None else 0.25
        ),
        secondary_target_bgr=(secondary_blue, secondary_green, secondary_red),
        secondary_max_channel_spread=(
            secondary.getint("max_channel_spread", fallback=10)
            if secondary is not None else 10
        ),
        secondary_core_tolerance=(
            secondary.getfloat("core_tolerance", fallback=32.0)
            if secondary is not None else 32.0
        ),
        secondary_edge_tolerance=(
            secondary.getfloat("edge_tolerance", fallback=50.0)
            if secondary is not None else 50.0
        ),
        edge_growth_iterations=mask.getint("edge_growth_iterations", fallback=3),
        close_kernel_size=mask.getint("close_kernel_size", fallback=3),
        min_component_area=mask.getint("min_component_area", fallback=2),
        isolation_distance=mask.getint("isolation_distance", fallback=4),
        dilation_size=mask.getint("dilation_size", fallback=3),
        dilation_iterations=mask.getint("dilation_iterations", fallback=1),
    )

    future_frames = (
        temporal.getint("future_mask_frames", fallback=0)
        if temporal is not None else 0
    )
    stability_frames = (
        temporal.getint("mask_stability_frames", fallback=0)
        if temporal is not None else 0
    )
    if stability_frames == 0:
        stability_frames = future_frames
    stability_iou = (
        temporal.getfloat("mask_stability_iou", fallback=0.995)
        if temporal is not None else 0.995
    )
    if future_frames < 0 or stability_frames < 0:
        raise ValueError("temporal frame counts must be non-negative")
    if not 0.0 <= stability_iou <= 1.0:
        raise ValueError("mask_stability_iou must be in [0, 1]")

    ocr_kwargs = {}
    if ocr is not None:
        option_map = {
            "det_limit_side_len": ("limit_side_len", int),
            "det_thresh": ("thresh", float),
            "det_box_thresh": ("box_thresh", float),
            "det_unclip_ratio": ("unclip_ratio", float),
            "sample_step": ("sample_step", int),
        }
        for argument, (option, converter) in option_map.items():
            value = _optional(ocr, option, converter)
            if argument == "sample_step" and value == 0:
                value = None
            if value is not None:
                ocr_kwargs[argument] = value
        ocr_kwargs.update(
            {
                "crop_before_ocr": ocr.getboolean(
                    "crop_before_ocr", fallback=False
                ),
                "crop_padding": ocr.getint("crop_padding", fallback=0),
                "crop_upscale": ocr.getfloat("crop_upscale", fallback=1.0),
                "crop_dedup_iou": ocr.getfloat(
                    "crop_dedup_iou", fallback=0.7
                ),
            }
        )

    return RefinedMaskRuntimeConfig(
        enabled=True,
        mask=mask_config,
        future_mask_frames=future_frames,
        mask_stability_frames=stability_frames,
        mask_stability_iou=stability_iou,
        preserve_future_mask_continuity=(
            temporal.getboolean("preserve_future_mask_continuity", fallback=True)
            if temporal is not None else True
        ),
        ocr_kwargs=ocr_kwargs,
        write_mask_preview=integration.getboolean(
            "write_mask_preview", fallback=False
        ),
        mask_preview_output=(
            integration.get("mask_preview_output", fallback="").strip() or None
        ),
        draw_ocr_boxes=(
            preview.getboolean("draw_ocr_boxes", fallback=True)
            if preview is not None else True
        ),
        draw_expanded_boxes=(
            preview.getboolean("draw_expanded_boxes", fallback=True)
            if preview is not None else True
        ),
    )


def resolve_mask_preview_output(video_path, runtime_config):
    """Resolve an explicit preview path or derive one beside the input video."""
    configured = runtime_config.mask_preview_output
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return str(path.resolve())
    input_path = Path(video_path).expanduser().resolve()
    return str(input_path.with_name(f"{input_path.stem}_mask_preview.mp4"))


def _draw_boxes(frame, boxes, colour):
    thickness = max(1, int(round(min(frame.shape[:2]) / 540)))
    for xmin, xmax, ymin, ymax in boxes:
        cv2.rectangle(
            frame,
            (int(xmin), int(ymin)),
            (int(xmax), int(ymax)),
            colour,
            thickness,
        )


def mask_iou(mask_a, mask_b) -> float:
    foreground_a = mask_a > 0
    foreground_b = mask_b > 0
    union = np.count_nonzero(foreground_a | foreground_b)
    if union == 0:
        return 1.0
    return np.count_nonzero(foreground_a & foreground_b) / union


def next_stability_run(previous_mask, current_mask, previous_run, iou_threshold):
    if not np.any(current_mask):
        return 0
    if previous_mask is None or not np.any(previous_mask):
        return 1
    if mask_iou(previous_mask, current_mask) >= iou_threshold:
        return previous_run + 1
    return 1


def compose_temporal_mask(
    raw_mask,
    future_masks,
    stability_run,
    stability_frames,
    previous_output_mask=None,
    preserve_continuity=True,
):
    lookahead_mask = raw_mask.copy()
    for future_mask in future_masks:
        cv2.bitwise_or(lookahead_mask, future_mask, dst=lookahead_mask)
    mask = lookahead_mask.copy() if stability_run < stability_frames else raw_mask.copy()
    if preserve_continuity and previous_output_mask is not None:
        retained = cv2.bitwise_and(previous_output_mask, lookahead_mask)
        cv2.bitwise_or(mask, retained, dst=mask)
    return mask


def build_refined_mask_cache(
    video_path,
    boxes_by_frame,
    runtime_config: RefinedMaskRuntimeConfig,
    max_frames=None,
    preview_output_path=None,
):
    """Generate temporal, frame-wise masks and store only non-empty PNGs."""
    cap = cv2.VideoCapture(get_readable_path(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open input video for refined masks: {video_path}")
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if width <= 0 or height <= 0:
        ok, first_frame = cap.read()
        if not ok or first_frame is None:
            cap.release()
            raise RuntimeError(f"cannot decode input video for refined masks: {video_path}")
        height, width = first_frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if fps <= 0:
        cap.release()
        raise RuntimeError(f"input video has invalid frame rate: {video_path}")
    if max_frames is not None:
        frame_count = min(frame_count, int(max_frames))
    cache = EncodedMaskCache((height, width))
    generator = SubtitleMaskGenerator(runtime_config.mask)
    preview_writer = None
    if preview_output_path:
        preview_path = Path(preview_output_path).expanduser().resolve()
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        preview_writer = cv2.VideoWriter(
            get_readable_path(str(preview_path)),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not preview_writer.isOpened():
            cap.release()
            raise RuntimeError(f"cannot create mask preview video: {preview_path}")
    buffer = deque()
    previous_raw = None
    previous_output = None
    stability_run = 0
    frame_no = 0

    def flush_head():
        nonlocal previous_output
        (
            oldest_frame_no,
            oldest_frame,
            original_boxes,
            expanded_boxes,
            raw_mask,
            oldest_stability,
        ) = buffer[0]
        future_masks = [
            entry[4]
            for entry in list(buffer)[1:runtime_config.future_mask_frames + 1]
        ]
        output_mask = compose_temporal_mask(
            raw_mask,
            future_masks,
            oldest_stability,
            runtime_config.mask_stability_frames,
            previous_output,
            runtime_config.preserve_future_mask_continuity,
        )
        cache.put(oldest_frame_no, output_mask)
        if preview_writer is not None:
            preview_frame = oldest_frame.copy()
            preview_frame[output_mask > 0] = 0
            if runtime_config.draw_expanded_boxes:
                _draw_boxes(preview_frame, expanded_boxes, (0, 165, 255))
            if runtime_config.draw_ocr_boxes:
                _draw_boxes(preview_frame, original_boxes, (0, 255, 0))
            preview_writer.write(preview_frame)
        previous_output = output_mask

    try:
        while frame_no < frame_count:
            ok, frame = cap.read()
            if not ok:
                break
            frame_no += 1
            original_boxes = boxes_by_frame.get(frame_no, [])
            raw_mask, expanded_boxes = generator.generate_with_expanded_boxes(
                frame, original_boxes
            )
            stability_run = next_stability_run(
                previous_raw,
                raw_mask,
                stability_run,
                runtime_config.mask_stability_iou,
            )
            previous_raw = raw_mask
            buffer.append(
                (
                    frame_no,
                    frame if preview_writer is not None else None,
                    original_boxes,
                    expanded_boxes,
                    raw_mask,
                    stability_run,
                )
            )
            if len(buffer) > runtime_config.future_mask_frames:
                flush_head()
                buffer.popleft()
        while buffer:
            flush_head()
            buffer.popleft()
    finally:
        cap.release()
        if preview_writer is not None:
            preview_writer.release()
    return cache
