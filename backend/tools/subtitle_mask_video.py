"""Preview pixel-level subtitle masks on a video.

Example:
    python -m backend.tools.subtitle_mask_video
    python -m backend.tools.subtitle_mask_video --config my_mask_config.ini
"""

import argparse
from collections import deque
import configparser
import os
from pathlib import Path
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from backend.tools.common_tools import get_readable_path
from backend.tools.subtitle_detect import SubtitleDetect
from backend.tools.subtitle_mask import SubtitleMaskConfig, SubtitleMaskGenerator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "subtitle_mask_config.ini"


class _DetectionProgress:
    """Minimal adapter required by SubtitleDetect.find_subtitle_frame_no."""

    def __init__(self):
        self.ab_sections = None
        self.progress_total = 0

    @staticmethod
    def append_output(message):
        print(message)


def _parse_rgb(value: str) -> Tuple[int, int, int]:
    value = value.strip()
    if value.startswith("#"):
        if len(value) != 7:
            raise argparse.ArgumentTypeError("hex colour must use #RRGGBB")
        try:
            return tuple(int(value[index:index + 2], 16) for index in (1, 3, 5))
        except ValueError as exc:
            raise argparse.ArgumentTypeError("invalid hex colour") from exc

    try:
        values = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("colour must be R,G,B or #RRGGBB") from exc
    if len(values) != 3 or any(channel < 0 or channel > 255 for channel in values):
        raise argparse.ArgumentTypeError("RGB channels must be in [0, 255]")
    return values


def _parse_area(value: str) -> Tuple[float, float, float, float]:
    try:
        values = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "area must contain four ratios: ymin,ymax,xmin,xmax"
        ) from exc
    if len(values) != 4:
        raise argparse.ArgumentTypeError(
            "area must contain four ratios: ymin,ymax,xmin,xmax"
        )
    if any(coordinate < 0.0 or coordinate > 1.0 for coordinate in values):
        raise argparse.ArgumentTypeError("area ratios must be in [0, 1]")
    ymin, ymax, xmin, xmax = values
    if ymin >= ymax or xmin >= xmax:
        raise argparse.ArgumentTypeError(
            "area must satisfy ymin < ymax and xmin < xmax"
        )
    return values


def _parse_areas(value: str) -> List[Tuple[float, float, float, float]]:
    """Parse one area per line; semicolons are also accepted as separators."""
    entries = []
    for raw_entry in value.replace(";", "\n").splitlines():
        entry = raw_entry.strip()
        if entry:
            entries.append(_parse_area(entry))
    if not entries:
        raise argparse.ArgumentTypeError("areas must contain at least one area")
    return entries


def _optional_value(section, option, converter):
    raw_value = section.get(option, fallback="").strip()
    if not raw_value or raw_value.lower() in {"none", "default"}:
        return None
    try:
        return converter(raw_value)
    except (TypeError, ValueError, argparse.ArgumentTypeError) as exc:
        raise ValueError(f"invalid value for [{section.name}] {option}: {raw_value}") from exc


def _resolve_config_path(value: str, config_dir: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return str(path.resolve())


def load_preview_config(config_path: str, input_override: Optional[str] = None):
    """Load and validate the annotated INI configuration."""
    config_file = Path(config_path).expanduser().resolve()
    if not config_file.is_file():
        raise FileNotFoundError(f"mask config file not found: {config_file}")

    # Keep '#' available inside values so colours such as #FFFFFF work.
    # Lines beginning with '#' are still treated as normal INI comments.
    ini = configparser.ConfigParser(inline_comment_prefixes=(";",))
    loaded = ini.read(config_file, encoding="utf-8")
    if not loaded:
        raise RuntimeError(f"cannot read mask config file: {config_file}")
    required_sections = {"video", "ocr", "mask", "temporal", "preview"}
    missing_sections = sorted(required_sections.difference(ini.sections()))
    if missing_sections:
        raise ValueError(f"missing config sections: {', '.join(missing_sections)}")

    video = ini["video"]
    ocr = ini["ocr"]
    mask = ini["mask"]
    secondary_mask = (
        ini["secondary_mask"] if ini.has_section("secondary_mask") else None
    )
    temporal = ini["temporal"]
    preview = ini["preview"]
    config_dir = config_file.parent

    input_value = input_override or video.get("input", "").strip()
    if not input_value:
        raise ValueError("[video] input must be set")
    input_path = _resolve_config_path(input_value, config_dir)

    output_value = video.get("output", "").strip()
    output_path = (
        _resolve_config_path(output_value, config_dir)
        if output_value
        else f"{os.path.splitext(input_path)[0]}_mask_preview.mp4"
    )
    write_mask_video = video.getboolean("write_mask_video", fallback=True)
    duration_seconds = video.getfloat("duration_seconds", fallback=0.0)
    if duration_seconds < 0:
        raise ValueError("[video] duration_seconds must be non-negative")
    mask_output_value = video.get("mask_output", "").strip()
    mask_output_path = None
    if write_mask_video:
        mask_output_path = (
            _resolve_config_path(mask_output_value, config_dir)
            if mask_output_value
            else f"{os.path.splitext(input_path)[0]}_mask.mp4"
        )

    areas = _optional_value(ocr, "areas", _parse_areas)
    if areas is None:
        # Backward compatibility with the original single-area option.
        legacy_area = _optional_value(ocr, "area", _parse_area)
        areas = [legacy_area] if legacy_area is not None else None
    sample_step = _optional_value(ocr, "sample_step", int)
    # A value of 0 means adaptive/model default in the annotated config.
    if sample_step == 0:
        sample_step = None

    return {
        "input_path": input_path,
        "output_path": output_path,
        "mask_output_path": mask_output_path,
        "duration_seconds": duration_seconds,
        "rgb": _parse_rgb(mask.get("color", "255,255,255")),
        "areas": areas,
        "max_channel_spread": mask.getint("max_channel_spread", fallback=15),
        "core_tolerance": mask.getfloat("core_tolerance", fallback=18.0),
        "edge_tolerance": mask.getfloat("edge_tolerance", fallback=38.0),
        "box_padding": mask.getint("box_padding", fallback=4),
        "box_expand_enabled": mask.getboolean("box_expand_enabled", fallback=True),
        "box_expand_step": mask.getint("box_expand_step", fallback=6),
        "box_expand_max_x": mask.getint("box_expand_max_x", fallback=80),
        "box_expand_max_y": mask.getint("box_expand_max_y", fallback=12),
        "box_expand_min_core_pixels": mask.getint(
            "box_expand_min_core_pixels", fallback=2
        ),
        "secondary_enabled": (
            secondary_mask.getboolean("enabled", fallback=False)
            if secondary_mask is not None else False
        ),
        "secondary_left_ratio": (
            secondary_mask.getfloat("left_ratio", fallback=0.25)
            if secondary_mask is not None else 0.25
        ),
        "secondary_rgb": (
            _parse_rgb(secondary_mask.get("color", "75,75,75"))
            if secondary_mask is not None else (75, 75, 75)
        ),
        "secondary_max_channel_spread": (
            secondary_mask.getint("max_channel_spread", fallback=10)
            if secondary_mask is not None else 10
        ),
        "secondary_core_tolerance": (
            secondary_mask.getfloat("core_tolerance", fallback=32.0)
            if secondary_mask is not None else 32.0
        ),
        "secondary_edge_tolerance": (
            secondary_mask.getfloat("edge_tolerance", fallback=50.0)
            if secondary_mask is not None else 50.0
        ),
        "edge_growth_iterations": mask.getint("edge_growth_iterations", fallback=3),
        "close_kernel_size": mask.getint("close_kernel_size", fallback=3),
        "min_component_area": mask.getint("min_component_area", fallback=2),
        "isolation_distance": mask.getint("isolation_distance", fallback=4),
        "dilation_size": mask.getint("dilation_size", fallback=3),
        "dilation_iterations": mask.getint("dilation_iterations", fallback=1),
        "draw_ocr_boxes": preview.getboolean("draw_ocr_boxes", fallback=True),
        "draw_expanded_boxes": preview.getboolean(
            "draw_expanded_boxes", fallback=True
        ),
        "future_mask_frames": temporal.getint("future_mask_frames", fallback=0),
        "mask_stability_frames": temporal.getint("mask_stability_frames", fallback=0),
        "mask_stability_iou": temporal.getfloat("mask_stability_iou", fallback=0.995),
        "preserve_future_mask_continuity": temporal.getboolean(
            "preserve_future_mask_continuity", fallback=True
        ),
        "ocr_limit_side_len": _optional_value(ocr, "limit_side_len", int),
        "ocr_thresh": _optional_value(ocr, "thresh", float),
        "ocr_box_thresh": _optional_value(ocr, "box_thresh", float),
        "ocr_unclip_ratio": _optional_value(ocr, "unclip_ratio", float),
        "ocr_sample_step": sample_step,
        "ocr_crop_before_ocr": ocr.getboolean(
            "crop_before_ocr", fallback=False
        ),
        "ocr_crop_padding": ocr.getint("crop_padding", fallback=0),
        "ocr_crop_upscale": ocr.getfloat("crop_upscale", fallback=1.0),
        "ocr_crop_dedup_iou": ocr.getfloat("crop_dedup_iou", fallback=0.7),
    }


def _area_to_pixels(
    area: Tuple[float, float, float, float], width: int, height: int
) -> Tuple[int, int, int, int]:
    ymin, ymax, xmin, xmax = area
    return (
        int(ymin * height),
        int(ymax * height),
        int(xmin * width),
        int(xmax * width),
    )


def _open_writer(path: str, fps: float, size: Tuple[int, int]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        os.path.abspath(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create output video: {path}")
    return writer


def _draw_ocr_boxes(frame, boxes, colour=(0, 255, 0)):
    """Draw OCR-derived rectangles on a preview frame."""
    thickness = max(1, int(round(min(frame.shape[:2]) / 540)))
    for xmin, xmax, ymin, ymax in boxes:
        cv2.rectangle(
            frame,
            (int(xmin), int(ymin)),
            (int(xmax), int(ymax)),
            colour,
            thickness,
        )


def _mask_iou(mask_a, mask_b) -> float:
    """Calculate binary-mask intersection over union."""
    foreground_a = mask_a > 0
    foreground_b = mask_b > 0
    union = np.count_nonzero(foreground_a | foreground_b)
    if union == 0:
        return 1.0
    intersection = np.count_nonzero(foreground_a & foreground_b)
    return intersection / union


def _next_stability_run(previous_mask, current_mask, previous_run, iou_threshold):
    """Count consecutive approximately-equal, non-empty masks."""
    if not np.any(current_mask):
        return 0
    if previous_mask is None or not np.any(previous_mask):
        return 1
    if _mask_iou(previous_mask, current_mask) >= iou_threshold:
        return previous_run + 1
    return 1


def _compose_temporal_mask(
    raw_mask,
    future_masks,
    stability_run,
    stability_frames,
    previous_output_mask=None,
    preserve_continuity=True,
):
    """Combine lookahead masks while preserving already-borrowed pixels."""
    lookahead_mask = raw_mask.copy()
    for future_mask in future_masks:
        cv2.bitwise_or(lookahead_mask, future_mask, dst=lookahead_mask)

    mask = (
        lookahead_mask.copy()
        if stability_run < stability_frames
        else raw_mask.copy()
    )
    if preserve_continuity and previous_output_mask is not None:
        retained_mask = cv2.bitwise_and(previous_output_mask, lookahead_mask)
        cv2.bitwise_or(mask, retained_mask, dst=mask)
    return mask


def _write_buffer_head(
    frame_buffer,
    future_mask_frames,
    stability_frames,
    previous_output_mask,
    preserve_future_mask_continuity,
    preview_writer,
    mask_writer,
    draw_ocr_boxes,
    draw_expanded_boxes,
):
    """Write the oldest buffered frame, optionally borrowing future masks."""
    _, frame, boxes, expanded_boxes, raw_mask, stability_run = frame_buffer[0]
    future_masks = [
        future_mask
        for _, _, _, _, future_mask, _
        in list(frame_buffer)[1:future_mask_frames + 1]
    ]
    mask = _compose_temporal_mask(
        raw_mask,
        future_masks,
        stability_run,
        stability_frames,
        previous_output_mask,
        preserve_future_mask_continuity,
    )

    preview = frame.copy()
    preview[mask > 0] = 0
    if draw_expanded_boxes:
        _draw_ocr_boxes(preview, expanded_boxes, colour=(0, 165, 255))
    if draw_ocr_boxes:
        _draw_ocr_boxes(preview, boxes)
    preview_writer.write(preview)
    if mask_writer is not None:
        mask_writer.write(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
    return mask


def generate_mask_preview_video(
    input_path: str,
    output_path: str,
    mask_output_path: Optional[str],
    duration_seconds: float,
    rgb: Tuple[int, int, int],
    area: Optional[Tuple[float, float, float, float]] = None,
    areas: Optional[List[Tuple[float, float, float, float]]] = None,
    max_channel_spread: int = 15,
    core_tolerance: float = 18.0,
    edge_tolerance: float = 38.0,
    box_padding: int = 4,
    box_expand_enabled: bool = True,
    box_expand_step: int = 6,
    box_expand_max_x: int = 80,
    box_expand_max_y: int = 12,
    box_expand_min_core_pixels: int = 2,
    secondary_enabled: bool = False,
    secondary_left_ratio: float = 0.25,
    secondary_rgb: Tuple[int, int, int] = (75, 75, 75),
    secondary_max_channel_spread: int = 10,
    secondary_core_tolerance: float = 32.0,
    secondary_edge_tolerance: float = 50.0,
    edge_growth_iterations: int = 3,
    close_kernel_size: int = 3,
    min_component_area: int = 2,
    isolation_distance: int = 4,
    dilation_size: int = 3,
    dilation_iterations: int = 1,
    draw_ocr_boxes: bool = True,
    draw_expanded_boxes: bool = True,
    future_mask_frames: int = 0,
    mask_stability_frames: Optional[int] = None,
    mask_stability_iou: float = 0.995,
    preserve_future_mask_continuity: bool = True,
    ocr_limit_side_len: Optional[int] = None,
    ocr_thresh: Optional[float] = None,
    ocr_box_thresh: Optional[float] = None,
    ocr_unclip_ratio: Optional[float] = None,
    ocr_sample_step: Optional[int] = None,
    ocr_crop_before_ocr: bool = False,
    ocr_crop_padding: int = 0,
    ocr_crop_upscale: float = 1.0,
    ocr_crop_dedup_iou: float = 0.7,
):
    """Run existing OCR, blacken refined-mask pixels, and write preview videos."""
    input_path = os.path.abspath(input_path)
    cap = cv2.VideoCapture(get_readable_path(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open input video: {input_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if width <= 0 or height <= 0 or fps <= 0:
        raise RuntimeError("input video has invalid dimensions or frame rate")
    if duration_seconds < 0:
        raise ValueError("duration_seconds must be non-negative")
    max_frames = frame_count
    if duration_seconds > 0:
        max_frames = min(frame_count, max(1, int(np.ceil(duration_seconds * fps))))
        print(
            f"Preview duration: {duration_seconds:g}s -> "
            f"{max_frames} frames at {fps:g} FPS"
        )

    if areas is None and area is not None:
        # Keep direct callers using the old keyword working.
        areas = [area]
    pixel_areas = (
        [_area_to_pixels(current, width, height) for current in areas]
        if areas else []
    )
    subtitle_areas = pixel_areas or [(0, height, 0, width)]
    for index, (ratio_area, pixel_area) in enumerate(
        zip(areas or [], pixel_areas), start=1
    ):
        print(f"OCR area {index}: {ratio_area} -> {pixel_area} pixels")
    if future_mask_frames < 0:
        raise ValueError("future_mask_frames must be non-negative")
    if mask_stability_frames is None or mask_stability_frames == 0:
        mask_stability_frames = future_mask_frames
    if mask_stability_frames < 0:
        raise ValueError("mask_stability_frames must be non-negative")
    if not 0.0 <= mask_stability_iou <= 1.0:
        raise ValueError("mask_stability_iou must be in [0, 1]")
    detector = SubtitleDetect(
        input_path,
        subtitle_areas,
        det_limit_side_len=ocr_limit_side_len,
        det_thresh=ocr_thresh,
        det_box_thresh=ocr_box_thresh,
        det_unclip_ratio=ocr_unclip_ratio,
        sample_step=ocr_sample_step,
        max_frames=max_frames,
        crop_before_ocr=ocr_crop_before_ocr,
        crop_padding=ocr_crop_padding,
        crop_upscale=ocr_crop_upscale,
        crop_dedup_iou=ocr_crop_dedup_iou,
    )
    ocr_overrides = {
        "limit_side_len": ocr_limit_side_len,
        "thresh": ocr_thresh,
        "box_thresh": ocr_box_thresh,
        "unclip_ratio": ocr_unclip_ratio,
        "sample_step": ocr_sample_step,
        "crop_before_ocr": ocr_crop_before_ocr,
        "crop_padding": ocr_crop_padding,
        "crop_upscale": ocr_crop_upscale,
        "crop_dedup_iou": ocr_crop_dedup_iou,
    }
    active_overrides = {
        key: value for key, value in ocr_overrides.items() if value is not None
    }
    if active_overrides:
        print(f"OCR parameter overrides: {active_overrides}")
    if future_mask_frames:
        print(f"Future mask lookahead: {future_mask_frames} frames")
        print(
            "Mask stability cutoff: "
            f"{mask_stability_frames} frames at IoU >= {mask_stability_iou}"
        )
    boxes_by_frame = detector.find_subtitle_frame_no(_DetectionProgress())
    print(f"OCR detected subtitle boxes on {len(boxes_by_frame)} frames")

    red, green, blue = rgb
    secondary_red, secondary_green, secondary_blue = secondary_rgb
    generator = SubtitleMaskGenerator(
        SubtitleMaskConfig(
            target_bgr=(blue, green, red),
            max_channel_spread=max_channel_spread,
            core_tolerance=core_tolerance,
            edge_tolerance=edge_tolerance,
            box_padding=box_padding,
            box_expand_enabled=box_expand_enabled,
            box_expand_step=box_expand_step,
            box_expand_max_x=box_expand_max_x,
            box_expand_max_y=box_expand_max_y,
            box_expand_min_core_pixels=box_expand_min_core_pixels,
            secondary_enabled=secondary_enabled,
            secondary_left_ratio=secondary_left_ratio,
            secondary_target_bgr=(
                secondary_blue,
                secondary_green,
                secondary_red,
            ),
            secondary_max_channel_spread=secondary_max_channel_spread,
            secondary_core_tolerance=secondary_core_tolerance,
            secondary_edge_tolerance=secondary_edge_tolerance,
            edge_growth_iterations=edge_growth_iterations,
            close_kernel_size=close_kernel_size,
            min_component_area=min_component_area,
            isolation_distance=isolation_distance,
            dilation_size=dilation_size,
            dilation_iterations=dilation_iterations,
        )
    )

    cap = cv2.VideoCapture(get_readable_path(input_path))
    preview_writer = _open_writer(output_path, fps, (width, height))
    mask_writer = (
        _open_writer(mask_output_path, fps, (width, height))
        if mask_output_path
        else None
    )
    try:
        show_terminal_progress = bool(
            sys.stderr
            and hasattr(sys.stderr, "isatty")
            and sys.stderr.isatty()
        )
        with tqdm(
            total=max_frames,
            unit="frame",
            desc="Mask preview",
            mininterval=1.0,
            dynamic_ncols=True,
            disable=not show_terminal_progress,
        ) as progress:
            frame_buffer = deque()
            previous_raw_mask = None
            previous_output_mask = None
            stability_run = 0
            frame_no = 0
            while True:
                if frame_no >= max_frames:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                frame_no += 1
                boxes = boxes_by_frame.get(frame_no, [])
                raw_mask, expanded_boxes = generator.generate_with_expanded_boxes(
                    frame, boxes
                )
                stability_run = _next_stability_run(
                    previous_raw_mask,
                    raw_mask,
                    stability_run,
                    mask_stability_iou,
                )
                previous_raw_mask = raw_mask
                frame_buffer.append(
                    (
                        frame_no,
                        frame,
                        boxes,
                        expanded_boxes,
                        raw_mask,
                        stability_run,
                    )
                )
                if len(frame_buffer) > future_mask_frames:
                    previous_output_mask = _write_buffer_head(
                        frame_buffer,
                        future_mask_frames,
                        mask_stability_frames,
                        previous_output_mask,
                        preserve_future_mask_continuity,
                        preview_writer,
                        mask_writer,
                        draw_ocr_boxes,
                        draw_expanded_boxes,
                    )
                    frame_buffer.popleft()
                    progress.update(1)

            while frame_buffer:
                previous_output_mask = _write_buffer_head(
                    frame_buffer,
                    future_mask_frames,
                    mask_stability_frames,
                    previous_output_mask,
                    preserve_future_mask_continuity,
                    preview_writer,
                    mask_writer,
                    draw_ocr_boxes,
                    draw_expanded_boxes,
                )
                frame_buffer.popleft()
                progress.update(1)
    finally:
        cap.release()
        preview_writer.release()
        if mask_writer is not None:
            mask_writer.release()

    print(f"Blackened-mask preview: {output_path}")
    if mask_output_path:
        print(f"Binary mask video: {mask_output_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Preview refined subtitle masks using an annotated INI config"
    )
    parser.add_argument(
        "input", nargs="?",
        help="optional input video override; normally set [video] input in the config",
    )
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG_PATH),
        help=f"INI config path (default: {DEFAULT_CONFIG_PATH})",
    )
    return parser


def main():
    args = build_parser().parse_args()
    try:
        settings = load_preview_config(args.config, args.input)
    except (
        FileNotFoundError,
        RuntimeError,
        ValueError,
        argparse.ArgumentTypeError,
        configparser.Error,
    ) as exc:
        build_parser().error(str(exc))
    print(f"Mask config: {Path(args.config).expanduser().resolve()}")
    generate_mask_preview_video(**settings)


if __name__ == "__main__":
    main()
