import argparse
from enum import Enum

from .constant import InpaintMode


def subtitle_area_ratios_to_pixels(areas, frame_width, frame_height):
    """Convert normalized (ymin, ymax, xmin, xmax) areas to pixel areas."""
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("video dimensions must be positive")
    pixel_areas = []
    for area in areas or []:
        if len(area) != 4:
            raise ValueError("each subtitle area must contain four ratios")
        ymin, ymax, xmin, xmax = (float(value) for value in area)
        if any(value < 0.0 or value > 1.0 for value in (ymin, ymax, xmin, xmax)):
            raise ValueError("subtitle area ratios must be in [0, 1]")
        if ymin >= ymax or xmin >= xmax:
            raise ValueError(
                "subtitle area ratios must satisfy ymin < ymax and xmin < xmax"
            )
        pixel_areas.append(
            (
                max(0, min(frame_height, round(ymin * frame_height))),
                max(0, min(frame_height, round(ymax * frame_height))),
                max(0, min(frame_width, round(xmin * frame_width))),
                max(0, min(frame_width, round(xmax * frame_width))),
            )
        )
    return pixel_areas

def parse_args():
    parser = argparse.ArgumentParser(
        description="Video Subtitle Remover Command Line Tool"
    )
    parser.add_argument(
        "--input", "-i", required=True, type=str,
        help="Input video file path"
    )
    parser.add_argument(
        "--output", "-o", required=False, type=str, default=None,
        help="Output video file path (optional)"
    )
    area_group = parser.add_mutually_exclusive_group()
    area_group.add_argument(
        "--subtitle-area-coords", "-c", action="append", nargs=4, type=int, metavar=("YMIN", "YMAX", "XMIN", "XMAX"),
        help="Subtitle area coordinates (ymin ymax xmin xmax). Can be specified multiple times for multiple areas."
    )
    area_group.add_argument(
        "--subtitle-area-ratios", "-r", action="append", nargs=4, type=float,
        metavar=("YMIN", "YMAX", "XMIN", "XMAX"),
        help=(
            "Normalized subtitle area ratios in [0,1] (ymin ymax xmin xmax). "
            "Can be specified multiple times for multiple areas."
        ),
    )
    parser.add_argument(
        "--inpaint-mode", type=str, default="sttn-auto",
        choices=[mode.name.lower().replace('_','-') for mode in InpaintMode],
        help="Inpaint mode, default is sttn-auto"
    )
    args = parser.parse_args()
    args.inpaint_mode = InpaintMode[args.inpaint_mode.replace('-','_').upper()]
    if args.subtitle_area_coords is None:
        args.subtitle_area_coords = []
    if args.subtitle_area_ratios is None:
        args.subtitle_area_ratios = []
    for area in args.subtitle_area_ratios:
        ymin, ymax, xmin, xmax = area
        if any(value < 0.0 or value > 1.0 for value in area):
            parser.error("--subtitle-area-ratios values must be in [0, 1]")
        if ymin >= ymax or xmin >= xmax:
            parser.error(
                "--subtitle-area-ratios must satisfy ymin < ymax and xmin < xmax"
            )
    return args
