"""Generate a pixel-level subtitle mask inside OCR text boxes.

The OCR detector is deliberately kept separate from this module.  Callers pass
the current BGR frame and the OCR boxes, and receive a binary uint8 mask whose
foreground value is 255.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np


SubtitleBox = Tuple[int, int, int, int]  # xmin, xmax, ymin, ymax


@dataclass(frozen=True)
class SubtitleMaskConfig:
    """Parameters used to refine an OCR rectangle into a subtitle mask.

    ``target_bgr`` follows OpenCV's channel order.  ``core_tolerance`` and
    ``edge_tolerance`` are distances in OpenCV's 8-bit Lab colour space.
    """

    target_bgr: Tuple[int, int, int] = (255, 255, 255)
    # Maximum difference between the brightest and darkest BGR channels.
    # A negative value disables the near-grayscale constraint.
    max_channel_spread: int = -1
    core_tolerance: float = 18.0
    edge_tolerance: float = 38.0
    box_padding: int = 4
    box_expand_enabled: bool = False
    box_expand_step: int = 6
    box_expand_max_x: int = 80
    box_expand_max_y: int = 12
    box_expand_min_core_pixels: int = 2
    secondary_enabled: bool = False
    secondary_left_ratio: float = 0.25
    secondary_target_bgr: Tuple[int, int, int] = (75, 75, 75)
    secondary_max_channel_spread: int = 10
    secondary_core_tolerance: float = 32.0
    secondary_edge_tolerance: float = 50.0
    edge_growth_iterations: int = 3
    close_kernel_size: int = 3
    min_component_area: int = 2
    isolation_distance: int = 4
    dilation_size: int = 3
    dilation_iterations: int = 1

    def __post_init__(self):
        if len(self.target_bgr) != 3 or any(not 0 <= value <= 255 for value in self.target_bgr):
            raise ValueError("target_bgr must contain three values in [0, 255]")
        if self.core_tolerance < 0:
            raise ValueError("core_tolerance must be non-negative")
        if not -1 <= self.max_channel_spread <= 255:
            raise ValueError("max_channel_spread must be in [-1, 255]")
        if self.edge_tolerance < self.core_tolerance:
            raise ValueError("edge_tolerance must be greater than or equal to core_tolerance")
        if (
            len(self.secondary_target_bgr) != 3
            or any(not 0 <= value <= 255 for value in self.secondary_target_bgr)
        ):
            raise ValueError("secondary_target_bgr must contain three values in [0, 255]")
        if not 0.0 <= self.secondary_left_ratio <= 1.0:
            raise ValueError("secondary_left_ratio must be in [0, 1]")
        if not -1 <= self.secondary_max_channel_spread <= 255:
            raise ValueError("secondary_max_channel_spread must be in [-1, 255]")
        if self.secondary_core_tolerance < 0:
            raise ValueError("secondary_core_tolerance must be non-negative")
        if self.secondary_edge_tolerance < self.secondary_core_tolerance:
            raise ValueError(
                "secondary_edge_tolerance must be greater than or equal to "
                "secondary_core_tolerance"
            )
        for name in (
            "box_padding",
            "box_expand_step",
            "box_expand_max_x",
            "box_expand_max_y",
            "box_expand_min_core_pixels",
            "edge_growth_iterations",
            "close_kernel_size",
            "min_component_area",
            "isolation_distance",
            "dilation_size",
            "dilation_iterations",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.box_expand_step == 0:
            raise ValueError("box_expand_step must be greater than zero")
        if self.box_expand_min_core_pixels == 0:
            raise ValueError("box_expand_min_core_pixels must be greater than zero")


class SubtitleMaskGenerator:
    """Create precise subtitle masks from frames and OCR rectangles."""

    def __init__(self, config: SubtitleMaskConfig = None):
        self.config = config or SubtitleMaskConfig()
        target = np.array([[self.config.target_bgr]], dtype=np.uint8)
        self._target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
        secondary_target = np.array(
            [[self.config.secondary_target_bgr]], dtype=np.uint8
        )
        self._secondary_target_lab = cv2.cvtColor(
            secondary_target, cv2.COLOR_BGR2LAB
        )[0, 0].astype(np.float32)

    def generate(self, frame: np.ndarray, boxes: Iterable[SubtitleBox]) -> np.ndarray:
        """Return a full-frame binary mask for ``frame``.

        Args:
            frame: OpenCV BGR image with shape ``(height, width, 3)``.
            boxes: OCR boxes in ``(xmin, xmax, ymin, ymax)`` order.
        """
        mask, _ = self.generate_with_expanded_boxes(frame, boxes)
        return mask

    def generate_with_expanded_boxes(
        self, frame: np.ndarray, boxes: Iterable[SubtitleBox]
    ):
        """Return the final mask together with colour-expanded OCR boxes."""
        prepared_boxes = self._prepare_boxes(frame, boxes)
        expanded_boxes = [box for box, _ in prepared_boxes]
        return self._generate_stages(frame, prepared_boxes)["final"], expanded_boxes

    def generate_with_debug(
        self, frame: np.ndarray, boxes: Iterable[SubtitleBox]
    ) -> Dict[str, np.ndarray]:
        """Return full-frame ``core``, ``edge``, ``cleaned`` and ``final`` masks."""
        return self._generate_stages(frame, self._prepare_boxes(frame, boxes))

    def expand_boxes(
        self, frame: np.ndarray, boxes: Iterable[SubtitleBox]
    ) -> List[SubtitleBox]:
        """Iteratively grow OCR boxes toward nearby core-colour pixels."""
        return [box for box, _ in self._prepare_boxes(frame, boxes)]

    def _prepare_boxes(self, frame, boxes):
        """Normalize, select a colour mode, and expand each OCR box."""
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a BGR numpy array with shape (H, W, 3)")
        height, width = frame.shape[:2]
        prepared_boxes = []
        for box in boxes or []:
            normalized = self._normalized_box(box, width, height)
            if normalized is None:
                continue
            use_secondary = self._should_use_secondary(
                frame, normalized, width
            )
            target_lab, core_tolerance, _, _ = self._mode_parameters(use_secondary)
            expanded = (
                self._expand_box(frame, normalized, target_lab, core_tolerance)
                if self.config.box_expand_enabled and not use_secondary
                else normalized
            )
            prepared_boxes.append((expanded, use_secondary))
        return prepared_boxes

    def _should_use_secondary(self, frame, box, frame_width):
        if not self.config.secondary_enabled:
            return False
        x1, x2, y1, y2 = box
        box_center_x = (x1 + x2) / 2.0
        if box_center_x > frame_width * self.config.secondary_left_ratio:
            return False
        roi = frame[y1:y2 + 1, x1:x2 + 1]
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
        primary_distance = np.linalg.norm(lab - self._target_lab, axis=2)
        return not np.any(primary_distance <= self.config.core_tolerance)

    def _mode_parameters(self, use_secondary):
        if use_secondary:
            return (
                self._secondary_target_lab,
                self.config.secondary_core_tolerance,
                self.config.secondary_edge_tolerance,
                self.config.secondary_max_channel_spread,
            )
        return (
            self._target_lab,
            self.config.core_tolerance,
            self.config.edge_tolerance,
            self.config.max_channel_spread,
        )

    @staticmethod
    def _normalized_box(box, frame_width, frame_height):
        if len(box) != 4:
            raise ValueError("each OCR box must be (xmin, xmax, ymin, ymax)")
        xmin, xmax, ymin, ymax = (int(value) for value in box)
        x1 = max(0, min(xmin, xmax))
        x2 = min(frame_width - 1, max(xmin, xmax))
        y1 = max(0, min(ymin, ymax))
        y2 = min(frame_height - 1, max(ymin, ymax))
        if x1 > x2 or y1 > y2:
            return None
        return x1, x2, y1, y2

    def _expand_box(
        self,
        frame: np.ndarray,
        box: SubtitleBox,
        target_lab: np.ndarray,
        core_tolerance: float,
    ) -> SubtitleBox:
        height, width = frame.shape[:2]
        original_x1, original_x2, original_y1, original_y2 = box
        max_x = self.config.box_expand_max_x
        max_y = self.config.box_expand_max_y
        search_x1 = max(0, original_x1 - max_x)
        search_x2 = min(width - 1, original_x2 + max_x)
        search_y1 = max(0, original_y1 - max_y)
        search_y2 = min(height - 1, original_y2 + max_y)
        search_roi = frame[search_y1:search_y2 + 1, search_x1:search_x2 + 1]
        lab = cv2.cvtColor(search_roi, cv2.COLOR_BGR2LAB).astype(np.float32)
        core_map = np.linalg.norm(lab - target_lab, axis=2) <= core_tolerance

        x1 = original_x1 - search_x1
        x2 = original_x2 - search_x1
        y1 = original_y1 - search_y1
        y2 = original_y2 - search_y1
        min_pixels = self.config.box_expand_min_core_pixels
        step = max(1, self.config.box_expand_step)

        # Grow left and right independently. Each strip spans the current text
        # height, which keeps horizontal growth aligned with the subtitle row.
        left_limit = original_x1 - search_x1
        left_limit = max(0, left_limit - max_x)
        while x1 > left_limit:
            strip_x1 = max(left_limit, x1 - step)
            strip = core_map[y1:y2 + 1, strip_x1:x1]
            ys, xs = np.nonzero(strip)
            if len(xs) < min_pixels:
                break
            x1 = strip_x1 + int(xs.min())

        right_limit = min(core_map.shape[1] - 1, x2 + max_x)
        while x2 < right_limit:
            strip_x2 = min(right_limit, x2 + step)
            strip = core_map[y1:y2 + 1, x2 + 1:strip_x2 + 1]
            ys, xs = np.nonzero(strip)
            if len(xs) < min_pixels:
                break
            x2 = x2 + 1 + int(xs.max())

        # Vertical growth uses the horizontally-expanded width but typically a
        # much smaller maximum distance to avoid neighbouring UI elements.
        top_limit = max(0, original_y1 - search_y1 - max_y)
        while y1 > top_limit:
            strip_y1 = max(top_limit, y1 - step)
            strip = core_map[strip_y1:y1, x1:x2 + 1]
            ys, xs = np.nonzero(strip)
            if len(ys) < min_pixels:
                break
            y1 = strip_y1 + int(ys.min())

        bottom_limit = min(core_map.shape[0] - 1, original_y2 - search_y1 + max_y)
        while y2 < bottom_limit:
            strip_y2 = min(bottom_limit, y2 + step)
            strip = core_map[y2 + 1:strip_y2 + 1, x1:x2 + 1]
            ys, xs = np.nonzero(strip)
            if len(ys) < min_pixels:
                break
            y2 = y2 + 1 + int(ys.max())

        return (
            x1 + search_x1,
            x2 + search_x1,
            y1 + search_y1,
            y2 + search_y1,
        )

    def _generate_stages(
        self, frame: np.ndarray, prepared_boxes
    ) -> Dict[str, np.ndarray]:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a BGR numpy array with shape (H, W, 3)")

        height, width = frame.shape[:2]
        stages = {
            name: np.zeros((height, width), dtype=np.uint8)
            for name in ("core", "edge", "cleaned", "final")
        }

        for box, use_secondary in prepared_boxes or []:
            bounds = self._padded_bounds(box, width, height)
            if bounds is None:
                continue
            x1, x2, y1, y2 = bounds
            roi_stages = self._refine_roi(
                frame[y1:y2, x1:x2], use_secondary
            )
            for name, roi_mask in roi_stages.items():
                target = stages[name][y1:y2, x1:x2]
                np.maximum(target, roi_mask, out=target)

        return stages

    def _padded_bounds(
        self, box: Sequence[int], frame_width: int, frame_height: int
    ):
        if len(box) != 4:
            raise ValueError("each OCR box must be (xmin, xmax, ymin, ymax)")
        xmin, xmax, ymin, ymax = (int(value) for value in box)
        padding = self.config.box_padding
        x1 = max(0, min(xmin, xmax) - padding)
        x2 = min(frame_width, max(xmin, xmax) + padding + 1)
        y1 = max(0, min(ymin, ymax) - padding)
        y2 = min(frame_height, max(ymin, ymax) + padding + 1)
        if x1 >= x2 or y1 >= y2:
            return None
        return x1, x2, y1, y2

    def _refine_roi(
        self, roi: np.ndarray, use_secondary: bool = False
    ) -> Dict[str, np.ndarray]:
        target_lab, core_tolerance, edge_tolerance, max_channel_spread = (
            self._mode_parameters(use_secondary)
        )
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
        distance = np.linalg.norm(lab - target_lab, axis=2)
        if max_channel_spread >= 0:
            channel_spread = np.ptp(roi.astype(np.int16), axis=2)
            near_grayscale = channel_spread <= max_channel_spread
        else:
            near_grayscale = np.ones(roi.shape[:2], dtype=bool)
        # High-confidence core pixels are accepted solely by Lab distance.
        # The near-grayscale constraint is only a guard for lower-confidence
        # edge candidates, so compression-induced colour shifts do not punch
        # holes in bright subtitle cores.
        core = distance <= core_tolerance
        relaxed = core | (
            (distance <= edge_tolerance) & near_grayscale
        )

        # Morphological reconstruction: relaxed pixels are accepted only when
        # they can be reached from a high-confidence subtitle-colour pixel.
        edge = core.copy()
        growth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        for _ in range(self.config.edge_growth_iterations):
            grown = cv2.dilate(edge.astype(np.uint8), growth_kernel, iterations=1).astype(bool)
            reconstructed = edge | (grown & relaxed)
            if np.array_equal(reconstructed, edge):
                break
            edge = reconstructed

        cleaned = edge.astype(np.uint8) * 255
        close_size = self._odd_kernel_size(self.config.close_kernel_size)
        if close_size > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        cleaned = self._remove_isolated_components(cleaned)

        final = cleaned.copy()
        dilation_size = self._odd_kernel_size(self.config.dilation_size)
        if dilation_size > 1 and self.config.dilation_iterations > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
            final = cv2.dilate(final, kernel, iterations=self.config.dilation_iterations)

        return {
            "core": core.astype(np.uint8) * 255,
            "edge": edge.astype(np.uint8) * 255,
            "cleaned": cleaned,
            "final": final,
        }

    def _remove_isolated_components(self, mask: np.ndarray) -> np.ndarray:
        min_area = self.config.min_component_area
        if min_area <= 1 or not np.any(mask):
            return mask

        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count <= 1:
            return mask

        large_labels: List[int] = [
            label
            for label in range(1, count)
            if stats[label, cv2.CC_STAT_AREA] >= min_area
        ]
        # If everything is small, retaining it is safer than deleting thin
        # punctuation or disconnected strokes.
        if not large_labels:
            return mask

        large_mask = np.isin(labels, large_labels).astype(np.uint8)
        result = large_mask.copy()
        distance = self.config.isolation_distance
        if distance > 0:
            size = self._odd_kernel_size(distance * 2 + 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            near_large = cv2.dilate(large_mask, kernel, iterations=1).astype(bool)
            for label in range(1, count):
                if label in large_labels:
                    continue
                component = labels == label
                if np.any(near_large & component):
                    result[component] = 1
        return result.astype(np.uint8) * 255

    @staticmethod
    def _odd_kernel_size(size: int) -> int:
        if size <= 1:
            return size
        return size if size % 2 == 1 else size + 1
