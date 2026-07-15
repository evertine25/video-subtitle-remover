import unittest
import tempfile

import numpy as np

from backend.tools.subtitle_mask import SubtitleMaskConfig, SubtitleMaskGenerator
from backend.tools.subtitle_mask_video import (
    _parse_areas,
    _compose_temporal_mask,
    _next_stability_run,
)
from backend.tools.subtitle_detect import SubtitleDetect
from backend.tools.args_handler import subtitle_area_ratios_to_pixels
from backend.tools.inpaint_tools import normalize_frame_masks, union_frame_masks
from backend.tools.refined_mask_runtime import (
    EncodedMaskCache,
    RefinedMaskRuntimeConfig,
    load_refined_mask_runtime_config,
    resolve_mask_preview_output,
)


class SubtitleMaskGeneratorTest(unittest.TestCase):
    def test_mask_is_limited_to_ocr_box(self):
        frame = np.zeros((50, 80, 3), dtype=np.uint8)
        frame[20:25, 20:30] = (255, 255, 255)
        frame[20:25, 60:70] = (255, 255, 255)
        generator = SubtitleMaskGenerator(
            SubtitleMaskConfig(box_padding=0, dilation_size=1)
        )

        mask = generator.generate(frame, [(18, 32, 18, 27)])

        self.assertTrue(np.all(mask[20:25, 20:30] == 255))
        self.assertTrue(np.all(mask[20:25, 60:70] == 0))

    def test_relaxed_edge_must_connect_to_core(self):
        frame = np.zeros((40, 60, 3), dtype=np.uint8)
        frame[15:20, 20:25] = (255, 255, 255)
        edge_roi = frame[14:21, 19:26]
        edge_roi[edge_roi.sum(axis=2) == 0] = (220, 220, 220)
        frame[5:8, 5:8] = (220, 220, 220)
        generator = SubtitleMaskGenerator(
            SubtitleMaskConfig(
                core_tolerance=10,
                edge_tolerance=45,
                box_padding=0,
                edge_growth_iterations=2,
                close_kernel_size=1,
                min_component_area=1,
                dilation_size=1,
            )
        )

        stages = generator.generate_with_debug(frame, [(0, 59, 0, 39)])

        self.assertEqual(stages["core"][15, 20], 255)
        self.assertEqual(stages["edge"][14, 20], 255)
        self.assertEqual(stages["edge"][6, 6], 0)

    def test_output_is_binary_uint8(self):
        frame = np.full((20, 30, 3), 255, dtype=np.uint8)
        mask = SubtitleMaskGenerator().generate(frame, [(2, 10, 3, 12)])
        self.assertEqual(mask.dtype, np.uint8)
        self.assertTrue(set(np.unique(mask)).issubset({0, 255}))

    def test_near_grayscale_constraint_rejects_coloured_pixels(self):
        frame = np.zeros((20, 30, 3), dtype=np.uint8)
        frame[5:10, 5:10] = (200, 200, 200)
        frame[5:10, 15:20] = (190, 200, 210)
        generator = SubtitleMaskGenerator(
            SubtitleMaskConfig(
                target_bgr=(200, 200, 200),
                max_channel_spread=10,
                core_tolerance=5,
                edge_tolerance=255,
                box_padding=0,
                close_kernel_size=1,
                min_component_area=1,
                dilation_size=1,
            )
        )

        mask = generator.generate(frame, [(0, 29, 0, 19)])

        self.assertEqual(mask[7, 7], 255)
        self.assertEqual(mask[7, 17], 0)

    def test_core_pixels_bypass_near_grayscale_constraint(self):
        frame = np.zeros((20, 30, 3), dtype=np.uint8)
        frame[5:10, 5:10] = (200, 208, 196)
        generator = SubtitleMaskGenerator(
            SubtitleMaskConfig(
                target_bgr=(200, 200, 200),
                max_channel_spread=0,
                core_tolerance=30,
                edge_tolerance=40,
                box_padding=0,
                close_kernel_size=1,
                min_component_area=1,
                dilation_size=1,
            )
        )

        mask = generator.generate(frame, [(4, 10, 4, 10)])

        self.assertEqual(mask[7, 7], 255)

    def test_box_expansion_follows_nearby_punctuation_core_pixels(self):
        frame = np.zeros((30, 70, 3), dtype=np.uint8)
        frame[12:17, 10:21] = (255, 255, 255)
        frame[14:16, 24:26] = (255, 255, 255)
        frame[14:16, 30:32] = (255, 255, 255)
        generator = SubtitleMaskGenerator(
            SubtitleMaskConfig(
                core_tolerance=5,
                edge_tolerance=5,
                box_padding=0,
                box_expand_enabled=True,
                box_expand_step=6,
                box_expand_max_x=20,
                box_expand_max_y=0,
                box_expand_min_core_pixels=2,
                close_kernel_size=1,
                min_component_area=1,
                dilation_size=1,
            )
        )

        mask, expanded = generator.generate_with_expanded_boxes(
            frame, [(10, 20, 12, 16)]
        )

        self.assertEqual(expanded, [(10, 31, 12, 16)])
        self.assertEqual(mask[14, 24], 255)
        self.assertEqual(mask[14, 30], 255)

    def test_box_expansion_stops_at_empty_strip_before_distant_white_pixels(self):
        frame = np.zeros((30, 80, 3), dtype=np.uint8)
        frame[12:17, 10:21] = (255, 255, 255)
        frame[14:16, 40:42] = (255, 255, 255)
        generator = SubtitleMaskGenerator(
            SubtitleMaskConfig(
                core_tolerance=5,
                edge_tolerance=5,
                box_padding=0,
                box_expand_enabled=True,
                box_expand_step=6,
                box_expand_max_x=40,
                box_expand_max_y=0,
                box_expand_min_core_pixels=2,
                close_kernel_size=1,
                min_component_area=1,
                dilation_size=1,
            )
        )

        mask, expanded = generator.generate_with_expanded_boxes(
            frame, [(10, 20, 12, 16)]
        )

        self.assertEqual(expanded, [(10, 20, 12, 16)])
        self.assertEqual(mask[14, 40], 0)

    def test_secondary_mode_detects_without_expanding_into_gray_background(self):
        frame = np.zeros((30, 100, 3), dtype=np.uint8)
        frame[12:17, 10:21] = (75, 75, 75)
        frame[14:16, 24:26] = (75, 75, 75)
        generator = SubtitleMaskGenerator(
            SubtitleMaskConfig(
                target_bgr=(255, 255, 255),
                core_tolerance=5,
                edge_tolerance=5,
                box_padding=0,
                box_expand_enabled=True,
                box_expand_step=6,
                box_expand_max_x=20,
                box_expand_max_y=0,
                box_expand_min_core_pixels=2,
                secondary_enabled=True,
                secondary_left_ratio=0.25,
                secondary_target_bgr=(75, 75, 75),
                secondary_core_tolerance=5,
                secondary_edge_tolerance=5,
                secondary_max_channel_spread=0,
                close_kernel_size=1,
                min_component_area=1,
                dilation_size=1,
            )
        )

        mask, expanded = generator.generate_with_expanded_boxes(
            frame, [(10, 20, 12, 16)]
        )

        self.assertEqual(expanded, [(10, 20, 12, 16)])
        self.assertEqual(mask[14, 10], 255)
        self.assertEqual(mask[14, 24], 0)

    def test_secondary_mode_does_not_activate_outside_left_ratio(self):
        frame = np.zeros((30, 100, 3), dtype=np.uint8)
        frame[12:17, 60:71] = (75, 75, 75)
        generator = SubtitleMaskGenerator(
            SubtitleMaskConfig(
                target_bgr=(255, 255, 255),
                core_tolerance=5,
                edge_tolerance=5,
                box_padding=0,
                secondary_enabled=True,
                secondary_left_ratio=0.25,
                secondary_target_bgr=(75, 75, 75),
                secondary_core_tolerance=5,
                secondary_edge_tolerance=5,
                close_kernel_size=1,
                min_component_area=1,
                dilation_size=1,
            )
        )

        mask = generator.generate(frame, [(60, 70, 12, 16)])

        self.assertFalse(np.any(mask))

    def test_primary_core_pixels_prevent_secondary_mode_switch(self):
        frame = np.zeros((30, 100, 3), dtype=np.uint8)
        frame[12:17, 10:15] = (255, 255, 255)
        frame[12:17, 16:21] = (75, 75, 75)
        generator = SubtitleMaskGenerator(
            SubtitleMaskConfig(
                target_bgr=(255, 255, 255),
                core_tolerance=5,
                edge_tolerance=5,
                box_padding=0,
                secondary_enabled=True,
                secondary_left_ratio=0.25,
                secondary_target_bgr=(75, 75, 75),
                secondary_core_tolerance=5,
                secondary_edge_tolerance=5,
                close_kernel_size=1,
                min_component_area=1,
                dilation_size=1,
            )
        )

        mask = generator.generate(frame, [(10, 20, 12, 16)])

        self.assertEqual(mask[14, 12], 255)
        self.assertEqual(mask[14, 18], 0)


class SubtitleMaskStabilityTest(unittest.TestCase):
    def test_identical_non_empty_masks_accumulate_stability(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:5, 2:5] = 255

        run = _next_stability_run(None, mask, 0, 0.995)
        run = _next_stability_run(mask, mask.copy(), run, 0.995)

        self.assertEqual(run, 2)

    def test_empty_masks_do_not_become_stable(self):
        empty = np.zeros((10, 10), dtype=np.uint8)

        run = _next_stability_run(empty, empty, 20, 0.995)

        self.assertEqual(run, 0)

    def test_changed_mask_restarts_stability(self):
        previous = np.zeros((10, 10), dtype=np.uint8)
        previous[2:5, 2:5] = 255
        current = np.zeros((10, 10), dtype=np.uint8)
        current[2:5, 6:9] = 255

        run = _next_stability_run(previous, current, 10, 0.995)

        self.assertEqual(run, 1)

    def test_borrowed_future_mask_does_not_disappear_at_stability_cutoff(self):
        raw = np.zeros((10, 10), dtype=np.uint8)
        raw[2, 2] = 255
        future = np.zeros((10, 10), dtype=np.uint8)
        future[2:5, 2:5] = 255
        previous_output = future.copy()

        result = _compose_temporal_mask(
            raw_mask=raw,
            future_masks=[future],
            stability_run=8,
            stability_frames=8,
            previous_output_mask=previous_output,
            preserve_continuity=True,
        )

        self.assertTrue(np.array_equal(result, future))

    def test_stable_mask_does_not_absorb_new_following_subtitle_pixels(self):
        current = np.zeros((10, 10), dtype=np.uint8)
        current[2:5, 2:5] = 255
        following = np.zeros((10, 10), dtype=np.uint8)
        following[6:9, 6:9] = 255

        result = _compose_temporal_mask(
            raw_mask=current,
            future_masks=[following],
            stability_run=8,
            stability_frames=8,
            previous_output_mask=current,
            preserve_continuity=True,
        )

        self.assertTrue(np.array_equal(result, current))


class SubtitleSamplingBlockTest(unittest.TestCase):
    def test_sampled_box_covers_complete_block(self):
        boxes = [(10, 30, 40, 50)]

        expanded = SubtitleDetect.expand_sampled_results(
            {1: boxes, 4: boxes}, sample_step=3, max_frame_no=8
        )

        self.assertEqual(sorted(expanded), [1, 2, 3, 4, 5, 6])
        self.assertEqual(expanded[3], boxes)
        self.assertEqual(expanded[6], boxes)

    def test_last_block_is_clipped_to_video_end(self):
        boxes = [(10, 30, 40, 50)]

        expanded = SubtitleDetect.expand_sampled_results(
            {7: boxes}, sample_step=3, max_frame_no=8
        )

        self.assertEqual(sorted(expanded), [7, 8])


class SubtitleCropOcrTest(unittest.TestCase):
    def test_crop_box_is_mapped_back_to_source_frame(self):
        mapped = SubtitleDetect._map_box_from_crop(
            (20, 100, 10, 40),
            crop_xmin=100,
            crop_ymin=200,
            scale_x=2.0,
            scale_y=2.0,
            frame_width=1920,
            frame_height=1080,
        )

        self.assertEqual(mapped, (110, 150, 205, 220))

    def test_overlapping_crop_detections_are_merged(self):
        boxes = [(100, 200, 300, 340), (105, 198, 302, 339), (400, 450, 300, 340)]

        merged = SubtitleDetect._deduplicate_boxes(boxes, 0.7)

        self.assertEqual(
            merged,
            [(100, 200, 300, 340), (400, 450, 300, 340)],
        )

    def test_zero_iou_threshold_disables_deduplication(self):
        boxes = [(10, 20, 30, 40), (10, 20, 30, 40)]

        merged = SubtitleDetect._deduplicate_boxes(boxes, 0)

        self.assertEqual(merged, boxes)


class FrameWiseInpaintMaskTest(unittest.TestCase):
    def test_legacy_single_mask_is_repeated_for_every_frame(self):
        mask = np.zeros((8, 10), dtype=np.uint8)
        mask[2:4, 3:5] = 255

        masks = normalize_frame_masks(mask, 3)

        self.assertEqual(len(masks), 3)
        self.assertTrue(all(np.array_equal(current, mask) for current in masks))

    def test_frame_wise_masks_remain_distinct(self):
        first = np.zeros((8, 10), dtype=np.uint8)
        second = np.zeros((8, 10), dtype=np.uint8)
        first[2, 2] = 255
        second[5, 7] = 255

        masks = normalize_frame_masks([first, second], 2)
        union = union_frame_masks(masks)

        self.assertEqual(masks[0][2, 2], 255)
        self.assertEqual(masks[0][5, 7], 0)
        self.assertEqual(masks[1][5, 7], 255)
        self.assertEqual(union[2, 2], 255)
        self.assertEqual(union[5, 7], 255)

    def test_frame_mask_count_must_match_frame_count(self):
        mask = np.zeros((8, 10), dtype=np.uint8)

        with self.assertRaises(ValueError):
            normalize_frame_masks([mask], 2)

    def test_encoded_mask_cache_round_trip_and_sparse_empty_frame(self):
        cache = EncodedMaskCache((8, 10))
        mask = np.zeros((8, 10), dtype=np.uint8)
        mask[2:4, 3:6] = 255

        cache.put(4, mask)

        self.assertEqual(list(cache.keys()), [4])
        self.assertTrue(np.array_equal(cache.get(4), mask))
        self.assertFalse(np.any(cache.get(5)))

    def test_runtime_switch_can_fall_back_without_mask_sections(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ini", encoding="utf-8", delete=False
        ) as config_file:
            config_file.write("[integration]\nuse_refined_mask = false\n")
            config_path = config_file.name
        try:
            runtime = load_refined_mask_runtime_config(config_path)
        finally:
            import os
            os.unlink(config_path)

        self.assertFalse(runtime.enabled)


class SubtitleAreaRatioTest(unittest.TestCase):
    def test_preview_config_accepts_multiple_ratio_areas(self):
        areas = _parse_areas(
            "0.70,0.80,0.05,0.95\n0.82,0.95,0.05,0.95"
        )

        self.assertEqual(
            areas,
            [
                (0.70, 0.80, 0.05, 0.95),
                (0.82, 0.95, 0.05, 0.95),
            ],
        )

    def test_ratio_area_is_converted_using_video_dimensions(self):
        areas = subtitle_area_ratios_to_pixels(
            [(0.7667, 0.8944, 0.0583, 0.8573)],
            frame_width=1920,
            frame_height=1080,
        )

        self.assertEqual(areas, [(828, 966, 112, 1646)])

    def test_multiple_ratio_areas_are_supported(self):
        areas = subtitle_area_ratios_to_pixels(
            [(0.0, 0.5, 0.0, 0.5), (0.5, 1.0, 0.5, 1.0)],
            frame_width=100,
            frame_height=80,
        )

        self.assertEqual(areas, [(0, 40, 0, 50), (40, 80, 50, 100)])

    def test_invalid_ratio_area_is_rejected(self):
        with self.assertRaises(ValueError):
            subtitle_area_ratios_to_pixels(
                [(0.9, 0.2, 0.1, 0.8)],
                frame_width=1920,
                frame_height=1080,
            )


class IntegratedMaskPreviewTest(unittest.TestCase):
    def test_default_preview_path_is_derived_from_input_video(self):
        runtime = RefinedMaskRuntimeConfig(
            enabled=True,
            mask=SubtitleMaskConfig(),
            write_mask_preview=True,
        )

        output = resolve_mask_preview_output(
            "D:/video/story.mp4", runtime
        ).replace("\\", "/")

        self.assertTrue(output.endswith("/video/story_mask_preview.mp4"))


if __name__ == "__main__":
    unittest.main()
