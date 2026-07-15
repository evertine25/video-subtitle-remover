# Video Subtitle Remover: Refined Subtitle Mask Edition

[![License](https://img.shields.io/badge/License-Apache%202-red.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)
![Supported OS](https://img.shields.io/badge/OS-Windows%2FmacOS%2FLinux-green.svg)
[![Docker](https://img.shields.io/badge/Docker-Image-blue?logo=docker)](https://hub.docker.com/r/eritpchy/video-subtitle-remover)

> This project is a fork of [Video-subtitle-remover (VSR)](https://github.com/YaoFANGUK/video-subtitle-remover). It is primarily optimized for specific video types, such as 2D animation and mobile-game story videos, and may not generalize well to all scenes.

## Overview

Based on the original project's OCR subtitle detection and video inpainting pipeline, this fork adds refined subtitle-mask generation and preview features:

- **Refined subtitle detection**: The original project generates a rectangular mask from each OCR detection box, which may remove a significant amount of background detail during inpainting. For scenes where subtitle and background colors are clearly different, this fork performs pixel-color classification inside the OCR boxes to generate a mask that follows the subtitle more closely.
- **OCR miss recovery**: OCR may perform poorly on punctuation such as ellipses, question marks, exclamation marks, and dashes. The OCR boxes can be expanded in four directions to capture subtitle pixels that were not covered by OCR.
- **Subtitle detection preview**: Before running video inpainting, you can output a preview video with detected mask regions painted black. This makes it easier to inspect the mask and tune the parameters.
- **Inpainting with refined masks**: The refined masks can be used in the original inpainting pipeline with `sttn-det`, `lama`, and `propainter` modes.

In current testing, LAMA provides a noticeable improvement in some scenes. ProPainter may still leave subtitle artifacts in certain cases, while STTN has not yet been tested extensively.

## Recommended Use Cases

This fork is best suited to videos that satisfy both of the following conditions:

- **The background contains substantial detail**, such as detailed 2D character artwork or scene images with limited motion;
- **The subtitle color is clearly different from the background color**.

If the subtitle color is close to the background, refined pixel-color masks may perform worse than the original rectangular masks. For example, white subtitles with black outlines may be difficult to distinguish from a large white region in the background.

The current implementation supports only one subtitle color at a time. Videos containing subtitles with multiple primary colors are not currently supported. If your video does not meet these conditions, consider using the original project's subtitle-removal pipeline.

## Usage

### Installation

Follow the installation instructions in the original project's README to install Python, the required models, and the appropriate runtime environment. A virtual environment is recommended.

### 1. Preview Subtitle Detection

Before removing subtitles, test the detection result with a short clip of approximately 10 seconds. The preview script paints detected mask regions black and writes a preview video:

```bash
python -m backend.tools.subtitle_mask_video "./yourinput.mp4"
```

Use the preview to check whether:

- the subtitles are fully covered;
- unrelated background or character artwork is being covered;
- OCR box expansion is sufficient;
- the subtitle color and tolerance settings are appropriate.

### 2. Tune the Subtitle-Mask Parameters

All refined-mask parameters are stored in `subtitle_mask_config.ini`. The goal is to make the mask **cover the subtitles completely while avoiding unnecessary background regions**.

#### Video Input

- `input`: Video path used only by the subtitle preview script. Once configured, the input path can be omitted from the preview command.
- `duration_seconds`: Only process the first specified number of seconds for quick testing. Set it to `0` to process the complete video.

#### OCR Parameters

- `areas`: OCR subtitle-detection regions. Subtitle detection is not performed outside these regions; multiple regions are supported.
- `thresh`: Text-pixel probability threshold. Lowering it can reduce missed subtitles, but may increase OCR false positives.
- `box_thresh`: Average text-box confidence threshold. Lowering it can reduce missed subtitles, but may increase OCR false positives.

Because the final refined mask is further filtered by pixel color inside the OCR boxes, an OCR false positive does not necessarily become a false-positive mask. For subtitles that are frequently missed, try lowering `thresh` and `box_thresh` moderately.

#### Pixel-Color Subtitle Detection

Subtitles usually contain a core color and an edge-transition color. Video compression may make edge pixels substantially different from the core color, so separate tolerances are used:

- `color`: Subtitle core color, in `R,G,B` or `#RRGGBB` format;
- `core_tolerance`: Color-distance threshold for subtitle core pixels;
- `edge_tolerance`: Color-distance threshold for transition pixels between the subtitle and background;
- `max_channel_spread`: Approximate grayscale constraint for black-and-white subtitles. Set it to `-1` to disable the constraint and allow color subtitles.

For outlined subtitles, edge-color detection may be less useful. You can try setting `edge_tolerance` equal to `core_tolerance` and adjust the mask dilation parameters to cover the outline.

#### Mask Dilation

The initial mask generated by pixel-color detection may not fully cover subtitle edges, especially when the subtitles contain outlines, anti-aliased pixels, or semi-transparent pixels. The following parameters can expand the final mask:

- `dilation_size`: Size of the structuring element used for mask dilation. Set it to `0` or `1` to disable dilation; common values are `3` and `5`. Larger values expand the mask farther outward, but may also cover more background.
- `dilation_iterations`: Number of times mask dilation is applied. Set it to `0` to disable dilation; larger values generally produce a wider mask.

Adjust `dilation_size` first, then tune `dilation_iterations` based on the preview. Increase these values if subtitle edges remain visible; decrease them if character artwork or background details are being covered. Dilation can only expand pixels already included by color detection or OCR boxes; it cannot recover subtitle regions that were missed entirely.

#### OCR Box Expansion

OCR boxes may not fully cover small or isolated characters such as ellipses and dashes. The following parameters search for missed subtitle pixels around each OCR box:

- `box_expand_step`: Width, in pixels, of each expansion band in the four directions;
- `box_expand_max_x`: Maximum horizontal expansion for one OCR box;
- `box_expand_max_y`: Maximum vertical expansion for one OCR box.

#### Temporal Optimization

- `future_mask_frames`: Merges masks from the next N frames into the current frame. This is useful when subtitles fade in or appear gradually. Excessive values may cause the next subtitle line to be included too early.

### 3. Remove the Subtitles

After obtaining a satisfactory preview, make sure refined-mask processing is enabled in `subtitle_mask_config.ini`:

```ini
[integration]
use_refined_mask = true
```

To write a black-mask preview video during the removal process, set:

```ini
write_mask_preview = true
```

Then run the inpainting pipeline:

```powershell
python backend/main.py `
  --input "yourinput.mp4" `
  --output "youroutput.mp4" `
  --inpaint-mode lama `
  -r 0.7667 0.8944 0.0583 0.3500 `
  -r 0.7667 0.8944 0.3100 0.6200 `
  -r 0.7667 0.8944 0.5800 0.8573
```

Notes:

- `-r` specifies a subtitle region in the order `ymin ymax xmin xmax`, with values normalized to `0~1`;
- pass `-r` multiple times to specify multiple subtitle regions;
- subtitle regions must be provided manually through the command line for the final removal process; changing `areas` in the INI file does not affect this process;
- `lama` is currently the recommended mode to try first. ProPainter may leave subtitle artifacts in some scenes, and STTN has not yet been tested extensively;
- when `use_refined_mask` is disabled, the pipeline falls back to the original OCR rectangular-mask workflow.

## Limitations

The additional features in this fork are primarily intended for a personal project and specific game-story videos. General-purpose performance is limited, and some features are still under development. Results may vary depending on subtitle style, background complexity, character motion, and particle effects.

The additional code was generated with assistance from GPT-5.6 and is intended for personal use.
