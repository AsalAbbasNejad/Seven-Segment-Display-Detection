# Seven-Segment Display Detection

This project solves an image-processing challenge where an equation is displayed using seven-segment displays.  
The goal is to detect the displayed equation, identify the active segments of each display, and determine which segment is broken.

## Task Overview

Each input is a `3000x3000` RGB PNG image containing up to 16 seven-segment displays.  
The display may be rotated, shifted, or placed on different backgrounds.

Exactly one segment in the image is broken:

- **Dead segment**: a segment that should be on but is off
- **Shorted segment**: a segment that should be off but is on

The equation is written in hexadecimal format and may contain:

- Digits: `0-9`
- Hex letters: `A, b, C, d, E, F`
- Operators: `+`, `-`, `*`, `=`

## Output Format

The program prints three lines:

```text
<number_of_displays>
<active_segments_for_each_display>
<broken_display_index> <broken_segment>
