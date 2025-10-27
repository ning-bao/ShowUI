# Coordinate Format Guide

## Overview

As of version 2.1, the annotation pipeline uses **absolute pixel coordinates** instead of normalized (0-1) coordinates.

## Why the Change?

1. **Accuracy**: GPT models can make mistakes when normalizing coordinates
2. **Clarity**: Pixel coordinates are more intuitive and easier to verify
3. **ShowUI RL**: Aligns with ShowUI RL training requirements
4. **Precision**: Eliminates floating-point rounding errors

## Format Specification

### Absolute Pixel Format (Current - v2.1+)

```json
{
  "img_size": [1920, 1080],
  "element": [
    {
      "instruction": "Click the submit button",
      "bbox": [100, 200, 300, 250],
      "point": [200, 225]
    }
  ]
}
```

**Key Points:**
- All coordinates are **integers** in pixel space
- `bbox`: `[x1_px, y1_px, x2_px, y2_px]`
- `point`: `[cx_px, cy_px]` (center point)
- Bounds: `0 ≤ x1 < x2 ≤ width`, `0 ≤ y1 < y2 ≤ height`
- Point must be strictly inside bbox: `x1 < cx < x2` and `y1 < cy < y2`

### Legacy Normalized Format (v1.0 - v2.0)

```json
{
  "img_size": [1920, 1080],
  "element": [
    {
      "instruction": "Click the submit button",
      "bbox": [0.052, 0.185, 0.156, 0.231],
      "point": [0.104, 0.208]
    }
  ]
}
```

**Key Points:**
- All coordinates are **floats** between 0.0 and 1.0
- `bbox`: `[x1/width, y1/height, x2/width, y2/height]`
- `point`: `[cx/width, cy/height]`

## Backward Compatibility

The system **automatically detects** which format is being used:

### Detection Logic

```python
# If all coordinates are in range [0, 1], assume normalized
if all(0 <= coord <= 1 for coord in bbox + point):
    # Convert to pixels
    x1 = bbox[0] * width
    y1 = bbox[1] * height
    # ...
else:
    # Use as absolute pixels
    x1 = bbox[0]
    y1 = bbox[1]
    # ...
```

This means:
- ✅ Old normalized annotations still work
- ✅ New pixel annotations work natively
- ✅ Mixed formats in same folder work
- ✅ No migration required

## OpenAI API Prompt

The prompt automatically injects image dimensions:

```xml
<img_size>[{width}, {height}]</img_size>
```

This tells GPT-4o:
- Exact image dimensions
- To use absolute pixel coordinates
- To validate bounds against image size

## Examples

### Example 1: Button

**Image:** 1920×1080 pixels  
**Button Location:** Top-right corner, ~200×50 pixels

```json
{
  "instruction": "Click the Settings button (gear icon)",
  "bbox": [1650, 20, 1850, 70],
  "point": [1750, 45]
}
```

**Verification:**
- Width: 1850 - 1650 = 200px ✓
- Height: 70 - 20 = 50px ✓
- Center: ((1650+1850)/2, (20+70)/2) = (1750, 45) ✓
- In bounds: 0 < x < 1920, 0 < y < 1080 ✓

### Example 2: Input Field

**Image:** 800×600 pixels  
**Input Field:** Center of screen, ~300×40 pixels

```json
{
  "instruction": "Type your email address in the input field",
  "bbox": [250, 280, 550, 320],
  "point": [400, 300]
}
```

**Verification:**
- Width: 550 - 250 = 300px ✓
- Height: 320 - 280 = 40px ✓
- Center: ((250+550)/2, (280+320)/2) = (400, 300) ✓
- In bounds: 0 < x < 800, 0 < y < 600 ✓

## Converting Between Formats

### Normalized → Absolute Pixels

```python
def normalize_to_pixels(bbox_norm, point_norm, img_width, img_height):
    bbox_px = [
        int(bbox_norm[0] * img_width),   # x1
        int(bbox_norm[1] * img_height),  # y1
        int(bbox_norm[2] * img_width),   # x2
        int(bbox_norm[3] * img_height)   # y2
    ]
    point_px = [
        int(point_norm[0] * img_width),  # cx
        int(point_norm[1] * img_height)  # cy
    ]
    return bbox_px, point_px
```

### Absolute Pixels → Normalized

```python
def pixels_to_normalized(bbox_px, point_px, img_width, img_height):
    bbox_norm = [
        bbox_px[0] / img_width,   # x1
        bbox_px[1] / img_height,  # y1
        bbox_px[2] / img_width,   # x2
        bbox_px[3] / img_height   # y2
    ]
    point_norm = [
        point_px[0] / img_width,  # cx
        point_px[1] / img_height  # cy
    ]
    return bbox_norm, point_norm
```

## Validation Rules

The system enforces these rules for absolute pixel coordinates:

1. **Integer Values**: All coordinates must be integers
   ```python
   assert all(isinstance(c, int) for c in bbox + point)
   ```

2. **Bounds Check**: Coordinates must be within image dimensions
   ```python
   assert 0 <= bbox[0] < bbox[2] <= width
   assert 0 <= bbox[1] < bbox[3] <= height
   ```

3. **Point Inside BBox**: Center point must be strictly inside bbox
   ```python
   assert bbox[0] < point[0] < bbox[2]
   assert bbox[1] < point[1] < bbox[3]
   ```

4. **Minimum Size**: Prefer minimum size ≥ 8×8 pixels
   ```python
   width = bbox[2] - bbox[0]
   height = bbox[3] - bbox[1]
   assert width >= 8 and height >= 8  # recommended
   ```

## Frontend Handling

The frontend automatically scales coordinates for display:

```javascript
// Get image's original size
const origWidth = currentAnnotation.img_size[0];
const origHeight = currentAnnotation.img_size[1];

// Get canvas display size
const displayWidth = canvas.width;
const displayHeight = canvas.height;

// Calculate scale factors
const scaleX = displayWidth / origWidth;
const scaleY = displayHeight / origHeight;

// Scale absolute pixel coordinates
const x1_display = bbox[0] * scaleX;
const y1_display = bbox[1] * scaleY;
```

This ensures annotations display correctly regardless of zoom level.

## Migration Notes

### For Users

No action required! The system automatically handles both formats.

### For Developers

If you're building tools that consume these annotations:

1. **Check format first**: Test if coordinates are 0-1 or >1
2. **Use img_size**: Always reference the img_size field
3. **Validate bounds**: Ensure coordinates are within image dimensions
4. **Handle both**: Support both formats for maximum compatibility

## ShowUI RL Integration

The ShowUI RL prompt enforces these rules:

- Outputs 1-5 unique elements per image
- Uses absolute pixel coordinates
- Validates all constraints
- Sorts elements top-to-bottom, left-to-right
- Ensures uniqueness (no duplicates)
- Instructions ≤120 characters
- Actionable, specific, imperative style

## Troubleshooting

### Issue: Annotations appear at wrong locations

**Cause:** System may be misdetecting coordinate format

**Solution:** Ensure your coordinates are clearly absolute (>1) or normalized (0-1). Avoid edge cases like small images where pixel coords might be <1.

### Issue: Validation errors when pasting JSON

**Cause:** Coordinates in wrong format

**Solution:** Check that:
- bbox has 4 integers/floats
- point has 2 integers/floats
- All values are numbers
- Bounds are valid

### Issue: Legacy annotations not displaying

**Cause:** Rare edge case in format detection

**Solution:** The visualizer checks if ALL coordinates are ≤1 to detect normalized format. If you have a 100×100 pixel image with absolute coords, it may be confused. Solution: always include img_size in your annotations.

## Summary

| Aspect | Normalized (v1-v2) | Absolute Pixels (v2.1+) |
|--------|-------------------|-------------------------|
| Format | 0.0 - 1.0 floats | Integer pixels |
| Precision | ~3 decimal places | Exact pixels |
| Example bbox | [0.1, 0.2, 0.3, 0.4] | [192, 216, 576, 432] |
| Current default | ❌ Legacy | ✅ Active |
| Supported | ✅ Backward compat | ✅ Primary |

---

**Version:** 2.1  
**Date:** October 26, 2025  
**Status:** ✅ Active



