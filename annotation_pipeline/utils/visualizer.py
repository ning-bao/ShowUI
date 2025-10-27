"""
Visualization Module
Handles visualization of annotations on images
"""

from PIL import Image, ImageDraw, ImageFont
import os
import json
import base64
from io import BytesIO
import random


def visualize_annotations(image_path, annotation):
    """
    Visualize annotations on an image
    
    Args:
        image_path (str): Path to the image file
        annotation (dict): Annotation data with bbox and point information
        
    Returns:
        str: Base64 encoded image with annotations
    """
    # Load image
    img = Image.open(image_path).convert('RGBA')
    width, height = img.size
    
    # Create overlay for transparency
    overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Define colors for different elements
    colors = [
        (255, 0, 0, 180),      # Red
        (0, 255, 0, 180),      # Green
        (0, 0, 255, 180),      # Blue
        (255, 255, 0, 180),    # Yellow
        (255, 0, 255, 180),    # Magenta
        (0, 255, 255, 180),    # Cyan
        (255, 128, 0, 180),    # Orange
        (128, 0, 255, 180),    # Purple
    ]
    
    # Draw each annotation
    for idx, element in enumerate(annotation.get('element', [])):
        bbox = element['bbox']
        point = element['point']
        
        # All coordinates are absolute pixels - use directly
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        px = int(point[0])
        py = int(point[1])
        
        # Select color
        color = colors[idx % len(colors)]
        color_solid = (color[0], color[1], color[2], 255)
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color_solid, width=3)
        
        # Draw semi-transparent fill
        fill_color = (color[0], color[1], color[2], 50)
        draw.rectangle([x1, y1, x2, y2], fill=fill_color)
        
        # Draw center point
        point_radius = 5
        draw.ellipse([px-point_radius, py-point_radius, 
                     px+point_radius, py+point_radius], 
                     fill=color_solid, outline=(255, 255, 255, 255))
        
        # Draw element number
        label = f"#{idx + 1}"
        
        # Draw label background
        bbox_label = draw.textbbox((x1, y1 - 20), label, font=font)
        draw.rectangle(bbox_label, fill=color_solid)
        draw.text((x1, y1 - 20), label, fill=(255, 255, 255, 255), font=font)
    
    # Composite overlay onto original image
    img = Image.alpha_composite(img, overlay)
    
    # Convert to RGB for JPEG encoding
    img = img.convert('RGB')
    
    # Convert to base64
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=95)
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return img_base64


def create_annotation_overlay(width, height, annotation):
    """
    Create just the annotation overlay without the base image
    
    Args:
        width (int): Image width
        height (int): Image height
        annotation (dict): Annotation data
        
    Returns:
        Image: PIL Image object with annotations
    """
    overlay = Image.new('RGBA', (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Similar drawing logic as visualize_annotations
    # This can be used for dynamic overlay in the frontend
    
    return overlay


def save_boxes_visualization(image_path, boxes, out_path=None):
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    for b in boxes:
        try:
            # support dicts with bbox/point or raw box lists
            if isinstance(b, dict):
                x1, y1, x2, y2 = map(int, b.get('bbox', [0,0,0,0]))
                px, py = map(int, b.get('point', [ (x1+x2)//2, (y1+y2)//2 ]))
            else:
                x1, y1, x2, y2 = map(int, b)
                px, py = ( (x1+x2)//2, (y1+y2)//2 )
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
            draw.ellipse([px-2, py-2, px+2, py+2], fill=(0, 255, 0))
        except Exception:
            continue
    if not out_path:
        base, ext = os.path.splitext(image_path)
        out_path = base + "_preprocess_boxes" + ext
    img.save(out_path)
    return out_path

