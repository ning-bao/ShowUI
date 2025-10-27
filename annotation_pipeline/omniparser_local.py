#!/usr/bin/env python3
"""
OmniParser v2 Local Inference
Runs Microsoft's OmniParser-v2.0 models locally for UI element detection.

Models:
  - Icon detection: YOLOv8 (AGPL license)
  - Icon caption: Florence-2 (MIT license)

Reference: https://huggingface.co/microsoft/OmniParser-v2.0
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image

# Check dependencies
try:
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM
    from ultralytics import YOLO
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("\nPlease install required packages:")
    print("  pip install torch torchvision transformers ultralytics pillow numpy")
    exit(1)


class OmniParserV2:
    """Local OmniParser v2 inference wrapper"""
    
    def __init__(self, device: str = None, cache_dir: str = None, min_confidence: float = 0.3):
        """
        Initialize OmniParser v2 models
        
        Args:
            device: 'cuda', 'cpu', or None (auto-detect)
            cache_dir: Directory to cache downloaded models
            min_confidence: Minimum confidence threshold to keep predictions (default: 0.5)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_dir = cache_dir or os.path.expanduser('~/.cache/omniparser')
        self.min_confidence = min_confidence
        
        print(f"[OmniParser] Initializing on device: {self.device}")
        print(f"[OmniParser] Cache directory: {self.cache_dir}")
        
        # Load icon detection model (YOLOv8)
        print("[OmniParser] Loading icon detection model (YOLOv8)...")
        try:
            self.detector = YOLO('microsoft/OmniParser-v2.0')  # Will auto-download
            print("  ✓ YOLOv8 detector loaded")
        except Exception as e:
            print(f"  ✗ Failed to load YOLOv8: {e}")
            print("  Attempting manual download from Hugging Face...")
            # Fallback: download manually
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(
                repo_id="microsoft/OmniParser-v2.0",
                filename="icon_detect/model.pt",
                cache_dir=self.cache_dir
            )
            self.detector = YOLO(model_path)
            print("  ✓ YOLOv8 detector loaded from manual download")
        
        # Load icon caption model (Florence-2)
        print("[OmniParser] Loading icon caption model (Florence-2)...")
        try:
            self.processor = AutoProcessor.from_pretrained(
                "microsoft/OmniParser-v2.0",
                trust_remote_code=True,
                cache_dir=self.cache_dir
            )
            self.caption_model = AutoModelForCausalLM.from_pretrained(
                "microsoft/OmniParser-v2.0",
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                cache_dir=self.cache_dir
            ).to(self.device)
            print("  ✓ Florence-2 caption model loaded")
        except Exception as e:
            print(f"  ✗ Failed to load Florence-2: {e}")
            print("  Will proceed with detection only (no captions)")
            self.caption_model = None
            self.processor = None
        
        print("[OmniParser] Initialization complete")
    
    def detect_elements(self, image_path: str, conf_threshold: float = 0.25) -> List[Dict[str, Any]]:
        """
        Detect interactive elements in a screenshot
        
        Args:
            image_path: Path to screenshot image
            conf_threshold: Detection confidence threshold (0-1)
            
        Returns:
            List of elements with bbox [x1,y1,x2,y2] and point [cx,cy]
        """
        # Run YOLOv8 detection
        results = self.detector(image_path, conf=conf_threshold, verbose=False)
        
        elements = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get bbox coordinates
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                
                # Compute center point
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                # Get confidence
                conf = float(box.conf[0])
                
                # Filter by minimum confidence
                if conf < self.min_confidence:
                    continue
                
                element = {
                    'bbox': [x1, y1, x2, y2],
                    'point': [cx, cy],
                    'confidence': conf
                }
                
                elements.append(element)
        
        return elements
    
    def caption_elements(self, image_path: str, elements: List[Dict]) -> List[Dict]:
        """
        Add captions to detected elements using Florence-2
        
        Args:
            image_path: Path to screenshot image
            elements: List of elements with bbox
            
        Returns:
            Elements with added 'caption' field
        """
        if not self.caption_model or not self.processor:
            print("[OmniParser] Caption model not available, skipping captions")
            return elements
        
        image = Image.open(image_path).convert('RGB')
        
        for elem in elements:
            try:
                # Crop element region
                bbox = elem['bbox']
                crop = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                
                # Generate caption
                inputs = self.processor(images=crop, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    generated_ids = self.caption_model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=False
                    )
                
                caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                elem['caption'] = caption.strip()
            except Exception as e:
                print(f"[OmniParser] Failed to caption element {elem['bbox']}: {e}")
                elem['caption'] = ""
        
        return elements
    
    def parse(self, image_path: str, conf_threshold: float = 0.25, with_captions: bool = False) -> Dict:
        """
        Full parsing pipeline: detect and optionally caption elements
        
        Args:
            image_path: Path to screenshot
            conf_threshold: Detection confidence threshold
            with_captions: Whether to generate captions (slower)
            
        Returns:
            Dict with 'img_size' and 'elements'
        """
        # Get image size
        with Image.open(image_path) as img:
            W, H = img.size
        
        # Detect elements
        elements = self.detect_elements(image_path, conf_threshold)
        
        # Optionally add captions
        if with_captions:
            elements = self.caption_elements(image_path, elements)
        
        return {
            'img_size': [W, H],
            'elements': elements
        }


def main():
    parser = argparse.ArgumentParser(description='OmniParser v2 Local Inference')
    parser.add_argument('--image', required=True, help='Path to screenshot image')
    parser.add_argument('--conf', type=float, default=0.25, help='Detection confidence threshold (0-1)')
    parser.add_argument('--min-conf', type=float, default=0.5, help='Minimum confidence to keep predictions (default: 0.5)')
    parser.add_argument('--captions', action='store_true', help='Generate captions for elements (slower)')
    parser.add_argument('--device', default=None, help='Device: cuda or cpu (auto-detect if not set)')
    parser.add_argument('--output', default=None, help='Output JSON path (prints to stdout if not set)')
    parser.add_argument('--cache-dir', default=None, help='Model cache directory')
    
    args = parser.parse_args()
    
    # Initialize parser
    omni = OmniParserV2(device=args.device, cache_dir=args.cache_dir, min_confidence=args.min_conf)
    
    # Parse image
    print(f"\n[OmniParser] Parsing: {args.image}")
    result = omni.parse(args.image, conf_threshold=args.conf, with_captions=args.captions)
    
    print(f"[OmniParser] Found {len(result['elements'])} elements")
    
    # Output
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"[OmniParser] Saved to: {args.output}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

