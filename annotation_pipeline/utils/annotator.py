"""
OpenAI Annotator Module
This module handles the annotation of images using OpenAI API with ShowUI RL prompt.

Supports configurable model selection with an optional "flex" tier to save cost.
Environment variables:
  - OPENAI_MODEL: Preferred model (default: "gpt-5")
  - OPENAI_SERVICE_TIER:  "flex" to enable Flex Processing tier (optional)
  - OPENAI_MAX_COMPLETION_TOKENS: Max completion tokens (preferred)
  - OPENAI_MAX_TOKENS: Back-compat for max completion tokens (default: 4096)
  - OPENAI_TIMEOUT_SECONDS: Request timeout in seconds (default: 900)
  - OPENAI_ENABLE_CODE_INTERPRETER: Enable Code Interpreter via Responses API (default: false)
  - ANNOTATOR_PREPROCESS_ENABLE: Enable OCR+CV preprocessing hints (default: true)
  - ANNOTATOR_PREPROCESS_MAX_ELEMENTS: Max hint elements (0 or 'all' = unlimited)
  - ANNOTATOR_PREPROCESS_VISUALIZE: Save image with preprocessed boxes (default: false)
  - ANNOTATOR_PREPROCESS_VISUALIZE_PATH: Output path for visualization (optional)
  - ANNOTATOR_PREPROCESS_EDGE_SNAP: Refine boxes to edges (default: true)
  - ANNOTATOR_PREPROCESS_COLOR_BUTTONS: Enable multi-color button detection (default: true)
  - ANNOTATOR_PREPROCESS_REFINE_INPUT_INNER: Shrink input boxes to inner editable field (default: true)
  - ANNOTATOR_PREPROCESS_BUTTON_SENSITIVITY: Button detector sensitivity scalar (default: 1.0)
  - ANNOTATOR_PREPROCESS_BACKEND: 'omni' (OmniParser-v2) or 'cv' (default: 'omni')
  - OMNIPARSER_URL: HTTP endpoint for OmniParser-v2 service
  - OMNIPARSER_API_KEY: Optional auth header for OmniParser-v2
  - OMNIPARSER_TIMEOUT: Request timeout seconds for OmniParser-v2 (default: 30)
"""

import json
import base64
import os
import time
import math
from PIL import Image
from openai import OpenAI
from string import Template
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from typing import List, Tuple, Optional
try:
    import requests  # type: ignore
    _REQUESTS_AVAILABLE = True
except Exception:
    requests = None  # type: ignore
    _REQUESTS_AVAILABLE = False

# Optional deps for image preprocessing (CV). Fallback gracefully if missing.
try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    _PREPROC_DEPS_AVAILABLE = True
except Exception:
    cv2 = None  # type: ignore
    np = None  # type: ignore
    _PREPROC_DEPS_AVAILABLE = False

# -----------------------
# Preprocessing utilities
# -----------------------

def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def _center_of(box: List[int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    cx = _clamp(cx, x1 + 1, x2 - 1)
    cy = _clamp(cy, y1 + 1, y2 - 1)
    return int(cx), int(cy)

def _area(box: List[int]) -> int:
    return max(0, box[2]-box[0]) * max(0, box[3]-box[1])

def _sort_tblr_key(box: List[int]) -> Tuple[int, int]:
    cx, cy = _center_of(box)
    return (cy, cx)

def _clip_box(box: List[int], W: int, H: int) -> List[int]:
    x1, y1, x2, y2 = box
    x1 = _clamp(int(round(x1)), 0, W-1)
    y1 = _clamp(int(round(y1)), 0, H-1)
    x2 = _clamp(int(round(x2)), 0, W)
    y2 = _clamp(int(round(y2)), 0, H)
    x1 = min(x1, x2-1)
    y1 = min(y1, y2-1)
    return [x1, y1, x2, y2]

def _min_size_ok(box: List[int], min_side: int = 8) -> bool:
    return (box[2]-box[0] >= min_side) and (box[3]-box[1] >= min_side)

def _iou(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / max(union, 1e-6)

def _nearest_label_text(target_box: List[int], labels: List[Tuple[Tuple[int,int], str]], max_dist: int = 80) -> Optional[str]:
    tcx, tcy = _center_of(target_box)
    best_txt = None
    bestd = 1e9
    for (wcx, wcy), text in labels:
        d = math.hypot(wcx - tcx, wcy - tcy)
        if d < max_dist and d < bestd:
            bestd = d
            best_txt = text
    return best_txt

def _detect_blue_regions(img_bgr):
    if not _PREPROC_DEPS_AVAILABLE:
        return None
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array([90, 70, 80])
    upper1 = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower1, upper1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)), iterations=2)
    return mask

def _detect_button_like_regions(img_bgr):
    if not _PREPROC_DEPS_AVAILABLE:
        return None
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    masks = []
    # Blue
    masks.append(cv2.inRange(hsv, np.array([90, 70, 80]), np.array([130, 255, 255])))
    # Green
    masks.append(cv2.inRange(hsv, np.array([40, 60, 60]), np.array([85, 255, 255])))
    # Red (wrap-around)
    masks.append(cv2.inRange(hsv, np.array([0, 70, 70]), np.array([10, 255, 255])))
    masks.append(cv2.inRange(hsv, np.array([170, 70, 70]), np.array([180, 255, 255])))
    # Neutral gray buttons (low saturation, higher value)
    sat = hsv[:,:,1]; val = hsv[:,:,2]
    gray_mask = cv2.inRange(sat, 0, 35) & cv2.inRange(val, 120, 255)
    masks.append(gray_mask)
    mask = masks[0]
    for m in masks[1:]:
        mask = cv2.bitwise_or(mask, m)
    # Morph to consolidate interior of capsules/icons
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)), iterations=2)
    return mask

def _find_rect_candidates(img_bgr) -> List[List[int]]:
    if not _PREPROC_DEPS_AVAILABLE:
        return []
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, None, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[List[int]] = []
    H, W = gray.shape[:2]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w*h < 64:
            continue
        ar = w / float(h)
        if (w < 12 or h < 12) or ar > 20 or ar < 0.08:
            continue
        boxes.append([x, y, x+w, y+h])
    # clip now for safety
    boxes = [_clip_box(b, W, H) for b in boxes]
    return boxes

def _find_regions_mser(img_bgr) -> List[List[int]]:
    if not _PREPROC_DEPS_AVAILABLE:
        return []
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    boxes: List[List[int]] = []
    try:
        mser = cv2.MSER_create(_delta=5, _min_area=60, _max_area=50000)
        regions, _ = mser.detectRegions(gray)
        H, W = gray.shape[:2]
        for pts in regions:
            x, y, w, h = cv2.boundingRect(pts)
            if w*h < 64:
                continue
            ar = w / float(max(1, h))
            if (w < 12 or h < 12) or ar > 20 or ar < 0.08:
                continue
            boxes.append(_clip_box([x, y, x+w, y+h], W, H))
    except Exception:
        return []
    return boxes

def _refine_box_edges(img_bgr, box: List[int]) -> List[int]:
    if not _PREPROC_DEPS_AVAILABLE:
        return box
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    x1, y1, x2, y2 = box
    H, W = gray.shape[:2]
    margin = 6
    # left
    cmin = max(0, x1 - margin); cmax = min(W-1, x1 + margin)
    best_col = x1
    best_val = -1
    for cx in range(cmin, cmax+1):
        s = int(edges[y1:y2, cx].sum())
        if s > best_val:
            best_val = s; best_col = cx
    x1n = best_col
    # right
    cmin = max(1, x2 - margin); cmax = min(W, x2 + margin)
    best_col = x2
    best_val = -1
    for cx in range(cmin-1, cmax):
        s = int(edges[y1:y2, cx-1].sum()) if cx-1 >= 0 else 0
        if s > best_val:
            best_val = s; best_col = cx
    x2n = best_col
    # top
    rmin = max(0, y1 - margin); rmax = min(H-1, y1 + margin)
    best_row = y1
    best_val = -1
    for ry in range(rmin, rmax+1):
        s = int(edges[ry, x1:x2].sum())
        if s > best_val:
            best_val = s; best_row = ry
    y1n = best_row
    # bottom
    rmin = max(1, y2 - margin); rmax = min(H, y2 + margin)
    best_row = y2
    best_val = -1
    for ry in range(rmin-1, rmax):
        s = int(edges[ry-1, x1:x2].sum()) if ry-1 >= 0 else 0
        if s > best_val:
            best_val = s; best_row = ry
    y2n = best_row
    refined = _clip_box([x1n, y1n, x2n, y2n], W, H)
    if not _min_size_ok(refined):
        return box
    return refined

def _shrink_to_inner_field(img_bgr, box: List[int]) -> List[int]:
    """
    For input-like boxes: shrink to the inner editable area by trimming border bands
    where edge density is highest. Falls back if result is invalid.
    """
    if not _PREPROC_DEPS_AVAILABLE:
        return box
    x1, y1, x2, y2 = box
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape[:2]
    x1 = _clamp(x1, 0, W-1); x2 = _clamp(x2, 1, W)
    y1 = _clamp(y1, 0, H-1); y2 = _clamp(y2, 1, H)
    roi = gray[y1:y2, x1:x2]
    edges = cv2.Canny(roi, 50, 150)
    h, w = edges.shape[:2]
    if h < 6 or w < 6:
        return box
    band = max(2, min(6, min(h, w)//12))
    # Measure border edge sums
    top_sum = int(edges[:band, :].sum())
    bot_sum = int(edges[-band:, :].sum())
    left_sum = int(edges[:, :band].sum())
    right_sum = int(edges[:, -band:].sum())
    # Trim inward slightly past high-edge bands to approximate inner field
    trim_top = band if top_sum > 0 else 0
    trim_bot = band if bot_sum > 0 else 0
    trim_left = band if left_sum > 0 else 0
    trim_right = band if right_sum > 0 else 0
    inner = [x1 + trim_left, y1 + trim_top, x2 - trim_right, y2 - trim_bot]
    inner = _clip_box(inner, W, H)
    if _min_size_ok(inner):
        return inner
    return box

def _classify_candidate(box: List[int], img_bgr, labels: List[Tuple[Tuple[int,int], str]]):
    if not _PREPROC_DEPS_AVAILABLE:
        return ("generic", 0.1, None)
    x1, y1, x2, y2 = box
    crop = img_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mean_hsv = hsv.reshape(-1,3).mean(axis=0)
    sat = mean_hsv[1]; val = mean_hsv[2]
    is_blueish = (90 <= mean_hsv[0] <= 130) and (sat > 60) and (val > 80)
    is_neutral_capsule = (sat < 40 and val > 150)
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_crop, 40, 120)
    border_strip = np.zeros_like(edges)
    t = 2
    border_strip[:t,:] = 1; border_strip[-t:,:] = 1; border_strip[:,:t] = 1; border_strip[:,-t:] = 1
    edge_border_ratio = (edges[border_strip==1] > 0).mean() if int(border_strip.sum()) > 0 else 0
    fill_ratio = float((gray_crop < 245).mean()) if gray_crop.size > 0 else 0.0
    inside_text = ""
    ar = (x2-x1) / max(1,(y2-y1))
    area_px = (x2-x1) * (y2-y1)
    kind = 'generic'
    score = 0.1
    height_ok = 20 <= (y2-y1) <= 160
    width_ok = 32 <= (x2-x1) <= 800
    if (is_blueish or is_neutral_capsule or fill_ratio > 0.12) and height_ok and width_ok:
        kind = 'button'; score = max(0.9, 0.6 + 0.4*min(1.0, fill_ratio*2.5))
    elif edge_border_ratio > 0.22 and ar >= 1.6 and area_px > 400:
        kind = 'input'; score = 0.75
    elif area_px < 30000 and (ar >= 2.0 or ar <= 0.6):
        kind = 'link'; score = 0.55
    else:
        score = 0.3
    text = _nearest_label_text(box, labels) if labels else None
    return (kind, float(score), text)

def _non_max_dedup(boxes_with_scores: List[Tuple[List[int], float]]) -> List[List[int]]:
    boxes_with_scores = sorted(boxes_with_scores, key=lambda t: t[1], reverse=True)
    kept: List[List[int]] = []
    for b, _s in boxes_with_scores:
        keep = True
        for kb in kept:
            if _iou(b, kb) > 0.5:
                keep = False; break
            bc = _center_of(b); kbc = _center_of(kb)
            if abs(bc[0]-kbc[0]) < 10 and abs(bc[1]-kbc[1]) < 10:
                keep = False; break
        if keep:
            kept.append(b)
    return kept


# Global OmniParser instance cache
_OMNI_PARSER_INSTANCE = None

class GPTAnnotator:
    """Handles image annotation using OpenAI API"""
    
    def __init__(self, api_key=None, model=None, max_tokens=None, service_tier=None, timeout_seconds=None, use_code_interpreter=None):
        """
        Initialize the annotator
        
        Args:
            api_key (str, optional): OpenAI API key. Defaults to None (loads from env).
            model (str, optional): Model to use. Defaults to 'gpt-5'.
            max_tokens (int, optional): Max tokens for response. Defaults to 4096.
            service_tier (str, optional): Service tier, e.g., 'flex'. Defaults from env.
            timeout_seconds (float, optional): Request timeout seconds. Defaults from env.
        
        Raises:
            ValueError: If no API key is provided
        """
        # Ensure environment variables are loaded even if caller didn't
        dotenv_path = find_dotenv(usecwd=True)
        if dotenv_path:
            load_dotenv(dotenv_path=dotenv_path, override=False)
        else:
            script_env = Path(__file__).resolve().parent.parent / '.env'
            if script_env.exists():
                load_dotenv(dotenv_path=str(script_env), override=False)

        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        # Model (default GPT-5)
        self.model = model or os.getenv('OPENAI_MODEL') or 'gpt-5'
        # Max completion tokens (prefer new env, fall back to legacy)
        max_ct_env = os.getenv('OPENAI_MAX_COMPLETION_TOKENS')
        legacy_max_env = os.getenv('OPENAI_MAX_TOKENS')
        self.max_completion_tokens = (
            int(max_tokens) if isinstance(max_tokens, int) else
            int(max_ct_env) if max_ct_env else
            int(legacy_max_env) if legacy_max_env else
            4096
        )
        # Optional Flex Processing tier
        env_service_tier = os.getenv('OPENAI_SERVICE_TIER', '').strip().lower()
        self.service_tier = (service_tier or env_service_tier) or None
        # Optional timeout
        env_timeout = os.getenv('OPENAI_TIMEOUT_SECONDS', '').strip()
        self.timeout_seconds = (
            float(timeout_seconds) if timeout_seconds is not None else (float(env_timeout) if env_timeout else 900.0)
        )
        # Optional Code Interpreter via Responses API
        env_enable_ci = os.getenv('OPENAI_ENABLE_CODE_INTERPRETER', '').strip().lower()
        if use_code_interpreter is not None:
            self.use_code_interpreter = bool(use_code_interpreter)
        else:
            self.use_code_interpreter = env_enable_ci in ('1', 'true', 'yes', 'on')
        # Optional CI container config (JSON string)
        self.ci_container_config = None
        env_ci_container = os.getenv('OPENAI_CI_CONTAINER_JSON', '').strip()
        if env_ci_container:
            try:
                self.ci_container_config = json.loads(env_ci_container)
            except Exception:
                print("[Annotator] Warning: OPENAI_CI_CONTAINER_JSON is not valid JSON. Ignoring.")
        # Preprocessing config (defaults enabled)
        env_pre = os.getenv('ANNOTATOR_PREPROCESS_ENABLE', '').strip().lower()
        self.preprocess_enable = True if env_pre == '' else (env_pre in ('1','true','yes','on'))
        raw_max = os.getenv('ANNOTATOR_PREPROCESS_MAX_ELEMENTS', '').strip().lower()
        if raw_max in ('', '0', 'all', 'unlimited', 'none'):
            self.preprocess_max_elements = 0
        else:
            try:
                self.preprocess_max_elements = int(raw_max)
            except Exception:
                self.preprocess_max_elements = 0
        env_edge_snap = os.getenv('ANNOTATOR_PREPROCESS_EDGE_SNAP', '').strip().lower()
        self.preprocess_edge_snap = True if env_edge_snap == '' else (env_edge_snap in ('1','true','yes','on'))
        env_color_btn = os.getenv('ANNOTATOR_PREPROCESS_COLOR_BUTTONS', '').strip().lower()
        self.preprocess_color_buttons = True if env_color_btn == '' else (env_color_btn in ('1','true','yes','on'))
        env_vis = os.getenv('ANNOTATOR_PREPROCESS_VISUALIZE', '').strip().lower()
        self.preprocess_visualize = (env_vis in ('1','true','yes','on')) if env_vis != '' else False
        self.preprocess_visualize_path = os.getenv('ANNOTATOR_PREPROCESS_VISUALIZE_PATH', '').strip()
        env_refine_inner = os.getenv('ANNOTATOR_PREPROCESS_REFINE_INPUT_INNER', '').strip().lower()
        self.preprocess_refine_input_inner = True if env_refine_inner == '' else (env_refine_inner in ('1','true','yes','on'))
        try:
            self.preprocess_button_sensitivity = float(os.getenv('ANNOTATOR_PREPROCESS_BUTTON_SENSITIVITY', '1.0'))
        except Exception:
            self.preprocess_button_sensitivity = 1.0
        # Backend selection
        env_backend = os.getenv('ANNOTATOR_PREPROCESS_BACKEND', '').strip().lower()
        self.preprocess_backend = env_backend if env_backend in ('omni','omniparser','cv') else 'omni'
        # OmniParser HTTP config
        self.omni_url = os.getenv('OMNIPARSER_URL', '').strip()
        self.omni_api_key = os.getenv('OMNIPARSER_API_KEY', '').strip()
        try:
            self.omni_timeout = float(os.getenv('OMNIPARSER_TIMEOUT', '30'))
        except Exception:
            self.omni_timeout = 30.0
        
        # Preload OmniParser model if using local backend
        if self.preprocess_backend in ('omni','omniparser') and (not self.omni_url or self.omni_url.lower() in ('local', 'localhost')):
            self._preload_omniparser()
        # No OCR dependencies used any more
        
        # Validate API key
        if not self.api_key or self.api_key.startswith('your_'):
            raise ValueError(
                "OpenAI API key is required. Please set OPENAI_API_KEY in your .env file.\n"
                "Get your API key from: https://platform.openai.com/api-keys"
            )
        
        # Initialize OpenAI client
        try:
            self.client = OpenAI(api_key=self.api_key)
            print("✓ OpenAI API initialized successfully")
            print(f"  Model: {self.model}")
            print(f"  Service tier: {self.service_tier or 'default'}")
            if self.use_code_interpreter:
                ci_mode = 'enabled'
                if self.ci_container_config:
                    ci_mode += ' (with container)'
                print(f"  Code Interpreter: {ci_mode}")
            else:
                print("  Code Interpreter: disabled")
            if self.preprocess_enable:
                if _PREPROC_DEPS_AVAILABLE:
                    print(f"  Preprocessing hints: enabled (max {self.preprocess_max_elements or 'all'})")
                    print(f"    - Backend: {self.preprocess_backend}  | Edge snap: {'on' if self.preprocess_edge_snap else 'off'}  | Color buttons: {'on' if self.preprocess_color_buttons else 'off'}  | Visualize: {'on' if self.preprocess_visualize else 'off'}")
                else:
                    print("  Preprocessing hints: requested but disabled (missing cv2/numpy/pytesseract)")
                    self.preprocess_enable = False
            else:
                print("  Preprocessing hints: disabled")
        except Exception as e:
            raise ValueError(f"Failed to initialize OpenAI client: {e}")
    
    def _preload_omniparser(self):
        """Preload OmniParser model into memory/GPU for faster inference"""
        global _OMNI_PARSER_INSTANCE
        
        if _OMNI_PARSER_INSTANCE is not None:
            print("  OmniParser: already loaded (cached)")
            return
        
        try:
            import omniparser_local
            print("  OmniParser: loading models into memory...")
            _OMNI_PARSER_INSTANCE = omniparser_local.OmniParserV2(min_confidence=0.7)
            print("  ✓ OmniParser models preloaded and ready")
        except ImportError:
            print("  ⚠️  OmniParser: omniparser_local module not found")
            print("     Models will be loaded on first use")
        except Exception as e:
            print(f"  ⚠️  OmniParser: preload failed - {e}")
            print("     Models will be loaded on first use")
    
    def annotate(self, image_path):
        """
        Generate annotations for an image using OpenAI API
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Annotation data in the specified format
            
        Raises:
            Exception: If OpenAI API call fails
        """
        # Get image size
        with Image.open(image_path) as img:
            width, height = img.size
        
        # Call OpenAI API
        annotation = self._call_openai_api(image_path, width, height)
        return annotation
    
    def _call_openai_api(self, image_path, width, height):
        """
        Call OpenAI API to annotate image
        
        Args:
            image_path (str): Path to the image file
            width (int): Image width
            height (int): Image height
            
        Returns:
            dict: Annotation data from OpenAI
        """
        # Encode full image as base64
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Determine image format
        image_ext = os.path.splitext(image_path)[1].lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp'
        }
        mime_type = mime_types.get(image_ext, 'image/jpeg')
        
        # Optional: generate preprocessing hints with cropped element images
        hints: List[dict] = []
        element_crops: List[str] = []  # Base64 encoded crops
        if self.preprocess_enable and _PREPROC_DEPS_AVAILABLE:
            try:
                hints = self._compute_preprocess_hints(image_path, max_elements=self.preprocess_max_elements)
                print(f"[Annotator] Preprocessing produced {len(hints)} hint(s)")
                
                # Crop and encode each bounding box
                if hints:
                    element_crops = self._crop_elements(image_path, hints)
                    print(f"[Annotator] Cropped {len(element_crops)} element images")
                
                try:
                    print("[Annotator] Hints:")
                    print(json.dumps(hints, ensure_ascii=False))
                except Exception as e_json:
                    print(f"[Annotator] Failed to serialize hints: {e_json}")
            except Exception as e:
                print(f"[Annotator] Preprocessing failed: {e}. Continuing without hints.")
                hints = []
                element_crops = []

        # Construct the prompt with image dimensions, hints, and element crops
        prompt = self._get_annotation_prompt(width, height, hints, element_crops)
        
        # Decide token parameter names and service tier support
        model_lower = str(self.model).lower()
        uses_completion_tokens = (model_lower.startswith('gpt-5') or model_lower.startswith('o3'))
        token_param_name_chat = 'max_completion_tokens' if uses_completion_tokens else 'max_tokens'
        token_param_name_resp = 'max_output_tokens'
        # Only include service_tier for models that support it (e.g., gpt-5/o3)
        allow_service_tier = uses_completion_tokens

        # Call OpenAI API (optionally using Flex Processing tier)
        def build_request_kwargs_chat(token_limit: int):
            # Build content array with prompt, full image, and element crops
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}},
            ]
            
            # Add cropped element images if available
            for idx, crop_data in enumerate(element_crops):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{crop_data}"}
                })
            
            kwargs = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                "response_format": {"type": "json_object"},
            }
            kwargs[token_param_name_chat] = token_limit
            if allow_service_tier and self.service_tier:
                kwargs["service_tier"] = self.service_tier
            if self.timeout_seconds:
                kwargs["timeout"] = self.timeout_seconds
            return kwargs

        def build_request_kwargs_responses(token_limit: int, include_tools: bool = True):
            # Build content array with prompt, full image, and element crops
            resp_content = [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": f"data:{mime_type};base64,{image_data}"},
            ]
            
            # Add cropped element images if available
            for idx, crop_data in enumerate(element_crops):
                resp_content.append({
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{crop_data}"
                })
            
            kwargs = {
                "model": self.model,
                "input": [
                    {
                        "role": "user",
                        "content": resp_content,
                    }
                ],
                # Some SDK versions do not support response_format in Responses API
                # Use prompt-instruction to enforce JSON and parse client-side
                "tool_choice": "auto",
            }
            if include_tools:
                tool_def = {"type": "code_interpreter", "container": {"type": "auto"}}
                kwargs["tools"] = [tool_def]
            kwargs[token_param_name_resp] = token_limit
            if allow_service_tier and self.service_tier:
                kwargs["service_tier"] = self.service_tier
            if self.timeout_seconds:
                kwargs["timeout"] = self.timeout_seconds
            return kwargs

        # Compute initial and retry token limits
        base_limit = int(self.max_completion_tokens)
        attempts = [base_limit, min(max(base_limit * 2, 2048), 8192)]
        response = None
        last_error = None
        for attempt_index, max_ct in enumerate(attempts):
            # Log a concise request summary for debugging
            try:
                if self.use_code_interpreter:
                    print(f"[Annotator] Sending request (responses+CI): model={self.model}, param={token_param_name_resp}, limit={max_ct}, tier={self.service_tier if allow_service_tier else 'n/a'}, timeout={self.timeout_seconds}")
                else:
                    print(f"[Annotator] Sending request (chat): model={self.model}, param={token_param_name_chat}, limit={max_ct}, tier={self.service_tier if allow_service_tier else 'n/a'}, timeout={self.timeout_seconds}")
            except Exception:
                pass
            request_start = time.time()
            try:
                used_path = "chat"
                if self.use_code_interpreter:
                    try:
                        response = self.client.responses.create(**build_request_kwargs_responses(max_ct, include_tools=True))
                        used_path = "responses"
                    except Exception as e_ci:
                        # If CI requires container per newer API, retry without tools (no code interpreter)
                        err_msg = str(e_ci).lower()
                        if "tools[0].container" in err_msg or "missing required parameter" in err_msg:
                            print("[Annotator] Code Interpreter not available without container. Retrying Responses API without tools...")
                            response = self.client.responses.create(**build_request_kwargs_responses(max_ct, include_tools=False))
                            used_path = "responses"
                        elif isinstance(e_ci, TypeError):
                            print(f"[Annotator] Responses API TypeError: {e_ci}. Falling back to Chat Completions...")
                            response = self.client.chat.completions.create(**build_request_kwargs_chat(max_ct))
                            used_path = "chat"
                        else:
                            raise
                else:
                    response = self.client.chat.completions.create(**build_request_kwargs_chat(max_ct))
                    used_path = "chat"
                elapsed = time.time() - request_start
                print(f"[Annotator] Response received via {used_path} in {elapsed:.1f}s")
            except Exception as e:
                elapsed = time.time() - request_start
                print(f"[Annotator] Request failed after {elapsed:.1f}s: {e}")
                last_error = e
                break

            # Best-effort capture of raw response for debugging on failures
            raw_response_serialized = None
            try:
                if hasattr(response, "model_dump"):
                    raw_response_serialized = json.dumps(response.model_dump(), indent=2, default=str)
                elif hasattr(response, "to_dict"):
                    raw_response_serialized = json.dumps(response.to_dict(), indent=2, default=str)
                else:
                    # Fallback string representation
                    raw_response_serialized = str(response)
            except Exception:
                raw_response_serialized = str(response)

            # Extract content and finish reason
            try:
                if self.use_code_interpreter and used_path == "responses":
                    # Responses API
                    finish_reason = getattr(response, "finish_reason", None)
                    content = None
                    try:
                        content = getattr(response, "output_text", None)
                    except Exception:
                        content = None
                    if not content:
                        try:
                            outputs = getattr(response, "output", None) or getattr(response, "outputs", None) or []
                            texts = []
                            for out in outputs:
                                parts = getattr(out, "content", None) or getattr(out, "contents", None) or []
                                for p in parts:
                                    t = None
                                    if hasattr(p, "text"):
                                        t_obj = getattr(p, "text")
                                        if isinstance(t_obj, str):
                                            t = t_obj
                                        else:
                                            try:
                                                t = getattr(t_obj, "value", None)
                                            except Exception:
                                                t = None
                                    if isinstance(t, str) and t.strip():
                                        texts.append(t)
                            if texts:
                                content = "\n".join(texts)
                        except Exception:
                            content = None
                else:
                    # Chat Completions API
                    choice0 = response.choices[0]
                    finish_reason = getattr(choice0, "finish_reason", None)
                    content = choice0.message.content
            except Exception as e:
                print("[Annotator] Error accessing response content. Full response follows:")
                print(raw_response_serialized)
                raise RuntimeError(f"Failed to access response content: {e}")

            # If we hit length or empty content on first attempt, retry with higher cap
            if (not content or str(content).strip() == "") and finish_reason == "length" and attempt_index == 0:
                if self.use_code_interpreter:
                    print(f"[Annotator] Empty content due to length with {token_param_name_resp}={max_ct}. Retrying with higher cap...")
                else:
                    print(f"[Annotator] Empty content due to length with {token_param_name_chat}={max_ct}. Retrying with higher cap...")
                continue

            # Try to parse JSON
            try:
                annotation = json.loads(content)
            except Exception as e:
                # If cut short due to length on first attempt, retry
                if finish_reason == "length" and attempt_index == 0:
                    if self.use_code_interpreter:
                        print(f"[Annotator] JSON parse failed with length finish ({token_param_name_resp}={max_ct}). Retrying...")
                    else:
                        print(f"[Annotator] JSON parse failed with length finish ({token_param_name_chat}={max_ct}). Retrying...")
                    continue
                print("[Annotator] JSON parse failed. Raw assistant content follows:")
                try:
                    preview = content if len(str(content)) < 4000 else (str(content)[:4000] + "... [truncated]")
                except Exception:
                    preview = "<unavailable>"
                print(preview)
                print("[Annotator] Full raw OpenAI response for debugging:")
                print(raw_response_serialized)
                raise RuntimeError(f"Failed to parse JSON from model response: {e}")

            # Success: ensure format and return
            if 'img_size' not in annotation:
                annotation['img_size'] = [width, height]
            if 'element' not in annotation:
                annotation['element'] = []
            return annotation

        # If we reached here, request failed entirely
        if last_error is not None:
            raise RuntimeError(f"OpenAI request failed: {last_error}")
        raise RuntimeError("Annotation failed after retries (empty or invalid JSON response)")
    
    def _crop_elements(self, image_path: str, hints: List[dict]) -> List[str]:
        """
        Crop bounding boxes from image and encode as base64
        
        Args:
            image_path: Path to full screenshot
            hints: List of elements with bbox coordinates
            
        Returns:
            List of base64-encoded PNG images (one per element)
        """
        try:
            from io import BytesIO
            crops = []
            
            with Image.open(image_path) as img:
                for idx, hint in enumerate(hints):
                    try:
                        bbox = hint.get('bbox', [])
                        if len(bbox) != 4:
                            continue
                        
                        x1, y1, x2, y2 = map(int, bbox)
                        
                        # Crop the element region
                        crop = img.crop((x1, y1, x2, y2))
                        
                        # Encode as PNG base64
                        buffer = BytesIO()
                        crop.save(buffer, format='PNG')
                        crop_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                        crops.append(crop_b64)
                    except Exception as e:
                        print(f"[Annotator] Failed to crop element {idx}: {e}")
                        continue
            
            return crops
        except Exception as e:
            print(f"[Annotator] Crop elements failed: {e}")
            return []
    
    def _get_annotation_prompt(self, width, height, hints: Optional[List[dict]] = None, element_crops: Optional[List[str]] = None):
        """
        Get the annotation prompt for OpenAI API
        
        Args:
            width (int): Image width in pixels
            height (int): Image height in pixels
            hints (List[dict], optional): Precomputed hint elements to guide the model
            element_crops (List[str], optional): Base64 encoded cropped element images
            
        Returns:
            str: The prompt text with width/height injected
        """
        hints_text = "[]"
        has_crops = element_crops and len(element_crops) > 0
        
        try:
            if hints:
                # Only keep relevant keys and ensure ints (no instruction)
                norm = []
                for idx, h in enumerate(hints):
                    bbox = [int(h['bbox'][0]), int(h['bbox'][1]), int(h['bbox'][2]), int(h['bbox'][3])]
                    point = [int(h['point'][0]), int(h['point'][1])]
                    elem = {
                        "bbox": bbox,
                        "point": point,
                    }
                    # Reference crop image index if available
                    if has_crops and idx < len(element_crops):
                        elem["crop_index"] = idx + 1  # 1-indexed for user-friendly reference
                    norm.append(elem)
                hints_text = json.dumps(norm, ensure_ascii=False)
        except Exception:
            hints_text = "[]"

        # Add crop instruction if crops are available
        crop_instruction = ""
        if has_crops:
            crop_instruction = f"""
  <crop_images>
    After the main screenshot, {len(element_crops)} cropped UI element images are attached.
    Each crop corresponds to a candidate in <hints> by crop_index (1-indexed).
    Use these crops to visually inspect each element and generate a precise, context-aware instruction.
  </crop_images>"""

        template = Template("""<SYSTEM>
  You are UI Grounding Labeler for ShowUI RL.

  GOAL:
    From one attached screenshot, return precise, actionable UI groundings for 1–5 UNIQUE elements.

  OUTPUT (JSON ONLY):
    A single JSON object with keys:
      - "img_size": [WIDTH_PX, HEIGHT_PX]                 // integers
      - "element": [
          {
            "instruction": string,                        // one concise, actionable description per element
            "bbox": [x1_px, y1_px, x2_px, y2_px],         // ABSOLUTE PIXELS, not normalized
            "point": [cx_px, cy_px]                       // ABSOLUTE PIXELS, click/caret hotspot
          },
          ...
        ]

  HARD RULES (PIXEL SPACE):
    1) Do NOT include any image path/URL keys (no "img_url").
    2) If <img_size> is provided, use it exactly; otherwise infer exact pixel size from the image.
    3) All coordinates are ABSOLUTE PIXELS (no normalization).
       - Integers only (round to nearest).
       - Bounds: 0 ≤ x1 < x2 ≤ WIDTH_PX, 0 ≤ y1 < y2 ≤ HEIGHT_PX.
       - "point" must be strictly inside its bbox: x1 < cx < x2 and y1 < cy < y2.
       - Prefer minimum target size ≥ 8×8 px unless the control is clearly primary.
    4) Output an adaptive number of elements N in [1,5] based on visual richness:
         - Sparse UI: 1–2
         - Typical desktop/app: 3–4
         - Dense panels/tables/chat lists: up to 5
       Prioritize clearly actionable controls (buttons, inputs, tabs, menu items, toggles, icons with affordance, selectable rows/cards).
    5) UNIQUENESS (no duplicates of the same target):
         - Do NOT produce multiple entries for one UI control.
         - Treat two candidates as duplicates if IoU(px) > 0.5 OR their centers differ by < 10 px on both axes,
           OR they refer to the same visible control (same text/icon/role).
         - If duplicates arise, keep the single most actionable/specific description and drop the rest.
    6) INSTRUCTION STYLE (one per element):
         - Single sentence (≤120 chars), imperative and specific (e.g., "Type in the message box", "Open Settings (gear icon)").
         - Include brief disambiguators if needed (color/icon/nearby label/region).
         - Avoid vague words ("maybe", "seems"); avoid internal IDs (e.g., no snake_case labels).
    7) Sort elements by bbox center: top-to-bottom, then left-to-right.
    8) Return ONLY valid JSON (no prose/markdown).

  BBOX ACCURACY GUIDE (SNAP-TO-EDGE HEURISTICS):
    A) General:
       - Snap each bbox edge to the visible boundary of the clickable region.
       - Include the full clickable hit area; EXCLUDE outer drop-shadows, glows, and tooltips.
       - If the control has a focus ring/hover glow, align to the underlying control’s solid edge, not the glow.
       - If uncertain, shrink the bbox by 1–2 px on each side to guarantee the "point" lies strictly inside.
       - Round to nearest integer after snapping.

    B) By control type:
       - Buttons/toolbar buttons: include the entire button capsule/rectangle seen on hover/press; do not include shadow.
       - Icon-only buttons: prefer the square hoverable container (if visible/typical for the toolbar); otherwise tightly bound the icon with ~2–4 px padding.
       - Text inputs/search boxes: include the inner editable field (inside the border). Do not include the floating label/placeholder outside the field.
       - Selectable list rows/cards: span the full clickable row/card interior, from its left visual edge to its right edge; exclude inter-row gutters.
       - Menus/tabs: bound the individual item/tab face; exclude the menu container’s padding unless it expands the hit area.
       - Close/chevron icons: tightly bound the icon glyph or its hover target if present.

    C) Sanity checks for each element (must all be true):
       - x1 < x2 and y1 < y2; width = x2 - x1 ≥ 8 and height = y2 - y1 ≥ 8.
       - 0 ≤ x1, y1 and x2 ≤ WIDTH_PX, y2 ≤ HEIGHT_PX.
       - Choose point = visual center of the control or its primary icon/text: 
         cx = round((x1 + x2)/2), cy = round((y1 + y2)/2), then adjust 1 px inward if equal to an edge.
       - Ensure the point is strictly inside: x1 < cx < x2 and y1 < cy < y2.

    D) Anti-drift rules (to avoid loose boxes):
       - Do not include blank margins around controls unless part of the hit area.
       - Do not cross into neighboring controls; if edges are ambiguous, use the divider or alignment guides on screen.
       - For rounded controls, approximate with the tightest enclosing axis-aligned rectangle of the clickable area.

  DUPLICATE PRUNING (light NMS):
    - If two boxes overlap with IoU > 0.5, keep the more specific/actionable one (e.g., the button vs. its container).
    - If centers are within 10 px both horizontally and vertically, treat as the same target and keep one.

  SELF-CHECK BEFORE FINALIZING:
    - Schema correct? keys = {"img_size","element"} only.
    - 1 ≤ len(element) ≤ 5.
    - All coords are integers in pixel bounds; x1<x2, y1<y2.
    - Each point lies strictly inside its bbox.
    - No duplicate elements per UNIQUENESS rules.
    - Instructions are clear, actionable, unambiguous.

</SYSTEM>

<USER>
  <task>Produce ShowUI grounding JSON for the attached screenshot.</task>
  <img_size>[$width, $height]</img_size>
  <max_elements>5</max_elements>
  <hints>$hints</hints>$crop_instruction
  <exclude>[]</exclude>
  <requirements>
    - Output a single JSON object with "img_size" and "element".
    - Use ABSOLUTE PIXEL coordinates only; do NOT normalize.
    - For each element in <hints> with a crop_index, examine the corresponding cropped image.
    - Generate a precise, descriptive instruction by analyzing what you see in the crop.
    - Identify the element type (button, input, icon, link, etc.) and its purpose from visual context.
    - Include visible text, icons, or distinctive features in the instruction.
    - One instruction per element; no variants for the same target.
    - Apply uniqueness, pixel bounds, snapping, and sorting rules from SYSTEM.
  </requirements>
  <return>JSON only. No markdown, no commentary.</return>
</USER>""")

        return template.substitute(width=width, height=height, hints=hints_text, crop_instruction=crop_instruction)

    def _compute_preprocess_hints(self, image_path: str, max_elements: int = 5) -> List[dict]:
        # Omni backend: local OmniParser-v2 inference or external HTTP service
        if getattr(self, 'preprocess_backend', 'omni') in ('omni','omniparser'):
            # Try local inference first if omniparser_local is available
            if not self.omni_url or self.omni_url.lower() in ('local', 'localhost'):
                try:
                    global _OMNI_PARSER_INSTANCE
                    import omniparser_local
                    
                    # Use cached instance if available, otherwise create new one
                    if _OMNI_PARSER_INSTANCE is None:
                        print("[Annotator] Loading OmniParser-v2 (first use)...")
                        _OMNI_PARSER_INSTANCE = omniparser_local.OmniParserV2(min_confidence=0.7)
                    else:
                        print("[Annotator] Using cached OmniParser-v2 instance...")
                    
                    result = _OMNI_PARSER_INSTANCE.parse(image_path, conf_threshold=0.25, with_captions=False)
                    elements = result.get('elements', [])
                    # Normalize format
                    normalized = []
                    for e in elements:
                        bbox = e.get('bbox', [])
                        point = e.get('point', [(bbox[0]+bbox[2])//2, (bbox[1]+bbox[3])//2] if len(bbox)==4 else [0,0])
                        normalized.append({'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])], 'point': [int(point[0]), int(point[1])]})
                    if max_elements and max_elements > 0:
                        normalized = normalized[:max_elements]
                    print(f"[Annotator] OmniParser-v2 found {len(normalized)} elements")
                    return normalized
                except ImportError:
                    print("[Annotator] omniparser_local not found. Set OMNIPARSER_URL for HTTP service or install dependencies.")
                except Exception as e:
                    print(f"[Annotator] Local OmniParser failed: {e}. Falling back to CV backend.")
            # Try HTTP service if URL is set
            elif self.omni_url and _REQUESTS_AVAILABLE:
                try:
                    with open(image_path, 'rb') as f:
                        img_bytes = f.read()
                    headers = {'Accept': 'application/json'}
                    if self.omni_api_key:
                        headers['Authorization'] = f"Bearer {self.omni_api_key}"
                    files = { 'image': (os.path.basename(image_path), img_bytes, 'application/octet-stream') }
                    params = { 'return': 'elements', 'format': 'json' }
                    resp = requests.post(self.omni_url, headers=headers, files=files, data=params, timeout=self.omni_timeout)
                    resp.raise_for_status()
                    data = resp.json() if hasattr(resp, 'json') else json.loads(resp.text)
                    elements: List[dict] = []
                    raw_elems = data.get('elements', []) if isinstance(data, dict) else []
                    for e in raw_elems:
                        try:
                            bbox = [int(e['bbox'][0]), int(e['bbox'][1]), int(e['bbox'][2]), int(e['bbox'][3])]
                            point = [int(e.get('point',[ (bbox[0]+bbox[2])//2, (bbox[1]+bbox[3])//2 ])[0]), int(e.get('point',[ (bbox[0]+bbox[2])//2, (bbox[1]+bbox[3])//2 ])[1])]
                            elements.append({ 'bbox': bbox, 'point': point })
                        except Exception:
                            continue
                    if max_elements and max_elements > 0:
                        elements = elements[:max_elements]
                    return elements
                except Exception as e:
                    print(f"[Annotator] OmniParser HTTP backend failed: {e}. Falling back to CV backend.")
                    # Fall through to CV backend
        # CV backend
        if not _PREPROC_DEPS_AVAILABLE:
            return []
        img = cv2.imread(image_path)
        if img is None:
            return []
        H, W = img.shape[:2]

        labels: List[Tuple[Tuple[int,int], str]] = []
        rects = _find_rect_candidates(img)

        # Add multi-color button regions if enabled
        if self.preprocess_color_buttons:
            mask = _detect_button_like_regions(img)
            if mask is not None:
                try:
                    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for c in cnts:
                        x, y, w, h = cv2.boundingRect(c)
                        if w*h >= 400:
                            rects.append([x, y, x+w, y+h])
                except Exception:
                    pass

        # Add MSER regions to capture subtle UI controls
        rects.extend(_find_regions_mser(img))

        # Clip + min-size
        rects = [_clip_box(r, W, H) for r in rects]
        rects = [r for r in rects if _min_size_ok(r)]

        # Classify and score
        boxes_scored: List[Tuple[List[int], float, str]] = []
        kind_pref = {'button': 3, 'input': 2, 'link': 1, 'generic': 0}
        for r in rects:
            kind, score, text = _classify_candidate(r, img, labels)
            # combine score with kind preference for dedup ranking
            boxes_scored.append((r, float(score) + kind_pref.get(kind, 0), text or ""))

        # Deduplicate by IoU/center
        dedup_boxes = _non_max_dedup([(b, s) for (b, s, _t) in boxes_scored])

        # Optional edge snapping refinement
        if self.preprocess_edge_snap:
            try:
                refined = []
                for b in dedup_boxes:
                    rb = _refine_box_edges(img, b)
                    refined.append(rb)
                dedup_boxes = refined
            except Exception:
                pass

        # If enabled, shrink input-like boxes to inner editable field
        if self.preprocess_refine_input_inner:
            try:
                refined2 = []
                for r in dedup_boxes:
                    kind, _score, _text = _classify_candidate(r, img, labels)
                    if kind == 'input':
                        refined2.append(_shrink_to_inner_field(img, r))
                    else:
                        refined2.append(r)
                dedup_boxes = refined2
            except Exception:
                pass

        # Rebuild with kind/text/score for ranking and selection
        scored_map = {tuple(b): (b, 0.0, "", "generic") for (b, _s, _t) in boxes_scored}
        enriched: List[Tuple[List[int], float, str, str]] = []
        for r in dedup_boxes:
            kind, score, text = _classify_candidate(r, img, labels)
            enriched.append((r, float(score), text or "", kind))

        # Final ranking: kind, score desc, smaller area first
        kind_rank = {'button': 3, 'input': 2, 'link': 1, 'generic': 0}
        enriched.sort(key=lambda t: (-kind_rank.get(t[3],0), -t[1], _area(t[0])))

        chosen = enriched if (not max_elements or max_elements <= 0) else enriched[:max_elements]
        chosen.sort(key=lambda t: _sort_tblr_key(t[0]))

        elements: List[dict] = []
        for (box, _score, text, kind) in chosen:
            x1, y1, x2, y2 = map(int, box)
            cx, cy = _center_of([x1, y1, x2, y2])
            elements.append({
                "bbox": [x1, y1, x2, y2],
                "point": [int(cx), int(cy)],
            })

        # Optional visualization
        if self.preprocess_visualize:
            try:
                from .visualizer import save_boxes_visualization
                out_path = self.preprocess_visualize_path if self.preprocess_visualize_path else None
                saved = save_boxes_visualization(image_path, elements, out_path=out_path)
                print(f"[Annotator] Saved preprocessing boxes visualization to: {saved}")
            except Exception as e:
                print(f"[Annotator] Visualization failed: {e}")

        return elements

    def preprocess_only(self, image_path: str, max_elements: Optional[int] = None) -> dict:
        """
        Run the preprocessing pipeline only and return boxes/points without calling the LLM.
        Returns a dict with keys {"img_size", "element"}, where each element has {bbox, point}.
        """
        if not _PREPROC_DEPS_AVAILABLE:
            return {"img_size": [0, 0], "element": []}
        with Image.open(image_path) as img:
            W, H = img.size
        n = int(max_elements) if isinstance(max_elements, int) else int(self.preprocess_max_elements)
        elements = self._compute_preprocess_hints(image_path, max_elements=n)
        return {"img_size": [W, H], "element": elements}

