# ai_pipeline.py — MINIMAL & FREE-TIER SAFE

import logging
import threading
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(**name**)

# ─── Model singleton ────────────────────────────────────────────────────────

_model_lock = threading.Lock()
_model: Optional[Any] = None

def _load_model(weights: str = "yolov8n-seg.pt") -> Any:
global _model
if _model is not None:
return _model
with _model_lock:
if _model is None:
from ultralytics import YOLO
logger.info("Loading YOLO model...")
_model = YOLO(weights)
return _model

def detect_buildings(
img_bgr: np.ndarray,
polygon_mask: Optional[np.ndarray] = None,
) -> Dict:

```
# 🔥 FORCE SMALL IMAGE
h, w = img_bgr.shape[:2]
scale = 384 / max(h, w)
if scale < 1:
    img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    h, w = img_bgr.shape[:2]

model = _load_model()

# 🔥 LIGHT INFERENCE
results = model(img_bgr, conf=0.4, max_det=80, verbose=False)

mask = np.zeros((h, w), dtype=np.uint8)

if results[0].masks is not None:
    masks = results[0].masks.data.cpu().numpy()
    for m in masks:
        m = cv2.resize(m, (w, h))
        m = (m > 0.5).astype(np.uint8) * 255
        mask = cv2.bitwise_or(mask, m)

# Apply polygon mask
if polygon_mask is not None:
    polygon_mask = cv2.resize(polygon_mask, (w, h))
    mask = cv2.bitwise_and(mask, polygon_mask)

# Simple contours
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter noise
filtered = []
for c in contours:
    area = cv2.contourArea(c)
    if area > 100:
        filtered.append(c)

building_mask = np.zeros((h, w), dtype=np.uint8)
cv2.drawContours(building_mask, filtered, -1, 255, -1)

return {
    "count": len(filtered),
    "contours": filtered,
    "built_up_pixels": int(np.count_nonzero(building_mask)),
    "total_pixels": h * w,
}
```
