# ai_pipeline.py — OPTIMIZED FOR SPEED + NO 502

import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

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
logger.info("Loading YOLOv8 model...")
_model = YOLO(weights)
logger.info("Model loaded.")
return _model

# ─── Constants ───────────────────────────────────────────────────────────────

CONF_THRESHOLD = 0.25
MIN_CONTOUR_AREA = 150

def _yolo_masks_to_binary(result, img_shape: Tuple[int, int]) -> np.ndarray:
h, w = img_shape
combined = np.zeros((h, w), dtype=np.uint8)

```
if result.masks is None:
    return combined

masks_data = result.masks.data.cpu().numpy()
confs = result.boxes.conf.cpu().numpy()

for mask, conf in zip(masks_data, confs):
    if conf < CONF_THRESHOLD:
        continue

    m = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    m = (m > 0.5).astype(np.uint8) * 255
    combined = cv2.bitwise_or(combined, m)

return combined
```

def _filter_contours(contours: List[np.ndarray], img_area: int) -> List[np.ndarray]:
filtered = []
for c in contours:
area = cv2.contourArea(c)
if area < MIN_CONTOUR_AREA:
continue
if area > 0.6 * img_area:
continue
filtered.append(c)
return filtered

# ─── MAIN PIPELINE ───────────────────────────────────────────────────────────

def detect_buildings(
img_bgr: np.ndarray,
polygon_mask: Optional[np.ndarray] = None,
model_weights: str = "yolov8n-seg.pt",
) -> Dict[str, Any]:

```
# 🔥 STEP 0: Resize (CRITICAL for Render)
h, w = img_bgr.shape[:2]
scale = 480 / max(h, w)

if scale < 1:
    img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    h, w = img_bgr.shape[:2]

img_area = h * w

model = _load_model(model_weights)

logger.info("Running YOLO inference...")

# 🔥 OPTIMIZED INFERENCE
results = model(img_bgr, conf=0.35, max_det=100, verbose=False)

yolo_mask = _yolo_masks_to_binary(results[0], (h, w))

# Apply polygon mask
if polygon_mask is not None:
    polygon_mask = cv2.resize(polygon_mask, (w, h))
    yolo_mask = cv2.bitwise_and(yolo_mask, polygon_mask)

# 🔥 FAST CONTOUR DETECTION (NO WATERSHED)
contours, _ = cv2.findContours(
    yolo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

contours = _filter_contours(contours, img_area)

count = len(contours)
logger.info(f"Detected buildings: {count}")

# ── Metrics ───────────────────────────────────────────────────────────

building_mask = np.zeros((h, w), dtype=np.uint8)
cv2.drawContours(building_mask, contours, -1, 255, thickness=cv2.FILLED)

if polygon_mask is not None:
    total_pixels = int(np.count_nonzero(polygon_mask))
else:
    total_pixels = img_area

built_up_pixels = int(np.count_nonzero(building_mask))

return {
    "count": count,
    "contours": contours,
    "building_mask": building_mask,
    "built_up_pixels": built_up_pixels,
    "total_pixels": total_pixels,
}
```

# ─── OVERLAY ────────────────────────────────────────────────────────────────

def draw_overlay(
img_bgr: np.ndarray,
contours: List[np.ndarray],
polygon_coords_px: Optional[np.ndarray] = None,
fill_alpha: float = 0.35,
) -> np.ndarray:

```
overlay = img_bgr.copy()
output = img_bgr.copy()

cv2.drawContours(overlay, contours, -1, (255, 200, 0), thickness=cv2.FILLED)
cv2.addWeighted(overlay, fill_alpha, output, 1 - fill_alpha, 0, output)

cv2.drawContours(output, contours, -1, (0, 220, 255), thickness=2)

if polygon_coords_px is not None:
    cv2.polylines(output, [polygon_coords_px], True, (0, 0, 255), 3)

label = f"Buildings: {len(contours)}"
cv2.rectangle(output, (0, 0), (220, 36), (0, 0, 0), -1)
cv2.putText(
    output, label,
    (6, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
    (0, 220, 255), 2, cv2.LINE_AA
)

return output
```
