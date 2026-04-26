import cv2
import numpy as np
from ultralytics import YOLO

# Load once

model = YOLO("yolov8n-seg.pt")

def detect_buildings(img_bgr, polygon_mask=None):

```
# 🔥 FORCE SMALL IMAGE
h, w = img_bgr.shape[:2]
scale = 384 / max(h, w)

if scale < 1:
    img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))

results = model(img_bgr, conf=0.4, max_det=80, verbose=False)

mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)

if results[0].masks is not None:
    for m in results[0].masks.data.cpu().numpy():
        m = cv2.resize(m, (img_bgr.shape[1], img_bgr.shape[0]))
        m = (m > 0.5).astype(np.uint8) * 255
        mask = cv2.bitwise_or(mask, m)

# Apply polygon mask
if polygon_mask is not None:
    polygon_mask = cv2.resize(polygon_mask, (img_bgr.shape[1], img_bgr.shape[0]))
    mask = cv2.bitwise_and(mask, polygon_mask)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

filtered = []
for c in contours:
    if cv2.contourArea(c) > 100:
        filtered.append(c)

building_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
cv2.drawContours(building_mask, filtered, -1, 255, -1)

return {
    "count": len(filtered),
    "contours": filtered,
    "built_up_pixels": int(np.count_nonzero(building_mask)),
    "total_pixels": img_bgr.shape[0] * img_bgr.shape[1],
}
```
