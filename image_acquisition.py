# image_acquisition.py — LIGHT VERSION

import os
import base64
from io import BytesIO

import requests
import numpy as np
from PIL import Image

STATIC_MAPS_URL = "https://maps.googleapis.com/maps/api/staticmap"

MAX_TILE_PX = 384
SCALE = 1

def fetch_satellite_image(bbox):
api_key = os.getenv("GOOGLE_MAPS_API_KEY")

```
center_lat = (bbox["min_lat"] + bbox["max_lat"]) / 2
center_lng = (bbox["min_lng"] + bbox["max_lng"]) / 2

params = {
    "center": f"{center_lat},{center_lng}",
    "zoom": 19,
    "size": f"{MAX_TILE_PX}x{MAX_TILE_PX}",
    "scale": SCALE,
    "maptype": "satellite",
    "key": api_key,
}

response = requests.get(STATIC_MAPS_URL, params=params, timeout=10)

img = Image.open(BytesIO(response.content)).convert("RGB")
return np.array(img)
```

def base64_to_image(b64_str: str) -> np.ndarray:
if "," in b64_str:
b64_str = b64_str.split(",")[1]

```
raw = base64.b64decode(b64_str)
img = Image.open(BytesIO(raw)).convert("RGB")
return np.array(img)[:, :, ::-1]
```

def image_to_base64(img_bgr: np.ndarray) -> str:
img_rgb = img_bgr[:, :, ::-1]
pil_img = Image.fromarray(img_rgb)
buf = BytesIO()
pil_img.save(buf, format="JPEG", quality=70)
return base64.b64encode(buf.getvalue()).decode()
