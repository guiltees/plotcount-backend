import os
import requests
import numpy as np
from PIL import Image
from io import BytesIO

STATIC_MAPS_URL = "https://maps.googleapis.com/maps/api/staticmap"

# 🔥 FREE TIER SAFE VALUES

SIZE = 384
ZOOM = 20

def fetch_satellite_image(bbox):
api_key = os.getenv("GOOGLE_MAPS_API_KEY")

```
center_lat = (bbox["min_lat"] + bbox["max_lat"]) / 2
center_lng = (bbox["min_lng"] + bbox["max_lng"]) / 2

params = {
    "center": f"{center_lat},{center_lng}",
    "zoom": ZOOM,
    "size": f"{SIZE}x{SIZE}",
    "scale": 1,
    "maptype": "satellite",
    "key": api_key,
}

response = requests.get(STATIC_MAPS_URL, params=params, timeout=10)

img = Image.open(BytesIO(response.content)).convert("RGB")
return np.array(img)
```
