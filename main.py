# main.py — MINIMAL WORKING VERSION

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

import numpy as np

from ai_pipeline import detect_buildings
from image_acquisition import fetch_satellite_image, base64_to_image

app = FastAPI()

class LatLng(BaseModel):
lat: float
lng: float

class AnalyzeRequest(BaseModel):
polygon: Optional[List[LatLng]] = None
source: str
image: Optional[str] = None

@app.get("/health")
def health():
return {"status": "ok"}

@app.post("/analyze")
def analyze(req: AnalyzeRequest):

```
if req.source == "map":
    if not req.polygon:
        raise HTTPException(400, "Polygon required")

    coords = [{"lat": p.lat, "lng": p.lng} for p in req.polygon]

    bbox = {
        "min_lat": min(c["lat"] for c in coords),
        "max_lat": max(c["lat"] for c in coords),
        "min_lng": min(c["lng"] for c in coords),
        "max_lng": max(c["lng"] for c in coords),
    }

    img = fetch_satellite_image(bbox)
    img = img[:, :, ::-1]

elif req.source == "image":
    if not req.image:
        raise HTTPException(400, "Image required")
    img = base64_to_image(req.image)

else:
    raise HTTPException(400, "Invalid source")

# 🔥 HARD LIMIT
if img.shape[0] * img.shape[1] > 400 * 400:
    raise HTTPException(400, "Zoom more")

result = detect_buildings(img)

return {
    "count": result["count"],
    "density": 0,
    "built_up_percentage": 0,
    "overlay_image": ""  # disabled for speed
}
```
