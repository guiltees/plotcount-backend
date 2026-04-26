from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from ai_pipeline import detect_buildings
from image_acquisition import fetch_satellite_image

app = FastAPI()

class LatLng(BaseModel):
lat: float
lng: float

class AnalyzeRequest(BaseModel):
polygon: Optional[List[LatLng]] = None
source: str

@app.get("/health")
def health():
return {"status": "ok"}

@app.post("/analyze")
def analyze(req: AnalyzeRequest):

```
if req.source != "map":
    raise HTTPException(400, "Only map supported for now")

if not req.polygon or len(req.polygon) < 3:
    raise HTTPException(400, "Polygon required")

coords = [{"lat": p.lat, "lng": p.lng} for p in req.polygon]

bbox = {
    "min_lat": min(c["lat"] for c in coords),
    "max_lat": max(c["lat"] for c in coords),
    "min_lng": min(c["lng"] for c in coords),
    "max_lng": max(c["lng"] for c in coords),
}

img = fetch_satellite_image(bbox)

# 🔥 HARD LIMIT
if img.shape[0] * img.shape[1] > 400 * 400:
    raise HTTPException(400, "Zoom more")

img = img[:, :, ::-1]  # RGB → BGR

result = detect_buildings(img)

return {
    "count": result["count"],
    "density": 0,
    "built_up_percentage": 0,
    "overlay_image": ""
}
```
