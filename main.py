from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

# ✅ FIXED INDENTATION

class LatLng(BaseModel):
lat: float
lng: float

class AnalyzeRequest(BaseModel):
polygon: Optional[List[LatLng]] = None
source: str

@app.get("/health")
def health():
return {"status": "ok"}

# 🔥 TEMP DUMMY (to confirm backend works)

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
return {
"count": 5,
"density": 0,
"built_up_percentage": 0,
"overlay_image": ""
}
