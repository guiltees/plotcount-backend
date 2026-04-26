"""
main.py — PlotCount AI FastAPI entry point.

Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
import os
from typing import List, Optional

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from ai_pipeline import detect_buildings, draw_overlay
from image_acquisition import (
    base64_to_image,
    fetch_satellite_image,
    image_to_base64,
    resize_if_needed,
)
from kml_parser import flatten_polygon, parse_kml
from polygon_utils import (
    bounding_box,
    compute_density,
    coords_to_pixels,
    geo_area_km2,
    make_polygon_mask,
)

# ─── Bootstrap ───────────────────────────────────────────────────────────────

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PlotCount AI",
    description="Detect and count buildings within a geographic polygon using AI segmentation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Request / Response models ───────────────────────────────────────────────

class LatLng(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)


class AnalyzeRequest(BaseModel):
    polygon: Optional[List[LatLng]] = Field(
        None,
        description="List of lat/lng points defining the boundary polygon. "
                    "Required for source='map' and source='kml' (after parsing).",
    )
    source: str = Field(
        ...,
        description="One of: 'map', 'image', 'kml'",
    )
    image: Optional[str] = Field(
        None,
        description="Base64-encoded image. Required when source='image'.",
    )
    kml: Optional[str] = Field(
        None,
        description="Raw KML file content as string. Required when source='kml'.",
    )

    @validator("source")
    def source_must_be_valid(cls, v):
        if v not in {"map", "image", "kml"}:
            raise ValueError("source must be 'map', 'image', or 'kml'")
        return v


class AnalyzeResponse(BaseModel):
    count: int
    density: float                # buildings per km²
    built_up_percentage: float    # 0–100
    overlay_image: str            # base64-encoded JPEG


# ─── Helper ──────────────────────────────────────────────────────────────────

def _coords_list(polygon: List[LatLng]) -> List[dict]:
    return [{"lat": p.lat, "lng": p.lng} for p in polygon]


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "service": "PlotCount AI"}


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy"}


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
async def analyze(req: AnalyzeRequest):
    """
    Analyze a geographic area and count buildings.

    - **map**: Fetches satellite imagery using the bounding box of the polygon.
    - **kml**: Parses KML to extract the polygon, then same as map.
    - **image**: Uses the uploaded base64 image directly.
    """

    logger.info("Received /analyze request: source=%s", req.source)

    # ── 1. Resolve polygon + image ────────────────────────────────────────

    img_bgr: Optional[np.ndarray] = None
    coords: Optional[List[dict]] = None
    bbox_geo: Optional[dict] = None
    area_km2: float = 0.0

    if req.source == "kml":
        # Parse KML → polygon, then treat like 'map'
        if not req.kml:
            raise HTTPException(400, "kml field is required when source='kml'")
        try:
            polygons = parse_kml(req.kml)
            coords = flatten_polygon(polygons)
        except ValueError as e:
            raise HTTPException(422, f"KML parse error: {e}")
        bbox_geo = bounding_box(coords)
        area_km2 = geo_area_km2(coords)
        try:
            img_rgb = fetch_satellite_image(bbox_geo)
        except EnvironmentError as e:
            raise HTTPException(500, str(e))
        except RuntimeError as e:
            raise HTTPException(502, str(e))
        # Convert RGB → BGR
        img_bgr = img_rgb[:, :, ::-1].copy()

    elif req.source == "map":
        if not req.polygon or len(req.polygon) < 3:
            raise HTTPException(400, "polygon must have ≥ 3 points for source='map'")
        coords = _coords_list(req.polygon)
        bbox_geo = bounding_box(coords)
        area_km2 = geo_area_km2(coords)
        try:
            img_rgb = fetch_satellite_image(bbox_geo)
        except EnvironmentError as e:
            raise HTTPException(500, str(e))
        except RuntimeError as e:
            raise HTTPException(502, str(e))
        img_bgr = img_rgb[:, :, ::-1].copy()

    elif req.source == "image":
        if not req.image:
            raise HTTPException(400, "image field is required when source='image'")
        try:
            img_bgr = base64_to_image(req.image)
        except Exception as e:
            raise HTTPException(422, f"Failed to decode image: {e}")

        if req.polygon and len(req.polygon) >= 3:
            coords = _coords_list(req.polygon)
            area_km2 = 0.0          # No geo reference for uploaded images
        # If no polygon provided, full image is used

    # ── 2. Resize image ───────────────────────────────────────────────────

    img_bgr = resize_if_needed(img_bgr, max_edge=1024)
h, w = img_bgr.shape[:2]

# 🔥 LIMIT AREA (CRITICAL FOR FREE TIER)
if h * w > 500 * 500:
    raise HTTPException(
        status_code=400,
        detail="Area too large. Please zoom in more."
    )

    # ── 3. Polygon mask (restrict analysis to ROI) ────────────────────────

    polygon_mask: Optional[np.ndarray] = None
    polygon_px: Optional[np.ndarray] = None

    if coords and bbox_geo:
        polygon_mask = make_polygon_mask(coords, bbox_geo, w, h)
        polygon_px = coords_to_pixels(coords, bbox_geo, w, h)
    elif coords and req.source == "image":
        # For uploaded images, treat polygon coords as normalised 0–1 fractions
        # if values are in [0,1], or as pixel coords if larger.
        pts = []
        for c in coords:
            px = int(c["lng"] * w) if c["lng"] <= 1 else int(c["lng"])
            py = int(c["lat"] * h) if c["lat"] <= 1 else int(c["lat"])
            pts.append([[px, py]])
        polygon_px = np.array(pts, dtype=np.int32)
        polygon_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(polygon_mask, [polygon_px], 255)

    # ── 4–7: AI pipeline ──────────────────────────────────────────────────

    try:
        result = detect_buildings(img_bgr, polygon_mask=polygon_mask)
    except RuntimeError as e:
        raise HTTPException(500, f"AI pipeline error: {e}")

    count = result["count"]
    built_up_pixels = result["built_up_pixels"]
    total_pixels = result["total_pixels"]
    contours = result["contours"]

    # ── Metrics ───────────────────────────────────────────────────────────

    built_up_pct = round(built_up_pixels / max(total_pixels, 1) * 100, 2)
    density = compute_density(count, area_km2)

    # ── Step 8: Overlay visualization ────────────────────────────────────

    overlay_bgr = draw_overlay(img_bgr, contours, polygon_coords_px=polygon_px)
    overlay_b64 = image_to_base64(overlay_bgr, fmt="JPEG")

    logger.info(
        "Result: count=%d density=%.2f built_up=%.2f%%",
        count, density, built_up_pct,
    )

    return AnalyzeResponse(
        count=count,
        density=density,
        built_up_percentage=built_up_pct,
        overlay_image=overlay_b64,
    )


# ─── Error handlers ───────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s", request.url)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
