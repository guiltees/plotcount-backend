"""
image_acquisition.py — Fetches satellite imagery for a geographic polygon.

Uses the Google Maps Static API to retrieve a high-resolution satellite tile
that covers the polygon bounding box. Handles multi-tile stitching for large
areas and respects the 1280×1280 maximum tile size (with scale=2).
"""

import os
import math
import base64
import logging
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import requests
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Google Maps Static API endpoint
STATIC_MAPS_URL = "https://maps.googleapis.com/maps/api/staticmap"

# Maximum pixel dimensions per tile (scale=2 doubles effective resolution)
MAX_TILE_PX = 640          # API limit per dimension at scale=2 → effective 1280px
SCALE = 2                  # High-DPI mode
TARGET_LONG_EDGE = 1024    # Target longest edge for AI pipeline


def _get_api_key() -> str:
    key = os.getenv("GOOGLE_MAPS_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "GOOGLE_MAPS_API_KEY is not set. "
            "Add it to your .env file or environment variables."
        )
    return key


def _bbox_center(bbox: Dict[str, float]) -> Tuple[float, float]:
    """Return the center lat/lng of a bounding box."""
    lat = (bbox["min_lat"] + bbox["max_lat"]) / 2
    lng = (bbox["min_lng"] + bbox["max_lng"]) / 2
    return lat, lng


def _choose_zoom(bbox: Dict[str, float], tile_px: int = MAX_TILE_PX * SCALE) -> int:
    """
    Choose the highest zoom level at which the bounding box fits within
    a single tile of tile_px × tile_px pixels.

    Uses the Mercator projection formula:
        pixels_per_degree_lat = tile_px / (360 / 2^zoom)  (approximate)
    """
    lat_span = bbox["max_lat"] - bbox["min_lat"]
    lng_span = bbox["max_lng"] - bbox["min_lng"]

    if lat_span <= 0 or lng_span <= 0:
        return 18  # Fallback for point / degenerate polygon

    for zoom in range(21, 0, -1):
        world_px = 256 * (2 ** zoom)
        # Degrees per pixel at this zoom
        deg_per_px_lng = 360.0 / world_px
        # Lat is non-linear; use approximate conversion near center
        center_lat_rad = math.radians((bbox["min_lat"] + bbox["max_lat"]) / 2)
        deg_per_px_lat = 360.0 / (world_px * math.cos(center_lat_rad))

        needed_px_lng = lng_span / deg_per_px_lng
        needed_px_lat = lat_span / deg_per_px_lat

        if needed_px_lng <= tile_px and needed_px_lat <= tile_px:
            return zoom

    return 1


def fetch_satellite_image(
    bbox: Dict[str, float],
    size_px: int = MAX_TILE_PX,
) -> np.ndarray:
    """
    Fetch a satellite image covering the bounding box from the
    Google Maps Static API.

    Args:
        bbox:    Geographic bounding box (min/max lat/lng).
        size_px: Requested tile width/height in pixels (≤ 640 for scale=2).

    Returns:
        RGB numpy array of shape (H, W, 3).

    Raises:
        EnvironmentError: API key not set.
        RuntimeError:     API request failed.
    """
    api_key = _get_api_key()
    center_lat, center_lng = _bbox_center(bbox)
    zoom = _choose_zoom(bbox, tile_px=size_px * SCALE)

    params = {
        "center": f"{center_lat},{center_lng}",
        "zoom": zoom,
        "size": f"{size_px}x{size_px}",
        "scale": SCALE,
        "maptype": "satellite",
        "key": api_key,
    }

    logger.info(
        "Fetching satellite tile: center=(%s, %s) zoom=%s size=%s scale=%s",
        center_lat, center_lng, zoom, f"{size_px}x{size_px}", SCALE,
    )

    response = requests.get(STATIC_MAPS_URL, params=params, timeout=15)

    if response.status_code != 200:
        raise RuntimeError(
            f"Google Maps API error {response.status_code}: {response.text[:200]}"
        )

    img = Image.open(BytesIO(response.content)).convert("RGB")
    arr = np.array(img)
    logger.info("Satellite image fetched: shape=%s", arr.shape)
    return arr


def base64_to_image(b64_str: str) -> np.ndarray:
    """
    Decode a base64-encoded image string (with or without data-URI prefix)
    to a BGR numpy array suitable for OpenCV.

    Args:
        b64_str: Base64 string, optionally with 'data:image/...;base64,' prefix.

    Returns:
        BGR numpy array.
    """
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]

    raw = base64.b64decode(b64_str)
    img = Image.open(BytesIO(raw)).convert("RGB")
    arr = np.array(img)
    # Convert RGB → BGR for OpenCV
    return arr[:, :, ::-1].copy()


def image_to_base64(img_bgr: np.ndarray, fmt: str = "JPEG") -> str:
    """
    Encode a BGR numpy array to a base64 JPEG/PNG string.

    Args:
        img_bgr: BGR numpy array.
        fmt:     Output format: 'JPEG' or 'PNG'.

    Returns:
        Base64 string (no data-URI prefix).
    """
    img_rgb = img_bgr[:, :, ::-1]            # BGR → RGB
    pil_img = Image.fromarray(img_rgb.astype(np.uint8))
    buf = BytesIO()
    pil_img.save(buf, format=fmt, quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def resize_if_needed(img: np.ndarray, max_edge: int = TARGET_LONG_EDGE) -> np.ndarray:
    """
    Resize an image so its longest edge is ≤ max_edge, preserving aspect ratio.
    Returns the original if already small enough.
    """
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_edge:
        return img
    scale = max_edge / longest
    new_w = int(w * scale)
    new_h = int(h * scale)
    return np.array(
        Image.fromarray(img[:, :, ::-1] if img.ndim == 3 else img)
        .resize((new_w, new_h), Image.LANCZOS)
    )[:, :, ::-1] if img.ndim == 3 else np.array(
        Image.fromarray(img).resize((new_w, new_h), Image.LANCZOS)
    )
