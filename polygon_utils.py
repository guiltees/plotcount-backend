"""
polygon_utils.py — Geo/pixel polygon utilities.

Converts geographic polygons (lat/lng) to pixel masks and computes
bounding boxes, areas, and density metrics.
"""

import math
from typing import List, Dict, Tuple

import numpy as np
import cv2
from shapely.geometry import Polygon as ShapelyPolygon


# ─── Geo utilities ──────────────────────────────────────────────────────────

def bounding_box(coords: List[Dict[str, float]]) -> Dict[str, float]:
    """Return the geographic bounding box of a polygon."""
    lats = [c["lat"] for c in coords]
    lngs = [c["lng"] for c in coords]
    return {
        "min_lat": min(lats),
        "max_lat": max(lats),
        "min_lng": min(lngs),
        "max_lng": max(lngs),
    }


def geo_area_km2(coords: List[Dict[str, float]]) -> float:
    """
    Approximate geographic area of a polygon in km².
    Uses the Shoelace formula with a local flat-earth projection.
    Accurate for small areas (< few hundred km²).
    """
    if len(coords) < 3:
        return 0.0

    # Earth radius in km
    R = 6371.0
    lats = [c["lat"] for c in coords]
    lngs = [c["lng"] for c in coords]
    mean_lat = math.radians(sum(lats) / len(lats))

    # Convert to local Cartesian (km)
    xs = [math.radians(lng) * R * math.cos(mean_lat) for lng in lngs]
    ys = [math.radians(lat) * R for lat in lats]

    # Shoelace formula
    n = len(xs)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j]
        area -= xs[j] * ys[i]
    return abs(area) / 2.0


# ─── Pixel projection ────────────────────────────────────────────────────────

def coords_to_pixels(
    coords: List[Dict[str, float]],
    bbox: Dict[str, float],
    img_width: int,
    img_height: int,
) -> np.ndarray:
    """
    Map geographic coords to pixel coordinates within an image whose
    geographic extent is given by bbox.

    Returns an int32 numpy array of shape (N, 1, 2) suitable for cv2.fillPoly.
    """
    min_lat, max_lat = bbox["min_lat"], bbox["max_lat"]
    min_lng, max_lng = bbox["min_lng"], bbox["max_lng"]

    lat_span = max_lat - min_lat or 1e-9
    lng_span = max_lng - min_lng or 1e-9

    pts = []
    for c in coords:
        x = int((c["lng"] - min_lng) / lng_span * img_width)
        # Flip y: lat increases upward, pixel row increases downward
        y = int((max_lat - c["lat"]) / lat_span * img_height)
        pts.append([[x, y]])

    return np.array(pts, dtype=np.int32)


def make_polygon_mask(
    coords: List[Dict[str, float]],
    bbox: Dict[str, float],
    img_width: int,
    img_height: int,
) -> np.ndarray:
    """
    Create a binary mask (uint8) for the geographic polygon projected onto
    an image of the given dimensions.
    """
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    pts = coords_to_pixels(coords, bbox, img_width, img_height)
    cv2.fillPoly(mask, [pts], 255)
    return mask


# ─── Shapely helpers ─────────────────────────────────────────────────────────

def coords_to_shapely(coords: List[Dict[str, float]]) -> ShapelyPolygon:
    """Convert list of coord dicts to a Shapely Polygon."""
    return ShapelyPolygon([(c["lng"], c["lat"]) for c in coords])


def is_valid_polygon(coords: List[Dict[str, float]]) -> bool:
    """Return True if the polygon is geometrically valid."""
    if len(coords) < 3:
        return False
    poly = coords_to_shapely(coords)
    return poly.is_valid


def convex_hull(coords: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """Return the convex hull of the polygon coords."""
    poly = coords_to_shapely(coords)
    hull = poly.convex_hull
    return [{"lat": y, "lng": x} for x, y in hull.exterior.coords]


# ─── Density ─────────────────────────────────────────────────────────────────

def compute_density(count: int, area_km2: float) -> float:
    """Return houses per km², guarded against zero area."""
    if area_km2 <= 0:
        return 0.0
    return round(count / area_km2, 2)
