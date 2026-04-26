"""
kml_parser.py — Parses KML files and extracts polygon coordinates.
Supports both simple <coordinates> and multi-polygon KML documents.
"""

import xml.etree.ElementTree as ET
from typing import List, Dict, Any


# KML namespace
KML_NS = "http://www.opengis.net/kml/2.2"
# Fallback for files without explicit namespace
ALT_NS = ""


def _strip_ns(tag: str) -> str:
    """Strip XML namespace from a tag string."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _parse_coord_string(coord_str: str) -> List[Dict[str, float]]:
    """
    Parse a KML coordinate string into a list of lat/lng dicts.
    KML format: 'lng,lat,alt lng,lat,alt ...'
    """
    coords = []
    for token in coord_str.strip().split():
        parts = token.split(",")
        if len(parts) >= 2:
            try:
                lng = float(parts[0])
                lat = float(parts[1])
                coords.append({"lat": lat, "lng": lng})
            except ValueError:
                continue
    return coords


def _recursive_find_coordinates(element: ET.Element) -> List[str]:
    """Recursively walk XML tree and collect all <coordinates> text blocks."""
    results = []
    local = _strip_ns(element.tag)
    if local == "coordinates":
        if element.text:
            results.append(element.text)
    for child in element:
        results.extend(_recursive_find_coordinates(child))
    return results


def parse_kml(kml_content: str) -> List[List[Dict[str, float]]]:
    """
    Parse KML content string and return a list of polygons.
    Each polygon is a list of {"lat": float, "lng": float} dicts.

    Args:
        kml_content: Raw KML file content as string.

    Returns:
        List of polygons. Each polygon is a list of coordinate dicts.

    Raises:
        ValueError: If no coordinates are found or XML is malformed.
    """
    try:
        root = ET.fromstring(kml_content)
    except ET.ParseError as e:
        raise ValueError(f"Malformed KML XML: {e}")

    coord_strings = _recursive_find_coordinates(root)

    if not coord_strings:
        raise ValueError("No <coordinates> elements found in KML.")

    polygons = []
    for cs in coord_strings:
        coords = _parse_coord_string(cs)
        if len(coords) >= 3:          # A valid polygon needs at least 3 points
            polygons.append(coords)

    if not polygons:
        raise ValueError("KML contains coordinates but no valid polygons (need ≥ 3 points).")

    return polygons


def parse_kml_file(file_path: str) -> List[List[Dict[str, float]]]:
    """Convenience wrapper that reads a KML file from disk."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return parse_kml(content)


def flatten_polygon(polygons: List[List[Dict[str, float]]]) -> List[Dict[str, float]]:
    """Return the first (largest) polygon from a multi-polygon KML."""
    if not polygons:
        raise ValueError("No polygons to flatten.")
    # Pick the polygon with the most points (usually the outer boundary)
    return max(polygons, key=len)
