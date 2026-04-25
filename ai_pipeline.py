"""
ai_pipeline.py — Core AI building-detection pipeline.

Steps:
  1. Load / cache YOLOv8-seg model
  2. Run segmentation inference
  3. Apply instance-separation via distance transform + watershed
  4. Post-process contours (noise removal, merge broken fragments)
  5. Count buildings, compute metrics, draw overlay
"""

import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi

logger = logging.getLogger(__name__)

# ─── Model singleton ────────────────────────────────────────────────────────

_model_lock = threading.Lock()
_model: Optional[Any] = None  # ultralytics.YOLO instance


def _load_model(weights: str = "yolov8n-seg.pt") -> Any:
    """Load (or return cached) YOLOv8-seg model."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:
            try:
                from ultralytics import YOLO  # deferred import
                logger.info("Loading YOLOv8 model: %s", weights)
                _model = YOLO(weights)
                logger.info("Model loaded and cached.")
            except ImportError:
                raise RuntimeError(
                    "ultralytics package not installed. "
                    "Run: pip install ultralytics"
                )
    return _model


# ─── Constants ───────────────────────────────────────────────────────────────

CONF_THRESHOLD = 0.40          # Minimum detection confidence
MIN_CONTOUR_AREA = 150         # px² — ignore tiny blobs
MIN_CONTOUR_AREA_RATIO = 0.0001  # Fraction of image — ignore relative tiny blobs
BUILDING_CLASSES = {           # COCO classes that represent buildings
    "building", "house", "shed", "garage",
    # Fallback: treat any rectangular object as potential building
}
# COCO class IDs that most resemble buildings/rooftops when detected in aerial imagery
FALLBACK_CLASSES = {0, 2, 5, 7, 56, 57, 58, 59, 60, 61, 62, 63, 67, 72, 73, 76, 77, 78}


def _yolo_masks_to_binary(result, img_shape: Tuple[int, int]) -> np.ndarray:
    """
    Extract a single combined binary mask from a YOLO segmentation result.

    Args:
        result:    Single YOLO Results object.
        img_shape: (H, W) of the target image.

    Returns:
        uint8 mask of shape (H, W); 255 = building pixel.
    """
    h, w = img_shape
    combined = np.zeros((h, w), dtype=np.uint8)

    if result.masks is None:
        logger.warning("No segmentation masks in YOLO result.")
        return combined

    masks_data = result.masks.data.cpu().numpy()   # (N, H', W')
    confs = result.boxes.conf.cpu().numpy()         # (N,)
    class_ids = result.boxes.cls.cpu().numpy().astype(int)  # (N,)

    for i, (mask, conf, cls_id) in enumerate(zip(masks_data, confs, class_ids)):
        if conf < CONF_THRESHOLD:
            continue

        # Resize mask to image size
        m = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        m = (m > 0.5).astype(np.uint8) * 255
        combined = cv2.bitwise_or(combined, m)

    return combined


def _watershed_instance_separation(binary_mask: np.ndarray) -> np.ndarray:
    """
    Apply distance-transform + watershed to separate touching buildings.

    Args:
        binary_mask: uint8 binary mask (255 = foreground).

    Returns:
        Labelled array (int32) where each distinct building has a unique label > 0.
    """
    # Distance transform
    dist = ndi.distance_transform_edt(binary_mask)

    # Find local maxima as seeds (markers)
    # Use a structuring element proportional to expected building size
    kernel_size = max(3, int(min(binary_mask.shape) * 0.02))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    dilated_dist = cv2.dilate(dist.astype(np.float32), kernel)
    local_max = (dist == dilated_dist) & (dist > 0)

    # Label the local maxima as markers
    markers, _ = ndi.label(local_max)

    # Watershed
    # OpenCV watershed expects a 3-channel uint8 image
    img_3ch = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    markers_ws = markers.astype(np.int32)
    cv2.watershed(img_3ch, markers_ws)

    # Watershed labels: -1 = boundary, 0 = background, >0 = instance
    labelled = np.where(markers_ws > 0, markers_ws, 0).astype(np.int32)
    return labelled


def _filter_contours(
    contours: List[np.ndarray],
    img_area: int,
    min_area: int = MIN_CONTOUR_AREA,
) -> List[np.ndarray]:
    """Remove contours that are too small (noise) or too large (entire image)."""
    filtered = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        # Reject if > 60 % of the image (probably the boundary polygon itself)
        if area > 0.60 * img_area:
            continue
        filtered.append(c)
    return filtered


def _merge_nearby_contours(
    binary: np.ndarray,
    kernel_size: int = 5,
) -> np.ndarray:
    """
    Morphologically close nearby fragments so broken rooftop segments
    are merged into a single contour.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size)
    )
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return closed


def detect_buildings(
    img_bgr: np.ndarray,
    polygon_mask: Optional[np.ndarray] = None,
    model_weights: str = "yolov8n-seg.pt",
) -> Dict[str, Any]:
    """
    Run the full building-detection pipeline on a BGR image.

    Args:
        img_bgr:       Input image (BGR, uint8).
        polygon_mask:  Optional uint8 mask restricting the analysis area.
                       255 = inside polygon. If None, entire image is used.
        model_weights: Path to YOLOv8 weights file.

    Returns:
        Dict with keys:
          - count (int)
          - contours (list of np.ndarray)
          - building_mask (np.ndarray)
          - built_up_pixels (int)
          - total_pixels (int)
    """
    model = _load_model(model_weights)
    h, w = img_bgr.shape[:2]
    img_area = h * w

    # ── Step 3: YOLOv8 inference ───────────────────────────────────────────
    logger.info("Running YOLOv8 inference on image (%dx%d)…", w, h)
    results = model(img_bgr, verbose=False)

    # ── Step 2 (mask application) + Step 3 output ─────────────────────────
    yolo_mask = _yolo_masks_to_binary(results[0], (h, w))

    # Apply polygon restriction if provided
    if polygon_mask is not None:
        yolo_mask = cv2.bitwise_and(yolo_mask, polygon_mask)

    # ── Step 5a: Merge nearby fragments (broken rooftops) ─────────────────
    merged_mask = _merge_nearby_contours(yolo_mask, kernel_size=7)

    # ── Step 4: Watershed instance separation ─────────────────────────────
    labelled = _watershed_instance_separation(merged_mask)

    # ── Step 5b: Extract per-instance contours ────────────────────────────
    num_instances = labelled.max()
    raw_contours = []
    for label in range(1, num_instances + 1):
        instance_mask = ((labelled == label) * 255).astype(np.uint8)
        cnts, _ = cv2.findContours(
            instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if cnts:
            # Keep the largest contour for this instance
            raw_contours.append(max(cnts, key=cv2.contourArea))

    # ── Step 5c: Post-process — remove noise & validate ───────────────────
    contours = _filter_contours(raw_contours, img_area)

    # ── Step 6: Count ─────────────────────────────────────────────────────
    count = len(contours)
    logger.info("Building count after post-processing: %d", count)

    # ── Step 7: Metrics ───────────────────────────────────────────────────
    # Rebuild clean mask from accepted contours
    building_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(building_mask, contours, -1, 255, thickness=cv2.FILLED)

    if polygon_mask is not None:
        total_pixels = int(np.count_nonzero(polygon_mask))
    else:
        total_pixels = img_area

    built_up_pixels = int(np.count_nonzero(building_mask))

    return {
        "count": count,
        "contours": contours,
        "building_mask": building_mask,
        "built_up_pixels": built_up_pixels,
        "total_pixels": total_pixels,
    }


def draw_overlay(
    img_bgr: np.ndarray,
    contours: List[np.ndarray],
    polygon_coords_px: Optional[np.ndarray] = None,
    fill_alpha: float = 0.35,
) -> np.ndarray:
    """
    Draw building contours and optional boundary polygon on the image.

    Args:
        img_bgr:           Original BGR image.
        contours:          List of building contours.
        polygon_coords_px: Pixel-space polygon boundary (N,1,2 int32 array).
        fill_alpha:        Opacity of the building fill overlay.

    Returns:
        Annotated BGR image.
    """
    overlay = img_bgr.copy()
    output = img_bgr.copy()

    # Draw filled building polygons (semi-transparent cyan)
    cv2.drawContours(overlay, contours, -1, (255, 200, 0), thickness=cv2.FILLED)

    # Blend overlay
    cv2.addWeighted(overlay, fill_alpha, output, 1 - fill_alpha, 0, output)

    # Draw building outlines (solid bright cyan)
    cv2.drawContours(output, contours, -1, (0, 220, 255), thickness=2)

    # Draw user polygon boundary (red dashed effect via thick then thin)
    if polygon_coords_px is not None:
        cv2.polylines(output, [polygon_coords_px], isClosed=True, color=(0, 0, 255), thickness=3)

    # Label count in top-left corner
    label = f"Buildings: {len(contours)}"
    cv2.rectangle(output, (0, 0), (220, 36), (0, 0, 0), -1)
    cv2.putText(
        output, label,
        (6, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2, cv2.LINE_AA,
    )

    return output
