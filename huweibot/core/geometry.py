from __future__ import annotations

from math import sqrt

BBox = tuple[float, float, float, float]


def bbox_area(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return w * h


def bbox_center(bbox: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def intersection_area(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    return bbox_area((x1, y1, x2, y2))


def iou(a: BBox, b: BBox) -> float:
    inter = intersection_area(a, b)
    union = bbox_area(a) + bbox_area(b) - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def intersects(a: BBox, b: BBox) -> bool:
    return intersection_area(a, b) > 0.0


def center_distance(a: BBox, b: BBox) -> float:
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    return sqrt((ax - bx) ** 2 + (ay - by) ** 2)
