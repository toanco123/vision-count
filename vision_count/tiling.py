"""Chế độ vật nhỏ (ý tưởng của SAHI): chia ảnh lớn thành các ô nhỏ chồng nhau.

YOLO luôn thu ảnh về 640px trước khi nhận diện, nên trong ảnh lớn, vật nhỏ bị thu còn vài pixel
và mất. Nhận diện từng ô 640px giữ nguyên độ phân giải nên bắt được vật nhỏ.
"""

from __future__ import annotations

from vision_count.detector import Detection


def make_tiles(width: int, height: int, tile: int = 640, overlap: float = 0.2) -> list[tuple[int, int, int, int]]:
    """Trả về danh sách ô (x1, y1, x2, y2) phủ kín ảnh; ô cuối mỗi hàng/cột sát mép ảnh."""
    if width <= tile and height <= tile:
        return [(0, 0, width, height)]
    step = max(1, int(tile * (1 - overlap)))

    def starts(size: int) -> list[int]:
        if size <= tile:
            return [0]
        return sorted(set(range(0, size - tile, step)) | {size - tile})

    return [(x, y, min(x + tile, width), min(y + tile, height)) for y in starts(height) for x in starts(width)]


def _ios(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Phần giao chia cho diện tích khung NHỎ hơn (khác IoU: bắt được nửa khung nằm trong khung đầy đủ)."""
    inter_w = min(a[2], b[2]) - max(a[0], b[0])
    inter_h = min(a[3], b[3]) - max(a[1], b[1])
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter_w * inter_h / smaller if smaller > 0 else 0.0


def merge_overlaps(detections: list[Detection], ios_threshold: float = 0.6) -> list[Detection]:
    """Gộp khung trùng CÙNG LOẠI giữa các ô: giữ khung tin cậy cao hơn, bỏ khung chồng lên nó quá ngưỡng."""
    kept: list[Detection] = []
    for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
        if all(k.class_id != det.class_id or _ios(k.box, det.box) <= ios_threshold for k in kept):
            kept.append(det)
    return kept
