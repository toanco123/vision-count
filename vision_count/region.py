"""Vùng đếm: chỉ đếm vật có TÂM khung nằm trong một hình chữ nhật (tính cả cạnh)."""

from __future__ import annotations

from collections import Counter

from vision_count.detector import DetectionResult

# (x1, y1, x2, y2) theo pixel của ảnh gốc, x1 <= x2 và y1 <= y2
Region = tuple[float, float, float, float]


def normalize_region(p1: tuple[float, float], p2: tuple[float, float]) -> Region:
    """Tạo vùng từ 2 điểm bất kỳ (người dùng có thể bấm góc nào trước cũng được)."""
    (ax, ay), (bx, by) = p1, p2
    return (min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))


def filter_by_region(result: DetectionResult, region: Region | None) -> DetectionResult:
    """Giữ lại các vật có tâm nằm trong vùng và đếm lại. region=None thì trả về nguyên kết quả."""
    if region is None:
        return result
    x1, y1, x2, y2 = region
    kept = [
        d
        for d in result.detections
        if x1 <= (d.box[0] + d.box[2]) / 2 <= x2 and y1 <= (d.box[1] + d.box[3]) / 2 <= y2
    ]
    return DetectionResult(
        detections=kept,
        counts=dict(Counter(d.label for d in kept).most_common()),
        image_width=result.image_width,
        image_height=result.image_height,
    )
