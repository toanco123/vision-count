"""Đếm nhiều ảnh một lần và xuất bảng tổng hợp ra CSV."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from vision_count import config
from vision_count.detector import DetectionResult, InvalidImageError, ObjectDetector
from vision_count.export import write_csv
from vision_count.labels_vi import vi_label


@dataclass
class BatchItem:
    """Kết quả của một ảnh: có result nếu đếm được, có error nếu file lỗi."""

    name: str
    result: DetectionResult | None = None
    error: str | None = None


def count_many(
    detector: ObjectDetector,
    sources: list[str | Path],
    confidence: float = config.DEFAULT_CONFIDENCE,
    classes: list[str] | None = None,
    tiled: bool = False,
) -> list[BatchItem]:
    """Đếm lần lượt từng ảnh. File hỏng được ghi lỗi, KHÔNG làm dừng các ảnh còn lại."""
    items = []
    for source in sources:
        name = Path(source).name
        try:
            items.append(BatchItem(name, detector.detect(source, confidence=confidence, classes=classes, tiled=tiled)))
        except InvalidImageError as exc:
            items.append(BatchItem(name, error=str(exc)))
    return items


def write_batch_csv(items: list[BatchItem], path: str | Path) -> Path:
    """Mỗi ảnh một dòng, mỗi loại vật một cột (tên tiếng Việt), cuối bảng có dòng tổng cộng."""
    path = Path(path)
    totals: Counter[str] = Counter()
    for item in items:
        if item.result:
            totals.update(item.result.counts)
    labels = [label for label, _ in totals.most_common()]  # loại nhiều nhất đứng đầu

    rows = []
    for item in items:
        if item.result:
            rows.append([item.name, item.result.total, *[item.result.counts.get(l, 0) for l in labels], ""])
        else:
            rows.append([item.name, "", *[""] * len(labels), item.error])
    rows.append(["Tổng cộng", sum(totals.values()), *[totals[l] for l in labels], ""])

    write_csv(path, ["Ảnh", "Tổng", *[vi_label(l) for l in labels], "Lỗi"], rows)
    return path
