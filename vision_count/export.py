"""Xuất kết quả đếm ra file: ảnh đã khoanh khung + 2 file CSV.

CSV ghi bằng 'utf-8-sig' (có BOM) để Excel mở không bị lỗi dấu tiếng Việt.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from PIL import Image

from vision_count.detector import DetectionResult
from vision_count.labels_vi import vi_label

COUNT_HEADER = ["Loại vật thể", "Tên gốc (model)", "Số lượng"]
DETAIL_HEADER = ["#", "Loại vật thể", "Tên gốc (model)", "Độ tin cậy", "x1", "y1", "x2", "y2"]
# (tiền tố, đuôi) của 3 file xuất ra, theo đúng thứ tự trả về
EXPORT_FILES = [("ket_qua", ".jpg"), ("so_luong", ".csv"), ("chi_tiet", ".csv")]


def export_result(
    annotated: Image.Image,
    result: DetectionResult,
    out_dir: str | Path,
    timestamp: datetime | None = None,
) -> list[Path]:
    """Ghi 3 file vào out_dir và trả về danh sách đường dẫn [ảnh, số lượng, chi tiết]."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Dấu thời gian trong tên file để các lần tải về không trùng tên nhau
    stamp = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    # Nếu đã có file cùng tên (gọi 2 lần trong cùng 1 giây) thì thêm hậu tố _2, _3...
    base, n = stamp, 1
    while any((out_dir / f"{prefix}_{stamp}{ext}").exists() for prefix, ext in EXPORT_FILES):
        n += 1
        stamp = f"{base}_{n}"
    image_path, counts_path, detail_path = (out_dir / f"{prefix}_{stamp}{ext}" for prefix, ext in EXPORT_FILES)

    annotated.convert("RGB").save(image_path, quality=95)

    count_rows = [[vi_label(label), label, n] for label, n in result.counts.items()]
    count_rows.append(["Tổng cộng", "", result.total])
    write_csv(counts_path, COUNT_HEADER, count_rows)

    detail_rows = [
        [i, vi_label(d.label), d.label, d.confidence, *d.box]
        for i, d in enumerate(result.detections, start=1)
    ]
    write_csv(detail_path, DETAIL_HEADER, detail_rows)

    return [image_path, counts_path, detail_path]


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    """Ghi CSV bằng utf-8-sig (có BOM) để Excel mở không lỗi dấu tiếng Việt."""
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
