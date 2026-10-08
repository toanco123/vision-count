"""Kiểm tra bộ dữ liệu YOLO trước khi train, để không mất công train rồi mới biết dữ liệu sai.

Chạy: python training/check_dataset.py duong_dan/dataset
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _class_count(names) -> int:
    return len(names) if isinstance(names, (list, dict)) else 0


def _check_label_file(path: Path, n_classes: int) -> list[str]:
    errors = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        where = f"{path.name} dòng {i}"
        if len(parts) != 5:
            errors.append(f"{where}: cần đúng 5 số (class x y w h), đang có {len(parts)}")
            continue
        try:
            cls = int(parts[0])
            x, y, w, h = (float(v) for v in parts[1:])
        except ValueError:
            errors.append(f"{where}: có giá trị không phải số")
            continue
        if not 0 <= cls < n_classes:
            errors.append(f"{where}: class {cls} không có trong names của data.yaml (0..{n_classes - 1})")
        if not all(0 <= v <= 1 for v in (x, y, w, h)):
            errors.append(f"{where}: tọa độ phải nằm trong 0..1 (đã chia cho chiều rộng/cao ảnh)")
        elif w <= 0 or h <= 0:
            errors.append(f"{where}: chiều rộng/cao của khung phải lớn hơn 0")
    return errors


def check_dataset(root: str | Path) -> tuple[list[str], list[str]]:
    """Trả về (danh sách lỗi, danh sách cảnh báo). Không có lỗi thì có thể train."""
    root = Path(root)
    errors: list[str] = []
    warnings: list[str] = []
    yaml_path = root / "data.yaml"
    if not yaml_path.is_file():
        return [f"Không tìm thấy {yaml_path} (file khai báo đường dẫn và tên các loại)"], warnings
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    n_classes = _class_count(data.get("names"))
    if n_classes == 0:
        errors.append("data.yaml thiếu mục 'names' (danh sách tên các loại vật)")
    if "path" in data and not Path(str(data["path"])).is_absolute():
        # Ultralytics tính 'path' tương đối theo thư mục ĐANG CHẠY LỆNH, không theo vị trí data.yaml
        warnings.append(
            f"data.yaml có 'path: {data['path']}' (đường dẫn tương đối): Ultralytics sẽ tính theo thư mục "
            "đang chạy lệnh nên dễ không tìm thấy ảnh. Nên bỏ dòng 'path' (khi đó gốc là thư mục chứa data.yaml)."
        )

    for split in ("train", "val"):
        image_dir, label_dir = root / "images" / split, root / "labels" / split
        images = {p.stem: p for p in image_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS} if image_dir.is_dir() else {}
        labels = {p.stem: p for p in label_dir.glob("*.txt")} if label_dir.is_dir() else {}
        if not images:
            errors.append(f"Thư mục images/{split} chưa có ảnh nào")
        for stem, image in sorted(images.items()):
            if stem not in labels:
                warnings.append(f"{split}: ảnh {image.name} không có file nhãn (sẽ được coi là ảnh nền, không có vật)")
        for stem, label in sorted(labels.items()):
            if stem not in images:
                errors.append(f"{split}: file nhãn {label.name} không có ảnh tương ứng")
            errors.extend(f"{split}: {e}" for e in _check_label_file(label, n_classes))
    return errors, warnings


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Cách dùng: python training/check_dataset.py <thư_mục_dataset>")
        return 1
    errors, warnings = check_dataset(argv[1])
    for w in warnings:
        print(f"⚠️  {w}")
    for e in errors:
        print(f"❌ {e}")
    if errors:
        print(f"\nCó {len(errors)} lỗi: sửa xong rồi chạy lại.")
        return 1
    print(f"\n✅ Không có lỗi ({len(warnings)} cảnh báo). Bộ dữ liệu sẵn sàng để train.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
