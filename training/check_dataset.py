"""Kiểm tra bộ dữ liệu YOLO trước khi train, để không mất công train rồi mới biết dữ liệu sai.

Chạy: python training/check_dataset.py duong_dan/dataset
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml
from ultralytics.data.utils import IMG_FORMATS

# Mọi đuôi ảnh Ultralytics đọc được (jpg, png, tif, heic, webp...), so sánh không phân biệt hoa thường
IMAGE_EXTS = {"." + f for f in IMG_FORMATS}


def _class_count(names) -> int:
    return len(names) if isinstance(names, (list, dict)) else 0


def _read_lines(path: Path) -> list[str] | None:
    """Các dòng của file nhãn; None nếu file không phải văn bản UTF-8."""
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return None


def _check_label_lines(name: str, lines: list[str], n_classes: int) -> list[str]:
    errors = []
    for i, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        parts = line.split()
        where = f"{name} dòng {i}"
        # 5 số = khung (class x y w h). Nhiều hơn = polygon (class x1 y1 x2 y2 ...), Ultralytics tự đổi thành khung
        is_polygon = len(parts) >= 7 and (len(parts) - 1) % 2 == 0
        if len(parts) != 5 and not is_polygon:
            errors.append(
                f"{where}: cần đúng 5 số (class x y w h), hoặc class + các cặp tọa độ polygon "
                f"(ít nhất 3 điểm); đang có {len(parts)} số"
            )
            continue
        try:
            cls = int(parts[0])
            values = [float(v) for v in parts[1:]]
        except ValueError:
            errors.append(f"{where}: có giá trị không phải số")
            continue
        if not 0 <= cls < n_classes:
            errors.append(f"{where}: class {cls} không có trong names của data.yaml (0..{n_classes - 1})")
        if not all(0 <= v <= 1 for v in values):
            errors.append(f"{where}: tọa độ phải nằm trong 0..1 (đã chia cho chiều rộng/cao ảnh)")
        elif len(parts) == 5 and (values[2] <= 0 or values[3] <= 0):
            errors.append(f"{where}: chiều rộng/cao của khung phải lớn hơn 0")
    return errors


def _image_dir(root: Path, data: dict, key: str) -> Path:
    """Thư mục ảnh của một split, giải nghĩa giống Ultralytics.

    Tương đối so với thư mục chứa data.yaml; không tồn tại mà bắt đầu bằng '../' thì bỏ '../'
    (Roboflow xuất 'train: ../train/images').
    """
    value = data.get(key, f"images/{key}")
    value = str(value[0] if isinstance(value, list) and value else value)
    path = Path(value)
    if path.is_absolute():
        return path
    base = Path(str(data["path"])) if Path(str(data.get("path", ""))).is_absolute() else root
    resolved = (base / value).resolve()
    if not resolved.exists() and value.startswith("../"):
        resolved = (base / value[3:]).resolve()
    return resolved


def _label_dir(image_dir: Path) -> Path:
    """Quy tắc của Ultralytics: thay thư mục 'images' CUỐI CÙNG trong đường dẫn bằng 'labels'."""
    parts = list(image_dir.parts)
    if "images" not in parts:
        return image_dir  # không có 'images': file nhãn nằm cạnh ảnh
    idx = len(parts) - 1 - parts[::-1].index("images")
    parts[idx] = "labels"
    return Path(*parts)


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root.resolve()))
    except ValueError:
        return str(path)


def check_dataset(root: str | Path) -> tuple[list[str], list[str]]:
    """Trả về (danh sách lỗi, danh sách cảnh báo). Không có lỗi thì có thể train."""
    root = Path(root)
    errors: list[str] = []
    warnings: list[str] = []
    yaml_path = root / "data.yaml"
    if not yaml_path.is_file():
        return [f"Không tìm thấy {yaml_path} (file khai báo đường dẫn và tên các loại)"], warnings
    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except (yaml.YAMLError, UnicodeDecodeError) as exc:
        return [f"Không đọc được data.yaml (sai cú pháp YAML hoặc không phải UTF-8): {exc}"], warnings
    if not isinstance(data, dict):
        return ["Không đọc được data.yaml: nội dung phải là các dòng 'khóa: giá trị'"], warnings

    n_classes = _class_count(data.get("names"))
    if n_classes == 0:
        errors.append("data.yaml thiếu mục 'names' (danh sách tên các loại vật)")
    if "nc" in data and n_classes and data["nc"] != n_classes:
        errors.append(f"data.yaml có nc: {data['nc']} nhưng 'names' có {n_classes} tên; hai số phải bằng nhau")
    if "path" in data and not Path(str(data["path"])).is_absolute():
        # Ultralytics tính 'path' tương đối theo thư mục ĐANG CHẠY LỆNH, không theo vị trí data.yaml
        warnings.append(
            f"data.yaml có 'path: {data['path']}' (đường dẫn tương đối): Ultralytics sẽ tính theo thư mục "
            "đang chạy lệnh nên dễ không tìm thấy ảnh. Nên bỏ dòng 'path' (khi đó gốc là thư mục chứa data.yaml)."
        )

    for split in ("train", "val"):
        image_dir = _image_dir(root, data, split)
        label_dir = _label_dir(image_dir)
        images = {p.stem: p for p in image_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS} if image_dir.is_dir() else {}
        labels = {p.stem: p for p in label_dir.glob("*.txt")} if label_dir.is_dir() else {}
        if not images:
            errors.append(f"Thư mục {_rel(image_dir, root)} ({split}) chưa có ảnh nào")
        elif not labels:
            # Ultralytics vẫn train được nhưng coi mọi ảnh là ảnh nền: ra model không nhận ra gì
            errors.append(
                f"Thư mục {_rel(label_dir, root)} ({split}) không có file nhãn nào "
                "(thiếu thư mục, đặt sai tên, hoặc chưa xuất nhãn)"
            )
        for stem, image in sorted(images.items()):
            if stem not in labels:
                warnings.append(f"{split}: ảnh {image.name} không có file nhãn (sẽ được coi là ảnh nền, không có vật)")
        n_boxes = 0
        for stem, label in sorted(labels.items()):
            if stem not in images:
                errors.append(f"{split}: file nhãn {label.name} không có ảnh tương ứng")
            lines = _read_lines(label)
            if lines is None:
                errors.append(f"{split}: {label.name}: không phải văn bản UTF-8 (hãy xuất lại nhãn định dạng YOLO)")
                continue
            n_boxes += sum(1 for line in lines if line.strip())
            errors.extend(f"{split}: {e}" for e in _check_label_lines(label.name, lines, n_classes))
        if split == "train" and labels and n_boxes == 0:
            errors.append("train: các file nhãn đều trống, chưa có khung nào để model học")
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
