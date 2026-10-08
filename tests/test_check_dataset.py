"""Test script kiểm tra bộ dữ liệu fine-tune."""

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from training.check_dataset import check_dataset

ROOT = Path(__file__).resolve().parent.parent


def _make_dataset(root: Path, labels: dict, names=("oc_vit", "dai_oc")):
    (root / "images" / "train").mkdir(parents=True)
    (root / "images" / "val").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    (root / "labels" / "val").mkdir(parents=True)
    (root / "data.yaml").write_text(
        "train: images/train\nval: images/val\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)),
        encoding="utf-8",
    )
    for rel, text in labels.items():  # rel: "train/a" -> images/train/a.jpg + labels/train/a.txt
        split, stem = rel.split("/")
        Image.new("RGB", (64, 64)).save(root / "images" / split / f"{stem}.jpg")
        if text is not None:
            (root / "labels" / split / f"{stem}.txt").write_text(text)
    return root


def test_valid_dataset_has_no_errors(tmp_path):
    root = _make_dataset(tmp_path, {"train/a": "0 0.5 0.5 0.2 0.3\n1 0.1 0.1 0.05 0.05\n", "val/b": "1 0.5 0.5 0.4 0.4\n"})
    errors, warnings = check_dataset(root)
    assert errors == [] and warnings == []


def test_reports_bad_label_lines(tmp_path):
    root = _make_dataset(tmp_path, {
        "train/a": "5 0.5 0.5 0.2 0.3\n",  # class 5 không có trong names (chỉ có 0, 1)
        "train/b": "0 1.5 0.5 0.2 0.3\n",  # tọa độ ngoài 0..1
        "train/c": "0 0.5 0.5\n",  # thiếu số
        "val/d": "0 0.5 0.5 0.2 0.3\n",
    })
    errors, _ = check_dataset(root)
    joined = "\n".join(errors)
    assert "a.txt" in joined and "class" in joined
    assert "b.txt" in joined and "0..1" in joined
    assert "c.txt" in joined and "5 số" in joined


def test_warns_image_without_label_and_errors_label_without_image(tmp_path):
    root = _make_dataset(tmp_path, {"train/a": None, "val/b": "0 0.5 0.5 0.2 0.2\n"})
    (root / "labels" / "train" / "mo_coi.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    errors, warnings = check_dataset(root)
    assert any("a.jpg" in w for w in warnings)  # ảnh không có nhãn = ảnh nền (cho phép, nhưng nhắc)
    assert any("mo_coi.txt" in e for e in errors)


def test_missing_data_yaml_or_empty_val(tmp_path):
    errors, _ = check_dataset(tmp_path)
    assert any("data.yaml" in e for e in errors)
    root = _make_dataset(tmp_path / "ds", {"train/a": "0 0.5 0.5 0.2 0.2\n"})
    errors, _ = check_dataset(root)
    assert any("val" in e for e in errors)


def test_command_line_exit_code(tmp_path):
    root = _make_dataset(tmp_path, {"train/a": "0 0.5 0.5 0.2 0.2\n", "val/b": "0 0.5 0.5 0.2 0.2\n"})
    script = ROOT / "training" / "check_dataset.py"
    ok = subprocess.run([sys.executable, script, str(root)], capture_output=True, text=True)
    assert ok.returncode == 0 and "Không có lỗi" in ok.stdout
    bad = subprocess.run([sys.executable, script, str(tmp_path / "khong_co")], capture_output=True, text=True)
    assert bad.returncode == 1


def test_colab_notebook_is_valid():
    nb = json.loads((ROOT / "training" / "train_colab.ipynb").read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "model.train(" in code and "best.pt" in code


def test_data_yaml_template_parses():
    import yaml

    data = yaml.safe_load((ROOT / "training" / "data.yaml").read_text(encoding="utf-8"))
    assert {"train", "val", "names"} <= set(data)
    # Không có "path": Ultralytics sẽ lấy đúng thư mục chứa data.yaml làm gốc
    assert "path" not in data


def test_warns_relative_path_in_data_yaml(tmp_path):
    # "path: ." bị Ultralytics hiểu là thư mục đang chạy lệnh, không phải thư mục chứa data.yaml
    root = _make_dataset(tmp_path, {"train/a": "0 0.5 0.5 0.2 0.2\n", "val/b": "0 0.5 0.5 0.2 0.2\n"})
    yaml_path = root / "data.yaml"
    yaml_path.write_text("path: .\n" + yaml_path.read_text(encoding="utf-8"), encoding="utf-8")
    _, warnings = check_dataset(root)
    assert any("path" in w and "bỏ dòng" in w for w in warnings)


def test_split_without_label_folder_is_error(tmp_path):
    # Thiếu (hoặc đặt sai tên) thư mục labels/: Ultralytics sẽ coi mọi ảnh là ảnh nền, train ra model vô dụng
    root = _make_dataset(tmp_path, {"train/a": None, "val/b": None})
    errors, _ = check_dataset(root)
    assert any("labels/train" in e for e in errors)
    assert any("labels/val" in e for e in errors)


def test_train_with_no_boxes_is_error(tmp_path):
    root = _make_dataset(tmp_path, {"train/a": "", "train/b": "\n", "val/c": "0 0.5 0.5 0.2 0.2\n"})
    errors, _ = check_dataset(root)
    assert any("train" in e and "chưa có khung" in e for e in errors)


# ---------- Đợt sửa điểm nhỏ ----------

def test_accepts_all_ultralytics_image_formats(tmp_path):
    root = _make_dataset(tmp_path, {"train/a": "0 0.5 0.5 0.2 0.2\n", "val/b": "0 0.5 0.5 0.2 0.2\n"})
    Image.new("RGB", (64, 64)).save(root / "images" / "train" / "c.TIFF")
    (root / "labels" / "train" / "c.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    errors, warnings = check_dataset(root)
    assert errors == [] and warnings == []


def test_polygon_labels_are_accepted_but_checked(tmp_path):
    root = _make_dataset(tmp_path, {
        "train/a": "0 0.1 0.1 0.5 0.1 0.3 0.6\n",  # polygon 3 đỉnh: hợp lệ
        "train/b": "0 0.1 0.1 0.5 0.1 0.3\n",  # số tọa độ lẻ: sai
        "val/c": "1 0.1 0.1 0.5 0.1 1.3 0.6\n",  # tọa độ > 1: sai
    })
    errors, _ = check_dataset(root)
    assert not any("a.txt" in e for e in errors)
    assert any("b.txt" in e for e in errors) and any("c.txt" in e and "0..1" in e for e in errors)


def test_follows_paths_in_data_yaml_roboflow_layout(tmp_path):
    # Roboflow xuất dạng train/images, valid/images và data.yaml ghi "../train/images"
    for split in ("train", "valid"):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "labels").mkdir(parents=True)
        Image.new("RGB", (64, 64)).save(tmp_path / split / "images" / "x.jpg")
        (tmp_path / split / "labels" / "x.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (tmp_path / "data.yaml").write_text("train: ../train/images\nval: ../valid/images\nnc: 1\nnames: ['oc_vit']\n")
    errors, warnings = check_dataset(tmp_path)
    assert errors == [] and warnings == []


def test_nc_must_match_names(tmp_path):
    root = _make_dataset(tmp_path, {"train/a": "0 0.5 0.5 0.2 0.2\n", "val/b": "0 0.5 0.5 0.2 0.2\n"})
    (root / "data.yaml").write_text((root / "data.yaml").read_text() + "nc: 5\n")
    errors, _ = check_dataset(root)
    assert any("nc" in e for e in errors)


def test_bad_yaml_and_non_utf8_label_are_errors_not_crashes(tmp_path):
    root = _make_dataset(tmp_path, {"train/a": "0 0.5 0.5 0.2 0.2\n", "val/b": "0 0.5 0.5 0.2 0.2\n"})
    (root / "labels" / "train" / "a.txt").write_bytes(b"0 0.5 0.5 0.2 0.2 \xff\xfe\n")
    errors, _ = check_dataset(root)
    assert any("a.txt" in e and "UTF-8" in e for e in errors)
    (root / "data.yaml").write_text("names: [oc_vit\ntrain: images/train\n")
    errors, _ = check_dataset(root)
    assert any("data.yaml" in e and "đọc" in e for e in errors)


def test_colab_notebook_handles_any_image_and_upload_name():
    nb = json.loads((ROOT / "training" / "train_colab.ipynb").read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "next(iter(uploaded))" in code  # không cứng tên dataset.zip
    assert "IMG_FORMATS" in code and ".plot()" in code  # bỏ qua .DS_Store, không phụ thuộc đuôi ảnh
    # Không ghép đường dẫn ảnh kết quả từ save_dir + tên ảnh gốc (Ultralytics luôn lưu thành .jpg)
    assert "save_dir" not in code


def test_colab_cells_work_with_roboflow_layout(tmp_path):
    """Chạy thật mã ô 4 (kiểm tra dữ liệu) và ô 7 (thử dự đoán) trên bộ dữ liệu kiểu Roboflow."""
    from ultralytics import YOLO

    for split in ("train", "valid"):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "labels").mkdir(parents=True)
        Image.new("RGB", (64, 64)).save(tmp_path / split / "images" / "x.jpg")
        (tmp_path / split / "labels" / "x.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (tmp_path / ".DS_Store").write_text("")
    (tmp_path / "data.yaml").write_text("train: ../train/images\nval: ../valid/images\nnc: 1\nnames: ['oc_vit']\n")

    nb = json.loads((ROOT / "training" / "train_colab.ipynb").read_text(encoding="utf-8"))
    codes = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    check_cell = next(c for c in codes if "data.yaml" in c and "assert" in c)
    predict_cell = next(c for c in codes if "model.predict(" in c)
    env = {"model": YOLO(str(ROOT / "models" / "yolo11n.pt"))}
    exec(check_cell.replace("/content/dataset", str(tmp_path)), env)
    exec(predict_cell, env)  # không được lỗi FileNotFoundError với cấu trúc Roboflow
