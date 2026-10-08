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
