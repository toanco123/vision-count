"""Tạo lại training/train_colab.ipynb. Sửa nội dung notebook ở đây rồi chạy: python training/_build_notebook.py

Viết bằng Python + json.dump để chắc chắn file notebook luôn là JSON hợp lệ.
"""

import json
from pathlib import Path


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


CELLS = [
    md("""
# Fine-tune YOLO cho vật thể riêng (vision-count)

Notebook này train lại model YOLO để nhận ra **vật thể của bạn** (ốc vít, viên thuốc, cá giống...).

**Chuẩn bị trước** (xem `training/README.md` trong project):
1. Ảnh đã gán nhãn theo định dạng YOLO, kèm `dataset/data.yaml`. Cấu trúc chuẩn `dataset/images/{train,val}` + `dataset/labels/{train,val}`, hoặc cấu trúc Roboflow (`train/images`, `valid/images`) đều được: notebook đọc đường dẫn từ `data.yaml`.
2. Đã chạy `python training/check_dataset.py dataset` trên máy và **không có lỗi**.
3. Nén thư mục `dataset/` thành một file **.zip** (vd `dataset.zip`).

**Bật GPU miễn phí:** menu **Runtime → Change runtime type → T4 GPU → Save**. Sau đó chạy lần lượt từng ô (bấm ▶ ở mỗi ô, hoặc Shift + Enter).
"""),
    md("## 1. Kiểm tra đã có GPU chưa\nPhải thấy bảng có chữ **Tesla T4**. Nếu báo lỗi `command not found`: chưa bật GPU, xem hướng dẫn ở trên."),
    code("!nvidia-smi"),
    md("## 2. Cài Ultralytics"),
    code("!pip install -q ultralytics"),
    md("## 3. Tải bộ dữ liệu lên\nChạy ô dưới, bấm **Choose Files** và chọn file .zip. Sau khi giải nén phải có `/content/dataset/data.yaml`."),
    code("""
from google.colab import files

uploaded = files.upload()  # chọn file .zip của bộ dữ liệu
zip_name = next(iter(uploaded))  # tên file thật (tải lại lần 2, Colab sẽ đặt tên kiểu "dataset (1).zip")
print("Giải nén:", zip_name)
!unzip -q -o "{zip_name}" -d /content/
"""),
    md("## 4. Kiểm tra nhanh bộ dữ liệu"),
    code("""
from pathlib import Path

from ultralytics.data.utils import IMG_FORMATS, check_det_dataset

root = Path("/content/dataset")
assert (root / "data.yaml").is_file(), "Không thấy /content/dataset/data.yaml: kiểm tra lại file zip (bên trong phải là thư mục dataset/)"
# Đọc đường dẫn train/val đúng như lúc train (hỗ trợ cả cấu trúc chuẩn lẫn Roboflow)
data = check_det_dataset(str(root / "data.yaml"), autodownload=False)


def label_dir(image_dir):
    # Quy tắc của Ultralytics: thay thư mục 'images' CUỐI CÙNG trong đường dẫn bằng 'labels'
    parts = list(Path(image_dir).parts)
    if "images" not in parts:
        return Path(image_dir)
    parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
    return Path(*parts)


split_dirs = {}
for split in ("train", "val"):
    image_dir = Path(data[split][0] if isinstance(data[split], list) else data[split])
    split_dirs[split] = image_dir
    n_img = sum(1 for p in image_dir.iterdir() if p.suffix.lower().lstrip(".") in IMG_FORMATS)
    n_lbl = len(list(label_dir(image_dir).glob("*.txt")))
    print(f"{split}: {n_img} ảnh, {n_lbl} file nhãn ({image_dir})")
print(open(root / "data.yaml", encoding="utf-8").read())
"""),
    md("""
## 5. Train

- Bắt đầu từ model **pretrained** `yolo11n.pt` (transfer learning): học nhanh hơn và cần ít ảnh hơn nhiều so với train từ đầu.
- `epochs=100`: số vòng học. `patience=20`: tự dừng sớm nếu 20 vòng liền không tiến bộ.
- Với vài trăm ảnh, trên GPU T4 thường mất **15-40 phút**. Đừng đóng tab trong lúc chạy.
"""),
    code("""
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model.train(data="/content/dataset/data.yaml", epochs=100, imgsz=640, patience=20)
"""),
    md("""
## 6. Đánh giá

- **mAP50**: độ chính xác khi khung dự đoán trùng khung thật từ 50% trở lên. **Trên 0.8 là tốt**; dưới 0.5 thì nên thêm ảnh hoặc sửa nhãn.
- **mAP50-95**: chặt hơn (đòi khung khớp sát), thường thấp hơn mAP50.
"""),
    code("""
metrics = model.val()
print(f"mAP50    = {metrics.box.map50:.3f}")
print(f"mAP50-95 = {metrics.box.map:.3f}")
"""),
    md("## 7. Thử dự đoán một ảnh trong tập val"),
    code("""
from PIL import Image

# Chỉ lấy file ảnh (bỏ qua .DS_Store mà macOS hay cho vào file zip); thư mục val lấy từ ô 4
sample = next(p for p in sorted(split_dirs["val"].iterdir()) if p.suffix.lower().lstrip(".") in IMG_FORMATS)
pred = model.predict(sample, conf=0.25)
print(sample.name, "- đếm được:", len(pred[0].boxes), "vật")
Image.fromarray(pred[0].plot()[..., ::-1])  # plot() trả ảnh BGR, đảo sang RGB để hiển thị
"""),
    md("## 8. Tải model đã train về máy\nFile `best.pt` là model tốt nhất trong quá trình train."),
    code("""
best = Path(model.trainer.best)  # thường là /content/runs/detect/train/weights/best.pt
print(best)
files.download(str(best))
"""),
    md("""
## 9. Dùng model mới trong vision-count

1. Chép `best.pt` vào thư mục `models/` của project.
2. Mở `vision_count/config.py`, thêm một dòng vào `AVAILABLE_MODELS`:
   ```python
   "cua_toi": ("Model của tôi", "best.pt"),
   ```
3. Chạy lại `python app.py`, chọn **Model của tôi** trong ô **Model**. Ô **Chỉ đếm các loại** sẽ tự đổi sang các loại trong `names` của `data.yaml`.
"""),
]


def main() -> None:
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": CELLS,
    }
    out = Path(__file__).with_name("train_colab.ipynb")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"Đã tạo {out} ({len(CELLS)} ô)")


if __name__ == "__main__":
    main()
