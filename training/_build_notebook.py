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
1. Ảnh đã gán nhãn theo định dạng YOLO, xếp đúng cấu trúc `dataset/images/{train,val}` và `dataset/labels/{train,val}`, kèm `dataset/data.yaml`.
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

root = Path("/content/dataset")
assert (root / "data.yaml").is_file(), "Không thấy /content/dataset/data.yaml: kiểm tra lại file zip (bên trong phải là thư mục dataset/)"
for split in ("train", "val"):
    n_img = len(list((root / "images" / split).glob("*")))
    n_lbl = len(list((root / "labels" / split).glob("*.txt")))
    print(f"{split}: {n_img} ảnh, {n_lbl} file nhãn")
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
from ultralytics.data.utils import IMG_FORMATS

# Chỉ lấy file ảnh (bỏ qua .DS_Store mà macOS hay cho vào file zip)
sample = next(p for p in sorted((root / "images" / "val").iterdir()) if p.suffix.lower().lstrip(".") in IMG_FORMATS)
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
