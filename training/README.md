# Fine-tune: dạy model nhận ra vật thể của bạn

Model có sẵn chỉ biết 80 loại vật thông dụng (bộ COCO). Muốn đếm vật riêng như ốc vít, viên thuốc, cá giống hay bao hàng, bạn cần **fine-tune**: lấy model đã biết nhìn vật thể nói chung, rồi dạy thêm cho nó vật của bạn bằng ảnh có gán nhãn. Cách này cần ít ảnh hơn rất nhiều so với train từ đầu.

Thư mục này có sẵn:

| File | Dùng để |
|---|---|
| `data.yaml` | Mẫu file khai báo bộ dữ liệu (đường dẫn + tên các loại) |
| `check_dataset.py` | Kiểm tra bộ dữ liệu có lỗi không, **trước khi** mất công train |
| `train_colab.ipynb` | Notebook train trên Google Colab (GPU miễn phí) |
| `_build_notebook.py` | Script tạo lại notebook (muốn sửa notebook thì sửa file này rồi chạy) |

Các bước dưới đây làm theo thứ tự.

---

## Bước 1: Thu thập ảnh

- **Số lượng:** bắt đầu với khoảng **100-300 ảnh** cho mỗi loại vật. Kết quả chưa tốt thì thêm ảnh sau.
- **Đa dạng:** nhiều góc chụp, khoảng cách, ánh sáng, nền khác nhau; có cảnh vật nằm sát nhau, chồng lên nhau, bị che một phần.
- **Giống lúc dùng thật:** dùng đúng loại camera và cách chụp mà bạn sẽ dùng khi đếm. Ví dụ đếm trên băng chuyền thì chụp ảnh trên băng chuyền.
- **Thêm vài chục ảnh nền** (không có vật nào) để model bớt nhận nhầm.

## Bước 2: Gán nhãn (vẽ khung quanh từng vật)

Chọn một công cụ miễn phí:

- **[Label Studio](https://labelstud.io)**: chạy trên máy bạn, dữ liệu không rời khỏi máy. Nên cài trong một môi trường ảo **riêng**, để thư viện của nó không xung đột với vision-count:
  ```bash
  python3.12 -m venv ~/label-studio-env
  source ~/label-studio-env/bin/activate
  pip install label-studio
  label-studio start
  ```
  Trình duyệt sẽ mở http://localhost:8080. Tạo project, chọn mẫu **Object Detection with Bounding Boxes**, khai báo tên các loại, tải ảnh lên rồi vẽ khung.
- **[CVAT](https://www.cvat.ai)** hoặc **[Roboflow](https://roboflow.com)**: dùng trên web (bản miễn phí), tiện nhưng ảnh được tải lên máy chủ của họ.

Khi xuất dữ liệu, chọn định dạng **YOLO**. Mỗi ảnh sẽ đi kèm một file `.txt` cùng tên, mỗi dòng là một vật:

```
class_id  x_tâm  y_tâm  chiều_rộng  chiều_cao
0         0.512  0.430  0.120       0.085
```

Các số tọa độ đã được chia cho chiều rộng/cao ảnh, nên luôn nằm trong khoảng 0..1. `class_id` đánh số từ 0 theo thứ tự trong `names`.

Nếu công cụ xuất nhãn dạng **polygon** (`class x1 y1 x2 y2 x3 y3 ...`) thì cũng dùng được: Ultralytics tự đổi thành khung.

## Bước 3: Sắp xếp thư mục

Chia khoảng **80% ảnh để train** (học) và **20% để val** (kiểm tra). Ảnh val phải là ảnh model **chưa từng thấy** khi học.

> Nếu xuất từ **Roboflow**, bộ dữ liệu có sẵn cấu trúc `train/images`, `valid/images` và `data.yaml` ghi `train: ../train/images`. Có thể dùng luôn, không cần xếp lại; `check_dataset.py` đọc đúng đường dẫn trong `data.yaml`.

```
dataset/
├── data.yaml            ← chép từ training/data.yaml rồi sửa phần names
├── images/
│   ├── train/   anh001.jpg, anh002.jpg ...
│   └── val/     anh101.jpg ...
└── labels/
    ├── train/   anh001.txt, anh002.txt ...   (cùng tên với ảnh)
    └── val/     anh101.txt ...
```

Sửa `names` trong `data.yaml` cho đúng các loại của bạn. Tên này sẽ hiện trên giao diện vision-count, nên có thể đặt tiếng Việt:

```yaml
names:
  0: ốc vít
  1: đai ốc
```

> **Không** thêm dòng `path: .` vào `data.yaml`. Ultralytics sẽ hiểu dấu chấm là thư mục đang chạy lệnh (trên Colab là `/content`), chứ không phải thư mục chứa `data.yaml`, và sẽ báo không tìm thấy ảnh. Bỏ dòng `path` đi thì Ultralytics tự lấy đúng thư mục chứa `data.yaml`.

## Bước 4: Kiểm tra bộ dữ liệu

Ở thư mục project vision-count (đã `source .venv/bin/activate`):

```bash
python training/check_dataset.py duong_dan/toi/dataset
```

Script sẽ báo:
- ❌ **Lỗi** (phải sửa): file nhãn sai định dạng, `class_id` không có trong `names`, tọa độ ngoài 0..1, file nhãn không có ảnh, thư mục train/val trống, thiếu `data.yaml`.
- ❌ Lỗi còn gồm: thư mục nhãn trống, tập train không có khung nào, `nc` khác số tên trong `names`, file không đọc được (YAML sai cú pháp, nhãn không phải UTF-8).
- ⚠️ **Cảnh báo** (nên xem): ảnh không có file nhãn (sẽ được coi là ảnh nền), có dòng `path` tương đối.

Chạy lại tới khi thấy `✅ Không có lỗi`.

## Bước 5: Train trên Google Colab

1. Nén thư mục `dataset/` thành file .zip. Trên Mac: chuột phải vào thư mục, chọn **Compress**.
2. Vào https://colab.research.google.com, chọn **File → Upload notebook**, rồi chọn `training/train_colab.ipynb`.
3. Bật GPU: **Runtime → Change runtime type → T4 GPU → Save**.
4. Chạy lần lượt từng ô từ trên xuống. Ô "Tải bộ dữ liệu lên" sẽ hỏi file, chọn file .zip vừa tạo.
5. Train vài trăm ảnh trên GPU T4 thường mất **15-40 phút**. Đừng đóng tab trong lúc chạy; bản miễn phí có thể bị ngắt nếu để quá lâu không dùng.

## Bước 6: Đọc kết quả

Notebook in ra 2 chỉ số:
- **mAP50**: trên **0.8** là tốt, 0.5-0.8 là tạm được, dưới **0.5** là cần cải thiện.
- **mAP50-95**: chặt hơn (đòi khung khớp sát), thường thấp hơn mAP50.

Nếu kết quả kém:
- Thêm ảnh, nhất là những cảnh model hay sai.
- Kiểm tra lại nhãn: vẽ thiếu vật hoặc khung lệch là nguyên nhân phổ biến nhất.
- Cân bằng số ảnh giữa các loại.
- Thử model lớn hơn: đổi `yolo11n.pt` thành `yolo11s.pt` trong ô train.

## Bước 7: Dùng model mới trong vision-count

1. Ô cuối của notebook tải về file **`best.pt`**. Chép file này vào thư mục `models/` của project.
2. Mở `vision_count/config.py`, thêm một dòng vào `AVAILABLE_MODELS`:
   ```python
   AVAILABLE_MODELS = {
       "nano": ("Nano: nhanh nhất", "yolo11n.pt"),
       "small": ("Small: chính xác hơn, chậm hơn", "yolo11s.pt"),
       "cua_toi": ("Model của tôi", "best.pt"),
   }
   ```
3. Chạy lại `python app.py`, rồi chọn **Model của tôi** trong ô **Model**. Ô **Chỉ đếm các loại** tự đổi sang các loại trong `names` của bạn. Tab Một ảnh, Nhiều ảnh, Video và API đều dùng được model mới.
