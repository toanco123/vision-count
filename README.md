# vision-count: AI đếm vật thể trong ảnh

Tải một ảnh lên, hệ thống sẽ **phát hiện**, **khoanh khung** từng vật thể và **đếm số lượng** theo từng loại.

- Model: **YOLO11 nano** (`yolo11n.pt`) qua thư viện [Ultralytics]: nhỏ (~5MB), chạy được trên CPU, không cần GPU.
- Nhận được **80 loại vật thông dụng** của bộ dữ liệu COCO: người, xe đạp, ô tô, xe máy, xe buýt, chó, mèo, chai, cốc, ghế, điện thoại, laptop, trái cây...
- Giao diện: **Gradio**, mở bằng trình duyệt trên máy bạn.
- Mọi thứ chạy **offline trên máy bạn**, không dùng dịch vụ trả phí hay API bên ngoài. Chỉ cần mạng **một lần** để cài thư viện và tải file model.

[Ultralytics]: https://docs.ultralytics.com

---

## 1. Cấu trúc thư mục

```
vision-count/
├── app.py                  # Điểm khởi động: python app.py
├── requirements.txt        # Danh sách thư viện cần cài
├── pytest.ini              # Cấu hình chạy test
├── vision_count/           # ⭐ PHẦN AI, không phụ thuộc giao diện
│   ├── config.py           #   Cấu hình: tên model, ngưỡng mặc định...
│   ├── detector.py         #   Đọc ảnh, chạy YOLO, trả về khung + số đếm
│   └── drawing.py          #   Vẽ khung và nhãn lên ảnh
├── ui/
│   └── gradio_app.py       # Giao diện Gradio, chỉ gọi tới vision_count
├── tests/
│   └── test_detector.py    # Test tự động cho phần AI
├── models/                 # Nơi lưu file model (tự tải về lần đầu)
└── samples/                # Ảnh mẫu để thử
```

**Vì sao tách `vision_count/` và `ui/`?** Phần AI chỉ nhận ảnh vào và trả về dữ liệu (danh sách khung, số đếm), hoàn toàn không biết Gradio là gì. Nhờ vậy, sau này muốn làm API bằng FastAPI hay đổi giao diện sang React/Flutter thì **chỉ viết thêm lớp giao diện mới**, không phải sửa phần AI.

---

## 2. Cài đặt (chỉ làm một lần)

Mở **Terminal** và gõ lần lượt từng lệnh dưới đây.

### Bước 1: Cài Python 3.10 trở lên

macOS có sẵn Python 3.9, quá cũ cho project này. Cài Python 3.12 bằng Homebrew:

```bash
brew install python@3.12
```

Kiểm tra lại (phải thấy `Python 3.12.x`):

```bash
python3.12 --version
```

> Chưa có Homebrew? Cài theo hướng dẫn tại https://brew.sh, hoặc tải Python từ https://www.python.org/downloads/.

### Bước 2: Vào thư mục project

```bash
cd ~/Desktop/Project/vision-count
```

### Bước 3: Tạo môi trường ảo (virtual environment)

Môi trường ảo là một "hộp" riêng chứa thư viện cho project này, không làm lẫn với các project khác.

```bash
python3.12 -m venv .venv
```

### Bước 4: Kích hoạt môi trường ảo

```bash
source .venv/bin/activate
```

Thành công khi đầu dòng Terminal hiện chữ `(.venv)`.
⚠️ **Mỗi lần mở Terminal mới** để chạy project, bạn phải chạy lại lệnh này.

### Bước 5: Cài thư viện

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Bước này tải khoảng 1GB (chủ yếu là PyTorch) nên có thể mất 5-15 phút tùy tốc độ mạng.

---

## 3. Chạy ứng dụng

```bash
source .venv/bin/activate     # nếu chưa kích hoạt
python app.py
```

Lần chạy đầu, chương trình tự tải file model `yolo11n.pt` (~5MB) vào thư mục `models/`. Khi Terminal hiện:

```
* Running on local URL:  http://127.0.0.1:7860
```

mở trình duyệt và vào **http://127.0.0.1:7860**.

Muốn tắt ứng dụng: quay lại Terminal và bấm `Ctrl + C`.

### Cách dùng

1. Kéo thả hoặc bấm chọn ảnh ở ô **Ảnh cần đếm** (có thể thử ảnh `samples/bus.jpg`).
2. Chỉnh **Ngưỡng độ tin cậy** nếu cần (mặc định 0.25).
3. (Tùy chọn) Chọn vài loại trong ô **Chỉ đếm các loại**, ví dụ `person`, `car`. Để trống thì đếm tất cả.
4. Bấm **Đếm**.

> **Lần bấm "Đếm" đầu tiên** sau khi cài có thể mất 20-30 giây vì thư viện phải chuẩn bị một số thứ (ví dụ bộ nhớ đệm font của Matplotlib). Việc này chỉ xảy ra một lần; các lần sau việc nhận diện mỗi ảnh chỉ mất chưa tới 0.1 giây (đo trên Mac M1, chạy CPU).

Kết quả gồm:
- Ảnh đã khoanh khung, mỗi khung có nhãn dạng `#1 person 87%` (số thứ tự, tên loại, độ tin cậy).
- Tổng số vật thể.
- Bảng số lượng theo từng loại.
- Mục **Chi tiết từng khung** (bấm để mở) liệt kê độ tin cậy và tọa độ của từng khung.

### Ngưỡng độ tin cậy là gì?

Với mỗi khung, model đưa ra một con số từ 0 đến 1 cho biết nó "chắc chắn" đến đâu. Ngưỡng 0.25 nghĩa là chỉ giữ các khung có độ tin cậy từ 25% trở lên.

| Ngưỡng | Kết quả |
|---|---|
| **Thấp** (0.1-0.2) | Bắt được nhiều vật hơn (kể cả vật nhỏ, bị che), nhưng **dễ đếm nhầm** |
| **Vừa** (0.25-0.4) | Cân bằng, phù hợp đa số ảnh |
| **Cao** (0.5 trở lên) | Ít nhầm, nhưng **dễ bỏ sót** |

---

## 4. Chạy test

```bash
source .venv/bin/activate
pytest -v
```

Test kiểm tra: từ chối file không phải ảnh, ảnh trống trả về 0 vật thể, ảnh xe buýt mẫu đếm được người và xe buýt, bộ lọc loại vật, ngưỡng độ tin cậy...

---

## 5. Dùng phần AI trong code Python của bạn

Không cần giao diện, bạn có thể gọi thẳng phần AI:

```python
from vision_count import ObjectDetector, draw_detections, load_image

detector = ObjectDetector()                       # nạp model một lần
result = detector.detect("samples/bus.jpg", confidence=0.3)

print(result.total)      # tổng số vật, ví dụ: 5
print(result.counts)     # {'person': 4, 'bus': 1}
for d in result.detections:
    print(d.label, d.confidence, d.box)          # person 0.89 (48.0, 398.0, 245.0, 902.0)

# Chỉ đếm người
only_people = detector.detect("samples/bus.jpg", classes=["person"])

# Lưu ảnh đã khoanh khung
image = load_image("samples/bus.jpg")
draw_detections(image, result.detections).save("ket_qua.jpg")

# Dạng dict, sẵn sàng trả về JSON cho API
result.to_dict()
```

---

## 6. Lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `command not found: python3.12` | Chưa cài Python 3.12, làm lại Bước 1. |
| `ModuleNotFoundError: No module named 'ultralytics'` (hoặc `gradio`) | Quên kích hoạt môi trường ảo: chạy `source .venv/bin/activate`. |
| Báo lỗi "không phải là file ảnh hợp lệ" | File bị hỏng hoặc không phải ảnh. Dùng JPG, PNG, WEBP, BMP. |
| "Không tìm thấy vật thể nào" | Giảm ngưỡng độ tin cậy, bỏ bộ lọc loại vật, hoặc thử ảnh rõ hơn. Nếu vật không thuộc 80 loại COCO thì xem mục fine-tune bên dưới. |
| Lần chạy đầu báo lỗi tải model | Cần mạng ở lần đầu để tải `yolo11n.pt`. Hoặc tự tải file về rồi đặt vào thư mục `models/`. |
| `Address already in use` (cổng 7860 đang bận) | Ứng dụng đang chạy ở một Terminal khác. Tắt nó đi, hoặc đổi `server_port` trong `app.py`. |

---

## 7. Kế hoạch giai đoạn sau

### 7.1. Fine-tune khi model không nhận ra vật thể của bạn

Model pretrained chỉ biết 80 loại COCO. Với vật thể riêng (ốc vít, viên thuốc, cá giống, bao hàng...), cần dạy thêm cho model:

1. **Thu thập ảnh**: khoảng 100-300 ảnh cho mỗi loại để bắt đầu. Chụp ở nhiều góc, nhiều điều kiện ánh sáng, nhiều nền, có cả cảnh vật nằm sát hoặc chồng lên nhau, giống với lúc dùng thật.
2. **Gán nhãn** (vẽ khung quanh từng vật) bằng công cụ miễn phí:
   - [Label Studio](https://labelstud.io) hoặc [CVAT](https://www.cvat.ai): chạy được trên máy bạn.
   - [Roboflow](https://roboflow.com): bản miễn phí, dùng trên web, tiện xuất dữ liệu.
   - Xuất dữ liệu theo **định dạng YOLO**: mỗi ảnh đi kèm một file `.txt` chứa tọa độ khung, cùng một file `data.yaml` khai báo tên các loại.
3. **Chia dữ liệu**: khoảng 80% để train, 20% để kiểm tra (val).
4. **Train trên Google Colab** (miễn phí, có GPU): `Runtime → Change runtime type → T4 GPU`, rồi chạy:
   ```python
   !pip install ultralytics
   from ultralytics import YOLO
   model = YOLO("yolo11n.pt")      # bắt đầu từ model pretrained (transfer learning)
   model.train(data="data.yaml", epochs=100, imgsz=640)
   ```
5. **Đánh giá**: xem chỉ số mAP và các ảnh kết quả trong thư mục `runs/detect/train/`.
6. **Dùng model mới**: tải file `runs/detect/train/weights/best.pt` về, đặt vào `models/`, rồi sửa `DEFAULT_MODEL = "best.pt"` trong `vision_count/config.py`. **Không phải sửa thêm dòng code nào khác.**

### 7.2. Đếm trong video, có tracking để không đếm trùng

- Trong video, cùng một vật xuất hiện ở nhiều khung hình. Nếu đếm từng khung hình rồi cộng lại thì sẽ bị trùng.
- Giải pháp là **tracking**: gán cho mỗi vật một ID cố định qua các khung hình. Ultralytics có sẵn `model.track(source="video.mp4", persist=True, tracker="bytetrack.yaml")`.
- Hai cách đếm:
  - **Đếm số ID duy nhất**: tổng số vật đã xuất hiện trong video.
  - **Đếm khi vật đi qua một vạch/vùng**: phù hợp đếm xe qua cổng, người vào cửa. Ultralytics có sẵn `solutions.ObjectCounter`.
- Dự kiến thêm `vision_count/video.py` với hàm `count_video(path) -> VideoCountResult`, và thêm một tab "Video" trong Gradio.

### 7.3. Bọc thành API bằng FastAPI

- Thêm thư mục `api/` với file `api/main.py`, dùng lại nguyên `ObjectDetector`:
  - `POST /detect`: nhận file ảnh (multipart) cùng các tham số `confidence`, `classes`; trả về `result.to_dict()` dạng JSON.
  - `POST /detect/image`: trả về ảnh đã khoanh khung.
  - `GET /classes`: trả về danh sách loại vật model nhận được.
- Nạp model **một lần** lúc khởi động server (không nạp lại mỗi request).
- Chạy bằng `uvicorn api.main:app`. Sau đó app React hoặc Flutter gọi các API này để hiển thị.
