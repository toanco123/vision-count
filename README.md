# vision-count: AI đếm vật thể trong ảnh và video

Tải ảnh hoặc video lên, hệ thống sẽ **phát hiện**, **khoanh khung** từng vật thể và **đếm số lượng** theo từng loại.

- **Model:** YOLO11 qua thư viện [Ultralytics]:
  - **nano** (`yolo11n.pt`, ~5MB): mặc định, chạy được trên CPU, không cần GPU;
  - **small** (~19MB): chính xác hơn;
  - hoặc model bạn tự train.
- **Loại vật:** nhận được **80 loại vật thông dụng** của bộ dữ liệu COCO (người, xe đạp, ô tô, xe máy, xe buýt, chó, mèo, chai, cốc, ghế, điện thoại, laptop, trái cây...), tên hiển thị bằng tiếng Việt.
- **Chức năng:**
  - đếm một ảnh, hoặc chụp từ camera;
  - chỉ đếm trong một vùng;
  - chế độ vật nhỏ cho ảnh lớn;
  - đếm nhiều ảnh cùng lúc;
  - đếm video có tracking (không đếm trùng);
  - lịch sử đếm;
  - tải kết quả về (ảnh + CSV).
- **Giao diện:** Gradio, mở bằng trình duyệt. Có thêm **API FastAPI** để app khác (web, điện thoại) gọi vào.
- **Chạy offline** trên máy bạn, không dùng dịch vụ trả phí hay API bên ngoài. Chỉ cần mạng **một lần** để cài thư viện và tải file model.
- **Fine-tune:** có sẵn bộ công cụ trong `training/` để dạy model nhận ra vật thể riêng của bạn.

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
│   ├── drawing.py          #   Vẽ khung và nhãn lên ảnh
│   ├── registry.py         #   Chọn model nano/small, mỗi model chỉ nạp một lần
│   ├── labels_vi.py        #   Bảng dịch tên 80 loại vật sang tiếng Việt
│   ├── export.py           #   Xuất kết quả ra ảnh JPG + file CSV
│   ├── region.py           #   Vùng đếm: chỉ đếm vật có tâm nằm trong vùng
│   ├── tiling.py           #   Chế độ vật nhỏ: chia ô 640px, gộp khung trùng
│   ├── batch.py            #   Đếm nhiều ảnh, xuất CSV tổng hợp
│   ├── history.py          #   Lịch sử đếm (SQLite)
│   └── video.py            #   Đếm video có tracking (không đếm trùng)
├── api/
│   └── main.py             # API FastAPI (cho app React/Flutter...), chỉ gọi tới vision_count
├── ui/
│   ├── gradio_app.py       # Bố cục giao diện Gradio (3 tab) và nối sự kiện
│   └── handlers.py         # Hàm xử lý khi bấm nút, chỉ gọi tới vision_count
├── tests/                  # Test tự động (pytest)
├── training/               # Bộ công cụ fine-tune: hướng dẫn, data.yaml mẫu, notebook Colab
├── documents/              # Kế hoạch triển khai các tính năng
├── models/                 # Nơi lưu file model (tự tải về lần đầu)
├── data/                   # Lịch sử đếm (tạo khi chạy, không đưa lên git)
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

Giao diện có phần **Cài đặt** ở trên cùng (dùng chung) và 4 tab: **Một ảnh**, **Nhiều ảnh**, **Video**, **Lịch sử**.

**Cài đặt:**
- **Model**:
  - **Nano** (mặc định): nhanh nhất.
  - **Small**: chính xác hơn với vật nhỏ hoặc bị che, chậm hơn một chút (trên Mac M1: khoảng 0.08 giây/ảnh, so với 0.05 giây của Nano). Lần đầu chọn Small, ứng dụng sẽ tải file `yolo11s.pt` (~19MB) nên cần mạng; các lần sau chạy offline.
- **Ngưỡng độ tin cậy** (mặc định 0.25), xem giải thích bên dưới.
- **Chế độ vật nhỏ**: xem mục [Chế độ vật nhỏ](#chế-độ-vật-nhỏ).
- **Chỉ đếm các loại** (tùy chọn), ví dụ `người (person)`. Để trống thì đếm tất cả. Khi đổi **Model**, danh sách này tự đổi theo các loại model đó nhận được (quan trọng khi dùng model tự train), và các lựa chọn cũ bị xóa.
- **Kiểu nhãn trên ảnh**:
  - **Đầy đủ**: `#1 người 87%`.
  - **Chỉ số thứ tự**: `1`, `2`, `3`..., gọn hơn khi có nhiều vật.
  - **Chỉ khung**: không có chữ, dễ nhìn nhất khi vật dày đặc.

**Tab Một ảnh:**
1. Chọn nguồn ảnh:
   - Tab **Tải ảnh**: kéo thả hoặc bấm chọn ảnh ở ô **Ảnh cần đếm** (có thể thử ảnh `samples/bus.jpg`).
   - Tab **Camera**: bấm vào ô camera để bật webcam (lần đầu trình duyệt sẽ hỏi quyền, chọn **Cho phép/Allow**), rồi bấm nút chụp. Muốn chụp lại thì bấm nút xóa ảnh rồi chụp tiếp.
2. Bấm **Đếm**. Ứng dụng sẽ đếm ảnh của tab đang mở.
3. (Tùy chọn) Chỉ đếm trong một khu vực: xem mục [Vùng đếm](#vùng-đếm).

> **Lần bấm "Đếm" đầu tiên** sau khi cài có thể mất 20-30 giây vì thư viện phải chuẩn bị một số thứ (ví dụ bộ nhớ đệm font của Matplotlib). Việc này chỉ xảy ra một lần; các lần sau việc nhận diện mỗi ảnh chỉ mất chưa tới 0.1 giây (đo trên Mac M1, chạy CPU).

Kết quả gồm:
- Ảnh đã khoanh khung, mỗi khung có nhãn tiếng Việt dạng `#1 người 87%` (số thứ tự, tên loại, độ tin cậy).
- Tổng số vật thể.
- Bảng số lượng theo từng loại, tên hiện dạng `người (person)`: tiếng Việt kèm tên gốc của model.
- Mục **Chi tiết từng khung** (bấm để mở) liệt kê độ tin cậy và tọa độ của từng khung.
- Ô **Tải kết quả về** có 3 file, bấm vào tên file để tải:
  - `ket_qua_<ngày>_<giờ>.jpg`: ảnh đã khoanh khung.
  - `so_luong_<ngày>_<giờ>.csv`: số lượng theo loại, có dòng tổng cộng.
  - `chi_tiet_<ngày>_<giờ>.csv`: từng khung với độ tin cậy và tọa độ.

  File CSV mở trực tiếp bằng Excel, Numbers hoặc Google Sheets, không bị lỗi dấu tiếng Việt.
  Nếu Excel dồn hết dữ liệu vào **một cột** (thường gặp khi máy đặt định dạng vùng Việt Nam, vì Excel khi đó dùng dấu `;` để ngăn cột): mở Excel trống, vào **Data → From Text/CSV** (Dữ liệu → Từ văn bản/CSV), chọn file, ở mục **Delimiter** chọn **Comma** (dấu phẩy) rồi bấm **Load**.

> Ô lọc **Chỉ đếm các loại** cũng hiện tên tiếng Việt. Gõ "người", "xe" hoặc "chai" (có dấu) để tìm nhanh, hoặc gõ tên tiếng Anh như "person", "car".
>
> File kết quả được lưu trong thư mục tạm của máy và **tự xóa sau khoảng một ngày, hoặc khi tắt ứng dụng**. Muốn giữ lâu dài thì bấm tải về.

### Vùng đếm

Dùng khi chỉ muốn đếm trong một khu vực của ảnh, ví dụ một kệ hàng hay một làn đường.

1. Bấm **Đếm** một lần để có ảnh kết quả.
2. Bấm **2 góc đối diện** của khu vực lên ảnh kết quả (góc nào trước cũng được). Lần bấm 1 hiện một chấm vàng, lần bấm 2 hiện khung vàng, và dòng chữ bên trái ghi tọa độ vùng.
3. Bấm **Đếm** lại. Lúc này chỉ đếm vật nằm trong vùng, tóm tắt ghi "trong vùng đã chọn".
4. Muốn đếm cả ảnh trở lại: bấm **Xóa vùng** rồi bấm **Đếm**.

Quy tắc: một vật được tính là **trong vùng** khi **tâm** khung của nó nằm trong vùng. Vùng được nhớ theo tọa độ pixel, nên nếu đổi sang ảnh khác kích thước, nhớ chọn lại vùng.

### Chế độ vật nhỏ

YOLO luôn thu ảnh về 640px trước khi nhận diện. Với ảnh lớn (ví dụ ảnh 4000px từ điện thoại), vật nhỏ hoặc ở xa bị thu chỉ còn vài pixel và bị bỏ sót.

Khi bật **Chế độ vật nhỏ**, ảnh được chia thành các ô 640×640 chồng lên nhau 20%. Từng ô được nhận diện riêng (giữ nguyên độ phân giải), sau đó các khung trùng ở mép ô được gộp lại. Đây là ý tưởng của kỹ thuật SAHI.

- **Nên bật** khi: ảnh lớn, đám đông, hàng hóa chụp từ xa, vật chiếm rất ít diện tích ảnh.
- **Không cần bật** khi: ảnh nhỏ (dưới 640px thì chạy như thường), vật to rõ ràng.
- **Đổi lại**: chậm hơn, vì ảnh càng lớn thì càng nhiều ô phải nhận diện.
- **Lưu ý**: khi nhìn gần vào từng ô, model đôi khi nhận nhầm hoa văn hay chữ trên biển thành vật khác (độ tin cậy thấp, khoảng 25-40%). Nếu thấy vật lạ, hãy **tăng ngưỡng độ tin cậy** (ví dụ 0.4) hoặc dùng ô **Chỉ đếm các loại** để chỉ đếm loại bạn cần.

Ví dụ thử nghiệm (Mac M1, model Nano): một ảnh 2400×2400 có 16 người cao 60px. Chế độ thường đếm được 1 người (0.1 giây), chế độ vật nhỏ đếm được 15 người (khoảng 1.5 giây).

### Đếm nhiều ảnh

Tab **Nhiều ảnh**: chọn nhiều ảnh cùng lúc (giữ `Cmd` hoặc `Shift` khi chọn), rồi bấm **Đếm tất cả**. Phần **Cài đặt** ở trên vẫn được áp dụng, trừ vùng đếm.

Kết quả gồm:
- Bảng mỗi ảnh một dòng (tổng và chi tiết từng loại). File hỏng được ghi lỗi vào bảng, các ảnh khác vẫn được đếm.
- File `tong_hop_nhieu_anh.csv`: mỗi ảnh một dòng, mỗi loại vật một cột, có dòng tổng cộng ở cuối.
- Bộ ảnh đã khoanh khung (bấm vào để xem to).

### Đếm video

Tab **Video**: tải lên một file video (MP4, MOV, AVI...), rồi bấm **Đếm video**.

Trong video, cùng một người xuất hiện ở hàng chục khung hình. Nếu cộng số đếm từng khung lại sẽ ra con số rất lớn. Ứng dụng dùng **tracking** (thuật toán ByteTrack): mỗi vật được gán một **ID** và theo dõi qua các khung, nên kết quả là **số vật khác nhau** đã xuất hiện. Ví dụ thử nghiệm: video 30 khung có 3 người đi ngang, cộng từng khung ra 79, còn tracking ra đúng 3.

Kết quả gồm:
- Số vật khác nhau theo loại, số khung đã xử lý, độ dài video, và nhiều nhất bao nhiêu vật xuất hiện cùng lúc.
- **Video kết quả**: mỗi vật có khung kèm nhãn `người #12` (12 là ID), góc trên trái ghi **Đã đếm: N** cập nhật theo thời gian.
- File CSV số lượng.

Lưu ý:
- Một ID phải xuất hiện ít nhất **3 khung hình** mới được đếm, để bỏ các nhận diện chập chờn.
- **Tốc độ** (Mac M1, model Nano): khoảng 35-45 ms mỗi khung hình. Video 1 phút ở 30 fps (1800 khung) mất khoảng 1-1.5 phút. Thanh tiến độ hiện số khung đã xử lý.
- Thanh trượt **Xử lý 1 trên N khung hình**: đặt 2 hoặc 3 để chạy nhanh gấp 2-3 lần với video dài. Đổi lại, tracking dễ mất dấu vật di chuyển nhanh.
- **Giới hạn của tracking**: vật bị che khuất lâu, hoặc đi ra khỏi khung rồi quay lại, có thể bị cấp ID mới và **bị đếm thêm lần nữa**. Khi nhiều vật đè lên nhau nhiều, ID cũng dễ bị đổi. Hợp nhất với camera cố định, vật đi qua rõ ràng.
- Chế độ vật nhỏ không áp dụng cho video (sẽ quá chậm). Ứng dụng chỉ cho tải file video lên, chưa quay trực tiếp từ webcam.

### Lịch sử

Mỗi lần đếm (một ảnh, nhiều ảnh và video) được tự động lưu vào file `data/history.db` trên máy bạn. Đây là SQLite, một cơ sở dữ liệu dạng file có sẵn trong Python. Ứng dụng **chỉ lưu số liệu** (thời gian, tên ảnh, model, chế độ, số lượng), **không lưu ảnh**.

Tab **Lịch sử** hiện 50 lần đếm gần nhất. Bấm **Làm mới** để cập nhật, hoặc **Xóa lịch sử** để xóa hết. Thư mục `data/` không được đưa lên git.

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

Test kiểm tra: từ chối file không phải ảnh, ảnh trống trả về 0 vật thể, ảnh xe buýt mẫu đếm được người và xe buýt, bộ lọc loại vật, ngưỡng độ tin cậy, bảng dịch tiếng Việt, file CSV xuất ra...

---

## 5. Dùng phần AI trong code Python của bạn

Không cần giao diện, bạn có thể gọi thẳng phần AI:

```python
from vision_count import LABEL_NUMBER, draw_detections, export_result, get_detector, load_image, vi_label

detector = get_detector()                         # model nano, nạp một lần rồi giữ lại
# detector = get_detector("small")                # hoặc model small, chính xác hơn
result = detector.detect("samples/bus.jpg", confidence=0.3)

print(result.total)      # tổng số vật, ví dụ: 5
print(result.counts)     # {'person': 4, 'bus': 1}
for d in result.detections:
    print(d.label, d.confidence, d.box)          # person 0.89 (48.0, 398.0, 245.0, 902.0)

# Chỉ đếm người
only_people = detector.detect("samples/bus.jpg", classes=["person"])

# Tên tiếng Việt (model vẫn dùng tên tiếng Anh làm mã)
print(vi_label("person"))  # người

# Lưu ảnh đã khoanh khung, nhãn tiếng Việt (bỏ label_fn để giữ tên tiếng Anh)
image = load_image("samples/bus.jpg")
annotated = draw_detections(image, result.detections, label_fn=vi_label)
annotated.save("ket_qua.jpg")

# Kiểu nhãn gọn: chỉ số thứ tự (hoặc LABEL_NONE: chỉ khung)
draw_detections(image, result.detections, label_style=LABEL_NUMBER).save("ket_qua_gon.jpg")

# Xuất ảnh + 2 file CSV vào thư mục "ket_qua/"
files = export_result(annotated, result, "ket_qua")
print(files)  # [ket_qua_....jpg, so_luong_....csv, chi_tiet_....csv]

# Dạng dict, sẵn sàng trả về JSON cho API
result.to_dict()

# Chế độ vật nhỏ (chia ô)
tiled = detector.detect("anh_lon.jpg", tiled=True)

# Chỉ giữ vật có tâm nằm trong vùng (x1, y1, x2, y2)
from vision_count import filter_by_region
in_region = filter_by_region(result, (0, 0, 400, 1080))

# Đếm nhiều ảnh, xuất CSV tổng hợp
from vision_count import count_many, write_batch_csv
items = count_many(detector, ["samples/bus.jpg", "samples/zidane.jpg"])
write_batch_csv(items, "tong_hop.csv")

# Đọc lịch sử đếm
from vision_count import HistoryStore
for entry in HistoryStore().recent(limit=5):
    print(entry.created_at, entry.source, entry.total, entry.counts)

# Đếm video (có tracking), ghi kèm video đã vẽ khung + ID
from vision_count import count_video
video = count_video(detector, "video.mp4", output_path="video_ket_qua.mp4", label_fn=vi_label)
print(video.counts, video.frames_processed)   # {'person': 3} 30
```

---

## 6. API cho ứng dụng khác (FastAPI)

Phần AI cũng được bọc thành **API** để app khác (web React, app điện thoại Flutter...) gửi ảnh/video lên và nhận kết quả dạng JSON. API dùng chung đúng phần AI với giao diện Gradio.

Chạy API (trong một Terminal riêng, đã `source .venv/bin/activate`):

```bash
uvicorn api.main:app --port 8000
```

Mở **http://127.0.0.1:8000/docs** để xem và **thử từng API ngay trên trình duyệt** (bấm vào một API → **Try it out** → chọn file → **Execute**).

| API | Việc |
|---|---|
| `GET /health` | Kiểm tra API đang chạy |
| `GET /models` | Danh sách model (nano, small...) và đã tải về chưa |
| `GET /classes?model=nano` | Các loại vật model nhận được, kèm tên tiếng Việt |
| `POST /detect` | Gửi ảnh, nhận JSON: tổng, số lượng theo loại, từng khung |
| `POST /detect/image` | Gửi ảnh, nhận lại ảnh JPEG đã khoanh khung |
| `POST /video/count` | Gửi video, nhận số vật khác nhau (có tracking) |

Các tham số gửi kèm (đều không bắt buộc): `model` (`nano`/`small`), `confidence` (0..1), `classes` (vd `person,car`), `tiled` (`true` = chế độ vật nhỏ), `region` (vùng đếm `x1,y1,x2,y2`), `label_style` (`full`/`number`/`none`, chỉ cho `/detect/image`), `vid_stride` (chỉ cho video).

Ví dụ gọi bằng `curl` (chạy ở thư mục project):

```bash
curl -F "file=@samples/bus.jpg" http://127.0.0.1:8000/detect
curl -F "file=@samples/bus.jpg" -F "classes=person" -F "region=0,0,400,1080" http://127.0.0.1:8000/detect
curl -F "file=@samples/bus.jpg" -F "label_style=number" http://127.0.0.1:8000/detect/image -o ket_qua.jpg
curl -F "file=@video.mp4" http://127.0.0.1:8000/video/count
```

Kết quả của `/detect` có dạng:

```json
{"model": "nano", "total": 5, "counts": {"person": 4, "bus": 1}, "counts_vi": {"người": 4, "xe buýt": 1},
 "detections": [{"label": "bus", "label_vi": "xe buýt", "class_id": 5, "confidence": 0.9404, "box": [3.8, 229.4, 796.2, 728.3]}, ...],
 "image_width": 810, "image_height": 1080, "region": null}
```

Khi gửi sai (file không phải ảnh, model/loại không có, vùng sai định dạng), API trả mã **400** kèm thông báo tiếng Việt trong `detail`. Ngưỡng ngoài 0..1 trả mã **422**.

> API mặc định chỉ cho **chính máy bạn** gọi (127.0.0.1). Muốn điện thoại cùng Wi-Fi gọi được thì chạy `uvicorn api.main:app --host 0.0.0.0 --port 8000` và gọi bằng địa chỉ IP của máy. Khi đó **mọi máy cùng mạng** đều gọi được, nên chỉ làm vậy trong mạng tin cậy.

---

## 7. Lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `command not found: python3.12` | Chưa cài Python 3.12, làm lại Bước 1. |
| `ModuleNotFoundError: No module named 'ultralytics'` (hoặc `gradio`) | Quên kích hoạt môi trường ảo: chạy `source .venv/bin/activate`. |
| Báo lỗi "không phải là file ảnh hợp lệ" | File bị hỏng hoặc không phải ảnh. Dùng JPG, PNG, WEBP, BMP. |
| "Không tìm thấy vật thể nào" | Giảm ngưỡng độ tin cậy, bỏ bộ lọc loại vật, hoặc thử ảnh rõ hơn. Nếu vật không thuộc 80 loại COCO thì xem mục fine-tune bên dưới. |
| Lần chạy đầu báo lỗi tải model | Cần mạng ở lần đầu để tải `yolo11n.pt`. Hoặc tự tải file về rồi đặt vào thư mục `models/`. |
| Tab Camera không hiện hình / không hỏi quyền | Trình duyệt chỉ cho dùng camera khi mở bằng `http://127.0.0.1:7860` hoặc `http://localhost:7860` (hoặc `https`). Nếu đã lỡ bấm **Chặn**, bấm biểu tượng ổ khóa cạnh thanh địa chỉ để cấp lại quyền camera. Trên macOS còn cần bật quyền tại **System Settings → Privacy & Security → Camera** cho trình duyệt. |
| "Không nạp được model Small" | Lần đầu chọn Small cần mạng để tải `yolo11s.pt`. Kết nối mạng rồi thử lại, hoặc tự tải file `yolo11s.pt` từ trang Ultralytics và đặt vào thư mục `models/`. |
| `Address already in use` (cổng 7860 đang bận) | Ứng dụng đang chạy ở một Terminal khác. Tắt nó đi, hoặc đổi `server_port` trong `app.py`. |

---

## 8. Fine-tune cho vật thể riêng

Model có sẵn chỉ biết 80 loại COCO. Muốn đếm vật riêng (ốc vít, viên thuốc, cá giống, bao hàng...), xem hướng dẫn từng bước trong **[training/README.md](training/README.md)**:

1. Thu thập ảnh (khoảng 100-300 ảnh mỗi loại) và gán nhãn (Label Studio, CVAT hoặc Roboflow), xuất định dạng YOLO.
2. Xếp thư mục theo mẫu, sửa `training/data.yaml` cho đúng tên các loại.
3. Kiểm tra dữ liệu: `python training/check_dataset.py duong_dan/dataset`.
4. Train miễn phí trên Google Colab bằng notebook `training/train_colab.ipynb`.
5. Chép `best.pt` vào `models/`, thêm một dòng vào `AVAILABLE_MODELS` trong `vision_count/config.py`, rồi chọn model mới trên giao diện. Ô lọc loại vật tự đổi theo model.

---

## 9. Hướng phát triển tiếp

Những việc chưa làm, có thể làm sau:

- **Đếm vật đi qua một vạch** trong video (xe qua cổng, người vào cửa). Ultralytics có sẵn `solutions.ObjectCounter`.
- **Đếm trực tiếp từ webcam hoặc camera IP** theo thời gian thực. Cần cài thêm ffmpeg nếu quay video qua giao diện.
- **App giao diện khác** (web React, app điện thoại Flutter) gọi vào API ở mục 6.
