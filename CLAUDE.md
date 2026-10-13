# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Tổng quan

`vision-count`: ứng dụng đếm vật thể trong ảnh và video bằng YOLO11 (Ultralytics), chạy **offline trên máy** (macOS, Apple M1, CPU). Có hai cách dùng: giao diện Gradio và API FastAPI. Người dùng là người Việt mới học AI xử lý ảnh, nên **chú thích code, thông báo lỗi, README và commit message đều viết bằng tiếng Việt**. Hãy giữ nguyên quy ước này.

## Lệnh thường dùng

Dùng Python 3.12 trong `.venv`. Python 3.9 có sẵn của macOS quá cũ; Python 3.12 được cài qua Homebrew.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt   # cài đặt lần đầu
.venv/bin/python app.py                              # giao diện Gradio: http://127.0.0.1:7860
.venv/bin/uvicorn api.main:app --port 8000           # API, có tài liệu thử trực tiếp ở /docs
.venv/bin/pytest -q                                  # toàn bộ test (~155 test, ~30 giây)
.venv/bin/pytest tests/test_video.py -q              # một file test
.venv/bin/pytest tests/test_tiling.py::test_merge_overlaps_keeps_separate_boxes -q   # một test
.venv/bin/python training/check_dataset.py <thư_mục_dataset>   # kiểm tra bộ dữ liệu trước khi fine-tune
.venv/bin/python training/_build_notebook.py         # tạo lại training/train_colab.ipynb
```

- `pytest.ini` đặt `pythonpath = .`, nên test import được `vision_count`, `ui`, `api`, `training`. Chạy script lẻ thì cần `PYTHONPATH=.`.
- Không có linter hay formatter nào được cấu hình.
- **Không sửa tay `train_colab.ipynb`.** Sửa `training/_build_notebook.py` rồi chạy lại script để sinh notebook.
- Trong môi trường shell của Claude Code trên máy này, `grep` là một hàm bọc `ugrep`, có lúc treo rất lâu khi đọc từ pipe. Lọc output thì dùng `/usr/bin/grep`, hoặc ghi output ra file trước.

## Kiến trúc

Ba lớp, phụ thuộc một chiều: `vision_count` (lõi AI) ← `ui/handlers.py` ← `ui/gradio_app.py`. Còn `api/main.py` là một lớp vỏ song song, cũng chỉ gọi `vision_count`.

- **`vision_count/`**: lõi, **không được import Gradio hay FastAPI**. Mọi chức năng mới đặt ở đây trước, để cả UI lẫn API dùng chung.
- **`ui/handlers.py`**: các hàm thường xử lý sự kiện. Chúng nhận giá trị và trả tuple cho các ô output, ném `gr.Error` khi lỗi. Test gọi thẳng các hàm này, không cần trình duyệt.
- **`ui/gradio_app.py`**: chỉ bố cục và nối sự kiện. Gồm phần Cài đặt dùng chung (model, ngưỡng, chế độ vật nhỏ, lọc loại, kiểu nhãn) và 4 tab: Một ảnh (tải lên hoặc webcam, có vùng đếm), Nhiều ảnh, Video, Lịch sử. `build_app(history)` nhận `HistoryStore` để test truyền vào file DB tạm.
- **`api/main.py`**:
  - Endpoint: `/detect`, `/detect/image`, `/video/count`, `/models`, `/classes`, `/health`.
  - Lỗi đầu vào → 400 (`detail` tiếng Việt); file quá lớn → 413; lỗi kiểm tra tham số → 422, được chuyển sang tiếng Việt bởi handler `RequestValidationError`; không nạp được model → 503.
  - `_model_lock` bao quanh `detect()`, vì FastAPI chạy hàm `def` trên nhiều luồng.
  - Không dùng tên file người gửi khi lưu tạm (trình duyệt gửi tên `blob`, `..`...).

### Quy ước xuyên suốt nhiều file

- **Tên loại vật luôn là tiếng Anh ở bên trong:** `Detection.label`, `counts`, tham số `classes`. Tiếng Việt chỉ thêm ở lớp hiển thị, qua `labels_vi.vi_label` / `display_label` (`"người (person)"`) và tham số `label_fn` của các hàm vẽ. API trả cả `counts` lẫn `counts_vi`. Ô lọc của Gradio dùng tuple `(nhãn hiển thị, giá trị tiếng Anh)`.
- **Model:**
  - Danh sách nằm ở `config.AVAILABLE_MODELS`; file `.pt` để trong `models/` (bị gitignore, tự tải lần đầu).
  - Luôn lấy model qua `registry.get_detector(key)`: có cache và khóa, mỗi model chỉ nạp một lần.
  - Thêm model fine-tune = thêm một dòng vào `AVAILABLE_MODELS`. Ô lọc tự đổi theo `class_names` của model đang chọn, qua `handlers.class_choices`.
- **Lỗi đầu vào:** `InvalidImageError` / `InvalidVideoError` (lớp con của `ValueError`, thông báo tiếng Việt, nêu tên file). Handlers đổi thành `gr.Error`, API đổi thành 400.
- **`ObjectDetector.detect(image, confidence, classes, tiled=False)`:** gọi như cũ phải luôn ra kết quả như cũ. `load_image` nhận đường dẫn, bytes, PIL hoặc numpy, có xoay theo EXIF và tham số `name` để thông báo lỗi.
- **Chế độ vật nhỏ** (`tiling.py`):
  - Lượt 0 là cả ảnh; các lượt sau là từng ô 640px chồng nhau 20%.
  - `drop_cut_boxes` bỏ các khung chạm mép trong của ô.
  - `merge_overlaps` giữ mọi khung của lượt cả ảnh, và chỉ gộp giữa **các lượt khác nhau**: IoU > 0.5, hoặc khung của ô nằm >80% bên trong khung cùng loại to hơn của lượt cả ảnh ("mảnh vụn").
  - Khung trong **cùng một lượt** không bao giờ bị gộp (giữ được trẻ em đứng trước người lớn).
- **Vùng đếm:** vật thuộc vùng khi **tâm** khung nằm trong vùng (`region.filter_by_region`). UI giữ `base_state`, là ảnh kết quả chưa vẽ vùng, nên mọi dấu chọn vùng luôn được vẽ lại trên ảnh sạch. Ảnh kết quả phải để `interactive=False`.
- **Video** (`video.py`):
  - `count_video` tạo **bản YOLO riêng mỗi lần**, để trạng thái tracking không dính vào model dùng chung.
  - Dùng ByteTrack qua file yaml tạm do `_tracker_config(conf)` sinh ra. Ngưỡng của ByteTrack hạ theo khi `conf < 0.25`. File được ghi bằng `yaml.safe_dump`, vì viết tay `1e-05` sẽ bị YAML đọc thành chuỗi.
  - Một ID chỉ được đếm khi xuất hiện ≥ `MIN_TRACK_FRAMES` khung, có điều chỉnh theo `vid_stride`.
  - Chỉ nhận đuôi trong `VID_FORMATS` của Ultralytics, và từ chối file ảnh bị đổi đuôi.
  - Có cảnh báo `complete` / `warning` khi đọc được dưới 90% số khung.
  - Video kết quả ghi bằng codec `avc1` (H.264) để trình duyệt phát được; máy không có ffmpeg, nên `gr.Video` chỉ cho tải file lên.
- **File xuất ra:**
  - Mỗi lần đếm dùng một thư mục riêng do `handlers.new_export_dir()` tạo dưới `tempdir/vision_count`. Thư mục cũ hơn 1 giờ (tính theo file mới nhất bên trong) bị xóa ở lần tạo kế tiếp.
  - CSV luôn ghi bằng `utf-8-sig` (qua `export.write_csv`) để Excel đọc đúng dấu.
  - `gr.Blocks(delete_cache=(3600, 86400))` cố ý để 1 ngày: Gradio xóa cả ảnh người dùng tải lên theo tham số này.
- **Lịch sử:** SQLite ở `data/history.db` (bị gitignore). Chỉ lưu số liệu, không lưu ảnh. Mỗi thao tác mở một kết nối mới. `HistoryStore.add` nhận bất kỳ kết quả nào có `.total` và `.counts` (cả ảnh lẫn video).
- **Fine-tune** (`training/`):
  - `check_dataset.py` giải nghĩa đường dẫn train/val trong `data.yaml` giống Ultralytics: bỏ `../` khi không tồn tại, thay thư mục `images` cuối cùng bằng `labels`.
  - `data.yaml` **không được có `path: .`**, vì Ultralytics hiểu `.` là thư mục đang chạy lệnh.

## Test

- Test chạy model thật (`yolo11n.pt`, `yolo11s.pt`), nên lần đầu cần mạng để tải model.
- Ảnh mẫu lấy từ `ultralytics.utils.ASSETS` (`bus.jpg`: 4 người + 1 xe buýt; `zidane.jpg`).
- Fixture `synthetic_video` (session) trong `tests/conftest.py` sinh video 30 khung, 15 fps, có 3 người đi ngang. Kết quả kỳ vọng là đúng 3 ID.
- Test vật nhỏ dùng ảnh 2400×2400 có 16 người cao 60px: chế độ thường ≤ 3, chế độ vật nhỏ 13-16.
- `tests/test_ui.py` kiểm tra cấu trúc app qua `app.blocks`; logic giao diện được test ở `tests/test_handlers.py`.

## Cách làm việc trong repo này

- **Kế hoạch:** mỗi đợt tính năng có kế hoạch triển khai trong `documents/` (tên dạng `YYYY-MM-DD-<chủ-đề>.md`). Thư mục `.superpowers/` là nơi theo dõi tiến độ tạm thời, bị gitignore.
- **Quy trình:** làm trên nhánh riêng, viết test trước, review, rồi merge `--ff-only` vào `main`. **Hỏi người dùng trước khi push.**
- **README:** `README.md` và `training/README.md` là hướng dẫn tiếng Việt cho người mới, có ghi số liệu đo thật (tốc độ, kết quả thử). Khi đổi hành vi thì cập nhật README cho khớp, và đo lại trước khi ghi số liệu mới.
