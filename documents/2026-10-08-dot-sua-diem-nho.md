# Đợt sửa các điểm nhỏ còn tồn: Kế hoạch triển khai

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Mục tiêu:** Sửa 13 điểm nhỏ (Minor) mà lần review giai đoạn 3 để lại, được người dùng đồng ý ngày 2026-10-08. Không thêm chức năng mới.

**Kiến trúc:** Không đổi cấu trúc. Mỗi điểm sửa ở đúng module chứa nó, có test tái hiện viết trước.

**Spec:** Danh sách "minor (deferred)" của review giai đoạn 3 (ghi trong phần tổng kết gửi người dùng):
1. Lỗi 422 của API là tiếng Anh.
2. Vùng `nan`/`inf` được chấp nhận.
3. Ảnh cực lớn gây lỗi 500; `/detect` không giới hạn dung lượng.
4. Lần đầu dùng model khi mất mạng gây lỗi 500; `get_detector` chưa có khóa.
5. `peak_in_frame` tính cả ID chập chờn.
6. Gợi ý "giảm ngưỡng" cho video vô tác dụng dưới 0.25.
7. `MIN_TRACK_FRAMES` tính theo khung đã xử lý.
8. Video rất dài có thể bị xóa thư mục kết quả.
9. `check_dataset`: thiếu định dạng ảnh, nhãn polygon, cấu trúc Roboflow, `nc` khác `names`, file không UTF-8 / YAML lỗi.
10. Notebook: ảnh `.png`/`.JPG`/`.DS_Store`, tên `dataset.zip` cứng.
11. Đổi model luôn xóa ô lọc.
12. Chữ cũ.
13. Gộp vào điểm 3: DecompressionBomb trong đếm nhiều ảnh, vì `count_many` bắt `InvalidImageError`.

## Ràng buộc chung

- Không thêm thư viện. Giữ hành vi các chức năng hiện có (trừ đúng các điểm sửa). Chú thích tiếng Việt.

## Điểm cần chú ý khi review

1. **Ngưỡng tracking theo `conf`:** chỉ hạ ngưỡng của ByteTrack khi người dùng hạ `conf` dưới 0.25. Ngưỡng cao hơn thì giữ mặc định. Test: `test_tracker_config_follows_low_confidence`.
2. **Khóa của `get_detector`:** không được gây kẹt (deadlock) khi model đang tải mà có request khác. Test: `test_get_detector_loads_once_under_concurrency`.
3. **Giữ lựa chọn ô lọc khi đổi model:** loại không có ở model mới thì bỏ, loại có thì giữ. Test: `test_class_choices_keep_common_selection`.
4. **`check_dataset` theo đường dẫn trong `data.yaml`:** vẫn đúng với cấu trúc chuẩn cũ (các test cũ phải giữ nguyên kết quả).

---

### Task 1: Tăng độ chắc chắn cho API và đọc ảnh

**Files:** Sửa `api/main.py`, `vision_count/detector.py`, `vision_count/registry.py`. Test thêm vào `tests/test_api.py`, `tests/test_detector.py`, `tests/test_registry.py`.

- [ ] **Bước 1: Test (sẽ fail)**

`tests/test_api.py`, thêm:

```python
def test_api_validation_errors_are_vietnamese():
    resp = _post_image(confidence="2")
    assert resp.status_code == 422 and "Ngưỡng độ tin cậy" in resp.json()["detail"]
    resp = client.post("/detect", data={"model": "nano"})  # thiếu file
    assert resp.status_code == 422 and "file" in resp.json()["detail"] and "Thiếu" in resp.json()["detail"]


@pytest.mark.parametrize("region", ["nan,nan,nan,nan", "0,0,inf,inf"])
def test_api_region_must_be_finite(region):
    resp = _post_image(region=region)
    assert resp.status_code == 400 and "Vùng" in resp.json()["detail"]


def test_api_rejects_too_large_upload(monkeypatch):
    import api.main as api_main

    monkeypatch.setattr(api_main, "MAX_UPLOAD_MB", 0.01)  # ~10KB, ảnh mẫu 134KB
    resp = _post_image()
    assert resp.status_code == 413 and "quá lớn" in resp.json()["detail"]


def test_api_model_load_failure_is_503(monkeypatch):
    import api.main as api_main

    def offline(_key):
        raise RuntimeError("Download failure, environment may be offline")

    monkeypatch.setattr(api_main, "get_detector", offline)
    resp = _post_image()
    assert resp.status_code == 503 and "Không nạp được model" in resp.json()["detail"]
```

`tests/test_detector.py`, thêm:

```python
def test_load_image_rejects_decompression_bomb(monkeypatch):
    from PIL import Image as PILImage

    monkeypatch.setattr(PILImage, "MAX_IMAGE_PIXELS", 1000)  # ảnh mẫu 810x1080 vượt xa 2x ngưỡng
    with pytest.raises(InvalidImageError, match="quá lớn"):
        load_image(BUS_IMAGE)
```

`tests/test_registry.py`, thêm:

```python
def test_get_detector_loads_once_under_concurrency(monkeypatch):
    import threading
    import time

    from vision_count import registry

    built = []

    class SlowDetector:
        def __init__(self, weights):
            time.sleep(0.2)  # giả lập đang tải model
            built.append(weights)

    registry._load_detector.cache_clear()
    monkeypatch.setattr(registry, "ObjectDetector", SlowDetector)
    try:
        threads = [threading.Thread(target=registry.get_detector, args=("nano",)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert built == ["yolo11n.pt"]
    finally:
        registry._load_detector.cache_clear()  # bỏ SlowDetector khỏi cache cho các test sau
```

- [ ] **Bước 2:** Chạy `.venv/bin/pytest tests/test_api.py tests/test_detector.py tests/test_registry.py -q`. Kỳ vọng: các test mới FAIL.

- [ ] **Bước 3: Code**

`vision_count/detector.py`, trong `load_image`: thêm `Image.DecompressionBombError` vào phép bắt lỗi, với thông báo riêng (đặt trước `except (UnidentifiedImageError, ...)`):

```python
    except Image.DecompressionBombError as exc:
        raise InvalidImageError(
            f"'{name}' quá lớn (vượt {Image.MAX_IMAGE_PIXELS * 2:,} điểm ảnh). Hãy thu nhỏ ảnh rồi thử lại."
        ) from exc
```

`vision_count/registry.py`: khóa quanh lần nạp đầu tiên.

```python
import threading

_load_lock = threading.Lock()  # 2 yêu cầu cùng lúc lần đầu: chỉ nạp/tải model một lần


def get_detector(model_key: str = config.DEFAULT_MODEL_KEY) -> ObjectDetector:
    """Trả về detector của model được chọn. Mỗi model chỉ nạp một lần (an toàn khi gọi từ nhiều luồng)."""
    if model_key not in config.AVAILABLE_MODELS:
        raise ValueError(...)  # như cũ
    with _load_lock:
        return _load_detector(model_key)
```

`api/main.py`:
- Thêm `MAX_UPLOAD_MB = 50` cho ảnh. Đọc file bằng `file.file.read(limit + 1)`; nếu vượt giới hạn thì trả `HTTPException(413, "File quá lớn (tối đa 50MB).")`. Video dùng `MAX_VIDEO_MB = 2000` qua vòng chép có đếm byte.
- `_parse_region`: thêm `if not all(math.isfinite(v) for v in (x1, y1, x2, y2))` → 400 với cùng thông báo về vùng.
- Hàm `_get_model(model)`: `ValueError` → 400; lỗi khác → `HTTPException(503, f"Không nạp được model '{model}': {exc}. Lần đầu dùng model cần mạng để tải.")`. Thay mọi chỗ gọi `get_detector(model)` trong API bằng `_get_model(model)`.
- Bộ xử lý `RequestValidationError` trả 422 với `detail` tiếng Việt:

```python
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

_FIELD_MESSAGES = {
    "confidence": "Ngưỡng độ tin cậy (confidence) phải là số từ 0 đến 1.",
    "vid_stride": "vid_stride phải là số nguyên từ 1 đến 10.",
    "tiled": "tiled phải là true hoặc false.",
}


@app.exception_handler(RequestValidationError)
def _validation_error(request, exc: RequestValidationError):
    messages = []
    for err in exc.errors():
        field = str(err["loc"][-1]) if err.get("loc") else "?"
        if err.get("type") == "missing":
            messages.append(f"Thiếu trường bắt buộc '{field}'" + (" (file cần đếm)." if field == "file" else "."))
        else:
            messages.append(_FIELD_MESSAGES.get(field, f"Giá trị của '{field}' không hợp lệ."))
    return JSONResponse(status_code=422, content={"detail": " ".join(dict.fromkeys(messages))})
```

- [ ] **Bước 4:** `.venv/bin/pytest -q` → tất cả PASS.
- [ ] **Bước 5:** Commit `API chắc chắn hơn: lỗi 422 tiếng Việt, vùng hữu hạn, giới hạn dung lượng, model lỗi 503, ảnh quá lớn, khóa nạp model`.

---

### Task 2: Video: đỉnh cùng lúc, ngưỡng tracking, số khung tối thiểu, thư mục kết quả

**Files:** Sửa `vision_count/video.py`, `ui/handlers.py`. Test thêm vào `tests/test_video.py`, `tests/test_handlers.py`.

- [ ] **Bước 1: Test (sẽ fail)**

`tests/test_video.py`, thêm:

```python
def test_peak_counts_only_counted_tracks(synthetic_video):
    result = count_video(get_detector(), synthetic_video, min_track_frames=1000)
    assert result.total == 0 and result.peak_in_frame == 0


def test_min_frames_scale_with_stride():
    assert video_module._effective_min_frames(3, 1) == 3
    assert video_module._effective_min_frames(3, 2) == 2
    assert video_module._effective_min_frames(3, 5) == 2  # vẫn bỏ ID chỉ thấy 1 khung
    assert video_module._effective_min_frames(1, 5) == 1
    assert video_module._effective_min_frames(1000, 1) == 1000


def test_tracker_config_follows_low_confidence():
    import yaml

    low = yaml.safe_load(open(video_module._tracker_config(0.1), encoding="utf-8"))
    assert low["track_high_thresh"] == 0.1 and low["new_track_thresh"] == 0.1
    assert low["track_low_thresh"] <= 0.1
    default = yaml.safe_load(open(video_module._tracker_config(0.5), encoding="utf-8"))
    assert default["track_high_thresh"] == 0.25 and default["new_track_thresh"] == 0.25
    assert default["tracker_type"] == "bytetrack"


def test_low_confidence_still_counts(synthetic_video):
    assert count_video(get_detector(), synthetic_video, confidence=0.1).counts.get("person", 0) >= 3
```

`tests/test_handlers.py`, thêm:

```python
def test_new_export_dir_keeps_folder_with_recent_file(tmp_path):
    busy = tmp_path / "dang_ghi_video"
    busy.mkdir()
    (busy / "video_ket_qua.mp4").write_bytes(b"...")  # file vừa được ghi
    two_hours_ago = time.time() - 7200
    os.utime(busy, (two_hours_ago, two_hours_ago))  # thư mục tạo từ lâu
    handlers.new_export_dir(root=tmp_path, max_age_seconds=3600)
    assert busy.exists()
```

- [ ] **Bước 2:** Chạy các test trên. Kỳ vọng: FAIL.

- [ ] **Bước 3: Code** (`vision_count/video.py`)

```python
import math
import tempfile

_BYTETRACK_DEFAULTS = {
    "tracker_type": "bytetrack",
    "track_high_thresh": 0.25,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.25,
    "track_buffer": 30,
    "match_thresh": 0.8,
    "fuse_score": True,
}


def _tracker_config(confidence: float) -> str:
    """File cấu hình ByteTrack. Mặc định ByteTrack chỉ tạo ID mới cho khung có độ tin cậy >= 0.25,
    nên khi người dùng hạ ngưỡng dưới 0.25 thì hạ theo, để các vật mờ cũng được theo dõi."""
    cfg = dict(_BYTETRACK_DEFAULTS)
    if confidence < cfg["new_track_thresh"]:
        cfg["track_high_thresh"] = cfg["new_track_thresh"] = confidence
        cfg["track_low_thresh"] = min(cfg["track_low_thresh"], confidence)
    path = Path(tempfile.gettempdir()) / f"vision_count_bytetrack_{confidence:.3f}.yaml"
    path.write_text("".join(f"{k}: {v}\n" for k, v in cfg.items()), encoding="utf-8")
    return str(path)


def _effective_min_frames(min_track_frames: int, vid_stride: int) -> int:
    """Số khung ĐÃ XỬ LÝ tối thiểu. Bỏ bớt khung (stride) thì cần ít khung hơn cho cùng thời lượng,
    nhưng vẫn ít nhất 2 (nếu yêu cầu từ 2 trở lên) để bỏ ID chỉ thấy 1 khung."""
    return max(min(2, min_track_frames), math.ceil(min_track_frames / vid_stride))
```

Trong `count_video`:
- `tracker=_tracker_config(confidence)` thay cho `config.TRACKER`.
- `min_frames = _effective_min_frames(min_track_frames, vid_stride)` dùng ở cả `counted` (dòng "Đã đếm") lẫn bước đếm cuối.
- Lưu danh sách ID của từng khung (`frame_ids.append(ids)`). Cuối cùng: `counted_ids = {id có số khung >= min_frames}`, `peak = max((len(set(f) & counted_ids) for f in frame_ids), default=0)`. Bỏ phép tính `peak` trong vòng lặp.
- Trường `TRACKER` trong `config.py` thành chú thích (không dùng nữa) hoặc xóa. Chọn **xóa**, và ghi chú trong `_BYTETRACK_DEFAULTS` rằng đây là giá trị lấy từ `bytetrack.yaml` của Ultralytics.

`ui/handlers.py`:
- `new_export_dir`: tuổi của thư mục = thời điểm sửa **mới nhất** của thư mục và các file bên trong (video đang ghi dở có file mới được sửa). Bọc `stat`/`rmtree` trong `try/except OSError: continue` (2 tiến trình cùng dọn).
- Gợi ý khi video không đếm được: "Thử giảm ngưỡng độ tin cậy (ví dụ 0.15) hoặc bỏ bộ lọc loại vật" (giờ đã có tác dụng thật).

- [ ] **Bước 4:** `.venv/bin/pytest -q` → PASS.
- [ ] **Bước 5:** Commit `Video: đỉnh chỉ tính vật đã đếm, ngưỡng tracking theo conf, số khung tối thiểu theo stride, không xóa thư mục đang ghi`.

---

### Task 3: `check_dataset` và notebook

**Files:** Sửa `training/check_dataset.py`, `training/train_colab.ipynb` (tạo lại bằng script `training/_build_notebook.py`), `training/README.md`. Test thêm vào `tests/test_check_dataset.py`.

- [ ] **Bước 1: Test (sẽ fail)**

```python
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
    assert "sample.name" not in code
```

- [ ] **Bước 2:** Chạy `tests/test_check_dataset.py`. Kỳ vọng: các test mới FAIL.

- [ ] **Bước 3: Code**

`training/check_dataset.py`:
- `IMAGE_EXTS = {"." + f for f in IMG_FORMATS}` (import từ `ultralytics.data.utils`).
- Đọc YAML trong `try`: `yaml.YAMLError` → lỗi "Không đọc được data.yaml: ...". File nhãn: `UnicodeDecodeError` → lỗi "`<tên>`: không phải văn bản UTF-8".
- `nc` có mà khác `len(names)` → lỗi.
- Đường dẫn ảnh của mỗi split lấy từ `data["train"]` / `data["val"]`, giải nghĩa giống Ultralytics: tương đối so với thư mục chứa `data.yaml`; không tồn tại mà bắt đầu bằng `../` thì bỏ `../`. Thư mục nhãn = thay thành phần `images` **cuối cùng** trong đường dẫn bằng `labels` (quy tắc `img2label_paths`). Thiếu khóa thì dùng mặc định `images/train`, `images/val`. Thông báo lỗi ghi đường dẫn tương đối so với gốc dataset (giữ được chữ `images/val`, `labels/train` như các test cũ).
- Dòng nhãn: 5 số là khung (như cũ); nhiều hơn 5 thì phải là polygon (`class` + số tọa độ chẵn, ít nhất 6), mọi tọa độ trong 0..1.

`training/_build_notebook.py`: chuyển đoạn tạo notebook thành script (để sửa sau dễ hơn), rồi chạy nó để tạo lại `train_colab.ipynb`, với:
- Ô 3: `zip_name = next(iter(uploaded))`, rồi `!unzip -q -o "{zip_name}" -d /content/`.
- Ô 7: chọn ảnh mẫu theo đuôi trong `IMG_FORMATS` (bỏ qua `.DS_Store`), hiển thị bằng `Image.fromarray(pred[0].plot()[..., ::-1])`.

`training/README.md`: nhắc rằng hỗ trợ cả cấu trúc Roboflow (`train/images`, `valid/images`), và có thể dùng nhãn polygon.

- [ ] **Bước 4:** `.venv/bin/pytest -q` → PASS.
- [ ] **Bước 5:** Commit `check_dataset: theo đường dẫn data.yaml (Roboflow), polygon, đủ định dạng ảnh, nc/names, lỗi đọc file; notebook chắc chắn hơn`.

---

### Task 4: Giao diện: giữ lựa chọn ô lọc, chữ cũ

**Files:** Sửa `ui/handlers.py`, `ui/gradio_app.py`, `README.md`. Test thêm vào `tests/test_handlers.py`.

- [ ] **Bước 1: Test (sẽ fail)**

```python
def test_class_choices_keep_common_selection():
    update = handlers.class_choices("small", ["person", "khong_co_o_model_nay"])
    assert update["value"] == ["person"]


def test_class_choices_without_selection():
    assert handlers.class_choices("small")["value"] == []


def test_no_result_hint_points_to_training_guide():
    blank = Image.new("RGB", (320, 240), "white")
    path = Path(handlers.new_export_dir()) / "trang.png"
    blank.save(path)
    _, summary, *_ = handlers.count_single(SOURCE_UPLOAD, str(path), None, "nano", 0.25, [], LABEL_FULL, False, None, None)
    assert "training/README.md" in summary and "giai đoạn sau" not in summary
```

Sửa `test_class_choices_follow_model` cho hợp chữ ký mới (`handlers.class_choices("small", [])`, vẫn kỳ vọng `value == []`).

- [ ] **Bước 2:** Chạy. Kỳ vọng: FAIL.

- [ ] **Bước 3: Code**
- `class_choices(model_key, selected=None)`: `names = detector.class_names`; `value=[c for c in (selected or []) if c in names]`. Trong `gradio_app.py`, đổi `inputs=[model_input, class_input]`.
- `count_single`: câu gợi ý cuối thành "Nếu vật bạn cần đếm không nằm trong 80 loại có sẵn, xem cách tự train model trong `training/README.md`."
- `gradio_app.py`: tiêu đề `# vision-count: đếm vật thể trong ảnh và video`; chú thích phần Cài đặt: "dùng chung cho tab Một ảnh, Nhiều ảnh và Video".
- `README.md`: "(3 tab)" → "(4 tab)"; câu "các lựa chọn cũ bị xóa" thành "giữ lại các loại mà model mới cũng có".

- [ ] **Bước 4:** `.venv/bin/pytest -q` → PASS. Mở trình duyệt: chọn "người (person)", đổi sang Small → vẫn giữ "người (person)".
- [ ] **Bước 5:** Commit `Giao diện: giữ lựa chọn ô lọc khi đổi model, cập nhật chữ cũ`.
