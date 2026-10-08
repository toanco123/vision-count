"""Các hàm xử lý sự kiện của giao diện Gradio.

Tách khỏi phần bố cục (gradio_app.py) để test được trực tiếp, không cần mở trình duyệt.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

import gradio as gr
import pandas as pd

from vision_count import (
    LABEL_FULL,
    LABEL_NONE,
    LABEL_NUMBER,
    HistoryStore,
    InvalidImageError,
    InvalidVideoError,
    config,
    count_many,
    count_video,
    display_label,
    draw_detections,
    draw_region,
    export_result,
    filter_by_region,
    get_detector,
    is_model_downloaded,
    load_image,
    normalize_region,
    vi_label,
    write_batch_csv,
    write_video_csv,
)

COUNT_COLUMNS = ["Loại vật thể", "Số lượng"]
DETAIL_COLUMNS = ["#", "Loại vật thể", "Độ tin cậy", "Khung (x1, y1, x2, y2)"]
BATCH_COLUMNS = ["Ảnh", "Tổng", "Chi tiết"]
HISTORY_COLUMNS = ["Thời gian", "Nguồn", "Model", "Chế độ", "Tổng", "Chi tiết"]
VIDEO_COLUMNS = ["Loại vật thể", "Số vật khác nhau"]

# Hai nguồn ảnh: tải file lên hoặc chụp từ camera
SOURCE_UPLOAD = "upload"
SOURCE_WEBCAM = "webcam"

# Các kiểu nhãn cho người dùng chọn: (chữ hiển thị, giá trị)
LABEL_STYLE_CHOICES = [
    ("Đầy đủ: #1 người 87%", LABEL_FULL),
    ("Chỉ số thứ tự", LABEL_NUMBER),
    ("Chỉ khung", LABEL_NONE),
]

# Vùng đếm phải rộng/cao ít nhất 5px, tránh bấm 2 lần trùng một điểm tạo vùng rỗng
MIN_REGION_SIZE = 5
NO_REGION_TEXT = "Chưa chọn vùng: đếm cả ảnh. Muốn chỉ đếm một khu vực, bấm 2 góc lên ảnh kết quả."

# Ảnh trong bộ ảnh của tab Nhiều ảnh được thu nhỏ còn cạnh dài tối đa 1280px,
# để đếm nhiều ảnh điện thoại (12MP) không chiếm hàng GB bộ nhớ
GALLERY_MAX_SIDE = 1280

# Thư mục chứa file tải về của mọi lần đếm; thư mục con cũ hơn 1 giờ sẽ bị xóa
EXPORT_ROOT = Path(tempfile.gettempdir()) / "vision_count"


def new_export_dir(root: Path = EXPORT_ROOT, max_age_seconds: int = 3600) -> Path:
    """Tạo thư mục mới cho lần đếm này, đồng thời xóa các thư mục cũ để không đầy ổ đĩa."""
    root.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - max_age_seconds
    for child in root.iterdir():
        if child.is_dir() and child.stat().st_mtime < cutoff:
            shutil.rmtree(child, ignore_errors=True)
    return Path(tempfile.mkdtemp(dir=root))


def _model_name(model_key: str) -> str:
    return config.AVAILABLE_MODELS.get(model_key, (model_key,))[0]


def _load_model(model_key: str):
    """Nạp model; lỗi (sai tên, mất mạng khi tải lần đầu) thành thông báo trên giao diện."""
    if not is_model_downloaded(model_key):
        gr.Info(f"Đang tải model {_model_name(model_key)} lần đầu, vui lòng chờ...")
    try:
        return get_detector(model_key)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:  # Ví dụ: tải model thất bại vì mất mạng
        raise gr.Error(f"Không nạp được model {_model_name(model_key)}: {exc}") from exc


def _mode_text(tiled: bool, region) -> str:
    """Mô tả chế độ để hiện trong tóm tắt và lưu vào lịch sử."""
    parts = (["Vật nhỏ"] if tiled else []) + (["trong vùng đã chọn"] if region else [])
    return ", ".join(parts) or "Thường"


def _counts_text(counts: dict[str, int]) -> str:
    return ", ".join(f"{vi_label(label)}: {n}" for label, n in counts.items()) or "không có"


# ---------- Một ảnh ----------

def count_single(
    source_mode,
    image_path,
    webcam_image,
    model_key,
    confidence,
    selected_classes,
    label_style,
    tiled,
    region,
    history: HistoryStore | None = None,
):
    """Nút 'Đếm' của tab Một ảnh.

    Trả về (ảnh kết quả, tóm tắt, bảng số lượng, bảng chi tiết, file tải về, ảnh nền, các điểm).
    "Ảnh nền" là ảnh kết quả KHÔNG vẽ vùng đếm, dùng làm nền sạch khi người dùng chọn vùng mới;
    "các điểm" luôn rỗng để bỏ góc đã bấm dở từ trước lần đếm này.
    """
    if source_mode == SOURCE_WEBCAM:
        source, source_name = webcam_image, "Camera"  # Ảnh chụp từ camera, dạng mảng numpy
        if source is None:
            raise gr.Error("Vui lòng chụp ảnh từ camera trước.")
    else:
        source = image_path  # Đường dẫn file đã tải lên
        if not source:
            raise gr.Error("Vui lòng tải một ảnh lên trước.")
        source_name = Path(source).name

    detector = _load_model(model_key)
    try:
        image = load_image(source)
        result = detector.detect(image, confidence=confidence, classes=selected_classes or None, tiled=bool(tiled))
    except (InvalidImageError, ValueError) as exc:
        # gr.Error hiện thông báo lỗi màu đỏ trên giao diện thay vì làm sập app
        raise gr.Error(str(exc)) from exc

    result = filter_by_region(result, region)
    base = draw_detections(image, result.detections, label_fn=vi_label, label_style=label_style)
    annotated = draw_region(base, region) if region else base
    mode = _mode_text(bool(tiled), region)

    footer = f"_Model: {_model_name(model_key)} · Chế độ: {mode}_"
    if result.total == 0:
        hint = " Vùng đã chọn có thể không chứa vật nào: bấm **Xóa vùng** để đếm cả ảnh." if region else ""
        summary = (
            "### Không tìm thấy vật thể nào\n"
            "Thử **giảm ngưỡng độ tin cậy**, bỏ bớt bộ lọc loại vật, hoặc dùng ảnh rõ hơn. "
            "Nếu vật bạn cần đếm không nằm trong 80 loại COCO, cần fine-tune model (giai đoạn sau)."
            f"{hint}\n\n{footer}"
        )
    else:
        summary = f"### Tổng cộng: {result.total} vật thể ({len(result.counts)} loại)\n{footer}"

    counts_table = pd.DataFrame(
        [[display_label(label), n] for label, n in result.counts.items()], columns=COUNT_COLUMNS
    )
    detail_table = pd.DataFrame(
        [
            [i, display_label(d.label), f"{d.confidence:.0%}", ", ".join(f"{v:.0f}" for v in d.box)]
            for i, d in enumerate(result.detections, start=1)
        ],
        columns=DETAIL_COLUMNS,
    )
    # Mỗi lần đếm ghi vào một thư mục riêng, để không ghi đè file của lần trước
    files = export_result(annotated, result, new_export_dir())
    if history is not None:
        history.add(source_name, model_key, mode, result)
    return annotated, summary, counts_table, detail_table, [str(p) for p in files], base, []


# ---------- Chọn vùng bằng 2 lần bấm lên ảnh kết quả ----------

def add_region_point(base_image, points: list, region, xy):
    """Xử lý một lần bấm lên ảnh kết quả. Lần 1 đánh dấu góc thứ nhất, lần 2 tạo vùng.

    Luôn vẽ lên ảnh nền sạch (base_image, từ lần Đếm gần nhất) nên dấu cũ không bị cộng dồn.
    Trả về (ảnh hiển thị, các điểm đang chờ, vùng, dòng hướng dẫn).
    """
    if base_image is None:
        return None, [], region, "Hãy bấm **Đếm** một ảnh trước, rồi bấm 2 góc lên ảnh kết quả để chọn vùng."
    x, y = (float(v) for v in xy)
    if not points:
        return draw_region(base_image, point=(x, y)), [(x, y)], region, f"Đã chọn góc 1 ({x:.0f}, {y:.0f}). Bấm điểm thứ 2."
    new_region = normalize_region(points[0], (x, y))
    if new_region[2] - new_region[0] < MIN_REGION_SIZE or new_region[3] - new_region[1] < MIN_REGION_SIZE:
        # Bỏ chấm của lần bấm hỏng, hiện lại vùng đang dùng (nếu có)
        return draw_region(base_image, region), [], region, "Vùng quá nhỏ, hãy bấm lại 2 góc cách xa nhau hơn."
    x1, y1, x2, y2 = new_region
    info = f"Vùng đếm: ({x1:.0f}, {y1:.0f}) → ({x2:.0f}, {y2:.0f}). Bấm **Đếm** để chỉ đếm trong vùng này."
    return draw_region(base_image, region=new_region), [], new_region, info


def clear_region(base_image):
    """Nút 'Xóa vùng': quay lại đếm cả ảnh. Trả về (ảnh nền sạch, vùng, các điểm, dòng hướng dẫn)."""
    return base_image, None, [], NO_REGION_TEXT


# ---------- Nhiều ảnh ----------

def count_batch(
    file_paths, model_key, confidence, selected_classes, label_style, tiled, history: HistoryStore | None = None
):
    """Nút 'Đếm tất cả' của tab Nhiều ảnh. Trả về (tóm tắt, bảng, gallery, file CSV)."""
    if not file_paths:
        raise gr.Error("Vui lòng chọn ít nhất một ảnh.")
    detector = _load_model(model_key)
    try:
        items = count_many(
            detector, file_paths, confidence=confidence, classes=selected_classes or None, tiled=bool(tiled)
        )
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    mode = _mode_text(bool(tiled), None)
    gallery, rows = [], []
    for path, item in zip(file_paths, items):
        if item.result is None:
            rows.append([item.name, "", f"Lỗi: {item.error}"])
            continue
        rows.append([item.name, item.result.total, _counts_text(item.result.counts)])
        annotated = draw_detections(
            load_image(path), item.result.detections, label_fn=vi_label, label_style=label_style
        )
        annotated.thumbnail((GALLERY_MAX_SIDE, GALLERY_MAX_SIDE))  # chỉ giữ bản thu nhỏ trong bộ nhớ
        gallery.append((annotated, f"{item.name}: {item.result.total}"))
        if history is not None:
            history.add(item.name, model_key, f"Nhiều ảnh, {mode}", item.result)

    ok = [i for i in items if i.result]
    errors = len(items) - len(ok)
    summary = (
        f"### {len(items)} ảnh, tổng cộng {sum(i.result.total for i in ok)} vật thể"
        + (f" ({errors} lỗi)" if errors else "")
        + f"\n_Model: {_model_name(model_key)} · Chế độ: {mode}_"
    )
    csv_path = write_batch_csv(items, new_export_dir() / "tong_hop_nhieu_anh.csv")
    return summary, pd.DataFrame(rows, columns=BATCH_COLUMNS), gallery, str(csv_path)


# ---------- Ô lọc theo model ----------

def class_choices(model_key):
    """Đổi model thì cập nhật ô 'Chỉ đếm các loại' theo các loại model đó nhận được (vd model fine-tune)."""
    detector = _load_model(model_key)
    return gr.update(choices=[(display_label(n), n) for n in detector.class_names], value=[])


# ---------- Video ----------

def count_video_file(
    video_path, model_key, confidence, selected_classes, vid_stride, history: HistoryStore | None = None, progress=None
):
    """Nút 'Đếm video'. Trả về (tóm tắt, bảng, video kết quả, file CSV)."""
    if not video_path:
        raise gr.Error("Vui lòng tải một video lên trước.")
    detector = _load_model(model_key)
    stride = int(vid_stride)
    out_dir = new_export_dir()

    def report(done, total):
        if progress is not None:
            progress(done / total, desc=f"Đang xử lý khung hình {done}/{total}")

    try:
        result = count_video(
            detector,
            video_path,
            confidence=confidence,
            classes=selected_classes or None,
            vid_stride=stride,
            output_path=out_dir / "video_ket_qua.mp4",
            label_fn=vi_label,
            progress=report,
        )
    except (InvalidVideoError, ValueError) as exc:
        raise gr.Error(str(exc)) from exc

    mode = "Video" + (f", 1/{stride} khung hình" if stride > 1 else "")
    headline = (
        f"### {result.total} vật thể khác nhau trong video"
        if result.total
        else "### Không đếm được vật thể nào trong video\nThử giảm ngưỡng độ tin cậy hoặc bỏ bộ lọc loại vật."
    )
    warning = f"\n\n⚠️ **{result.warning}**\n" if result.warning else ""
    summary = (
        f"{headline}{warning}\n"
        f"{result.frames_processed} khung hình đã xử lý · video dài {result.duration_s:.1f} giây · "
        f"nhiều nhất {result.peak_in_frame} vật cùng lúc\n"
        f"_Model: {_model_name(model_key)} · Chế độ: {mode}_"
    )
    table = pd.DataFrame([[display_label(l), n] for l, n in result.counts.items()], columns=VIDEO_COLUMNS)
    csv_path = write_video_csv(result, out_dir / "so_luong_video.csv")
    if history is not None:
        history.add(Path(video_path).name, model_key, mode, result)
    video_out = str(result.output_path) if result.output_path else None
    return summary, table, video_out, str(csv_path)


# ---------- Lịch sử ----------

def history_table(history: HistoryStore, limit: int = 50) -> pd.DataFrame:
    """Bảng các lần đếm gần nhất (mới nhất trước)."""
    rows = [
        [e.created_at, e.source, _model_name(e.model_key), e.mode, e.total, _counts_text(e.counts)]
        for e in history.recent(limit)
    ]
    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


def clear_history(history: HistoryStore) -> pd.DataFrame:
    """Nút 'Xóa lịch sử'."""
    history.clear()
    return history_table(history)
