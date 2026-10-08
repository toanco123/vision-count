"""Giao diện Gradio. Chỉ lo phần hiển thị; mọi xử lý AI nằm trong package vision_count."""

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
    InvalidImageError,
    config,
    display_label,
    draw_detections,
    export_result,
    get_detector,
    is_model_downloaded,
    load_image,
    vi_label,
)

COUNT_COLUMNS = ["Loại vật thể", "Số lượng"]
DETAIL_COLUMNS = ["#", "Loại vật thể", "Độ tin cậy", "Khung (x1, y1, x2, y2)"]

# Hai nguồn ảnh: tải file lên hoặc chụp từ camera
SOURCE_UPLOAD = "upload"
SOURCE_WEBCAM = "webcam"

# Các kiểu nhãn cho người dùng chọn: (chữ hiển thị, giá trị)
LABEL_STYLE_CHOICES = [
    ("Đầy đủ: #1 người 87%", LABEL_FULL),
    ("Chỉ số thứ tự", LABEL_NUMBER),
    ("Chỉ khung", LABEL_NONE),
]

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


def build_app() -> gr.Blocks:
    """Tạo giao diện. Model được nạp qua get_detector (mỗi model một lần)."""

    def count_objects(source_mode, image_path, webcam_image, model_key, confidence, selected_classes, label_style):
        """Hàm được gọi khi bấm nút 'Đếm'. Trả về 5 giá trị cho 5 ô kết quả.

        source_mode cho biết người dùng đang ở tab nào: "upload" hoặc "webcam".
        """
        if source_mode == SOURCE_WEBCAM:
            source = webcam_image  # Ảnh chụp từ camera, dạng mảng numpy
            if source is None:
                raise gr.Error("Vui lòng chụp ảnh từ camera trước.")
        else:
            source = image_path  # Đường dẫn file đã tải lên
            if not source:
                raise gr.Error("Vui lòng tải một ảnh lên trước.")

        model_name = config.AVAILABLE_MODELS.get(model_key, (model_key,))[0]
        if not is_model_downloaded(model_key):
            gr.Info(f"Đang tải model {model_name} lần đầu, vui lòng chờ...")

        try:
            detector = get_detector(model_key)
            image = load_image(source)
            result = detector.detect(image, confidence=confidence, classes=selected_classes or None)
        except (InvalidImageError, ValueError) as exc:
            # gr.Error hiện thông báo lỗi màu đỏ trên giao diện thay vì làm sập app
            raise gr.Error(str(exc)) from exc
        except Exception as exc:  # Ví dụ: tải model thất bại vì mất mạng
            raise gr.Error(f"Không nạp được model {model_name}: {exc}") from exc

        annotated = draw_detections(image, result.detections, label_fn=vi_label, label_style=label_style)

        if result.total == 0:
            summary = (
                "### Không tìm thấy vật thể nào\n"
                "Thử **giảm ngưỡng độ tin cậy**, bỏ bớt bộ lọc loại vật, hoặc dùng ảnh rõ hơn. "
                "Nếu vật bạn cần đếm không nằm trong 80 loại COCO, cần fine-tune model (giai đoạn sau)."
                f"\n\n_Model: {model_name}_"
            )
        else:
            summary = f"### Tổng cộng: {result.total} vật thể ({len(result.counts)} loại)\n_Model: {model_name}_"

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
        return annotated, summary, counts_table, detail_table, [str(p) for p in files]

    # delete_cache: mỗi giờ Gradio xóa các file cũ hơn 1 ngày mà nó đã lưu, gồm bản sao file tải về
    # VÀ ảnh người dùng đã tải lên. Đặt 1 ngày (không phải 1 giờ) để ảnh không biến mất khi tab vẫn mở.
    with gr.Blocks(title="vision-count", delete_cache=(3600, 86400)) as app:
        gr.Markdown(
            "# vision-count: đếm vật thể trong ảnh\n"
            "Tải ảnh lên hoặc chụp bằng camera, chọn ngưỡng độ tin cậy rồi bấm **Đếm**. "
            "Model nhận được 80 loại vật thông dụng (người, xe, chó, mèo, chai, cốc...)."
        )

        with gr.Row():
            # Cột trái: đầu vào
            with gr.Column(scale=1):
                # Ghi nhớ tab đang mở để nút "Đếm" biết lấy ảnh từ đâu
                source_mode = gr.State(SOURCE_UPLOAD)

                with gr.Tabs():
                    with gr.Tab("Tải ảnh") as upload_tab:
                        # Dùng gr.File thay vì gr.Image để chính code của mình kiểm tra file,
                        # nhờ vậy file hỏng hoặc không phải ảnh sẽ có thông báo lỗi tiếng Việt rõ ràng.
                        image_input = gr.File(label="Ảnh cần đếm", file_types=["image"], type="filepath")
                    with gr.Tab("Camera") as webcam_tab:
                        # sources=["webcam"]: chỉ cho chụp từ camera. Trình duyệt sẽ hỏi quyền lần đầu.
                        webcam_input = gr.Image(label="Chụp ảnh từ camera", sources=["webcam"], type="numpy")

                # Khi người dùng bấm chuyển tab thì cập nhật source_mode
                upload_tab.select(fn=lambda: SOURCE_UPLOAD, outputs=source_mode)
                webcam_tab.select(fn=lambda: SOURCE_WEBCAM, outputs=source_mode)

                model_input = gr.Dropdown(
                    choices=[(label, key) for key, (label, _) in config.AVAILABLE_MODELS.items()],
                    value=config.DEFAULT_MODEL_KEY,
                    label="Model",
                    info="Small chính xác hơn với vật nhỏ/bị che. Lần đầu chọn sẽ tải file (~19MB).",
                )
                confidence_input = gr.Slider(
                    minimum=0.05,
                    maximum=0.95,
                    step=0.05,
                    value=config.DEFAULT_CONFIDENCE,
                    label="Ngưỡng độ tin cậy",
                    info="Cao: ít khung hơn nhưng chắc chắn hơn. Thấp: bắt được nhiều hơn nhưng dễ nhầm.",
                )
                class_input = gr.Dropdown(
                    # (chữ hiển thị, giá trị gửi về): hiện "người (person)" nhưng gửi về "person"
                    choices=[(display_label(name), name) for name in get_detector().class_names],
                    multiselect=True,
                    label="Chỉ đếm các loại (để trống = đếm tất cả)",
                )
                label_style_input = gr.Radio(
                    choices=LABEL_STYLE_CHOICES,
                    value=LABEL_FULL,
                    label="Kiểu nhãn trên ảnh",
                    info="Khi có nhiều vật, chọn kiểu gọn để nhãn không đè lên nhau.",
                )
                run_button = gr.Button("Đếm", variant="primary")

            # Cột phải: kết quả
            with gr.Column(scale=2):
                image_output = gr.Image(label="Kết quả", type="pil")
                summary_output = gr.Markdown()
                counts_output = gr.Dataframe(headers=COUNT_COLUMNS, label="Số lượng theo loại", interactive=False)
                with gr.Accordion("Chi tiết từng khung", open=False):
                    detail_output = gr.Dataframe(headers=DETAIL_COLUMNS, interactive=False)
                download_output = gr.File(
                    label="Tải kết quả về (ảnh + CSV)", file_count="multiple", interactive=False
                )

        run_button.click(
            fn=count_objects,
            inputs=[
                source_mode,
                image_input,
                webcam_input,
                model_input,
                confidence_input,
                class_input,
                label_style_input,
            ],
            outputs=[image_output, summary_output, counts_output, detail_output, download_output],
        )

    return app
