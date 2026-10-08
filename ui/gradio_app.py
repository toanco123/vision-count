"""Giao diện Gradio. Chỉ lo phần hiển thị; mọi xử lý AI nằm trong package vision_count."""

from __future__ import annotations

import tempfile

import gradio as gr
import pandas as pd

from vision_count import (
    InvalidImageError,
    ObjectDetector,
    config,
    display_label,
    draw_detections,
    export_result,
    load_image,
    vi_label,
)

COUNT_COLUMNS = ["Loại vật thể", "Số lượng"]
DETAIL_COLUMNS = ["#", "Loại vật thể", "Độ tin cậy", "Khung (x1, y1, x2, y2)"]

# Hai nguồn ảnh: tải file lên hoặc chụp từ camera
SOURCE_UPLOAD = "upload"
SOURCE_WEBCAM = "webcam"


def build_app(detector: ObjectDetector) -> gr.Blocks:
    """Tạo giao diện, nhận vào một detector đã nạp sẵn model."""

    def count_objects(source_mode, image_path, webcam_image, confidence, selected_classes):
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

        try:
            image = load_image(source)
            result = detector.detect(image, confidence=confidence, classes=selected_classes or None)
        except (InvalidImageError, ValueError) as exc:
            # gr.Error hiện thông báo lỗi màu đỏ trên giao diện thay vì làm sập app
            raise gr.Error(str(exc)) from exc

        annotated = draw_detections(image, result.detections, label_fn=vi_label)

        if result.total == 0:
            summary = (
                "### Không tìm thấy vật thể nào\n"
                "Thử **giảm ngưỡng độ tin cậy**, bỏ bớt bộ lọc loại vật, hoặc dùng ảnh rõ hơn. "
                "Nếu vật bạn cần đếm không nằm trong 80 loại COCO, cần fine-tune model (giai đoạn sau)."
            )
        else:
            summary = f"### Tổng cộng: {result.total} vật thể ({len(result.counts)} loại)"

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
        # Mỗi lần đếm ghi vào một thư mục tạm riêng, để không ghi đè file của lần trước
        files = export_result(annotated, result, tempfile.mkdtemp(prefix="vision_count_"))
        return annotated, summary, counts_table, detail_table, [str(p) for p in files]

    with gr.Blocks(title="vision-count") as app:
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
                    choices=[(display_label(name), name) for name in detector.class_names],
                    multiselect=True,
                    label="Chỉ đếm các loại (để trống = đếm tất cả)",
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
            inputs=[source_mode, image_input, webcam_input, confidence_input, class_input],
            outputs=[image_output, summary_output, counts_output, detail_output, download_output],
        )

    return app
