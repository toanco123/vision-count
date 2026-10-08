"""Giao diện Gradio. Chỉ lo phần hiển thị; mọi xử lý AI nằm trong package vision_count."""

from __future__ import annotations

import gradio as gr
import pandas as pd

from vision_count import InvalidImageError, ObjectDetector, config, draw_detections, load_image

COUNT_COLUMNS = ["Loại vật thể", "Số lượng"]
DETAIL_COLUMNS = ["#", "Loại vật thể", "Độ tin cậy", "Khung (x1, y1, x2, y2)"]


def build_app(detector: ObjectDetector) -> gr.Blocks:
    """Tạo giao diện, nhận vào một detector đã nạp sẵn model."""

    def count_objects(image_path, confidence, selected_classes):
        """Hàm được gọi khi bấm nút 'Đếm'. Trả về 4 giá trị cho 4 ô kết quả."""
        if not image_path:
            raise gr.Error("Vui lòng tải một ảnh lên trước.")

        try:
            image = load_image(image_path)
            result = detector.detect(image, confidence=confidence, classes=selected_classes or None)
        except (InvalidImageError, ValueError) as exc:
            # gr.Error hiện thông báo lỗi màu đỏ trên giao diện thay vì làm sập app
            raise gr.Error(str(exc)) from exc

        annotated = draw_detections(image, result.detections)

        if result.total == 0:
            summary = (
                "### Không tìm thấy vật thể nào\n"
                "Thử **giảm ngưỡng độ tin cậy**, bỏ bớt bộ lọc loại vật, hoặc dùng ảnh rõ hơn. "
                "Nếu vật bạn cần đếm không nằm trong 80 loại COCO, cần fine-tune model (giai đoạn sau)."
            )
        else:
            summary = f"### Tổng cộng: {result.total} vật thể ({len(result.counts)} loại)"

        counts_table = pd.DataFrame(list(result.counts.items()), columns=COUNT_COLUMNS)
        detail_table = pd.DataFrame(
            [
                [i, d.label, f"{d.confidence:.0%}", ", ".join(f"{v:.0f}" for v in d.box)]
                for i, d in enumerate(result.detections, start=1)
            ],
            columns=DETAIL_COLUMNS,
        )
        return annotated, summary, counts_table, detail_table

    with gr.Blocks(title="vision-count") as app:
        gr.Markdown(
            "# vision-count: đếm vật thể trong ảnh\n"
            "Tải ảnh lên, chọn ngưỡng độ tin cậy rồi bấm **Đếm**. "
            "Model nhận được 80 loại vật thông dụng (người, xe, chó, mèo, chai, cốc...)."
        )

        with gr.Row():
            # Cột trái: đầu vào
            with gr.Column(scale=1):
                # Dùng gr.File thay vì gr.Image để chính code của mình kiểm tra file,
                # nhờ vậy file hỏng hoặc không phải ảnh sẽ có thông báo lỗi tiếng Việt rõ ràng.
                image_input = gr.File(label="Ảnh cần đếm", file_types=["image"], type="filepath")
                confidence_input = gr.Slider(
                    minimum=0.05,
                    maximum=0.95,
                    step=0.05,
                    value=config.DEFAULT_CONFIDENCE,
                    label="Ngưỡng độ tin cậy",
                    info="Cao: ít khung hơn nhưng chắc chắn hơn. Thấp: bắt được nhiều hơn nhưng dễ nhầm.",
                )
                class_input = gr.Dropdown(
                    choices=detector.class_names,
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

        run_button.click(
            fn=count_objects,
            inputs=[image_input, confidence_input, class_input],
            outputs=[image_output, summary_output, counts_output, detail_output],
        )

    return app
