"""Giao diện Gradio: chỉ bố cục và nối sự kiện. Xử lý nằm ở ui/handlers.py, AI nằm ở vision_count."""

from __future__ import annotations

import gradio as gr

from ui import handlers
from ui.handlers import (
    BATCH_COLUMNS,
    COUNT_COLUMNS,
    DETAIL_COLUMNS,
    HISTORY_COLUMNS,
    LABEL_STYLE_CHOICES,
    NO_REGION_TEXT,
    SOURCE_UPLOAD,
    SOURCE_WEBCAM,
)
from vision_count import LABEL_FULL, HistoryStore, config, display_label, get_detector


def build_app(history: HistoryStore | None = None) -> gr.Blocks:
    """Tạo giao diện. history=None thì lưu lịch sử vào data/history.db."""
    history = history or HistoryStore()

    # delete_cache: mỗi giờ Gradio xóa các file cũ hơn 1 ngày mà nó đã lưu, gồm bản sao file tải về
    # VÀ ảnh người dùng đã tải lên. Đặt 1 ngày (không phải 1 giờ) để ảnh không biến mất khi tab vẫn mở.
    with gr.Blocks(title="vision-count", delete_cache=(3600, 86400)) as app:
        gr.Markdown(
            "# vision-count: đếm vật thể trong ảnh\n"
            "Model nhận được 80 loại vật thông dụng (người, xe, chó, mèo, chai, cốc...)."
        )

        # ---------- Cài đặt dùng chung cho tab Một ảnh và Nhiều ảnh ----------
        with gr.Accordion("Cài đặt", open=True):
            with gr.Row():
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
                tiled_input = gr.Checkbox(
                    value=False,
                    label="Chế độ vật nhỏ",
                    info="Chia ảnh thành ô 640px để bắt vật nhỏ/ở xa. Chậm hơn, hợp với ảnh lớn.",
                )
            with gr.Row():
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

        with gr.Tabs():
            # ---------- Tab Một ảnh ----------
            with gr.Tab("Một ảnh"):
                # Ghi nhớ tab nguồn đang mở, vùng đếm đã chọn và góc đang chờ góc thứ 2
                source_mode = gr.State(SOURCE_UPLOAD)
                region_state = gr.State(None)
                points_state = gr.State([])
                base_state = gr.State(None)  # ảnh kết quả chưa vẽ vùng: nền sạch để chọn vùng mới

                with gr.Row():
                    with gr.Column(scale=1):
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

                        region_info = gr.Markdown(NO_REGION_TEXT)
                        clear_region_button = gr.Button("Xóa vùng", size="sm")
                        run_button = gr.Button("Đếm", variant="primary")

                    with gr.Column(scale=2):
                        # interactive=False: ảnh kết quả chỉ để xem và bấm chọn vùng, không phải ô tải ảnh lên
                        # (vì nó có sự kiện select, Gradio có thể tự cho tải lên nếu không khóa lại)
                        image_output = gr.Image(
                            label="Kết quả (bấm 2 góc lên ảnh để chọn vùng đếm)", type="pil", interactive=False
                        )
                        summary_output = gr.Markdown()
                        counts_output = gr.Dataframe(
                            headers=COUNT_COLUMNS, label="Số lượng theo loại", interactive=False
                        )
                        with gr.Accordion("Chi tiết từng khung", open=False):
                            detail_output = gr.Dataframe(headers=DETAIL_COLUMNS, interactive=False)
                        download_output = gr.File(
                            label="Tải kết quả về (ảnh + CSV)", file_count="multiple", interactive=False
                        )

            # ---------- Tab Nhiều ảnh ----------
            with gr.Tab("Nhiều ảnh"):
                batch_input = gr.File(
                    label="Chọn nhiều ảnh", file_types=["image"], file_count="multiple", type="filepath"
                )
                batch_button = gr.Button("Đếm tất cả", variant="primary")
                batch_summary = gr.Markdown()
                batch_table = gr.Dataframe(headers=BATCH_COLUMNS, label="Kết quả từng ảnh", interactive=False)
                batch_download = gr.File(label="Tải bảng tổng hợp (CSV)", interactive=False)
                batch_gallery = gr.Gallery(label="Ảnh đã khoanh khung", columns=3, height="auto")

            # ---------- Tab Lịch sử ----------
            with gr.Tab("Lịch sử") as history_tab:
                gr.Markdown("Các lần đếm gần nhất (lưu trên máy, chỉ số liệu, không lưu ảnh).")
                with gr.Row():
                    refresh_button = gr.Button("Làm mới", size="sm")
                    clear_history_button = gr.Button("Xóa lịch sử", size="sm", variant="stop")
                history_output = gr.Dataframe(headers=HISTORY_COLUMNS, interactive=False)

        # ---------- Nối sự kiện ----------
        settings = [model_input, confidence_input, class_input, label_style_input, tiled_input]

        def on_count(mode, path, frame, model_key, conf, classes, style, tiled, region):
            return handlers.count_single(mode, path, frame, model_key, conf, classes, style, tiled, region, history)

        run_button.click(
            fn=on_count,
            inputs=[source_mode, image_input, webcam_input, *settings, region_state],
            outputs=[
                image_output,
                summary_output,
                counts_output,
                detail_output,
                download_output,
                base_state,
                points_state,
            ],
        )

        def on_image_click(base, points, region, evt: gr.SelectData):
            # evt.index = [x, y]: tọa độ pixel trên ảnh gốc nơi người dùng bấm
            return handlers.add_region_point(base, points, region, evt.index)

        image_output.select(
            fn=on_image_click,
            inputs=[base_state, points_state, region_state],
            outputs=[image_output, points_state, region_state, region_info],
        )
        clear_region_button.click(
            fn=handlers.clear_region,
            inputs=[base_state],
            outputs=[image_output, region_state, points_state, region_info],
        )

        def on_batch(paths, model_key, conf, classes, style, tiled):
            return handlers.count_batch(paths, model_key, conf, classes, style, tiled, history)

        batch_button.click(
            fn=on_batch,
            inputs=[batch_input, *settings],
            outputs=[batch_summary, batch_table, batch_gallery, batch_download],
        )

        def show_history():
            return handlers.history_table(history)

        history_tab.select(fn=show_history, outputs=history_output)
        refresh_button.click(fn=show_history, outputs=history_output)
        clear_history_button.click(fn=lambda: handlers.clear_history(history), outputs=history_output)

    return app
