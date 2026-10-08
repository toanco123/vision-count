"""Fixture dùng chung cho nhiều file test."""

import cv2
import numpy as np
import pytest
from PIL import Image
from ultralytics.utils import ASSETS


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory):
    """Video 30 khung, 15 fps, 640x480: 3 người (cắt từ ảnh xe buýt mẫu) đi ngang với tốc độ khác nhau."""
    from vision_count import get_detector

    bus = Image.open(ASSETS / "bus.jpg").convert("RGB")
    people = get_detector().detect(bus, classes=["person"]).detections[:3]
    crops = [bus.crop(tuple(int(v) for v in d.box)) for d in people]
    crops = [c.resize((int(c.width * 200 / c.height), 200)) for c in crops]

    path = tmp_path_factory.mktemp("video") / "synthetic.mp4"
    width, height, n_frames = 640, 480, 30
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"avc1"), 15, (width, height))
    for f in range(n_frames):
        frame = Image.new("RGB", (width, height), (110, 120, 110))
        for k, crop in enumerate(crops):
            # Di chuyển liên tục (không "dịch chuyển tức thời"), mỗi người một làn và một tốc độ
            x = int(20 + (width - 160) * f / (n_frames - 1) * (0.6 + 0.2 * k))
            frame.paste(crop, (x, 20 + k * 130))
        writer.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    writer.release()
    return path
