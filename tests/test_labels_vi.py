"""Test bảng dịch tên loại vật sang tiếng Việt."""

from vision_count import ObjectDetector, display_label, vi_label
from vision_count.labels_vi import COCO_VI


def test_translates_common_labels():
    assert vi_label("person") == "người"
    assert vi_label("car") == "ô tô"
    assert vi_label("bottle") == "chai"


def test_unknown_label_falls_back_to_original():
    # Model fine-tune có thể có loại mới chưa có trong bảng dịch
    assert vi_label("oc_vit") == "oc_vit"
    assert display_label("oc_vit") == "oc_vit"


def test_display_label_shows_both_names():
    assert display_label("person") == "người (person)"


def test_covers_all_80_coco_classes():
    names = ObjectDetector().class_names
    missing = [n for n in names if n not in COCO_VI]
    assert missing == []
    assert len(COCO_VI) == 80
