import csv
import json

import cv2
import numpy as np

from src.capture.raw_recorder import RawRgbdRecorder


def test_records_only_when_active_and_roundtrips(tmp_path):
    rec = RawRgbdRecorder(tmp_path / "p01_normal", {"width": 8, "height": 6, "depth_scale": 0.001})
    color = np.full((6, 8, 3), 120, np.uint8)
    depth = np.arange(48, dtype=np.uint16).reshape(6, 8) * 20 + 500
    rec.put(1.0, color, depth)          # active=False -> 저장 안 됨
    rec.active = True
    rec.put(2.0, color, depth)
    rec.put(3.0, color, depth + 1)
    assert rec.close() == 2
    rows = list(csv.DictReader(open(tmp_path / "p01_normal" / "frames.csv", encoding="utf-8")))
    assert [r["index"] for r in rows] == ["1", "2"]
    back = cv2.imread(str(tmp_path / "p01_normal" / rows[1]["depth_file"]), cv2.IMREAD_UNCHANGED)
    assert back.dtype == np.uint16 and (back == depth + 1).all()
    meta = json.load(open(tmp_path / "p01_normal" / "meta.json", encoding="utf-8"))
    assert meta["frames"] == 2 and meta["depth_scale"] == 0.001


def test_refuses_to_overwrite(tmp_path):
    import pytest
    r = RawRgbdRecorder(tmp_path / "x")
    r.active = True
    r.put(1.0, np.zeros((2, 2, 3), np.uint8), np.zeros((2, 2), np.uint16))
    r.close()
    with pytest.raises(SystemExit):
        RawRgbdRecorder(tmp_path / "x")


def test_frame_numbers_timestamps_and_meta(tmp_path):
    from src.capture.raw_recorder import frame_number_stats
    rec = RawRgbdRecorder(tmp_path / "s", {"depth_scale": 0.001})
    rec.session_info["pose"] = "P0"
    rec.active = True
    c = np.zeros((4, 4, 3), np.uint8)
    d = np.ones((4, 4), np.uint16)
    # 프레임 번호 10,11,13(12 누락), 33.3ms 간격
    for k, fn in enumerate([10, 11, 13]):
        rec.put(1000.0 + 33.3 * k, c, d, {"color_frame_number": fn, "depth_frame_number": fn,
                                         "depth_timestamp_ms": 1000.0 + 33.3 * k + 0.5,
                                         "timestamp_domain": "global_time"})
    assert rec.close() == 3
    rows = list(csv.DictReader(open(tmp_path / "s" / "frames.csv", encoding="utf-8")))
    assert rows[2]["color_frame_number"] == "13" and rows[0]["timestamp_domain"] == "global_time"
    assert rows[0]["color_file"].endswith(".png")
    meta = json.load(open(tmp_path / "s" / "meta.json", encoding="utf-8"))
    assert meta["pose"] == "P0"
    assert meta["color_frame_number_check"]["missing"] == 1
    assert abs(meta["saved_fps"] - 30.03) < 0.1
    assert abs(meta["rgb_depth_timestamp_diff_ms"]["max"] - 0.5) < 1e-6
    assert meta["depth_invalid_value"] == 0
    assert meta["segment"]["start_color_frame_number"] == 10 and meta["segment"]["end_index"] == 3
    assert frame_number_stats([1, 2, 2, 5])["missing"] == 2
    assert frame_number_stats([1, 2, 2, 5])["duplicates"] == 1


def test_jpg_option_and_png_lossless(tmp_path):
    color = (np.arange(48, dtype=np.uint8).reshape(4, 4, 3) * 5)
    rec = RawRgbdRecorder(tmp_path / "p", color_format="png")
    rec.active = True
    rec.put(1.0, color, np.zeros((4, 4), np.uint16))
    rec.close()
    back = cv2.imread(str(tmp_path / "p" / "color" / "000001.png"), cv2.IMREAD_UNCHANGED)
    assert (back == color).all()
    rec = RawRgbdRecorder(tmp_path / "j", color_format="jpg")
    rec.active = True
    rec.put(1.0, color, np.zeros((4, 4), np.uint16))
    rec.close()
    assert (tmp_path / "j" / "color" / "000001.jpg").exists()
    meta = json.load(open(tmp_path / "j" / "meta.json", encoding="utf-8"))
    assert "lossy" in meta["color_file"]
