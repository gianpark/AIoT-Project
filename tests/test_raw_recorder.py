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
