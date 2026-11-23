import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

from src.capture.raw_recorder import RawRgbdRecorder

REPO = Path(__file__).resolve().parents[1]


def _session(root, name, n=2):
    rec = RawRgbdRecorder(root / name, {"depth_scale": 0.001})
    rec.active = True
    for k in range(n):
        rec.put(1000.0 + 33.3 * k, np.zeros((4, 4, 3), np.uint8), np.ones((4, 4), np.uint16),
                {"color_frame_number": k + 1, "depth_frame_number": k + 1, "depth_timestamp_ms": 1000.0 + 33.3 * k})
    rec.close()


def _run(*args, cwd):
    return subprocess.run([sys.executable, str(REPO / "scripts" / "package_share.py"), *args],
                          cwd=cwd, capture_output=True, text=True, encoding="utf-8")


def test_package_filters_and_takes(tmp_path):
    raw = tmp_path / "data" / "raw_rgbd"
    for nm in ["a", "b", "c", "d"]:
        _session(raw, nm)
    sess = raw / "sessions.csv"
    with sess.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_dir", "participant", "pose", "pose_name", "take", "capture_date", "start_local",
                    "frames", "saved_fps", "missing_frames", "lr_convention", "note"])
        w.writerow(["data/raw_rgbd/a", "C01", "P0", "", 1, "2026-11-25", "", 2, "", 0, "", ""])
        w.writerow(["data/raw_rgbd/b", "C01", "P0", "", 2, "2026-11-25", "", 2, "", 0, "", ""])
        w.writerow(["data/raw_rgbd/c", "C01", "neck_extra", "", 1, "2026-11-25", "", 2, "", 0, "", ""])
        w.writerow(["data/raw_rgbd/d", "C02", "P3", "", 1, "2026-11-25", "", 2, "", 0, "", "중단됨 "])
    r = _run("--map", "share/share_map.csv", "--init-map", "data/raw_rgbd/sessions.csv", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    rows = list(csv.DictReader((tmp_path / "share" / "share_map.csv").open(encoding="utf-8-sig")))
    assert [x["source_dir"] for x in rows] == ["data/raw_rgbd/a", "data/raw_rgbd/b"]
    assert all(x["consent_signed"] == "N" for x in rows)

    # 동의 전: 전부 제외
    r = _run("--map", "share/share_map.csv", "--approval-date", "2026-11-24", "--recipient", "A",
             "--dry-run", cwd=tmp_path)
    assert r.returncode == 1 and "서면동의 없음" in r.stdout

    for x in rows:
        x["consent_signed"] = x["consent_share"] = "Y"
    with (tmp_path / "share" / "share_map.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    r = _run("--map", "share/share_map.csv", "--approval-date", "2026-11-26", "--recipient", "A",
             "--dry-run", cwd=tmp_path)
    assert "이전 촬영" in r.stdout and r.returncode == 1

    r = _run("--map", "share/share_map.csv", "--approval-date", "2026-11-24", "--recipient", "A", "--zip",
             cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    pkg = next(p for p in (tmp_path / "share" / "out").glob("C_to_A_*") if p.is_dir())
    assert pkg.is_dir() and (pkg / "C01" / "P0" / "t1" / "frames.csv").exists()
    assert (pkg / "C01" / "P0" / "t2" / "meta.json").exists()
    summ = list(csv.DictReader((pkg / "sessions_summary.csv").open(encoding="utf-8-sig")))
    assert [s["take"] for s in summ] == ["t1", "t2"] and summ[0]["missing_frames"] == "0"
    assert "share_map" not in " ".join(p.name for p in pkg.rglob("*"))
    assert "P3·P4" in (pkg / "README.txt").read_text(encoding="utf-8")


def test_encrypted_zip_roundtrip(tmp_path):
    import importlib.util
    import pytest
    pyzipper = pytest.importorskip("pyzipper")
    spec = importlib.util.spec_from_file_location("package_share", REPO / "scripts" / "package_share.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pkg = tmp_path / "out" / "C_to_A_x"
    (pkg / "C01" / "P0").mkdir(parents=True)
    (pkg / "C01" / "P0" / "a.txt").write_text("hello", encoding="utf-8")
    z = mod.encrypted_zip(pkg, tmp_path / "out", "C_to_A_x", password="correct horse")
    with pyzipper.AESZipFile(z) as zf:
        with pytest.raises(RuntimeError):
            zf.setpassword(b"wrong")
            zf.read("C_to_A_x/C01/P0/a.txt")
        zf.setpassword(b"correct horse")
        assert zf.read("C_to_A_x/C01/P0/a.txt") == b"hello"
