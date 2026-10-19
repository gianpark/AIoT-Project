"""깊이 없이 2D keypoint(x, y, score)만으로 만든 특징 — '깊이 융합 이점' 평가용 2D 기준선.

입력은 `--log`가 남긴 열(`<name>_x`, `<name>_y`, `<name>_score`). x는 프레임 가로, y는 세로 기준 0~1 정규화
좌표이므로 x에 가로/세로 비(기본 640/480)를 곱해 거리를 비슷한 단위로 맞춘다.
특징: 어깨중점 기준·어깨너비로 정규화한 상대 좌표(코, 눈, 귀, 팔꿈치, 손목), 어깨너비(프레임 높이 단위, 거리의 2D 대용),
어깨 기울기, 양 팔꿈치 각도, 손목-코 최소 거리/어깨너비. 점수 < MIN_SCORE인 점은 NaN(결측 처리는 호출 쪽).
엉덩이는 쓰지 않는다(책상에 가려져 신뢰할 수 없어 no_hip 설정과 같은 조건).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_SCORE = 0.3
REL_POINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
              "left_elbow", "right_elbow", "left_wrist", "right_wrist"]


def _pt(d: pd.DataFrame, name: str, aspect: float):
    x = d[f"{name}_x"].to_numpy(float) * aspect
    y = d[f"{name}_y"].to_numpy(float)
    ok = d[f"{name}_score"].to_numpy(float) >= MIN_SCORE
    return x, y, ok


def _angle(ax, ay, bx, by, cx, cy):
    """b에서 a-b-c가 이루는 각도(도)."""
    v1x, v1y, v2x, v2y = ax - bx, ay - by, cx - bx, cy - by
    n = np.hypot(v1x, v1y) * np.hypot(v2x, v2y)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (v1x * v2x + v1y * v2y) / n
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def features_2d(d: pd.DataFrame, aspect: float = 640 / 480) -> pd.DataFrame:
    lsx, lsy, lso = _pt(d, "left_shoulder", aspect)
    rsx, rsy, rso = _pt(d, "right_shoulder", aspect)
    sh_ok = lso & rso
    mx, my = (lsx + rsx) / 2, (lsy + rsy) / 2
    sw = np.hypot(lsx - rsx, lsy - rsy)
    sw = np.where(sh_ok & (sw > 1e-6), sw, np.nan)
    out = {}
    for n in REL_POINTS:
        x, y, ok = _pt(d, n, aspect)
        out[f"{n}_rx"] = np.where(ok, (x - mx) / sw, np.nan)
        out[f"{n}_ry"] = np.where(ok, (y - my) / sw, np.nan)
    out["shoulder_width_2d"] = sw
    out["shoulder_slope_2d"] = np.where(sh_ok, np.degrees(np.arctan2(rsy - lsy, rsx - lsx)), np.nan)
    for side in ("left", "right"):
        sx, sy, so = _pt(d, f"{side}_shoulder", aspect)
        ex, ey, eo = _pt(d, f"{side}_elbow", aspect)
        wx, wy, wo = _pt(d, f"{side}_wrist", aspect)
        out[f"elbow_{side}_2d"] = np.where(so & eo & wo, _angle(sx, sy, ex, ey, wx, wy), np.nan)
    nx, ny, no = _pt(d, "nose", aspect)
    dists = []
    for side in ("left", "right"):
        wx, wy, wo = _pt(d, f"{side}_wrist", aspect)
        dists.append(np.where(wo & no, np.hypot(wx - nx, wy - ny) / sw, np.nan))
    out["wrist_nose_2d"] = np.fmin(dists[0], dists[1])
    return pd.DataFrame(out, index=d.index)
