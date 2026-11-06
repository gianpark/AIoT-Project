"""경고 단위 평가: 로그를 앱과 같은 규칙(5초 간격 샘플, 자세 경고 연속 3회, 쿨다운)으로 재생해
'녹화 한 번(20초) 안에 자세 경고가 울렸는가'를 자세별로 센다.

- 샘플 시작 시점(위상)을 0~4.5초 사이 0.5초 간격으로 바꿔 10가지로 재생하고 평균한다.
- 보정: 참가자별 normal 첫 녹화의 처음 150프레임(약 5초)의 neck_forward_offset_m 중앙값을 REF_NF(0.143)로 옮기는
  이동량을 모든 프레임에 더한다. 그 150프레임은 normal 평가에서 제외한다. (좌우 치우침은 이번에 보정하지 않음). 환경 2 normal은 150프레임을 빼면 너무 짧아 기준 프레임을 포함해 평가한다.
- 좌표(x, y)가 없는 로그는 합성 keypoint로 재생되어 좌우 기울임 판정이 불가하므로 좌우 자세는 표에서 뺀다.

사용: python -m scripts.eval_alert_level [--out experiments/alert_level_eval.md]
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

from scripts.replay_judge_on_logs import DEPTH_COLS, num, real_keypoints, synth_keypoints
from src.logic.decision import AlertStateMachine, judge

REF_NF = 0.143
CALIB_FRAMES = 150
PHASES = np.arange(0.0, 5.0, 0.5)
LEAN = {"left_lean_neck", "right_lean_neck"}
EXPECT = {"normal": None, "back_hard": "slouch_back", "back_light": "slouch_back", "neck_hard": "neck_forward",
          "neck_light": "neck_forward", "forward": "neck_forward", "slouch": "neck_forward", "chin": "chin_rest",
          "back_neck": None, "left_lean_neck": "tilt_left", "right_lean_neck": "tilt_right"}


def load(dirpath: str, people: list[str]) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(os.path.join(dirpath, "side_*.csv"))):
        m = re.match(r"side_(p\d+)_(.+?)(_r\d)?\.csv", os.path.basename(f))
        if m[1] not in people:
            continue
        d = pd.read_csv(f)
        d["p"], d["pose"], d["rep"] = m[1], m[2], (m[3] or "_r1")[1:]
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def judge_row(row, shift: float):
    d = {c: num(row.get(c)) for c in DEPTH_COLS}
    if d["neck_forward_offset_m"] is not None:
        d["neck_forward_offset_m"] += shift
    real = real_keypoints(row)
    return judge(real, d, frame_aspect=640 / 480) if real else judge(synth_keypoints(row), d, frame_aspect=1.0)


def replay(judgements, t, phase):
    """한 녹화의 판정열을 phase초부터 5초 간격으로 샘플해 상태머신에 넣고 알림 목록을 돌려준다."""
    sm = AlertStateMachine()
    alerts = []
    k = 0
    while phase + 5 * k <= t[-1]:
        i = int(np.searchsorted(t, phase + 5 * k))
        if i >= len(judgements):
            break
        alerts += sm.update(5.0 * k, judgements[i])
        k += 1
    return alerts


def evaluate(D: pd.DataFrame, calibrate: bool):
    out = []
    for p, gp in D.groupby("p"):
        nn = gp[gp.pose == "normal"]
        calib_rep = sorted(nn.rep.unique())[0]  # 그 사람의 가장 이른 normal 녹화(p03은 r1이 없어 r2)
        n1 = nn[nn.rep == calib_rep].sort_values("timestamp")
        shift = 0.0
        calib_idx = set()
        if calibrate:
            first = n1.head(CALIB_FRAMES)
            shift = REF_NF - first["neck_forward_offset_m"].median()
            if not np.isfinite(shift):
                raise ValueError(f"{p}: 보정 기준을 계산할 수 없음")
            calib_idx = set(first.index)
        for (pose, rep), g in gp.groupby(["pose", "rep"]):
            g = g.sort_values("timestamp")
            g_eval = g[~g.index.isin(calib_idx)]
            # 보정 기준 녹화에서 150프레임을 빼면 20초 규칙(샘플 3회)을 돌릴 길이가 안 되는 경우가 있다(환경 2 normal).
            # 그때는 기준 프레임을 포함한 전체 녹화로 평가하고 표에 표시한다(정상 오경고가 낙관적으로 나온다).
            g = g_eval if len(g_eval) >= 300 else g
            if len(g) < 300:
                continue
            t = g["timestamp"].to_numpy(float)
            t = t - t[0]
            js = [judge_row(r, shift) for r in g.to_dict("records")]
            exp = EXPECT.get(pose)
            for ph in PHASES:
                al = replay(js, t, ph)
                post = [a for a in al if a.startswith("posture")]
                out.append(dict(p=p, pose=pose, rep=rep, phase=ph, posture=bool(post),
                                correct=bool(exp) and any(a == f"posture:{exp}" for a in post),
                                proximity=any(a == "proximity" for a in al)))
    return pd.DataFrame(out)


def table(R: pd.DataFrame, title: str, calibrate_note: str) -> list[str]:
    L = [f"### {title}", "", calibrate_note, "",
         "| 자세 | 자세 경고 발생률(%) | 기대 종류로 발생(%) | 근접 알림(%) | 녹화·위상 수 |", "|---|---|---|---|---|"]
    for pose, g in R.groupby("pose"):
        L.append(f"| {pose} | {g.posture.mean()*100:.0f} | "
                 f"{('%.0f' % (g.correct.mean()*100)) if EXPECT.get(pose) else '-'} | {g.proximity.mean()*100:.0f} | {len(g)} |")
    bad = R[(R.pose != "normal")]
    nm = R[R.pose == "normal"]
    L += ["", f"- 정상 자세 오경고(자세 경고 발생률): {nm.posture.mean()*100:.0f}%, 근접 알림 {nm.proximity.mean()*100:.0f}%",
          f"- 나쁜 자세 평균 검출(자세별 평균): {bad.groupby('pose').posture.mean().mean()*100:.0f}%", ""]
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/alert_level_eval.md")
    a = ap.parse_args()
    sets = [("환경 1, p01·p02 (임계값을 정할 때 쓴 데이터)", "data/side_logs", ["p01", "p02"]),
            ("환경 1, p03 (임계값 설정 뒤 수집)", "data/side_logs", ["p03"]),
            ("환경 2, p01·p02 (환경 변화)", "data/side_env2", ["p01", "p02"])]
    L = ["# 경고 단위 평가 (규칙 판정, 5초 간격·연속 3회)", "",
         "- 한 녹화(20초)에서 자세 경고가 한 번이라도 울렸는지를 센다. 샘플 시작 위상 10가지 평균.",
         "- 참가자 수가 적고 녹화도 자세당 1~3회라 예비 결과다. 좌우 기울임은 좌표가 있는 로그에서만 평가했고 이 표에서는 좌우 치우침을 보정하지 않았다.", ""]
    for title, path, people in sets:
        D = load(path, people)
        has_xy = D.groupby("p").apply(lambda g: g["nose_x"].notna().any() if "nose_x" in g else False)
        L += [f"## {title}", ""]
        for cal, note in ((False, "보정 없음"), (True, "보정: normal 첫 5초 기준으로 neck_forward_offset 이동")):
            R = evaluate(D, cal)
            if len(R) == 0:
                continue
            noxy = [p for p, v in has_xy.items() if not v]
            if noxy:
                R = R[~(R.p.isin(noxy) & R.pose.isin(LEAN))]
            L += table(R, note.split(":")[0], note)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
