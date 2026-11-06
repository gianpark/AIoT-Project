"""손목 점수 기준 완화 평가: wrist_face_distance는 코·어깨·손목 점수가 모두 0.3 이상일 때만 값을 낸다. 손목만 기준을 낮추면
(코·어깨는 0.3 유지) 턱 괴기 신호가 더 자주 잡히지만 정상 자세에서 오경고가 늘 수 있다. 경고 단위(eval_alert_level과 같은 틀)로 비교한다.
사용: python -m scripts.eval_wrist_score [--out experiments/wrist_score_eval.md]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import src.logic.decision as dec
from scripts import eval_alert_level as ev

_orig = dec.wrist_face_distance


def make(ws):
    def f(keypoints, frame_aspect=640 / 480, min_score=0.3):
        nose, l_sh, r_sh = keypoints[0], keypoints[5], keypoints[6]
        if nose.score < min_score or l_sh.score < min_score or r_sh.score < min_score:
            return None
        sw = float(np.hypot((l_sh.x - r_sh.x) * frame_aspect, l_sh.y - r_sh.y))
        if sw < 1e-3:
            return None
        ds = [float(np.hypot((w.x - nose.x) * frame_aspect, w.y - nose.y)) / sw for w in (keypoints[9], keypoints[10]) if w.score >= ws]
        return min(ds) if ds else None
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/wrist_score_eval.md")
    a = ap.parse_args()
    sets = [("환경 1, p01·p02", "data/side_logs", ["p01", "p02"]), ("환경 1, p03", "data/side_logs", ["p03"]),
            ("환경 2", "data/side_env2", ["p01", "p02"])]
    L = ["# 손목 점수 기준 완화 평가 (경고 단위, 보정 있음)", "",
         "손목 점수 기준만 바꾼다(코·어깨는 0.3). 현재 앱은 0.3. 값: 정상 오경고 / 명확한 자세 검출 / 경계 자세 검출 / chin 검출 / chin 종류 일치(%).",
         "환경 2 보정 후 정상은 기준 프레임을 포함해 낙관적이다.", "",
         "| 데이터 | 손목 점수 기준 | 정상 오경고 | 명확한 검출 | 경계 검출 | chin 검출 | chin 종류 일치 |", "|---|---|---|---|---|---|---|"]
    for title, path, people in sets:
        D = ev.load(path, people)
        for ws in (0.3, 0.2, 0.1):
            dec.wrist_face_distance = make(ws)
            ev.wrist_face_distance = dec.wrist_face_distance
            R = ev.evaluate(D, True)
            nm = R[R.pose == "normal"]
            bad = R[R.pose != "normal"]
            avg = lambda x: x.groupby("pose").posture.mean().mean() * 100
            ch = R[R.pose == "chin"]
            L.append(f"| {title} | {ws} | {nm.posture.mean()*100:.0f} | {avg(bad[~bad.pose.isin(ev.BORDER)]):.0f} | "
                     f"{avg(bad[bad.pose.isin(ev.BORDER)]):.0f} | {ch.posture.mean()*100:.0f} | {ch.correct.mean()*100:.0f} |")
    dec.wrist_face_distance = _orig
    open(a.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
