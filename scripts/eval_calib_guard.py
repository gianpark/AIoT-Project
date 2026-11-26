"""앱 개인 보정에 안정성 검사(분산 거부)와 이동 폭 상한을 넣었을 때를 녹화 재생으로 비교한다.

- 수집: eval_app_calibration.app_calibrate와 같은 앱 조건(점수 >= 0.5, near_invalid 아님, 0.15초 간격, 150개).
- 분산 검사: 수집한 150개의 10~90 백분위 폭이 기준을 넘으면 보정을 거부한다(재생에서는 보정 없음으로 둔다. 앱에서는 재보정 안내).
- 이동 폭 상한: |기준값 - 0.143|을 상한으로 자른다.
- 환경 1(p01~p07)은 eval_app_calibration과 같은 평가. 환경 2(p01·p02)는 정상 녹화가 약 14초 하나뿐이라 150개를 못 채우므로
  모인 샘플 전부(약 90개)의 중앙값을 쓰고, normal 오경고는 기준 구간을 포함한 전체 녹화로 센다(낙관적).

사용: python -m scripts.eval_calib_guard [--out experiments/calib_guard_eval.md]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from scripts.eval_1126 import NEW, OLD, load_all, summ
from scripts.eval_alert_level import EXPECT, LEAN, PHASES, REF_NF, judge_row, load, replay
from scripts.eval_app_calibration import app_calibrate

CONFIGS = [  # (이름, 분산 기준 m 또는 None, 이동 상한 m 또는 None)
    ("보정 없음", "off", None),
    ("앱 현재(검사 없음)", None, None),
    ("분산 0.04", 0.04, None),
    ("분산 0.03", 0.03, None),
    ("상한 0.02", None, 0.02),
    ("상한 0.03", None, 0.03),
    ("분산 0.04 + 상한 0.03", 0.04, 0.03),
]


def shift_for(samples, spread_max, cap):
    if spread_max == "off" or len(samples) == 0:
        return 0.0, "off"
    s = np.asarray(samples)
    spread = float(np.percentile(s, 90) - np.percentile(s, 10))
    if spread_max is not None and spread > spread_max:
        return 0.0, f"거부(폭 {spread:.3f})"
    off = float(np.median(s)) - REF_NF
    if cap is not None and abs(off) > cap:
        return -float(np.sign(off) * cap), f"상한 적용({-off:+.3f}→{-np.sign(off) * cap:+.3f})"
    return -off, "적용"


def run(D, cfg, env2=False):
    name, spread_max, cap = cfg
    out, info = [], []
    for p, gp in D.groupby("p"):
        nn = gp[gp.pose == "normal"]
        cal, excl, _, _ = app_calibrate(nn)
        samples = list(cal._samples)
        if env2:
            excl = set()  # 정상 녹화가 하나라 전체로 채점(낙관적)
        shift, note = shift_for(samples, spread_max, cap)
        if spread_max == "off":
            excl = set()
        info.append(dict(p=p, n=len(samples), note=note, shift=shift))
        for (pose, rep), g in gp.groupby(["pose", "rep"]):
            g = g[~g.index.isin(excl)].sort_values("timestamp")
            if len(g) < 300 or (pose in LEAN and not ("nose_x" in g and g.nose_x.notna().any())):
                continue
            t = g.timestamp.to_numpy(float); t -= t[0]
            js = [judge_row(r, shift) for r in g.to_dict("records")]
            exp = EXPECT.get(pose)
            for ph in PHASES:
                post = [a for a in replay(js, t, ph) if a.startswith("posture")]
                out.append(dict(p=p, pose=pose, rep=rep, phase=ph, posture=bool(post),
                                correct=bool(exp) and any(a == f"posture:{exp}" for a in post)))
    return pd.DataFrame(out), pd.DataFrame(info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/calib_guard_eval.md")
    args = ap.parse_args()
    D1 = load_all()
    D2 = load("data/side_env2", ["p01", "p02"])
    L = ["# 개인 보정 안정성 검사·이동 폭 상한 비교 (녹화 재생)", "",
         "- 앱 수집 조건 그대로 150개를 모은 뒤 분산 검사(10~90% 폭)와 이동 상한을 적용. 거부되면 보정 없음으로 채점.",
         "- 환경 1: p01~p07, 보정 구간은 평가에서 뺌. 환경 2: p01·p02, 정상 녹화가 짧아 모인 샘플 전부로 보정하고 normal은 전체로 채점(낙관적).",
         "- 좌우 lean_neck은 요약에서 제외. 5초 간격·경고 연속 3회·위상 10가지 평균.", ""]
    L += ["## 1. 요약", "",
          "| 설정 | 환경1 정상 오경고 | 환경1 명확 검출 | 환경1 경계 검출 | 새 참가자 명확 | 새 참가자 경계 | 환경2 정상 오경고 | 환경2 명확 검출 | 환경2 경계 검출 |",
          "|---|---|---|---|---|---|---|---|---|"]
    details = []
    for cfg in CONFIGS:
        R1, I1 = run(D1, cfg)
        R2, I2 = run(D2, cfg, env2=True)
        a, n, e = summ(R1), summ(R1[R1.p.isin(NEW)]), summ(R2)
        L.append(f"| {cfg[0]} | {a['fa']:.0f} | {a['clear']:.0f} | {a['border']:.0f} | {n['clear']:.0f} | {n['border']:.0f} | "
                 f"{e['fa']:.0f} | {e['clear']:.0f} | {e['border']:.0f} |")
        det = ["", f"### {cfg[0]}", "",
               "보정 처리: " + ", ".join(f"{r.p} {r.note} {r.shift:+.3f}" for r in I1.itertuples())
               + " / 환경2: " + ", ".join(f"{r.p}({r.n}개) {r.note} {r.shift:+.3f}" for r in I2.itertuples()), "",
               "| 사람 | 명확 검출 | 경계 검출 | back_light | slouch | forward | neck_hard |", "|---|---|---|---|---|---|---|"]
        for env, R in (("환경1", R1), ("환경2", R2)):
            for p, g in R.groupby("p"):
                s = summ(g)
                pr = lambda pose: g[g.pose == pose].posture.mean() * 100
                det.append(f"| {env} {p} | {s['clear']:.0f} | {s['border']:.0f} | {pr('back_light'):.0f} | {pr('slouch'):.0f} | "
                           f"{pr('forward'):.0f} | {pr('neck_hard'):.0f} |")
        details += det
        print(L[-1], flush=True)
    L += ["", "단위는 모두 %. 명확 = neck_hard·forward·slouch·back_hard·back_neck·chin, 경계 = neck_light·back_light.", "",
          "## 2. 설정별 상세"] + details
    open(args.out, "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
