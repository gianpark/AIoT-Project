"""보정 방식 비교: 세션 시작 normal 자세로 특징을 보정할 때 어떤 설계가 교차 환경에 강한가.

변형
 - 보정 구간: normal 첫 N초(3, 5, 10)만 사용 / normal 전체. 보정에 쓴 프레임은 학습·평가에서 제외(낙관 편향 제거).
 - 보정 연산: 빼기(중앙값), 빼기+나누기(MAD)
 - 보정 대상: 전체 / 거리·오프셋 계열 / 각도 계열
평가: 기존(env1) → 새(env2) 두 시나리오(S1 환경만 새것, S2 사람·환경 새것), 9·7클래스, 특징 세트 no_hip / all.
"""
from __future__ import annotations
import os
os.environ.setdefault('RF_TREES', '100')
import warnings, numpy as np, pandas as pd
from scripts.replay_judge_on_logs import load
from scripts.rf_lopo import SETS
from scripts.rf_cross_env import fit_eval, MERGE
warnings.filterwarnings("ignore")
FPS = 21
OFFS = ["head_depth_m", "chest_depth_m", "hip_depth_m", "neck_forward_offset_m", "torso_recline_offset_m", "head_up_m", "torso_len_sw", "wrist_face_sw"]
ANG = ["head_forward_deg", "torso_pitch_deg", "elbow_left_deg", "elbow_right_deg"]
ALL = sorted(set(SETS["all"]))
TARGETS = {"전체": ALL, "거리계열만": [c for c in ALL if c in OFFS], "각도계열만": [c for c in ALL if c in ANG]}

def calibrate(D, secs, op, cols):
    D = D.copy(); keep = pd.Series(True, index=D.index)
    for (p, env, rep), g in D.groupby(["p", "env", "rep"]):  # 회차(rep)를 하나의 세션으로 본다
        n = g[(g.pose == "normal")]
        if len(n) == 0:
            keep.loc[g.index] = False; continue  # 그 회차에 normal이 없으면 보정 불가 -> 제외
        n = n.iloc[: int(secs * FPS)] if secs else n
        if cols:
            med = n[cols].median(); mad = (n[cols] - med).abs().median().replace(0, np.nan)
            s = g[cols] - med
            if op == "div": s = s / (mad.fillna(mad.median()) + 1e-6)
            D.loc[g.index, cols] = s
        keep.loc[n.index] = False  # 보정에 쓴 프레임 제외
    return D[keep]

E1 = load("data/side_logs").assign(env=1); E2 = load("data/side_env2").assign(env=2)
E1 = E1[~E1.pose.str.contains("lean")]; E2 = E2[(~E2.pose.str.contains("lean")) & (E2.pose != "normal2")]
D0 = pd.concat([E1, E2], ignore_index=True)

def evalD(D, cls, sets=("no_hip", "all")):
    d = D if cls == 9 else D.assign(pose=D.pose.replace(MERGE))
    e1, e2 = d[d.env == 1], d[d.env == 2]
    res = {}
    for s in sets:
        v1 = np.mean([fit_eval(e1, e2[e2.p == w], SETS[s])[0] for w in ("p01", "p02")])
        v2 = np.mean([fit_eval(e1[e1.p != w], e2[e2.p == w], SETS[s])[0] for w in ("p01", "p02")])
        lo = np.mean([fit_eval(e1[e1.p != w], e1[e1.p == w], SETS[s])[0] for w in ("p01", "p02", "p03")])
        res[s] = (v1, v2, lo)
    return res

L = ["# 보정 방식 비교 (교차 환경)", "",
     "- S1: env1 전원→env2 두 참가자 평균, S2: 해당 참가자 제외→env2 평균, LOPO: env1 내 평균. 값은 Macro-F1.",
     "- 보정에 쓴 normal 프레임은 학습·평가에서 모두 제외(회차를 하나의 세션으로 보고 그 회차 normal의 첫 N초 사용; normal이 없는 p03 r1은 제외). 무보정도 5초 구간 프레임을 같이 제외해 공정 비교.",
     "- RF 100그루(속도 때문에 다른 문서의 300그루보다 적음), 참가자 2명(env2), 자세당 1회 촬영이므로 방향 확인용.", "",
     "| 클래스 | 보정 구간 | 연산 | 대상 | 세트 | S1 | S2 | LOPO |", "|---|---|---|---|---|---|---|---|"]
import os
DONE = set()
if os.path.exists("experiments/rf_calib_study.md"):
    for ln in open("experiments/rf_calib_study.md", encoding="utf-8"):
        c = [x.strip() for x in ln.strip().strip("|").split("|")]
        if len(c) == 8 and c[0] in ("7", "9"):
            DONE.add(tuple(c[:4])); L.append(ln.rstrip("\n"))
def dump():
    open("experiments/rf_calib_study.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
for cls in (7, 9):
    base = calibrate(D0, 5, "sub", [])
    if (str(cls), "없음", "-", "-") not in DONE:
        for s, (a, b, c) in evalD(base, cls).items():
            L.append(f"| {cls} | 없음 | - | - | {s} | {a:.3f} | {b:.3f} | {c:.3f} |")
        dump()
    for secs in (5, 3, 10, 0):
        for op in ("sub", "div"):
            for tname, cols in TARGETS.items():
                if secs != 5 and (op != "sub" or tname != "전체"):
                    continue
                if (str(cls), f"{secs or '전체'}초", '빼기' if op=='sub' else '빼기+MAD나눔', tname) in DONE:
                    continue
                d = calibrate(D0, secs, op, cols)
                for s, (a, b, c) in evalD(d, cls).items():
                    L.append(f"| {cls} | {secs or '전체'}초 | {'빼기' if op=='sub' else '빼기+MAD나눔'} | {tname} | {s} | {a:.3f} | {b:.3f} | {c:.3f} |")
                dump()
    print(cls, "done", flush=True)
print("ALLDONE", flush=True)
