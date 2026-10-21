"""교차 환경 + 세션별 normal 보정: (참가자, 환경)마다 normal 자세 평균을 빼서(시작 시 보정 가정) 환경 이동을 줄이는지 확인."""
from __future__ import annotations
import numpy as np, pandas as pd, warnings
from scripts.replay_judge_on_logs import load
from scripts.rf_lopo import SETS
from scripts.rf_cross_env import fit_eval, MERGE
warnings.filterwarnings("ignore")

def calib(D, cols):
    D = D.copy()
    for (p, env), g in D.groupby(["p", "env"]):
        base = g[g.pose == "normal"][cols].median()
        D.loc[g.index, cols] = g[cols] - base
    return D

E1 = load("data/side_logs").assign(env=1); E2 = load("data/side_env2").assign(env=2)
E1 = E1[~E1.pose.str.contains("lean")]; E2 = E2[(~E2.pose.str.contains("lean")) & (E2.pose != "normal2")]
D = pd.concat([E1, E2], ignore_index=True)
out = ["# 교차 환경 + normal 보정 (참가자·환경별 normal 중앙값을 뺌)", "",
       "- 보정 대상: 거리·각도 특징(상대 특징 세트 포함). 보정한 normal 프레임은 평가에도 들어가므로 normal 정확도는 낙관적이다.", "",
       "| 클래스 | 보정 | 평가 | no_hip | all |", "|---|---|---|---|---|"]
for name, f in (("9", None), ("7", MERGE)):
    d0 = D if f is None else D.assign(pose=D.pose.replace(f))
    for cal in (False, True):
        d = d0
        if cal:
            cols = sorted(set(SETS["all"])); d = calib(d0, cols)
        e1, e2 = d[d.env == 1], d[d.env == 2]
        for who in ("p01", "p02"):
            te = e2[e2.p == who]
            for tag, tr in (("S1 환경만 새것", e1), ("S2 사람·환경 새것", e1[e1.p != who])):
                r = [fit_eval(tr, te, SETS[s])[0] for s in ("no_hip", "all")]
                out.append(f"| {name} | {'O' if cal else 'X'} | {tag} {who} | {r[0]:.3f} | {r[1]:.3f} |")
        # env1 내 LOPO (보정 후)
        r = [np.mean([fit_eval(e1[e1.p != w], e1[e1.p == w], SETS[s])[0] for w in ("p01", "p02", "p03")]) for s in ("no_hip", "all")]
        out.append(f"| {name} | {'O' if cal else 'X'} | env1 LOPO 평균 | {r[0]:.3f} | {r[1]:.3f} |")
open("experiments/rf_cross_env_calib.md", "w", encoding="utf-8").write("\n".join(out) + "\n")
print("\n".join(out))
