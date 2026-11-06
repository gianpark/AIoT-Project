"""합성 연속 세션 평가: 한 사람의 20초 녹화를 이어 붙여 15분짜리 세션을 만들고 앱 상태머신(5초 샘플, 연속 3회, 쿨다운)을 돌린다.
정상 녹화는 사람당 3개(약 60초)뿐이라 정상 구간의 오경고율을 새로 알려 주지는 못한다. 이 평가가 보여 주는 것은
(1) 쿨다운 때문에 자세 에피소드가 알림 없이 지나가는 비율, (2) 경고만 쓸 때와 주의 알림을 더할 때의 차이, (3) 시간당 알림 횟수다.
- 세션: 20초 구간 45개. 6개 구간이 자세 에피소드(서로 떨어져 있음), 나머지는 정상 녹화를 무작위로 이어 붙임.
- 에피소드는 그 구간(20초) 안에 자세 알림이 울렸을 때 검출로 센다. 정상 구간 알림은 오경고로 센다.
- 보정은 eval_alert_level.py와 같다(그 사람의 가장 이른 normal 녹화 처음 150프레임, 정상 풀에서는 제외).
사용: python -m scripts.eval_session_sim [--sessions 300] [--out experiments/session_sim_eval.md]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from scripts.eval_alert_level import BORDER, CALIB_FRAMES, LEAN, REF_NF, judge_row, load
from src.logic.decision import NORMAL, AlertStateMachine

SEG_S, N_SEG, N_EP = 20.0, 45, 6


def build_pool(D, p, calibrate):
    gp = D[D.p == p]
    nn = gp[gp.pose == "normal"]
    n1 = nn[nn.rep == sorted(nn.rep.unique())[0]].sort_values("timestamp")
    shift, calib_idx = 0.0, set()
    if calibrate:
        first = n1.head(CALIB_FRAMES)
        shift = REF_NF - first["neck_forward_offset_m"].median()
        calib_idx = set(first.index)
    pool = {"normal": [], "ep": {}}
    for (pose, rep), g in gp.groupby(["pose", "rep"]):
        g = g.sort_values("timestamp")
        ge = g[~g.index.isin(calib_idx)]
        g = ge if len(ge) >= 300 else g
        if pose in LEAN and not ("nose_x" in g and g["nose_x"].notna().any()):
            continue
        if len(g) < 300:
            continue
        t = g["timestamp"].to_numpy(float)
        t = t - t[0]
        rec = (t, [judge_row(r, shift) for r in g.to_dict("records")])
        if pose == "normal":
            pool["normal"].append(rec)
        else:
            pool["ep"].setdefault(pose, []).append(rec)
    return pool


def run_session(pool, rng, cooldown, soft):
    ep_pos = sorted(rng.choice(np.arange(1, N_SEG - 1, 2), N_EP, replace=False))  # 홀수 구간, 서로 인접하지 않음
    poses = list(pool["ep"])
    segs = []
    for s in range(N_SEG):
        if s in ep_pos:
            pose = poses[rng.integers(len(poses))]
            segs.append((pose, pool["ep"][pose][rng.integers(len(pool["ep"][pose]))]))
        else:
            segs.append(("normal", pool["normal"][rng.integers(len(pool["normal"]))]))
    phase = rng.uniform(0, 5.0)  # 세션 시작 시점에 따른 샘플 위상(한 녹화의 같은 지점만 반복해 보지 않도록)
    sm = AlertStateMachine(posture_cooldown_s=cooldown)
    soft_streak, last_soft = 0, None
    ep = {pose: [] for pose in poses}
    fa = fa_w = 0
    nsamp = nwarn = ncaut = 0
    for si, (pose, (t, js)) in enumerate(segs):
        warned = softed = False
        for k in range(int(SEG_S // 5)):  # 구간 길이가 샘플 간격의 배수라 위상은 세션 내내 같다
            now = si * SEG_S + 5.0 * k
            j = js[min(int(np.searchsorted(t, phase + 5.0 * k)), len(js) - 1)]
            al = [a for a in sm.update(now, j) if a.startswith("posture")]
            warned |= bool(al)
            if soft:
                soft_streak = soft_streak + 1 if j.posture_level != NORMAL else 0
                if soft_streak >= 3 and (last_soft is None or now - last_soft >= cooldown):
                    softed, last_soft, soft_streak = True, now, 0
            if pose == "normal":
                nsamp += 1
                nwarn += j.posture_level == "경고"
                ncaut += j.posture_level != NORMAL
        if pose == "normal":
            fa_w += warned
            fa += warned or softed
        else:
            ep[pose].append((warned, warned or softed))
    return ep, fa_w, fa, nsamp, nwarn, ncaut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=300)
    ap.add_argument("--out", default="experiments/session_sim_eval.md")
    a = ap.parse_args()
    sets = [("환경 1", "data/side_logs", ["p01", "p02", "p03"]), ("환경 2", "data/side_env2", ["p01", "p02"])]
    L = ["# 합성 연속 세션 평가 (15분, 자세 에피소드 6개)", "",
         f"세션 {a.sessions}개 × 사람. 정상 녹화가 사람당 3개뿐이라 정상 구간 오경고율은 새 정보가 아니다. 환경 2 보정 후 정상은 기준 프레임을 포함해 낙관적이다.",
         "에피소드 검출률은 '그 20초 구간 안에 알림이 울린 비율'이고, 경고 / 경고+주의 알림 순서로 표기한다.", "",
         "| 데이터 | 보정 | 쿨다운 | 명확한 에피소드 검출 | 경계 에피소드 검출 | 정상 구간 알림(시간당) | 정상 샘플 경고 / 주의 이상 |", "|---|---|---|---|---|---|---|"]
    for title, path, people in sets:
        D = load(path, people)
        for cal in (False, True):
            pools = {p: build_pool(D, p, cal) for p in people}
            for cd in (300.0, 60.0):
                acc = {"clear": [0, 0, 0], "border": [0, 0, 0]}
                fa_w = fa = ns = nw = nc = 0
                rng = np.random.default_rng(0)
                for p in people:
                    for _ in range(a.sessions):
                        ep, w_, f_, s_, nw_, nc_ = run_session(pools[p], rng, cd, True)
                        fa_w += w_; fa += f_; ns += s_; nw += nw_; nc += nc_
                        for pose, res in ep.items():
                            g = "border" if pose in BORDER else "clear"
                            for w, e in res:
                                acc[g][0] += 1; acc[g][1] += w; acc[g][2] += e
                hrs = len(people) * a.sessions * N_SEG * SEG_S / 3600 * (N_SEG - N_EP) / N_SEG
                f = lambda g: f"{acc[g][1]/acc[g][0]*100:.0f}% / {acc[g][2]/acc[g][0]*100:.0f}%"
                L.append(f"| {title} | {'있음' if cal else '없음'} | {int(cd)}초 | {f('clear')} | {f('border')} | "
                         f"{fa_w/hrs:.1f} / {fa/hrs:.1f} | {nw/ns*100:.0f}% / {nc/ns*100:.0f}% |")
    open(a.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
