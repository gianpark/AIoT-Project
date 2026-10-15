"""MoveNet 수정(D: ROI 크롭, A: 깊이 보정) 전후 비교 — data/mod_logs/mod_<참가자>_<자세>_<조건>.csv.

실행: python -m scripts.analyze_mod_logs  → experiments/mod_compare.md
지표(조건별, 자세별):
 - frames / fps: 로그 타임스탬프 기준 실효 프레임률
 - wrist_score: 양 손목 점수 평균 (높을수록 잘 잡힘)
 - wrist_valid: 손목 점수 ≥0.3 프레임 비율
 - wrist_face_miss: wrist_face_sw 결측률
 - hip_miss: hip_depth_m 결측(NaN) 비율
 - jitter_sh: 양 어깨 좌표의 프레임 간 이동량 중앙값(정규화 좌표 ×1000, 작을수록 안정)
 - jitter_wr: 양 손목 좌표 프레임 간 이동량 중앙값(×1000)
"""
import glob, os, re
import numpy as np
import pandas as pd

CONDS = ["base", "roi", "refine", "both"]
OUT = "experiments/mod_compare.md"


def jitter(df, names):
    ds = []
    for n in names:
        x, y = df[f"{n}_x"].to_numpy(), df[f"{n}_y"].to_numpy()
        ds.append(np.hypot(np.diff(x), np.diff(y)))
    return float(np.nanmedian(np.concatenate(ds)) * 1000)


def metrics(path):
    d = pd.read_csv(path)
    t = d["timestamp"].to_numpy()
    dur = t[-1] - t[0] if len(t) > 1 else np.nan
    ws = d[["left_wrist_score", "right_wrist_score"]]
    return {
        "frames": len(d),
        "fps": (len(d) - 1) / dur if dur else np.nan,
        "wrist_score": float(ws.mean().mean()),
        "wrist_valid": float((ws >= 0.3).mean().mean()),
        "wrist_face_miss": float(d["wrist_face_sw"].isna().mean()),
        "hip_miss": float(d["hip_depth_m"].isna().mean()),
        "jitter_sh": jitter(d, ["left_shoulder", "right_shoulder"]),
        "jitter_wr": jitter(d, ["left_wrist", "right_wrist"]),
    }


def main():
    rows = []
    for f in sorted(glob.glob("data/mod_logs/mod_*.csv")):
        m = re.match(r"mod_(p\d+)_(.+)_(base|roi|refine|both)\.csv", os.path.basename(f))
        if not m:
            continue
        rows.append({"p": m[1], "pose": m[2], "cond": m[3], **metrics(f)})
    df = pd.DataFrame(rows)
    cols = ["frames", "fps", "wrist_score", "wrist_valid", "wrist_face_miss", "hip_miss", "jitter_sh", "jitter_wr"]
    out = ["# MoveNet 수정 전후 비교 (예비, 참가자 1명)\n",
           "자동 생성: `python -m scripts.analyze_mod_logs`. 효과 크기는 단일 참가자·단일 촬영이라 방향만 본다.\n",
           "## 조건별 평균 (자세 평균)\n"]
    g = df.groupby("cond")[cols].mean().reindex(CONDS)
    out.append(g.round(3).to_markdown() + "\n")
    out.append("## 자세 × 조건\n")
    for c in cols:
        out.append(f"### {c}\n")
        out.append(df.pivot(index="pose", columns="cond", values=c)[CONDS].round(3).to_markdown() + "\n")
    os.makedirs("experiments", exist_ok=True)
    open(OUT, "w", encoding="utf-8").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
