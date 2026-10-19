"""--compare 로그(같은 프레임에 base/roi/refine/both를 모두 돌린 기록) 분석.

실행: python -m scripts.analyze_compare_logs [폴더=data/mod_compare]  → experiments/compare_results.md
같은 프레임에서 비교하므로 촬영 때마다 달라지는 자세·가려짐 변동이 없다.
지표(변형별, 작을수록/클수록은 표시):
 - wrist_valid(↑): 양 손목 점수 ≥0.3 프레임 비율
 - wrist_face_miss(↓): 코-손목 거리(어깨너비 단위)를 못 구한 비율(손목 점수<0.3 이거나 어깨 점수<0.3)
 - hip_miss(↓): hip_depth_m 결측 비율
 - jitter_sh/jitter_wr(↓): 어깨/손목 좌표 프레임 간 이동량 중앙값(×1000)
 - shift_*: base 대비 해당 점의 평균 이동량(×1000) — 수정이 점을 얼마나 움직였는지
 - level_diff: 판정 등급이 base와 다른 프레임 비율
 - warn_rate/caution_rate/normal_rate: 판정 등급 분포(자세 라벨 기준: normal은 정상이, 나머지는 경고가 기대값)
"""
import glob, os, re, sys
import numpy as np
import pandas as pd

VARS = ["base", "roi", "refine", "both"]
NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
         "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
         "left_knee", "right_knee", "left_ankle", "right_ankle"]
MIN_SCORE = 0.3


def xy(d, v, n):
    return d[f"{v}_{n}_x"].to_numpy(), d[f"{v}_{n}_y"].to_numpy()


def jitter(d, v, names):
    out = []
    for n in names:
        x, y = xy(d, v, n)
        out.append(np.hypot(np.diff(x), np.diff(y)))
    return float(np.nanmedian(np.concatenate(out)) * 1000)


def wrist_face_missing(d, v):
    sw = np.hypot(*(np.subtract(xy(d, v, "left_shoulder"), xy(d, v, "right_shoulder"))))
    ok_sh = (d[f"{v}_left_shoulder_score"] >= MIN_SCORE) & (d[f"{v}_right_shoulder_score"] >= MIN_SCORE) & (sw > 1e-6)
    ok_w = (d[f"{v}_left_wrist_score"] >= MIN_SCORE) | (d[f"{v}_right_wrist_score"] >= MIN_SCORE)
    ok_n = d[f"{v}_nose_score"] >= MIN_SCORE
    return float((~(ok_sh & ok_w & ok_n)).mean())


def shift(d, v, names):
    vals = []
    for n in names:
        bx, by = xy(d, "base", n)
        x, y = xy(d, v, n)
        vals.append(np.hypot(x - bx, y - by))
    return float(np.nanmean(np.concatenate(vals)) * 1000)


def per_file(path):
    d = pd.read_csv(path)
    t = d["timestamp"].to_numpy()
    rows = []
    for v in VARS:
        ws = d[[f"{v}_left_wrist_score", f"{v}_right_wrist_score"]]
        rows.append({
            "cond": v,
            "frames": len(d),
            "fps": (len(d) - 1) / (t[-1] - t[0]) if len(d) > 1 and t[-1] > t[0] else np.nan,
            "wrist_valid": float((ws >= MIN_SCORE).mean().mean()),
            "wrist_face_miss": wrist_face_missing(d, v),
            "hip_miss": float(d[f"{v}_hip_depth_m"].isna().mean()),
            "jitter_sh": jitter(d, v, ["left_shoulder", "right_shoulder"]),
            "jitter_wr": jitter(d, v, ["left_wrist", "right_wrist"]),
            "shift_nose": shift(d, v, ["nose"]),
            "shift_wrist": shift(d, v, ["left_wrist", "right_wrist"]),
            "shift_elbow": shift(d, v, ["left_elbow", "right_elbow"]),
            "level_diff": float((d[f"{v}_level"].fillna("") != d["base_level"].fillna("")).mean()),
            "warn_rate": float((d[f"{v}_level"] == "경고").mean()),
            "caution_rate": float((d[f"{v}_level"] == "주의").mean()),
            "normal_rate": float((d[f"{v}_level"] == "정상").mean()),
        })
    return rows


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "data/mod_compare"
    rows = []
    for f in sorted(glob.glob(os.path.join(folder, "cmp_*.csv"))):
        m = re.match(r"cmp_(p\d+)_(.+?)(?:_r(\d))?\.csv", os.path.basename(f))
        if not m:
            continue
        if sum(1 for _ in open(f, encoding="utf-8")) < 21:  # 헤더 + 20행 미만(중단된 촬영)은 제외
            print("건너뜀(행 부족):", os.path.basename(f))
            continue
        for r in per_file(f):
            rows.append({"p": m[1], "pose": m[2], "round": int(m[3] or 1), **r})
    if not rows:
        print("cmp_*.csv가 없습니다:", folder)
        return
    df = pd.DataFrame(rows)
    cols = ["frames", "fps", "wrist_valid", "wrist_face_miss", "hip_miss", "jitter_sh", "jitter_wr",
            "shift_nose", "shift_wrist", "shift_elbow", "level_diff", "warn_rate", "caution_rate", "normal_rate"]
    out = ["# 같은 프레임 비교 (--compare)\n",
           "자동 생성: `python -m scripts.analyze_compare_logs`. 같은 프레임·같은 depth로 4경로를 돌린 결과라 "
           "촬영 간 변동이 없다. 참가자 수가 적어 방향만 본다.\n",
           "## 변형별 평균 (모든 자세)\n",
           df.groupby("cond")[cols].mean().reindex(VARS).round(3).to_markdown() + "\n"]
    for pose, g in df.groupby("pose"):
        out += [f"## 자세: {pose}\n", g.groupby("cond")[cols].mean().reindex(VARS).round(3).to_markdown() + "\n"]
    os.makedirs("experiments", exist_ok=True)
    open("experiments/compare_results.md", "w", encoding="utf-8").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
