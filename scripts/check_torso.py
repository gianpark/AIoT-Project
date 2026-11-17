"""구도 확인용: 로그 CSV의 torso_len_sw 중앙값을 출력한다.
사용: python scripts/check_torso.py data/_check.csv
p01 기준(side_logs 정상 자세): 약 1.00~1.11, p02/p03 기준: 1.05~1.14
"""
import csv
import statistics as st
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "data/_check.csv"
rows = list(csv.DictReader(open(path, encoding="utf-8")))
vals = []
for r in rows:
    try:
        v = float(r["torso_len_sw"])
        if v == v:
            vals.append(v)
    except (KeyError, ValueError):
        pass
if not vals:
    print("torso_len_sw 값이 없습니다")
else:
    print(f"{path}: 프레임 {len(rows)}, 유효 {len(vals)}, 중앙값 {st.median(vals):.3f}, 최소 {min(vals):.3f}, 최대 {max(vals):.3f}")
