"""IRB 승인 자료 중 공유 대상(P0~P5)만 골라 타 팀 제공용 패키지를 만든다.

사용 예 (저장소 루트, 가상환경 켠 상태):
    # 0) capture_raw로 찍은 세션 목록에서 share_map.csv 초안 만들기 (동의 열은 N으로 채워짐)
    python scripts/package_share.py --map share/share_map.csv --init-map data/raw_rgbd/sessions.csv
    python scripts/package_share.py --map share/share_map.csv --approval-date YYYY-MM-DD(승인 통지서의 승인일) --recipient A --out share/out --dry-run
    python scripts/package_share.py --map share/share_map.csv --approval-date YYYY-MM-DD(승인 통지서의 승인일) --recipient A --out share/out --zip

share_map.csv 열 (한 행 = 촬영 세션 폴더 1개):
    source_dir, internal_id, share_code, pose(P0~P5), capture_date(YYYY-MM-DD), consent_signed(Y/N), consent_share(Y/N)
internal_id와 share_map.csv는 패키지에 절대 복사하지 않는다.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import shutil
import sys
from pathlib import Path

ALLOWED_POSES = {f"P{i}" for i in range(6)}
REQUIRED_COLS = ["source_dir", "internal_id", "share_code", "pose",
                 "capture_date", "consent_signed", "consent_share"]

README_TEMPLATE = """바른자세 C팀 제공 자료 ({package_id})

제공 팀: C팀 (연구책임자 박기안)
받는 팀: {recipient}팀
제공일: {today}
내용: 공통 자세 P0~P5 RGB·Depth 원본, 참가자 {n_people}명, 세션 {n_sessions}개, 파일 {n_files}개
무결성: manifest.csv의 sha256으로 확인 (전체 파일 목록 겸용)
세션 요약: sessions_summary.csv (세션별 프레임 수, 실제 저장 fps, 프레임 번호 누락 수, RGB·Depth 시각 차)

1. 폴더 구조
  <공유코드>/<자세>/<회차>/     예: C01/P3/t1/
      frames.csv               프레임별 대응표 (index, 카메라 프레임 번호, color·depth 타임스탬프,
                               타임스탬프 기준, 노트북 수신 시각, 파일 이름)
      color/000001.png         RGB 원본 (BGR 순서로 저장된 8bit PNG, 무손실. jpg면 meta.json에 표시)
      depth/000001.png         uint16 depth 원본, 값 x depth_scale = 미터
      meta.json                카메라 정보와 세션 정보 (아래 2~4)
  같은 사람·자세를 여러 번 찍었으면 t1, t2 ... 로 나뉜다.

2. 자세 코드
  P0 정상 / P1 거북목 / P2 상체 후방 / P3 좌측 기울임 / P4 우측 기울임 / P5 전방
  P3·P4는 목을 내밀지 않고 상체를 옆으로 기울인 자세. 왼쪽·오른쪽 기준은 meta.json의 lr_convention.
  구간: 폴더 하나가 자세 하나. 카운트다운은 저장하지 않았고 저장 시작부터 끝까지가 지시 구간이다.
  사람이 검토한 안정 구간은 따로 표시하지 않았다.

3. 카메라·Depth 규격 (meta.json)
  카메라: Intel RealSense D455 (device_name, device_serial, firmware_version)
  해상도·설정 fps: width, height, fps_requested / 실제 저장 fps: saved_fps
  Depth: color 좌표계에 정렬(rs.align), 정렬 뒤에는 color_intrinsics를 쓴다.
         값은 카메라 광축 방향 거리(Z), depth_scale(보통 0.001 m/단위).
         0 = 무효(매칭 실패·범위 밖). 필터를 쓰지 않아 65535 같은 채움값은 없다.
         D455 권장 범위 약 0.6~6 m.
  보정: color_intrinsics(왜곡 모델·계수 포함), depth_intrinsics_native, depth_to_color_extrinsics(참고용)
  시간: color_timestamp_ms, depth_timestamp_ms, timestamp_domain. 대표 시각은 color 타임스탬프.
  프레임 누락: color_frame_number_check (카메라 프레임 번호 불연속 수)
  IMU는 저장하지 않았다.

4. 처리 이력
  리사이즈·크롭 없음, depth→color 정렬만 적용, depth 필터 없음, 프레임 제거 없음.
  scripts/capture_raw.py(자세 추정 없이 저장만)로 촬영한 세션은 capture_script에 표시된다.

5. 이용 조건
  1) 각 팀 IRB 승인 연구 목적 안에서만 사용한다. 집계 결과(정확도·처리 시간 등)는 각 팀 승인 범위 안에서 논문·발표에 쓸 수 있다.
  2) 연구원 외 제3자에게 재배포하거나 공개 저장소에 올리지 않는다.
  3) 참가자 신원을 확인하거나 다른 정보와 결합해 식별하려 하지 않는다.
  4) 논문·발표에 얼굴이 보이는 원본 이미지를 싣지 않는다.
  5) 보유 기한 {expire}이 지나면 사본까지 모두 파기하고 C팀에 알린다.
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_forbid(path):
    if not path:
        return []
    words = Path(path).read_text(encoding="utf-8").splitlines()
    return [w.strip().lower() for w in words if w.strip()]


def check_row(row, approval, repo):
    pose = row["pose"].strip().upper()
    if pose not in ALLOWED_POSES:
        return None, f"공유 대상 자세 아님({row['pose']})"
    try:
        cap = dt.date.fromisoformat(row["capture_date"].strip())
    except ValueError:
        return None, f"촬영일 형식 오류({row['capture_date']})"
    if cap < approval:
        return None, f"승인일({approval}) 이전 촬영({cap})"
    if row["consent_signed"].strip().upper() != "Y":
        return None, "서면동의 없음"
    if row["consent_share"].strip().upper() != "Y":
        return None, "타 팀 제공 동의 없음"
    if not row["share_code"].strip():
        return None, "share_code 비어 있음"
    src = Path(row["source_dir"].strip())
    if not src.is_absolute():
        src = repo / src
    if not src.is_dir():
        return None, f"폴더 없음({src})"
    return src, None


def init_map(sessions_csv: Path, map_path: Path, only_sample: bool = False) -> int:
    """capture_raw가 남긴 sessions.csv에서 P0~P5 세션만 골라 share_map.csv 초안을 만든다."""
    if map_path.exists():
        print(f"{map_path}가 이미 있습니다. 덮어쓰지 않습니다 — 다른 이름을 주세요.", file=sys.stderr)
        return 2
    with sessions_csv.open(encoding="utf-8-sig", newline="") as f:
        sess = list(csv.DictReader(f))
    out = []
    for r in sess:
        if r["pose"].upper() not in ALLOWED_POSES or r.get("note", "").startswith("중단됨"):
            continue
        if (r.get("label", "") == "sample") != only_sample:
            continue
        out.append({"source_dir": r["source_dir"], "internal_id": r["participant"],
                    "share_code": r["participant"], "pose": r["pose"].upper(),
                    "capture_date": r["capture_date"], "consent_signed": "N", "consent_share": "N"})
    map_path.parent.mkdir(parents=True, exist_ok=True)
    with map_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REQUIRED_COLS)
        w.writeheader()
        w.writerows(out)
    print(f"{map_path}에 {len(out)}행을 썼습니다. 동의서를 확인한 행만 consent_signed·consent_share를 Y로 바꾸세요.")
    print("같은 사람·자세를 여러 번 찍은 행은 패키지에서 t1, t2 ... 로 나뉩니다. 보내지 않을 회차는 행을 지우세요.")
    return 0


def encrypted_zip(pkg: Path, out_root: Path, package_id: str, password: str | None = None) -> str:
    """패키지 폴더를 AES-256 암호화 zip으로 묶는다. 비밀번호는 화면에 보이지 않게 두 번 입력받는다."""
    try:
        import pyzipper
    except ImportError:
        raise SystemExit("암호화에는 pyzipper가 필요합니다: pip install pyzipper")
    if password is None:
        import getpass
        while True:
            pw1 = getpass.getpass("zip 비밀번호(12자 이상 권장): ")
            pw2 = getpass.getpass("한 번 더: ")
            if pw1 and pw1 == pw2:
                break
            print("비밀번호가 비었거나 서로 다릅니다. 다시 입력하세요.")
        password = pw1
    dst = out_root / f"{package_id}_enc.zip"
    if dst.exists():
        raise SystemExit(f"이미 있는 파일입니다: {dst}")
    with pyzipper.AESZipFile(dst, "w", compression=pyzipper.ZIP_DEFLATED, encryption=pyzipper.WZ_AES) as zf:
        zf.setpassword(password.encode("utf-8"))
        zf.setencryption(pyzipper.WZ_AES, nbits=256)
        for p in sorted(pkg.rglob("*")):
            if p.is_file():
                zf.write(p, (Path(package_id) / p.relative_to(pkg)).as_posix())
    return str(dst)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True)
    ap.add_argument("--init-map", metavar="SESSIONS_CSV", help="capture_raw의 sessions.csv로 share_map.csv 초안을 만들고 끝냄(동의 열은 N)")
    ap.add_argument("--sample-only", action="store_true", help="--init-map에서 샘플(label=sample)만 고름. 기본은 샘플 제외")
    ap.add_argument("--tag", default="", help="패키지 이름 뒤에 붙일 말 (예: sample)")
    ap.add_argument("--approval-date", required=False)
    ap.add_argument("--recipient", required=False)
    ap.add_argument("--out", default="share/out")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--forbid")
    ap.add_argument("--expire", default="2027-12-31")
    ap.add_argument("--zip", action="store_true")
    ap.add_argument("--encrypt", action="store_true", help="AES-256 암호화 zip으로 묶음(비밀번호는 실행 중에 입력, 따로 전달). pip install pyzipper 필요")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.init_map:
        return init_map(Path(args.init_map), Path(args.map), args.sample_only)
    if not args.approval_date or not args.recipient:
        ap.error("--approval-date와 --recipient가 필요합니다")
    approval = dt.date.fromisoformat(args.approval_date)
    repo = Path(args.repo).resolve()
    forbid = load_forbid(args.forbid)
    today = dt.date.today()
    package_id = f"C_to_{args.recipient}_{today:%Y%m%d}" + (f"_{args.tag}" if args.tag else "")
    out_root = Path(args.out).resolve()
    pkg = out_root / package_id

    with open(args.map, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    missing = [c for c in REQUIRED_COLS if rows and c not in rows[0]]
    if not rows or missing:
        print(f"share_map.csv 열 확인 필요: 없음={missing}", file=sys.stderr)
        return 2

    plan = []
    print(f"[패키지] {package_id}  승인일 {approval}")
    for i, row in enumerate(rows, start=2):
        src, reason = check_row(row, approval, repo)
        label = f"행{i} {row['internal_id']}->{row['share_code']} {row['pose']} {row['source_dir']}"
        if reason:
            print(f"  제외  {label}: {reason}")
            continue
        files = sorted(p for p in src.rglob("*") if p.is_file())
        if not files:
            print(f"  제외  {label}: 파일 없음")
            continue
        iid = row["internal_id"].strip().lower()
        bad = [p for p in files if any(w in p.relative_to(src).as_posix().lower() for w in forbid)]
        if bad:
            print(f"  제외  {label}: 금지 문자열 포함 파일 {len(bad)}개 (예: {bad[0].name})")
            continue
        if iid and any(iid in p.relative_to(src).as_posix().lower() for p in files):
            print(f"  주의  {label}: 파일 이름에 내부 코드 '{iid}'가 들어 있음")
        print(f"  포함  {label}: 파일 {len(files)}개")
        plan.append((row, src, files))

    if not plan:
        print("포함할 세션이 없습니다.")
        return 1

    takes: dict = {}
    for row, src, _ in plan:
        key = (row["share_code"].strip(), row["pose"].strip().upper())
        takes[key] = takes.get(key, 0) + 1
        row["_take"] = f"t{takes[key]}"

    if args.dry_run:
        print("[dry-run] 복사하지 않았습니다.")
        return 0

    if pkg.exists():
        print(f"이미 있는 패키지 폴더입니다: {pkg}", file=sys.stderr)
        return 2
    pkg.mkdir(parents=True)

    manifest_rows = []
    for row, src, files in plan:
        code, pose = row["share_code"].strip(), row["pose"].strip().upper()
        for p in files:
            dst = pkg / code / pose / row["_take"] / p.relative_to(src)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
            manifest_rows.append({"share_code": code, "pose": pose, "take": row["_take"],
                                  "path": dst.relative_to(pkg).as_posix(),
                                  "bytes": dst.stat().st_size, "sha256": sha256(dst)})

    with (pkg / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["share_code", "pose", "take", "path", "bytes", "sha256"])
        w.writeheader()
        w.writerows(manifest_rows)

    import json
    summ = []
    for row, src, files in plan:
        code, pose = row["share_code"].strip(), row["pose"].strip().upper()
        m = {}
        mp = src / "meta.json"
        if mp.exists():
            try:
                m = json.loads(mp.read_text(encoding="utf-8"))
            except ValueError:
                m = {}
        chk = m.get("color_frame_number_check") or {}
        sync = m.get("rgb_depth_timestamp_diff_ms") or {}
        summ.append({"share_code": code, "pose": pose, "take": row["_take"], "label": m.get("label", ""), "capture_date": row["capture_date"].strip(),
                     "frames": m.get("frames", ""), "saved_fps": m.get("saved_fps", ""),
                     "missing_frames": chk.get("missing", ""), "duplicate_frames": chk.get("duplicates", ""),
                     "rgb_depth_diff_ms_max": sync.get("max", ""),
                     "color_file": m.get("color_file", ""), "capture_script": m.get("capture_script", "앱(--save-raw)")})
        if m.get("note"):
            print(f"  주의  {code} {pose}: meta.json에 메모가 있음 — 실명 등이 없는지 확인: {m['note']!r}")
    with (pkg / "sessions_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)

    codes = sorted({r["share_code"].strip() for r, _, _ in plan})
    (pkg / "README.txt").write_text(README_TEMPLATE.format(
        package_id=package_id, recipient=args.recipient, today=today,
        n_people=len(codes), n_sessions=len(plan), n_files=len(manifest_rows),
        expire=args.expire), encoding="utf-8")

    archive = ""
    if args.encrypt:
        archive = encrypted_zip(pkg, out_root, package_id)
        print(f"암호화 zip: {archive}  sha256={sha256(Path(archive))}")
        print("비밀번호는 zip과 다른 경로(예: 전화·대면)로 전달하세요. 암호화 전 폴더는 보내지 마세요.")
    elif args.zip:
        archive = shutil.make_archive(str(pkg), "zip", root_dir=out_root, base_dir=package_id)
        print(f"zip: {archive}  sha256={sha256(Path(archive))}")

    log_path = out_root / "share_log.csv"
    new_log = not log_path.exists()
    with log_path.open("a", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        if new_log:
            w.writerow(["package_id", "created", "recipient", "share_codes", "poses",
                        "sessions", "files", "total_bytes", "zip_sha256", "expire"])
        w.writerow([package_id, today.isoformat(), args.recipient, " ".join(codes),
                    " ".join(sorted({r["pose"].strip().upper() for r, _, _ in plan})),
                    len(plan), len(manifest_rows), sum(r["bytes"] for r in manifest_rows),
                    sha256(Path(archive)) if archive else "", args.expire])

    print(f"완료: {pkg}  (세션 {len(plan)}개, 파일 {len(manifest_rows)}개)")
    print(f"기록: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
