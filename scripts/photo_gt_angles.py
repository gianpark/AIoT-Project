"""측면 사진에서 MoveNet(Thunder)으로 코·귀·어깨 점을 찾아 '어깨→코 선의 수직선 기준 각도'를 계산한다.

- 각도 정의: 어깨(견봉 근처, 어깨 관절 점)에서 코로 가는 선이 수직선과 이루는 각도. 머리가 어깨보다 앞(보는 방향)에 있으면 +.
- 보는 방향: 코가 귀보다 화면 왼쪽이면 왼쪽을 본다고 판단(오른쪽도 같은 방식).
- 보이는 쪽 어깨(점수가 높은 쪽)를 사용하고, 귀-어깨 각도도 함께 기록한다(참고용).
- 사진은 읽기만 한다. 사진·점 표시 이미지는 저장소에 두지 않는다(--overlay-dir은 저장소 밖에 지정).
사용: python3 -m scripts.photo_gt_angles --photos "<폴더>" --out data/side_gt/gt_angles.csv --overlay-dir <저장소 밖 폴더>
"""
from __future__ import annotations
import argparse, csv, math, os, re
import numpy as np
from PIL import Image, ImageDraw, ImageOps

KP = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder"]
IDX = {n: i for i, n in enumerate(["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
                                   "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
                                   "left_knee", "right_knee", "left_ankle", "right_ankle"])}


def load_interp(path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tflite_runtime.interpreter import Interpreter
    it = Interpreter(model_path=path); it.allocate_tensors(); return it


def infer(it, img: Image.Image):
    w, h = img.size; s = max(w, h)
    canvas = Image.new("RGB", (s, s), (0, 0, 0)); ox, oy = (s - w) // 2, (s - h) // 2
    canvas.paste(img, (ox, oy))
    inp = it.get_input_details()[0]; size = inp["shape"][1]
    arr = np.asarray(canvas.resize((size, size), Image.BILINEAR))[None]
    arr = arr.astype(inp["dtype"]) if inp["dtype"] != np.float32 else arr.astype(np.float32)
    it.set_tensor(inp["index"], arr); it.invoke()
    out = it.get_tensor(it.get_output_details()[0]["index"])[0, 0]  # 17×(y,x,score)
    pts = {}
    for n, i in IDX.items():
        y, x, sc = out[i]; pts[n] = (x * s - ox, y * s - oy, float(sc))
    return pts


def angles(p):
    nose = p["nose"]
    ear = max((p["left_ear"], p["right_ear"]), key=lambda t: t[2])
    sh = max((p["left_shoulder"], p["right_shoulder"]), key=lambda t: t[2])
    face = -1 if nose[0] < ear[0] else 1  # -1: 왼쪽을 봄
    def ang(a, b):  # b(어깨)→a(코/귀) 선의 수직 대비 각도, 보는 방향이 +
        dx = (a[0] - b[0]) * face; dy = b[1] - a[1]
        return math.degrees(math.atan2(dx, dy))
    return ang(nose, sh), ang(ear, sh), face, nose[2], ear[2], sh[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", required=True); ap.add_argument("--out", default="data/side_gt/gt_angles.csv")
    ap.add_argument("--model", default="models/movenet_thunder_int8.tflite"); ap.add_argument("--overlay-dir", default=None)
    a = ap.parse_args()
    it = load_interp(a.model)
    rows = []
    for f in sorted(os.listdir(a.photos)):
        m = re.match(r"gt_env2_(p\d+)_(.+)\.jpe?g$", f, re.I)
        if not m: print("건너뜀(이름 형식 아님):", f); continue
        img = ImageOps.exif_transpose(Image.open(os.path.join(a.photos, f)).convert("RGB"))
        pts = infer(it, img)
        n_ang, e_ang, face, sn, se, ss = angles(pts)
        rows.append({"participant": f"{m[1]}_env2", "pose": m[2], "angle_deg": round(n_ang, 1), "ear_shoulder_deg": round(e_ang, 1),
                     "facing": "left" if face < 0 else "right", "score_nose": round(sn, 2), "score_ear": round(se, 2), "score_shoulder": round(ss, 2)})
        if a.overlay_dir:
            os.makedirs(a.overlay_dir, exist_ok=True)
            d = ImageDraw.Draw(img); r = max(img.size) // 90
            for n in KP:
                x, y, sc = pts[n]; d.ellipse([x - r, y - r, x + r, y + r], outline=(255, 0, 0) if sc > 0.3 else (0, 0, 255), width=6)
            sh = max((pts["left_shoulder"], pts["right_shoulder"]), key=lambda t: t[2])
            d.line([sh[0], sh[1], pts["nose"][0], pts["nose"][1]], fill=(255, 255, 0), width=8)
            d.line([sh[0], sh[1], sh[0], sh[1] - img.size[1] // 3], fill=(0, 255, 0), width=6)
            img.thumbnail((700, 900)); img.save(os.path.join(a.overlay_dir, f"ov_{m[1]}_{m[2]}.jpg"), quality=80)
    order = {"normal": 0, "neck_light": 1, "neck_hard": 2, "forward": 3, "back_hard": 4}
    rows.sort(key=lambda r: (r["participant"], order.get(r["pose"], 9)))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows: print(r)


if __name__ == "__main__":
    main()
