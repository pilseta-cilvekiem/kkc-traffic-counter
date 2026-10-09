#!/usr/bin/env python3
"""Prepare a YOLO model for ACUITY compilation to a `.nb` for the A733 NPU.

Does the three workstation-side steps, then prints the commands to run inside Allwinner's
ACUITY container (which is where the actual quantisation and compilation happen):

  1. export ONNX at the ROI's aspect ratio, opset 11, static shape;
  2. cut the detection head off, at the six convolutions Allwinner's own recipe cuts at —
     8-bit quantisation of the head's box arithmetic costs far more accuracy than the time it
     saves, so the head is decoded on the CPU instead (see `_decode_yolo11` in detectors.py);
  3. write a calibration set of real ROI crops from the deployment footage.

Calibrating on this camera rather than on COCO is the difference between a usable int8 model
and a bad one: the quantisation scales get fitted to the activation ranges this angle, scale
and lighting actually produce.

    python export_npu.py --weights yolo11n.pt --long 960 --video some.mp4

See NPU.md for the container side.
"""

from __future__ import annotations

import argparse
import glob
import shutil
from pathlib import Path

import cv2
import numpy as np

from bikecount.config import Config
from export_openvino import net_shape, roi_aspect

# The six head convolutions: per scale, `cv2` is the DFL box branch and `cv3` the class
# branch. Ultralytics numbers the head module 23 for every YOLO11 size.
HEAD_OUTPUTS = [
    f"/model.23/cv{b}.{s}/cv{b}.{s}.2/Conv_output_0" for s in range(3) for b in (2, 3)
]


def letterbox(image, net_h: int, net_w: int):
    h, w = image.shape[:2]
    r = min(net_h / h, net_w / w)
    nw, nh = int(w * r), int(h * r)
    out = np.full((net_h, net_w, 3), 114, np.uint8)
    top, left = (net_h - nh) // 2, (net_w - nw) // 2
    out[top : top + nh, left : left + nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    return out


def write_calibration(cfg: Config, videos: list[str], out_dir: Path, count: int, net_h: int, net_w: int) -> int:
    """Spread frames across every clip, so day and night are both represented."""
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)

    written: list[str] = []
    per = max(1, count // max(1, len(videos)))
    for v in videos:
        cap = cv2.VideoCapture(v)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fw, fh = int(cap.get(3)), int(cap.get(4))
        poly, pad = cfg.roi.polygon, cfg.roi.crop_padding
        xs = [p[0] * fw for p in poly]
        ys = [p[1] * fh for p in poly]
        x0, y0 = int(max(0, min(xs) - pad)), int(max(0, min(ys) - pad))
        x1, y1 = int(min(fw, max(xs) + pad)), int(min(fh, max(ys) + pad))

        for i in range(per):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (i + 0.5) / per))
            ok, frame = cap.read()
            if not ok:
                continue
            name = f"cal_{len(written):04d}.jpg"
            cv2.imwrite(str(out_dir / name), letterbox(frame[y0:y1, x0:x1], net_h, net_w))
            written.append(name)
        cap.release()

    (out_dir / "dataset.txt").write_text("".join(f"./{n}\n" for n in written))
    return len(written)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--long", type=int, default=960, help="long side of the network input")
    ap.add_argument("--frame", nargs=2, type=int, default=(1280, 720), metavar=("W", "H"))
    ap.add_argument("--video", nargs="*", help="clips to calibrate on (default: ./*.mp4)")
    ap.add_argument("--calib", type=int, default=64, help="calibration frames")
    ap.add_argument("--out", default="npu_build", help="directory to assemble into")
    ap.add_argument("--zoo", help="path to the unpacked allwinner model zoo, to copy the "
                                  "pegasus_*.sh scripts from (see NPU.md)")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    crop_w, crop_h = roi_aspect(cfg, *args.frame)
    net_h, net_w = net_shape(crop_w, crop_h, args.long)
    stem = Path(args.weights).stem
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[roi]   crop {crop_w}x{crop_h}  ->  network input {net_w}x{net_h}")

    from ultralytics import YOLO  # lazy: pulls in torch

    produced = Path(YOLO(args.weights).export(
        format="onnx", imgsz=[net_h, net_w], dynamic=False, simplify=True,
        opset=11, nms=False, batch=1, device="cpu",
    ))
    # ACUITY's scripts `pushd` into a directory named after the model and expect
    # `<name>/<name>.onnx` inside it, with config_yml.py and the scripts in the parent.
    model_dir = out / f"{stem}_6"
    model_dir.mkdir(exist_ok=True)
    # pegasus_export_ovx_nbg.sh copies the finished .nb to ../model/ and then unconditionally
    # `rm -rf`s its workspace — so if that directory is missing, the compile succeeds and the
    # output is deleted a line later.
    (out / "model").mkdir(exist_ok=True)
    full = out / f"{stem}_full.onnx"
    shutil.move(str(produced), str(full))

    import onnx

    cut = model_dir / f"{stem}_6.onnx"
    onnx.utils.extract_model(str(full), str(cut), ["images"], HEAD_OUTPUTS)

    # ACUITY 6.30 ships onnx 1.12, which refuses anything above IR version 8. Recent onnx
    # stamps 10 or 11 by default; the opset is already 11, so only the envelope needs lowering.
    model = onnx.load(str(cut))
    if model.ir_version > 8:
        print(f"[onnx]  lowering ir_version {model.ir_version} -> 8 for ACUITY's onnx 1.12")
        model.ir_version = 8
        onnx.save(model, str(cut))
    print(f"[onnx]  {cut}  ({len(HEAD_OUTPUTS)} head outputs, decode moved to the CPU)")

    videos = args.video or sorted(glob.glob("*.mp4"))
    if not videos:
        print("[warn]  no videos found — calibrate on real footage or the int8 model will be poor")
    else:
        n = write_calibration(cfg, videos, out / "dataset", args.calib, net_h, net_w)
        print(f"[calib] {n} letterboxed ROI crops in {out}/dataset/")

    if args.zoo:
        scripts = Path(args.zoo) / "scripts_model_convert"
        for name in ("pegasus_import.sh", "pegasus_quantize.sh",
                     "pegasus_inference.sh", "pegasus_export_ovx_nbg.sh"):
            shutil.copy(scripts / name, out / name)
            (out / name).chmod(0o755)
        # The stock config_yml.py points at a COCO sample set; ours points at the crops
        # written above. Everything else matches Allwinner's YOLO recipe.
        cfg_py = (Path(args.zoo) / "examples/yolo11/convert_model/config_yml.py").read_text()
        cfg_py = cfg_py.replace("'../../dataset/coco_12/dataset.txt'", "'../dataset/dataset.txt'")
        # pegasus_import.sh resolves config_yml.py as `$(dirname $0)/config_yml.py` *after*
        # it has already pushd'd into the model directory, so a relative invocation looks for
        # it there. Write both copies rather than depend on how the script was invoked.
        (out / "config_yml.py").write_text(cfg_py)
        (model_dir / "config_yml.py").write_text(cfg_py)
        print(f"[zoo]   pegasus scripts + config_yml.py copied into {out}/")

    print(f"""
Next, inside the ACUITY container (see NPU.md):

  export ACUITY_PATH=$(echo ~/acuity-toolkit-whl-*)/bin
  cd /workspace/{out.name}
  ./pegasus_import.sh         {stem}_6
  ./pegasus_quantize.sh       {stem}_6 uint8 {min(args.calib, 64)}
  ./pegasus_export_ovx_nbg.sh {stem}_6 uint8 a733

Then copy {stem}_6_uint8_a733.nb to the board and run:

  python detect_bikes.py video.mp4 --model {stem}_6_uint8_a733.nb""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
