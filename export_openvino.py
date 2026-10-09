#!/usr/bin/env python3
"""Export YOLO weights to an OpenVINO IR shaped for *this* camera's ROI.

The single largest speed win on the ARM board is not the runtime or the quantisation — it is
not asking the model to look at grey padding. The ROI crop on this camera is a 1144x310
strip; fed to a square 640x640 network, about 70% of the multiply-accumulates land on the
letterbox bars. Exporting at 192x640 instead does identical work on identical pixels for a
third of the cost.

That shape is derived here from the polygon in `config.yaml`, so moving the camera or
redrawing the ROI just means re-running this — there is no hand-tuned constant to forget.

    python export_openvino.py --weights yolo11n.pt         # -> yolo11n_rect_openvino_model/
    python export_openvino.py --weights yolo11n.pt --long 512   # smaller/faster

Run it on a workstation, not the board: it needs torch. Copy the resulting directory over.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from bikecount.config import Config

STRIDE = 32  # YOLO's maximum downsample; both input dimensions must be a multiple of it


def roi_aspect(cfg: Config, width: int, height: int) -> tuple[int, int]:
    """Crop box size in pixels, matching `Roi.__init__` exactly."""
    poly = cfg.roi.polygon
    if not poly or len(poly) < 3 or not cfg.roi.crop:
        return width, height
    pad = cfg.roi.crop_padding
    xs = [p[0] * width for p in poly]
    ys = [p[1] * height for p in poly]
    x0, y0 = max(0, min(xs) - pad), max(0, min(ys) - pad)
    x1, y1 = min(width, max(xs) + pad), min(height, max(ys) + pad)
    return int(x1 - x0), int(y1 - y0)


def net_shape(crop_w: int, crop_h: int, long_side: int) -> tuple[int, int]:
    """(h, w) rounded up to the stride, never smaller than one stride."""
    if crop_w >= crop_h:
        w = long_side
        h = max(STRIDE, -(-round(long_side * crop_h / crop_w) // STRIDE) * STRIDE)
    else:
        h = long_side
        w = max(STRIDE, -(-round(long_side * crop_w / crop_h) // STRIDE) * STRIDE)
    return h, w


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="yolo11n.pt", help="source .pt weights")
    ap.add_argument("--config", default="config.yaml", help="config holding the ROI polygon")
    ap.add_argument("--long", type=int, default=640, help="long side of the network input")
    ap.add_argument("--frame", nargs=2, type=int, default=(1280, 720), metavar=("W", "H"),
                    help="source resolution the ROI fractions resolve against")
    ap.add_argument("--out", help="output directory (default: <stem>_rect_openvino_model)")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    crop_w, crop_h = roi_aspect(cfg, *args.frame)
    h, w = net_shape(crop_w, crop_h, args.long)

    square = args.long * args.long
    print(f"[roi]   crop {crop_w}x{crop_h}  (aspect {crop_w / crop_h:.2f}:1)")
    print(f"[shape] network input {w}x{h} — {100 * (1 - h * w / square):.0f}% less work "
          f"than a square {args.long}x{args.long} export")

    from ultralytics import YOLO  # lazy: pulls in torch

    out = Path(args.out or f"{Path(args.weights).stem}_rect_openvino_model")
    produced = Path(YOLO(args.weights).export(format="openvino", imgsz=[h, w]))
    if produced.resolve() != out.resolve():
        shutil.rmtree(out, ignore_errors=True)
        shutil.move(str(produced), str(out))
    print(f"[done]  {out}/  — copy this directory to the board")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
