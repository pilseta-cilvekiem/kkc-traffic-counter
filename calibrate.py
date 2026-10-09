#!/usr/bin/env python3
"""Score detector configurations against hand-labelled clips in `groundtruth.yaml`.

COCO mAP says nothing about whether a specific night-time street camera sees its cyclists.
This does: for every labelled window it runs the real pipeline and compares counted line
crossings against what a human saw.

    python calibrate.py                                    # every video, current config
    python calibrate.py --models yolo11n.pt yolo11s.pt --imgsz 640 960 1280
    python calibrate.py --stride 3 5 8                     # how far can we decimate?
    python calibrate.py --video-filter 134146              # daytime clip only
    python calibrate.py --speeds                           # per-track px/s, for tuning the gate

Windows are padded by --pad seconds on each side, because a rider labelled "18:18-18:25" may
cross the line a beat before or after the window a human wrote down.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import yaml

from bikecount.config import Config
from bikecount.detectors import build_detector
from bikecount.pipeline import Pipeline
from bikecount.source import VideoSource


def load_groundtruth(path: str) -> list[tuple[str, list[dict]]]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"{path} not found")
    raw = yaml.safe_load(p.read_text()) or {}
    if "videos" in raw:
        return [(v["file"], v.get("windows", [])) for v in raw["videos"]]
    raise SystemExit(f"{path} must contain a `videos:` list")


def run_window(cfg: Config, detector, video: str, win: dict, pad: float, stride: int):
    start = max(0.0, float(win["start"]) - pad)
    duration = float(win["end"]) + pad - start

    source = VideoSource(video, stride=stride, start=start, duration=duration)
    pipeline = Pipeline(cfg, detector, source.width, source.height, source.fps)

    frames = hits = 0
    best = 0.0
    # The ground truth labels riders, so that is what is scored — not pedestrians or cars.
    countable = set(cfg.motion.rider_classes)
    with source:
        for frame in source:
            result = pipeline.process(frame.index, frame.timestamp, frame.image)
            frames += 1
            confs = [
                float(c)
                for c, cid in zip(result.detections.confidence, result.detections.class_id)
                if int(cid) in countable
            ]
            if confs:
                hits += 1
                best = max(best, max(confs))
    return pipeline, frames, hits, best


def report_speeds(cfg: Config, detector, videos, pad: float, stride: int) -> None:
    """Dump per-track motion, so the `motion.*` gates can be set from evidence.

    Three numbers per track, because no one of them is sufficient:
      * median px/s  -> `min_speed_px_s`, but it is perspective-bound,
      * heights/s    -> `min_rider_heights_per_s`, the same speed measured against the track's
                        own box, which is what separates a walker from a rider at any depth,
      * straightness -> `min_straightness`; anything that only looks like it moved reads low.
    """
    print(f"{'window':<32} {'trk':>4} {'cls':>4} {'n':>4} {'median px/s':>12} "
          f"{'box h':>6} {'h/s':>6} {'strt':>5}")
    print("-" * 80)
    for video, windows in videos:
        for win in windows:
            pipeline, *_ = run_window(cfg, detector, video, win, pad, stride)
            label = str(win.get("label", ""))[:31]
            seen = set()
            for track in pipeline.tracker.tracks:
                seen.add(track.track_id)
            # Tracks still alive at the end plus those already retired are both interesting,
            # but only live ones are reachable; run short windows so few are lost.
            for track in sorted(pipeline.tracker.tracks, key=lambda t: -len(t.speeds)):
                if len(track.speeds) < 3:
                    continue
                print(
                    f"{label:<32} {track.track_id:>4} {track.class_id:>4} "
                    f"{len(track.speeds):>4} {track.median_speed:>12.1f} "
                    f"{track.median_height:>6.0f} {track.heights_per_s:>6.2f} "
                    f"{track.straightness:>5.2f}"
                )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--groundtruth", default="groundtruth.yaml")
    p.add_argument("--models", nargs="+", default=[None], help="weights to compare")
    p.add_argument("--imgsz", type=int, nargs="+", default=[None])
    p.add_argument("--stride", type=int, nargs="+", default=[None])
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--pad", type=float, default=4.0, help="seconds of slack around each window")
    p.add_argument("--video-filter", help="only videos whose filename contains this")
    p.add_argument("--speeds", action="store_true", help="dump per-track speeds instead")
    args = p.parse_args(argv)

    videos = load_groundtruth(args.groundtruth)
    if args.video_filter:
        videos = [(f, w) for f, w in videos if args.video_filter in f]
    videos = [(f, w) for f, w in videos if Path(f).exists()]
    if not videos:
        raise SystemExit("no ground-truth videos found on disk")

    base = Config.load(args.config)

    if args.speeds:
        cfg = base.with_overrides(weights=args.models[0], imgsz=args.imgsz[0], conf=args.conf)
        report_speeds(cfg, build_detector(cfg.model), videos, args.pad,
                      args.stride[0] or base.runtime.stride)
        return 0

    hdr = (f"{'model':<12} {'size':>9} {'st':>3} {'window':<26} {'exp':>3} {'got':>3} "
           f"{'hits':>5} {'best':>5} {'ms':>6}")

    for weights, imgsz, stride in itertools.product(args.models, args.imgsz, args.stride):
        cfg = base.with_overrides(
            weights=weights, imgsz=imgsz, conf=args.conf, stride=stride
        )
        detector = build_detector(cfg.model)
        eff_stride = cfg.runtime.stride

        # The OpenVINO and NPU backends carry a static input shape baked into the export, so
        # `model.imgsz` says nothing about what they actually ran at. Report the real one.
        net_h = getattr(detector, "net_h", None)
        net_w = getattr(detector, "net_w", None)
        size = f"{net_w}x{net_h}" if net_h else str(cfg.model.imgsz)

        print(f"\n=== {cfg.model.weights}  imgsz={size}  "
              f"conf={cfg.model.conf}  stride={eff_stride} ===")
        print(hdr)
        print("-" * len(hdr))

        tot_exp = tot_got = 0
        infer_ms: list[float] = []
        for video, windows in videos:
            for win in windows:
                pipeline, frames, hits, best = run_window(
                    cfg, detector, video, win, args.pad, eff_stride
                )
                expect = int(win.get("expect", 0))
                got = pipeline.rider_total
                tot_exp += expect
                tot_got += got
                infer_ms.append(pipeline.timings.mean("infer"))

                flag = ""
                if got < expect:
                    flag = "   <-- MISSED"
                elif got > expect:
                    flag = "   <-- OVERCOUNT"
                print(
                    f"{cfg.model.weights:<12} {size:>9} {eff_stride:>3} "
                    f"{str(win.get('label',''))[:25]:<26} {expect:>3} {got:>3} "
                    f"{hits:>5} {best:>5.2f} {pipeline.timings.mean('infer'):>6.1f}{flag}"
                )

        print(f"{'TOTAL':<12} {'':>9} {'':>3} {'':<26} {tot_exp:>3} {tot_got:>3} "
              f"{'':>5} {'':>5} {np.mean(infer_ms):>6.1f}")

    print(
        "\nexp/got = riders a human saw vs line crossings counted. Both under- and "
        "over-counting matter.\nhits    = processed frames containing a countable detection; "
        "hits>0 with got=0 means\n          it was seen but never held a track across the line."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
