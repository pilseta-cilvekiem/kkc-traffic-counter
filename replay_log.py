#!/usr/bin/env python3
"""Re-count a recorded run from its `--debug-log`, using the current config.

`--debug-log` records every detection box the model produced, frame by frame. That is enough
to drive the tracker and the counting gates again — no model, no video, no NPU. So a tuning
change can be scored against a recording of the exact scene it was written for, from a
development machine, in under a second:

    python replay_log.py live.log -o replayed.log
    python validate_log.py replayed.log --video 134146 --offset 6.67

What it does *not* re-run is inference: the boxes are fixed at what the model saw on the day,
so changing `model.conf` or the weights needs a real run. Everything downstream of detection —
tracking, dedup, the speed and coherence gates — is the production code path.

It is close to, but not exactly, the original run. The log holds only detections that were
assigned a track, so the weak unmatched boxes the tracker saw and discarded are gone, and the
association can diverge slightly from there. Expect the same crossings, not the same count to
the last event: replaying `live.log` with every gate disabled reproduces 8 of its 10. Use it
to compare two configs over one recording, which is what it is exact about, rather than as a
substitute for the run itself.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import supervision as sv

from bikecount.config import Config
from bikecount.debug_log import DebugLog
from bikecount.pipeline import Pipeline


class _NoDetector:
    """Stands in for the model. `Pipeline` only reads `.names` off it when replaying."""

    names = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def detections_from(record: dict) -> sv.Detections:
    dets = record.get("det") or []
    if not dets:
        return sv.Detections.empty()
    class_id = np.array([int(d["c"]) for d in dets], dtype=int)
    return sv.Detections(
        xyxy=np.array([d["b"] for d in dets], dtype=np.float32),
        class_id=class_id,
        confidence=np.array([float(d["p"]) for d in dets], dtype=np.float32),
        # Logs written before the raw class was recorded have none; the merged class stands in.
        data={"raw_class_id": np.array([int(d.get("rc", d["c"])) for d in dets], dtype=int)},
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log", help="JSONL written by detect_bikes.py --debug-log")
    p.add_argument("-c", "--config", default="config.yaml")
    p.add_argument("-o", "--out", help="write a replayed log here, for validate_log.py")
    args = p.parse_args(argv)

    cfg = Config.load(args.config)

    meta: dict = {}
    frames: list[dict] = []
    with open(args.log) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("t") == "meta":
                meta = rec
            elif rec.get("t") == "f":
                frames.append(rec)
    if not frames:
        raise SystemExit(f"{args.log} records no frames")

    pipeline = Pipeline(
        cfg,
        _NoDetector(),
        int(meta.get("width", 1280)),
        int(meta.get("height", 720)),
        float(meta.get("source_fps") or meta.get("video_fps") or 15.0),
        adaptive_dt=bool(cfg.runtime.adaptive_dt),
    )

    out = DebugLog(args.out, {**meta, "replayed_from": args.log}) if args.out else None
    events = []
    for rec in frames:
        result = pipeline.count(
            int(rec.get("i", 0)), float(rec["ts"]), detections_from(rec)
        )
        events.extend(result.events)
        if out is not None:
            # Keep the original video frame number, so the replayed log still lines the
            # crossings up with the recorded debug video in debug_viewer.py.
            out.frame(int(rec.get("v", 0)), result, pipeline.tracker)
    if out is not None:
        out.close(interrupted=False, counts=pipeline.counts)

    print(f"{len(frames)} frames replayed, {len(events)} crossings "
          f"(in {pipeline.in_total} / out {pipeline.out_total})")
    for label, c in pipeline.counts.items():
        print(f"  {label:<14} in {c['in']:>4}  out {c['out']:>4}")
    print(f"\n  {'#':>3} {'ts':>9} {'trk':>4} {'class':<14} {'conf':>5} {'px/s':>7}  dir")
    for i, ev in enumerate(events, 1):
        print(f"  {i:>3} {ev.timestamp:>9.2f} {ev.track_id:>4} {ev.class_name[:13]:<14} "
              f"{ev.confidence:>5.2f} {ev.speed_px_s:>7.1f}  {ev.direction}")
    if out is not None:
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
