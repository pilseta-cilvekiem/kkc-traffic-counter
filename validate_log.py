#!/usr/bin/env python3
"""Score a recorded run (`--debug-log` JSONL) against `groundtruth.yaml`.

`calibrate.py` re-runs the pipeline over the labelled clips. That is the right tool when the
detector is runnable on the machine you are sitting at — but the NPU build only runs on the
box, and the interesting failures happen on the live stream, not on a file. This scores what
actually happened: every crossing the log recorded, matched against the windows a human
labelled.

    python validate_log.py live.log --video 134146 --auto-offset

Two things it has to reconcile:

* **Timebase.** Log timestamps start when the pipeline connected; ground-truth seconds are
  positions in the source file. Streaming lag puts a constant offset between them, so pass
  `--offset` or let `--auto-offset` fit the one that lines the most events up.
* **Coverage.** A run that was cut short has not had the chance to miss anything later, so
  windows past the last logged frame are excluded rather than scored as misses.

Anything counted outside a labelled window is a false positive: the ground truth lists every
rider in the stretch a human watched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def read_log(path: str) -> tuple[dict, list[dict], float]:
    meta: dict = {}
    events: list[dict] = []
    last_ts = 0.0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = rec.get("t")
            if kind == "meta":
                meta = rec
            elif kind == "e":
                events.append(rec)
            elif kind == "f":
                last_ts = max(last_ts, float(rec.get("ts", 0.0)))
    return meta, events, last_ts


def load_windows(path: str, video_filter: str | None) -> tuple[str, list[dict]]:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    videos = [(v["file"], v.get("windows", [])) for v in raw.get("videos", [])]
    if video_filter:
        videos = [(f, w) for f, w in videos if video_filter in f]
    if len(videos) != 1:
        names = "\n  ".join(f for f, _ in videos) or "(none)"
        raise SystemExit(
            f"--video must select exactly one ground-truth file; {len(videos)} matched:\n  {names}"
        )
    file, windows = videos[0]
    return file, sorted(windows, key=lambda w: float(w["start"]))


def match(events: list[dict], windows: list[dict], offset: float, pad: float, horizon: float):
    """Greedily assign events to windows. Returns (per-window counts, false positives)."""
    live = [w for w in windows if float(w["end"]) <= horizon]
    got = [0] * len(live)
    false_pos: list[tuple[dict, float]] = []
    for ev in events:
        vt = float(ev["ts"]) + offset
        for i, win in enumerate(live):
            if float(win["start"]) - pad <= vt <= float(win["end"]) + pad:
                got[i] += 1
                break
        else:
            false_pos.append((ev, vt))
    return live, got, false_pos


def score(events, windows, offset, pad, horizon) -> tuple[int, int]:
    """(matched-but-not-over-counted, total error) — the objective --auto-offset minimises."""
    live, got, fp = match(events, windows, offset, pad, horizon)
    hit = sum(min(g, int(w.get("expect", 0))) for w, g in zip(live, got))
    err = sum(abs(g - int(w.get("expect", 0))) for w, g in zip(live, got)) + len(fp)
    return hit, err


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log", help="JSONL written by detect_bikes.py --debug-log")
    p.add_argument("--groundtruth", default="groundtruth.yaml")
    p.add_argument("--video", help="substring picking the ground-truth file this log covers")
    p.add_argument("--offset", type=float, default=0.0,
                   help="seconds to add to log timestamps to reach source-video time")
    p.add_argument("--auto-offset", action="store_true",
                   help="fit the offset that lines the most events up with labelled windows")
    p.add_argument("--search", type=float, default=60.0,
                   help="+/- seconds scanned by --auto-offset (default 60)")
    p.add_argument("--pad", type=float, default=4.0, help="slack around each window")
    p.add_argument("--labels", nargs="+", default=["bicycle", "motorcycle", "rider(person)"],
                   help="event classes the ground truth covers; pedestrians and cars are "
                        "counted too, but nobody labelled them, so they are not scored")
    args = p.parse_args(argv)

    meta, events, last_ts = read_log(args.log)
    events = [ev for ev in events if ev.get("class") in args.labels]
    if not events:
        raise SystemExit(f"{args.log} records no crossings")
    file, windows = load_windows(args.groundtruth, args.video)

    offset = args.offset
    if args.auto_offset:
        best = None
        step = 0.25
        n = int(args.search / step)
        for k in range(-n, n + 1):
            cand = args.offset + k * step
            hit, err = score(events, windows, cand, args.pad, last_ts + cand)
            key = (-hit, err, abs(cand - args.offset))
            if best is None or key < best[0]:
                best = (key, cand)
        offset = best[1]

    horizon = last_ts + offset
    live, got, false_pos = match(events, windows, offset, args.pad, horizon)

    print(f"log        {args.log}  ({len(events)} crossings, {last_ts:.1f}s of stream)")
    print(f"video      {file}")
    print(f"offset     {offset:+.2f}s  (video_time = log_ts {offset:+.2f})")
    print(f"covered    video 0 - {horizon:.1f}s, {len(live)} of {len(windows)} labelled windows\n")

    hdr = f"{'window':<28} {'video time':>13} {'exp':>4} {'got':>4}"
    print(hdr)
    print("-" * len(hdr))
    exp_tot = got_tot = 0
    for win, g in zip(live, got):
        e = int(win.get("expect", 0))
        exp_tot += e
        got_tot += g
        flag = "   <-- MISSED" if g < e else ("   <-- OVERCOUNT" if g > e else "")
        span = f"{float(win['start']):.0f}-{float(win['end']):.0f}"
        print(f"{str(win.get('label',''))[:27]:<28} {span:>13} {e:>4} {g:>4}{flag}")
    print(f"{'(outside any window)':<28} {'':>13} {0:>4} {len(false_pos):>4}"
          f"{'   <-- FALSE POSITIVES' if false_pos else ''}")
    print("-" * len(hdr))
    print(f"{'TOTAL':<28} {'':>13} {exp_tot:>4} {got_tot + len(false_pos):>4}")

    if false_pos:
        print("\nfalse positives (nothing a human labelled is near these):")
        print(f"  {'#':>3} {'log ts':>9} {'video':>8} {'trk':>4} {'class':<14} "
              f"{'conf':>5} {'px/s':>7}  dir")
        for ev, vt in false_pos:
            print(f"  {ev.get('n', 0):>3} {float(ev['ts']):>9.2f} {vt:>8.1f} "
                  f"{ev.get('track_id', -1):>4} {str(ev.get('class', ''))[:13]:<14} "
                  f"{float(ev.get('conf', 0)):>5.2f} {float(ev.get('speed_px_s', 0)):>7.1f}"
                  f"  {ev.get('direction', '')}")
        video_out = meta.get("video") or "<annotated video>"
        print(f"\nWatch them with:  python debug_viewer.py {video_out} --log {args.log}"
              "\n(the viewer pauses on every crossing, in the order listed above)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
