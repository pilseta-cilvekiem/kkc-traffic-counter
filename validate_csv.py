#!/usr/bin/env python3
"""Score a run against a human traffic tally (`vid.csv`): every object, every class, both ways.

`validate_log.py` scores riders against short labelled windows. This scores the whole
recording against a sheet where someone logged *everything* that passed — one row per
object, with when it was in view and which way it went:

    Transporta veids,Sākuma laiks,Beigu laiks,Virziens
    Auto,11:59:59,12:00:01,Uz centru

    python validate_csv.py vid_events.jsonl                # --events output
    python validate_csv.py annotated.log --horizon 300     # or a --debug-log

Two clocks, again. The sheet is in the camera's burnt-in clock; events are seconds into the
file. On `vid.mp4` the burnt-in clock starts at 11:59:59 and runs at 0.6x file time (checked
at 0 s, 3000 s and 5990 s), so `clock = --clock-start + ts * --clock-rate`.

An event matches a row of the same class whose [start, end] (padded by `--pad` clock seconds)
contains it; the matching is maximum-cardinality, preferring pairs that agree on direction.
Direction is then scored separately, so a right count going the wrong way shows up as
such rather than as a miss plus a false positive. Totals per class are the other half: for a
traffic count the number per hour matters more than which row each event paired with.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

# Sheet label -> the `class` the pipeline emits.
CLASS_MAP = {
    "Auto": "car",
    "Gājējs": "pedestrian",
    "Velo": "bicycle",
    "Motociklists": "bicycle",   # merged into bicycle by `model.merge_classes`
    "Skūtera vadītājs": "rider(person)",
}
# `out` is the direction cars go on this one-way street.
DIR_MAP = {"Uz centru": "out", "Prom no centra": "in"}
# Scooters are counted as `bicycle` whenever the model boxes the scooter itself, so for the
# rider total the two are one group.
GROUPS = {"car": "car", "pedestrian": "pedestrian", "bicycle": "two-wheeler",
          "rider(person)": "two-wheeler"}


def hms(s: str) -> int:
    h, m, sec = (int(x) for x in s.strip().split(":"))
    return h * 3600 + m * 60 + sec


def fmt(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def read_sheet(path: str, max_span: float) -> tuple[list[dict], list[str]]:
    rows, notes = [], []
    with open(path, newline="") as fh:
        for n, rec in enumerate(csv.reader(fh), start=1):
            if n == 1 or not rec:
                continue
            kind, a, b, d = (x.strip() for x in rec[:4])
            s, e = hms(a), hms(b)
            if e < s or e - s > max_span:
                notes.append(f"line {n}: {kind} {a}-{b} -> treated as {a}-{fmt(s + 60)}")
                e = s + 60
            rows.append({"line": n, "kind": kind, "cls": CLASS_MAP[kind], "dir": DIR_MAP[d],
                         "start": s, "end": e})
    return rows, notes


def read_events(path: str) -> tuple[list[dict], float]:
    events, last = [], 0.0
    with open(path) as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = rec.get("t")
            if kind == "f":
                last = max(last, float(rec.get("ts", 0.0)))
            elif kind in (None, "e") and "direction" in rec:
                events.append(rec)
                last = max(last, float(rec["ts"]))
    return events, last


def match(evs: list[dict], rows: list[dict], pad: float) -> list[tuple[dict, dict]]:
    """Maximum matching of events to the rows whose padded interval contains them; among
    those, the most same-direction pairs, then the smallest distance to the interval."""
    if not evs or not rows:
        return []
    t = np.array([e["clock"] for e in evs])[:, None]
    s = np.array([r["start"] for r in rows])[None, :] - pad
    e = np.array([r["end"] for r in rows])[None, :] + pad
    ok = (t >= s) & (t <= e)
    same = np.array([ev["direction"] for ev in evs])[:, None] == \
        np.array([r["dir"] for r in rows])[None, :]
    dist = np.maximum(0, np.maximum(s + pad - t, t - e + pad))
    # One match outweighs every direction bonus, and a direction bonus every distance.
    w = np.where(ok, 1e6 + 1e3 * same - np.minimum(dist, 999), 0.0)
    ri, ci = linear_sum_assignment(w, maximize=True)
    return [(evs[i], rows[j]) for i, j in zip(ri, ci) if ok[i, j]]


def prf(tp: int, n_ev: int, n_gt: int) -> str:
    p = tp / n_ev if n_ev else 0.0
    r = tp / n_gt if n_gt else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return f"{p:6.1%} {r:6.1%} {f:6.1%}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("events", help="--events JSONL or --debug-log JSONL")
    p.add_argument("--sheet", default="vid.csv")
    p.add_argument("--clock-start", default="11:59:59", help="burnt-in clock at file 0 s")
    p.add_argument("--clock-rate", type=float, default=0.6,
                   help="clock seconds per file second (default 0.6)")
    p.add_argument("--start", type=float, default=0.0, help="file seconds the run started at")
    p.add_argument("--horizon", type=float,
                   help="file seconds the run reached (default: last frame/event)")
    p.add_argument("--tail", type=float, default=120.0,
                   help="on a partial run, clock seconds before its end left unscored")
    p.add_argument("--pad", type=float, default=3.0, help="clock seconds of slack per row")
    p.add_argument("--max-span", type=float, default=600.0,
                   help="rows longer than this (clock s) are typos; use start+60 s instead")
    p.add_argument("--bucket", type=float, default=600.0,
                   help="clock seconds per row of the time table (default 600)")
    p.add_argument("--list", choices=["fp", "fn", "all"], help="print unmatched items")
    args = p.parse_args(argv)

    rows, notes = read_sheet(args.sheet, args.max_span)
    events, last = read_events(args.events)
    c0 = hms(args.clock_start)
    horizon = args.horizon if args.horizon is not None else last
    lo, hi = c0 + args.start * args.clock_rate, c0 + horizon * args.clock_rate
    for ev in events:
        ev["clock"] = c0 + float(ev["ts"]) * args.clock_rate
    # A run cut short: score the rows that started at least `--tail` before it stopped (a
    # queued car can take that long to reach the line), and keep an event from that tail
    # only if it is one of those rows crossing.
    full = hi + args.pad >= max(r["end"] for r in rows)
    cut = hi if full else hi - args.tail
    rows = [r for r in rows if lo - args.pad <= r["start"] and (r["start"] <= cut if not full
                                                                 else r["end"] <= hi + args.pad)]
    events = [e for e in events if lo <= e["clock"] <= hi + args.pad]
    if not full:
        late_ok = set()
        for grp in set(GROUPS.values()):
            pairs = match([e for e in events if GROUPS.get(e["class"]) == grp],
                          [r for r in rows if GROUPS[r["cls"]] == grp], args.pad)
            late_ok |= {id(e) for e, _ in pairs}
        events = [e for e in events if e["clock"] <= cut or id(e) in late_ok]
        hi = cut

    print(f"events   {args.events}  ({len(events)} crossings)")
    print(f"covered  clock {fmt(lo)} - {fmt(hi)}  ({len(rows)} rows of {args.sheet})")
    for n in notes:
        print(f"  note   {n}")
    print()

    hdr = (f"{'class':<14} {'truth':>5} {'got':>5} {'diff':>6} {'match':>5}  "
           f"{'prec':>6} {'recall':>6} {'F1':>6}   {'dir ok':>6}   truth in/out   got in/out")
    print(hdr)
    print("-" * len(hdr))
    unmatched: list[tuple[str, str, dict]] = []

    def line(name, evs, gts, key):
        pairs = match(evs, gts, args.pad)
        dir_ok = sum(1 for ev, r in pairs if ev["direction"] == r["dir"])
        gd = Counter(r["dir"] for r in gts)
        ed = Counter(e["direction"] for e in evs)
        diff = len(evs) - len(gts)
        print(f"{name:<14} {len(gts):>5} {len(evs):>5} {diff:>+6} {len(pairs):>5}  "
              f"{prf(len(pairs), len(evs), len(gts))}   "
              f"{dir_ok / len(pairs) if pairs else 0:6.1%}   "
              f"{gd['in']:>5}/{gd['out']:<5}  {ed['in']:>5}/{ed['out']:<5}")
        if key:
            got_ev = {id(ev) for ev, _ in pairs}
            got_gt = {id(r) for _, r in pairs}
            unmatched.extend(("FP", name, e) for e in evs if id(e) not in got_ev)
            unmatched.extend(("FN", name, r) for r in gts if id(r) not in got_gt)
        return len(pairs)

    by_cls = defaultdict(list)
    for e in events:
        by_cls[e["class"]].append(e)
    gt_cls = defaultdict(list)
    for r in rows:
        gt_cls[r["cls"]].append(r)
    for cls in ["car", "pedestrian", "bicycle", "rider(person)"]:
        line(cls, by_cls[cls], gt_cls[cls], key=False)
    others = sorted(set(by_cls) - set(GROUPS))
    for cls in others:
        line(cls, by_cls[cls], [], key=False)
    print("-" * len(hdr))
    tp = 0
    for grp in ["car", "pedestrian", "two-wheeler"]:
        evs = [e for e in events if GROUPS.get(e["class"]) == grp]
        gts = [r for r in rows if GROUPS[r["cls"]] == grp]
        tp += line(grp, evs, gts, key=True)
    print("-" * len(hdr))
    print(f"{'ALL':<14} {len(rows):>5} {len(events):>5} {len(events) - len(rows):>+6} "
          f"{tp:>5}  {prf(tp, len(events), len(rows))}")

    # Counts per time bucket: the number a traffic survey actually reports.
    print(f"\ncounts per {args.bucket / 60:.0f} min of clock (truth/got)")
    grps = ["car", "pedestrian", "two-wheeler"]
    print(f"{'from':<9}" + "".join(f"{g:>16}" for g in grps))
    b0 = lo - (lo - c0) % args.bucket
    t = b0
    abs_err = Counter()
    tot = Counter()
    while t < hi:
        cells = []
        for g in grps:
            nt = sum(1 for r in rows if GROUPS[r["cls"]] == g and t <= r["start"] < t + args.bucket)
            ng = sum(1 for e in events if GROUPS.get(e["class"]) == g
                     and t <= e["clock"] < t + args.bucket)
            abs_err[g] += abs(ng - nt)
            tot[g] += nt
            cells.append(f"{nt:>7}/{ng:<5} {ng - nt:+3d}")
        print(f"{fmt(t):<9}" + "".join(f"{c:>16}" for c in cells))
        t += args.bucket
    print(f"{'|err|/n':<9}" + "".join(
        f"{abs_err[g] / tot[g] if tot[g] else 0:>16.1%}" for g in grps))

    if args.list:
        print()
        for tag, grp, x in sorted(unmatched, key=lambda u: u[2].get("clock", u[2].get("start"))):
            if args.list != "all" and tag.lower() != args.list:
                continue
            if tag == "FP":
                print(f"FP {grp:<12} clock {fmt(x['clock'])}  file {float(x['ts']):8.1f}s  "
                      f"trk {x.get('track_id')} {x['class']} {x['direction']} "
                      f"conf {float(x.get('conf', 0)):.2f} {float(x.get('speed_px_s', 0)):.0f}px/s")
            else:
                fs = (x["start"] - c0) / args.clock_rate
                fe = (x["end"] - c0) / args.clock_rate
                print(f"FN {grp:<12} clock {fmt(x['start'])}-{fmt(x['end'])}  "
                      f"file {fs:6.1f}-{fe:6.1f}s  {x['kind']} {x['dir']}  (csv line {x['line']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
