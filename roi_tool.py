#!/usr/bin/env python3
"""Define the detection ROI polygon, the counting line and the zones on a still frame.

    python roi_tool.py video.mp4 --at 1200

Controls
    left click    add a point to the shape being edited
    right click   undo the last point of the shape being edited
    l             edit the counting line (again: back to the polygon)
    z             edit the next zone (cycles through `roi.zones`, then back to the polygon)
    c             clear the shape being edited
    s             save to the config file and quit (needs 3+ polygon points)
    q / ESC       quit WITHOUT saving

Every shape is always drawn; the one being edited is highlighted and named in the header.

The two shapes do different jobs:

    polygon = WHERE to look.  Inference runs on its bounding box only, and detections whose
              bottom-centre lands outside it are dropped. Speed, plus fewer false positives.

    zones   = WHAT it can be. Regions such as the pavements, where a detection whose box
              centre is inside is relabelled (`relabel`: a `car` there is a person). Draw
              them where those boxes' centres sit, i.e. somewhat above the pavement itself. Edited
              points keep the zone's name and relabelling; only its outline changes.

    line    = WHEN to count.  A tracked object with its anchor on one side in one frame and
              the other side in the next scores exactly one crossing. Without a line you get
              detections and tracks but no counts, because "how many are in the polygon right
              now" cannot be summed over frames without counting the same rider repeatedly.

Point order on the line only decides direction: crossing to the RIGHT of the arrow counts as
OUT, to the LEFT as IN. Reverse the two points to swap them. Keep the line inside the polygon —
detections are discarded outside it, so a line reaching beyond it cannot be crossed there.

Headless? Use --dump-frame frame.png, read the pixel coordinates off the image in any editor,
and put them into config.yaml yourself — they are normalised, i.e. x/width and y/height.
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

from bikecount.config import Config
from bikecount.source import grab_frame

POLY_COLOUR = (0, 200, 255)
LINE_COLOUR = (0, 255, 0)
ZONE_COLOUR = (255, 160, 60)  # as the debug video draws them


class RoiEditor:
    def __init__(self, frame: np.ndarray, cfg: Config):
        self.frame = frame
        self.h, self.w = frame.shape[:2]
        self.cfg = cfg
        self.polygon: list[tuple[int, int]] = [
            (int(x * self.w), int(y * self.h)) for x, y in cfg.roi.polygon
        ]
        self.line: list[tuple[int, int]] = [
            (int(x * self.w), int(y * self.h)) for x, y in cfg.roi.line
        ]
        self.zones: list[list[tuple[int, int]]] = [
            [(int(x * self.w), int(y * self.h)) for x, y in z.polygon] for z in cfg.roi.zones
        ]
        self.line_mode = False
        self.zone_index: int | None = None  # the zone being edited, if any

    @property
    def target(self) -> list[tuple[int, int]]:
        if self.zone_index is not None:
            return self.zones[self.zone_index]
        return self.line if self.line_mode else self.polygon

    @property
    def target_name(self) -> str:
        if self.zone_index is not None:
            return f"ZONE '{self.cfg.roi.zones[self.zone_index].name}' (3+ clicks)"
        return "LINE (2 clicks)" if self.line_mode else "POLYGON (3+ clicks)"

    def next_zone(self) -> None:
        self.line_mode = False
        if not self.zones:
            return
        i = -1 if self.zone_index is None else self.zone_index
        self.zone_index = i + 1 if i + 1 < len(self.zones) else None

    def toggle_line(self) -> None:
        self.zone_index = None
        self.line_mode = not self.line_mode

    def on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.line_mode and self.zone_index is None:
                if len(self.line) >= 2:
                    self.line = []
                self.line.append((x, y))
            else:
                self.target.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.target:
                self.target.pop()

    def draw(self) -> np.ndarray:
        canvas = self.frame.copy()

        for i, (pts, zone) in enumerate(zip(self.zones, self.cfg.roi.zones)):
            active = i == self.zone_index
            if len(pts) >= 3:
                overlay = canvas.copy()
                cv2.fillPoly(overlay, [np.array(pts, dtype=np.int32)], ZONE_COLOUR)
                cv2.addWeighted(overlay, 0.3 if active else 0.15, canvas,
                                0.7 if active else 0.85, 0, dst=canvas)
            if len(pts) >= 2:
                cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], isClosed=len(pts) >= 3,
                              color=ZONE_COLOUR, thickness=2 if active else 1)
            if pts:
                _label(canvas, pts, f"{zone.name}: {_describe(zone.relabel)}")
            if active:
                for j, pt in enumerate(pts):
                    cv2.circle(canvas, pt, 4, ZONE_COLOUR, -1)
                    cv2.putText(canvas, str(j), (pt[0] + 6, pt[1] - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, ZONE_COLOUR, 1, cv2.LINE_AA)

        if len(self.polygon) >= 3:
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [np.array(self.polygon, dtype=np.int32)], POLY_COLOUR)
            cv2.addWeighted(overlay, 0.2, canvas, 0.8, 0, dst=canvas)
        if len(self.polygon) >= 2:
            cv2.polylines(
                canvas, [np.array(self.polygon, dtype=np.int32)],
                isClosed=len(self.polygon) >= 3, color=POLY_COLOUR, thickness=2,
            )
        for i, pt in enumerate(self.polygon):
            cv2.circle(canvas, pt, 4, POLY_COLOUR, -1)
            cv2.putText(canvas, str(i), (pt[0] + 6, pt[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, POLY_COLOUR, 1, cv2.LINE_AA)

        if len(self.line) == 2:
            cv2.arrowedLine(canvas, self.line[0], self.line[1], LINE_COLOUR, 2, tipLength=0.05)
            # Label which side is which, so the direction of the arrow is not a guess.
            (x1, y1), (x2, y2) = self.line
            mx, my = (x1 + x2) // 2, (y1 + y2) // 2
            dx, dy = x2 - x1, y2 - y1
            norm = max(1.0, (dx * dx + dy * dy) ** 0.5)
            # Right of the arrow is (dy, -dx) negated in image coords, i.e. (-dy, dx).
            ox, oy = int(-dy / norm * 42), int(dx / norm * 42)
            cv2.putText(canvas, "OUT", (mx + ox - 14, my + oy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, LINE_COLOUR, 2, cv2.LINE_AA)
            cv2.putText(canvas, "IN", (mx - ox - 10, my - oy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, LINE_COLOUR, 2, cv2.LINE_AA)
        for pt in self.line:
            cv2.circle(canvas, pt, 5, LINE_COLOUR, -1)

        hint = (f"[{self.target_name}]  L=line  Z=next zone  RMB=undo  C=clear  S=save  "
                f"Q=quit(no save)   points={len(self.target)}")
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (canvas.shape[1], 28), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, dst=canvas)
        cv2.putText(canvas, hint, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    def normalised(self) -> tuple[list[list[float]], list[list[float]]]:
        poly = [[round(x / self.w, 5), round(y / self.h, 5)] for x, y in self.polygon]
        line = [[round(x / self.w, 5), round(y / self.h, 5)] for x, y in self.line]
        return poly, (line if len(line) == 2 else [])

    def normalised_zones(self) -> list[list[list[float]]]:
        return [[[round(x / self.w, 5), round(y / self.h, 5)] for x, y in pts]
                for pts in self.zones]


_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def _describe(relabel: dict[int, int]) -> str:
    """{2: 0, 5: 0, 3: 1} -> 'car/bus -> person, motorcycle -> bicycle'."""
    by_target: dict[int, list[str]] = {}
    for src, dst in relabel.items():
        by_target.setdefault(dst, []).append(_NAMES.get(src, str(src)))
    return ", ".join(f"{'/'.join(srcs)} -> {_NAMES.get(dst, dst)}"
                     for dst, srcs in by_target.items()) or "no relabelling"


def _label(canvas: np.ndarray, pts: list[tuple[int, int]], text: str) -> None:
    """The zone's name on a dark plate, centred on its vertices and kept inside the frame."""
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.5
    (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
    cx = int(np.mean([p[0] for p in pts]))
    cy = int(np.mean([p[1] for p in pts]))
    x = min(max(4, cx - tw // 2), canvas.shape[1] - tw - 4)
    y = min(max(th + 4, cy), canvas.shape[0] - 6)
    cv2.rectangle(canvas, (x - 4, y - th - 5), (x + tw + 4, y + 5), (0, 0, 0), -1)
    cv2.putText(canvas, text, (x, y), font, scale, ZONE_COLOUR, 1, cv2.LINE_AA)


def save_roi(cfg: Config, path: str) -> None:
    """Rewrite only the `roi:` block of the config file, keeping everything else as written.

    `Config.save` would round-trip the whole file through YAML and drop every comment in it —
    and the comments in `config.yaml` are where the tuning decisions are recorded.
    """
    from dataclasses import asdict

    import yaml

    block = yaml.safe_dump({"roi": asdict(cfg.roi)}, sort_keys=False, default_flow_style=None,
                           width=100)
    try:
        lines = open(path).read().splitlines(keepends=True)
    except FileNotFoundError:
        lines = []
    start = next((i for i, ln in enumerate(lines) if ln.startswith("roi:")), None)
    if start is None:
        open(path, "w").write(block + "".join(lines))
        return
    end = next((i for i in range(start + 1, len(lines))
                if lines[i].strip() and not lines[i][0].isspace() and not lines[i].startswith("#")),
               len(lines))
    open(path, "w").write("".join(lines[:start]) + block + "".join(lines[end:]))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("--at", type=float, default=0.0, help="seconds into the video")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--dump-frame", help="just write the frame to this path and exit")
    args = p.parse_args(argv)

    frame = grab_frame(args.video, args.at)

    if args.dump_frame:
        cv2.imwrite(args.dump_frame, frame)
        h, w = frame.shape[:2]
        print(f"wrote {args.dump_frame} ({w}x{h}) — divide pixel coords by {w} and {h}")
        return 0

    cfg = Config.load(args.config)
    editor = RoiEditor(frame, cfg)

    window = "roi_tool"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, editor.w, editor.h)
    cv2.setMouseCallback(window, editor.on_mouse)

    while True:
        cv2.imshow(window, editor.draw())
        key = cv2.waitKey(20) & 0xFF

        if key in (ord("q"), 27):
            print("not saved")
            break
        if key == ord("l"):
            editor.toggle_line()
        elif key == ord("z"):
            editor.next_zone()
        elif key == ord("c"):
            editor.target.clear()
        elif key == ord("s"):
            poly, line = editor.normalised()
            if len(poly) < 3:
                print("need at least 3 polygon points", file=sys.stderr)
                continue
            zones = editor.normalised_zones()
            short = [z.name for z, pts in zip(cfg.roi.zones, zones) if len(pts) < 3]
            if short:
                print(f"zone(s) {', '.join(short)} need at least 3 points", file=sys.stderr)
                continue
            cfg.roi.polygon = poly
            cfg.roi.line = line
            for zone, pts in zip(cfg.roi.zones, zones):
                zone.polygon = pts
            save_roi(cfg, args.config)
            print(f"saved {len(poly)} polygon points, "
                  f"{'a counting line' if line else 'no line'} and {len(zones)} zone(s) "
                  f"to {args.config}")
            break

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
