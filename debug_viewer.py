#!/usr/bin/env python3
"""Review an annotated debug video by jumping between the crossings it counted.

`detect_bikes.py --debug-out live.mp4` writes `live.log` beside the video: which video frame
each crossing happened on, and which frame its track was first seen on. This plays the video
against that log, so checking a run is pressing `n` twelve times rather than scrubbing an
hour of footage hoping to catch something.

    python detect_bikes.py video.mp4 --debug-out live.mp4 --events events.jsonl
    python debug_viewer.py live.mp4          # finds live.log automatically

Jumping to an event starts at the **beginning of its track**, not at the line — you watch
the approach the detector watched, which is the only way to tell a real rider from a
mislabelled pedestrian who happened to be moving.

    n / p     next / previous event        space   play / pause
    r         replay this event            , / .   step one frame back / forward
    <- / ->   seek 1 second                [ / ]   seek 10 seconds
    - / =     slower / faster              a       auto-pause after an event on/off
    0         back to the start            h       hide / show help
    click the timeline to seek             q       quit

Headless (over ssh, no display): `--list` prints the events and their timestamps instead.
"""

from __future__ import annotations

import argparse
import sys
from bisect import bisect_right
from pathlib import Path

import cv2

from bikecount.debug_log import DebugLogData, load_debug_log

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_IN_COLOUR = (120, 255, 120)
_OUT_COLOUR = (80, 190, 255)
# Timeline ticks are coloured by what crossed (BGR); which side of the line they sit on says
# the direction, so the colour is free for the class.
_CLASS_COLOURS = {
    "bicycle": (80, 220, 80),
    "rider(person)": (220, 220, 60),
    "pedestrian": (60, 210, 255),
    "car": (90, 90, 255),
}
_OTHER_COLOUR = (200, 200, 200)
_BAR_H = 54     # timeline strip above the frame
_LINE_H = 24    # one line of the status stack
# The status stack sits on the top of the frame, right of the pipeline's burnt-in HUD box
# (`debug_view.py`, 430 px wide): up there is only building front above the detection area,
# while the bottom of the frame is the pavement the riders and pedestrians use.
_PANEL_X = 440

# Arrow keys arrive as different codes depending on the highgui backend, so both sets are
# accepted rather than one being guessed at.
_LEFT = {81, 65361, 2424832}
_RIGHT = {83, 65363, 2555904}
_UP = {82, 65362, 2490368}
_DOWN = {84, 65364, 2621440}


class Viewer:
    def __init__(
        self,
        video: str,
        data: DebugLogData,
        lead: float = 6.0,
        min_lead: float = 2.0,
        tail: float = 1.5,
        auto_pause: bool = True,
    ):
        self.data = data
        self.events = data.events
        # Crossing frames per direction, for the totals as of the frame on screen. The log's
        # per-frame totals change when a crossing is judged, which is a beat after it happens.
        self._crossed = {
            d: sorted(int(e["v"]) for e in self.events if e["direction"] == d)
            for d in ("in", "out")
        }
        self.cap = cv2.VideoCapture(video)
        if not self.cap.isOpened():
            raise SystemExit(f"cannot open {video}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or data.video_fps
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or len(data.frames)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.lead_frames = int(lead * self.fps)
        self.min_lead_frames = int(min_lead * self.fps)
        self.tail_frames = int(tail * self.fps)
        self.auto_pause = auto_pause

        self.pos = 0
        self.frame = None
        self.playing = False
        self.speed = 1.0
        self.event_idx = -1
        self.stop_at: int | None = None
        self.show_help = True
        self.window = "bikecount debug"

    # -- playback ---------------------------------------------------------------------

    def seek(self, target: int) -> None:
        target = max(0, min(int(target), max(0, self.total - 1)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        self.pos = target
        self._read()

    def _read(self) -> bool:
        ok, image = self.cap.read()
        if ok:
            self.frame = image
            self.pos += 1  # POS_FRAMES is now one past the frame we are showing
            return True
        self.playing = False
        return False

    def goto_event(self, index: int) -> None:
        if not self.events:
            return
        self.event_idx = max(0, min(index, len(self.events) - 1))
        event = self.events[self.event_idx]
        start = self.data.event_start(
            event, max_lead_frames=self.lead_frames, min_lead_frames=self.min_lead_frames
        )
        self.stop_at = int(event["v"]) + self.tail_frames if self.auto_pause else None
        self.seek(start)
        self.playing = True

    # -- drawing ----------------------------------------------------------------------

    def render(self):
        shown = max(0, self.pos - 1)  # the frame currently on screen
        record = self.data.frame_at(shown)

        # The timeline gets its own strip above the frame rather than covering the top of it,
        # where the camera's burnt-in clock and the pipeline's HUD sit; the status lines hang
        # below it, over the buildings. Nothing is drawn over the street.
        canvas = cv2.copyMakeBorder(self.frame, _BAR_H, 0, 0, 0, cv2.BORDER_CONSTANT)
        self._draw_timeline(canvas, shown, 0)
        y = self._draw_status(canvas, shown, record, _BAR_H)
        if self.show_help:
            self._draw_help(canvas, y)
        return canvas

    def _draw_status(self, canvas, shown: int, record, top: int) -> int:
        """Draw the status lines downwards from `top`; returns the y below them."""
        parts = [f"frame {shown}/{max(0, self.total - 1)}", _hms(shown / self.fps)]
        if record is not None:
            # The source timestamp, which is what the events file and the original recording
            # are indexed by — the video's own timeline is a different clock.
            parts.append(f"src {_hms(record['ts'])} (#{record['i']})")
            parts.append(f"det {len(record.get('det', []))}")
        parts.append(
            f"IN {bisect_right(self._crossed['in'], shown)} "
            f"OUT {bisect_right(self._crossed['out'], shown)}"
        )
        if not self.playing:
            parts.append("PAUSED")
        elif self.speed != 1.0:
            parts.append(f"x{self.speed:g}")
        text = "   ".join(parts)

        _shade(canvas, _PANEL_X, top, canvas.shape[1] - _PANEL_X, _LINE_H)
        cv2.putText(canvas, text, (_PANEL_X + 8, top + 17), _FONT, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
        top += _LINE_H

        if self.event_idx < 0:
            return top
        event = self.events[self.event_idx]
        colour = _IN_COLOUR if event["direction"] == "in" else _OUT_COLOUR
        crossing = int(event["v"])
        if shown < crossing:
            when = f"crossing in {(crossing - shown) / self.fps:0.1f}s"
        else:
            when = f"crossed {(shown - crossing) / self.fps:0.1f}s ago"
        label = (
            f"event {self.event_idx + 1}/{len(self.events)}  {event['direction'].upper()}  "
            f"#{event['track_id']} {event['class']}  {event['speed_px_s']:.0f}px/s  "
            f"conf {event['conf']:.2f}  @ {_hms(event['ts'])}  [{when}]"
        )
        _shade(canvas, _PANEL_X, top, canvas.shape[1] - _PANEL_X, _LINE_H)
        cv2.putText(canvas, label, (_PANEL_X + 8, top + 17), _FONT, 0.5, colour, 1, cv2.LINE_AA)
        return top + _LINE_H

    def _draw_timeline(self, canvas, shown: int, top: int) -> None:
        w = canvas.shape[1]
        _shade(canvas, 0, top, w, _BAR_H, alpha=0.65)
        y = top + 33
        cv2.line(canvas, (8, y), (w - 8, y), (90, 90, 90), 1)

        span = max(1, self.total - 1)

        def x_of(frame: int) -> int:
            return int(8 + (w - 16) * min(max(frame, 0), span) / span)

        # IN ticks stand above the line, OUT ticks hang below it; the colour is the class.
        current_tick = None
        for i, event in enumerate(self.events):
            x = x_of(int(event["v"]))
            colour = _CLASS_COLOURS.get(event["class"], _OTHER_COLOUR)
            side = -1 if event["direction"] == "in" else 1
            if i == self.event_idx:
                current_tick = (x, side, colour)  # drawn last so neighbours don't cover it
                continue
            cv2.line(canvas, (x, y + side * 2), (x, y + side * 12), colour, 2)
        if current_tick is not None:
            x, side, colour = current_tick
            cv2.line(canvas, (x, y + side * 2), (x, y + side * 16), colour, 3)
            cv2.circle(canvas, (x, y + side * 16), 3, colour, -1)

        x = x_of(shown)
        cv2.line(canvas, (x, y - 19), (x, y + 19), (255, 255, 255), 2)

        text_y = top + 13
        x = _put(canvas, f"{len(self.events)} events", 8, text_y, _OTHER_COLOUR)
        x = _put(canvas, "IN above / OUT below", x + 16, text_y, (150, 150, 150))
        seen = {e["class"] for e in self.events}
        legend = [c for c in _CLASS_COLOURS if c in seen]
        legend += sorted(seen - set(_CLASS_COLOURS))
        for name in legend:
            colour = _CLASS_COLOURS.get(name, _OTHER_COLOUR)
            cv2.rectangle(canvas, (x + 16, text_y - 8), (x + 24, text_y), colour, -1)
            x = _put(canvas, name, x + 28, text_y, colour)
        cv2.putText(canvas, _hms(span / self.fps), (w - 70, text_y), _FONT, 0.45,
                    _OTHER_COLOUR, 1, cv2.LINE_AA)

    def _draw_help(self, canvas, top: int) -> None:
        line = ("n/p event  r replay  space play  ,/. frame  <-/-> 1s  [/] 10s  -/= speed  "
                "a auto-pause  0 start  click timeline  h help  q quit")
        _shade(canvas, _PANEL_X, top, canvas.shape[1] - _PANEL_X, 22, alpha=0.55)
        cv2.putText(canvas, line, (_PANEL_X + 8, top + 15), _FONT, 0.42, (220, 220, 220), 1,
                    cv2.LINE_AA)

    # -- main loop --------------------------------------------------------------------

    def on_mouse(self, event, x, y, flags, _param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or self.frame is None:
            return
        w = self.frame.shape[1]
        if y >= _BAR_H:
            return
        span = max(1, self.total - 1)
        self.stop_at = None
        self.seek(round((x - 8) / max(1, w - 16) * span))

    def run(self) -> int:
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, self.width, self.height + _BAR_H)
        cv2.setMouseCallback(self.window, self.on_mouse)

        # Open on the first event rather than on frame 0: an hour of empty pavement is not
        # what anyone opened this for.
        if self.events:
            self.goto_event(0)
            self.playing = False
        else:
            self.seek(0)

        while True:
            if self.frame is None:
                break
            cv2.imshow(self.window, self.render())

            delay = max(1, int(1000 / (self.fps * self.speed))) if self.playing else 30
            key = cv2.waitKeyEx(delay)
            if key != -1 and not self._handle(key):
                break

            if self.playing:
                if self.stop_at is not None and self.pos - 1 >= self.stop_at:
                    self.playing = False
                    self.stop_at = None
                elif not self._read():
                    self.playing = False
            if cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1:
                break

        self.cap.release()
        cv2.destroyAllWindows()
        return 0

    def _handle(self, key: int) -> bool:
        """False to quit."""
        char = chr(key & 0xFF) if 0 <= (key & 0xFF) < 128 else ""
        step = int(self.fps)

        if char in ("q", "\x1b"):
            return False
        elif char == " ":
            self.playing = not self.playing
            self.stop_at = None
        elif char == "n":
            self.goto_event(self.event_idx + 1)
        elif char == "p":
            self.goto_event(self.event_idx - 1)
        elif char == "r":
            self.goto_event(self.event_idx if self.event_idx >= 0 else 0)
        elif char == ",":
            self.playing = False
            self.seek(self.pos - 2)
        elif char == ".":
            self.playing = False
            self._read() or self.seek(self.pos - 1)
        elif char == "[":
            self.seek(self.pos - 1 - 10 * step)
        elif char == "]":
            self.seek(self.pos - 1 + 10 * step)
        elif char == "-":
            self.speed = max(0.125, self.speed / 2)
        elif char in ("=", "+"):
            self.speed = min(8.0, self.speed * 2)
        elif char == "a":
            self.auto_pause = not self.auto_pause
            self.stop_at = None
        elif char == "0":
            self.event_idx = -1
            self.stop_at = None
            self.seek(0)
        elif char == "h":
            self.show_help = not self.show_help
        elif key in _LEFT:
            self.seek(self.pos - 1 - step)
        elif key in _RIGHT:
            self.seek(self.pos - 1 + step)
        elif key in _UP:
            self.goto_event(self.event_idx + 1)
        elif key in _DOWN:
            self.goto_event(self.event_idx - 1)
        return True


def _shade(canvas, x: int, y: int, w: int, h: int, alpha: float = 0.6) -> None:
    y2, x2 = min(y + h, canvas.shape[0]), min(x + w, canvas.shape[1])
    if y2 <= y or x2 <= x:
        return
    patch = canvas[y:y2, x:x2]
    cv2.addWeighted(patch, 1 - alpha, patch * 0, alpha, 0, dst=patch)


def _put(canvas, text: str, x: int, y: int, colour) -> int:
    """Timeline-strip label at (x, y); returns the x just past it."""
    cv2.putText(canvas, text, (x, y), _FONT, 0.45, colour, 1, cv2.LINE_AA)
    return x + cv2.getTextSize(text, _FONT, 0.45, 1)[0][0]


def _hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}.{int((seconds % 1) * 10)}"


def print_events(data: DebugLogData, video_fps: float) -> None:
    meta = data.meta
    print(f"source : {meta.get('source')}")
    print(f"video  : {meta.get('video')}  ({video_fps:.1f} fps, {len(data.frames)} frames)")
    print(f"model  : {meta.get('model')}  stride={meta.get('stride')}")
    if data.summary:
        print(
            f"totals : IN={data.summary.get('in_total')} OUT={data.summary.get('out_total')}"
            f"{'  (interrupted)' if data.summary.get('interrupted') else ''}"
        )
    if not data.events:
        print("\nno crossings counted")
        return
    print(f"\n{'#':>3} {'dir':<4} {'class':<16} {'px/s':>6} {'conf':>5} "
          f"{'source t':>10} {'video t':>9} {'watch from':>10}")
    for i, event in enumerate(data.events, 1):
        start = data.event_start(
            event, max_lead_frames=int(6 * video_fps), min_lead_frames=int(2 * video_fps)
        )
        print(
            f"{i:>3} {event['direction']:<4} {event['class']:<16} "
            f"{event['speed_px_s']:>6.0f} {event['conf']:>5.2f} "
            f"{_hms(event['ts']):>10} {_hms(event['v'] / video_fps):>9} "
            f"{_hms(start / video_fps):>10}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Jump between the crossings in an annotated debug video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("video", help="annotated video from --debug-out")
    p.add_argument("--log", help="the sidecar log (default: the video's path with .log)")
    p.add_argument(
        "--lead", type=float, default=6.0,
        help="seconds of approach to show before a crossing, at most; playback starts at the "
             "track's first sighting when that is more recent",
    )
    p.add_argument(
        "--min-lead", type=float, default=2.0,
        help="seconds of approach to show even when the track appeared later than that",
    )
    p.add_argument("--tail", type=float, default=1.5, help="seconds to keep playing after a crossing")
    p.add_argument("--no-auto-pause", action="store_true", help="keep playing past an event")
    p.add_argument("--list", action="store_true", help="print the events and exit (no window)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    video = Path(args.video)
    log = Path(args.log) if args.log else video.with_suffix(".log")
    if not log.exists():
        raise SystemExit(
            f"no debug log at {log} — run detect_bikes.py with --debug-out {video} "
            f"(the log is written alongside it)"
        )
    data = load_debug_log(log)

    if args.list:
        print_events(data, data.video_fps)
        return 0
    if not video.exists():
        raise SystemExit(f"no such video: {video}")

    viewer = Viewer(
        str(video), data,
        lead=args.lead, min_lead=args.min_lead, tail=args.tail,
        auto_pause=not args.no_auto_pause,
    )
    try:
        return viewer.run()
    except cv2.error as exc:
        # No display is the usual reason, and it has an obvious fallback.
        print(f"[error] cannot open a window ({exc.err or exc}).", file=sys.stderr)
        print("[info] falling back to --list:\n", file=sys.stderr)
        print_events(data, data.video_fps)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
