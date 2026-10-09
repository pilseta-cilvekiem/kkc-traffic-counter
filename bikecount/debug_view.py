"""All debug rendering. Kept apart from the pipeline so the deployment path never draws."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
import supervision as sv

from .pipeline import CrossingEvent, FrameResult, Pipeline

_HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class _Shown:
    """A counted crossing, and the first written frame the debug video can show it on."""

    event: CrossingEvent
    from_index: int  # source frame index
    from_ts: float


@dataclass
class _Pending:
    """A frame with the scene drawn, waiting for the counts to be known."""

    result: FrameResult
    canvas: np.ndarray
    infer_line: str


class DebugRenderer:
    """Draws the debug video, with each count on the frame where the crossing happened.

    A crossing is only judged `resolve_delay_frames` after it happens (longer for a person,
    and longer again for a vehicle held on the line), so drawing the counts as each frame is
    processed puts them visibly after the object crossed. Instead the scene — boxes, trails,
    labels — is drawn at once and the frame is held back `delay_frames`, and the counts and
    the crossing flash are drawn when it is written, from the crossing frame of every event.
    Only a crossing judged later than that (a held vehicle) still shows late, and says so.

    Frames go in with `push()` and come out of it (and of `flush()` at the end) in order.
    """

    FLASH_S = 1.5  # how long a crossing stays flashed in the HUD

    def __init__(self, pipeline: Pipeline, width: int, height: int, source_fps: float,
                 delay_frames: int | None = None):
        self.pipeline = pipeline
        self.width = width
        self.height = height
        self.source_fps = source_fps
        track = pipeline.cfg.track
        self.delay_frames = (
            max(track.resolve_delay_frames, track.person_resolve_delay_frames) + 2
            if delay_frames is None else delay_frames
        )

        thickness = sv.calculate_optimal_line_thickness(resolution_wh=(width, height))
        text_scale = sv.calculate_optimal_text_scale(resolution_wh=(width, height))

        self.box_annotator = sv.BoxAnnotator(thickness=thickness)
        self.label_annotator = sv.LabelAnnotator(
            text_scale=text_scale, text_thickness=1, text_padding=3
        )
        # From the box centre, the point the counting line triggers on.
        self.trace_annotator = sv.TraceAnnotator(
            position=sv.Position.CENTER, trace_length=40, thickness=thickness
        )
        self._recent_ms: deque[float] = deque(maxlen=30)

        self._queue: deque[_Pending] = deque()
        self._uncounted: list[_Shown] = []  # known, but after the last written frame
        self._shown: deque[_Shown] = deque()  # counted, for the flash
        self.in_total = 0
        self.out_total = 0
        self.counts: dict[str, dict[str, int]] = {}

    def push(self, result: FrameResult, image: np.ndarray) -> list[np.ndarray]:
        """Add a processed frame; returns the frames now ready to write, oldest first."""
        # The first frame still to be written: an event judged after its crossing frame was
        # written can be shown from here on at the earliest.
        nxt = self._queue[0].result if self._queue else result
        for event in result.events:
            late = event.frame_index < nxt.frame_index
            self._uncounted.append(_Shown(
                event,
                nxt.frame_index if late else event.frame_index,
                nxt.timestamp if late else event.timestamp,
            ))

        self._queue.append(_Pending(result, self._draw_scene(result, image),
                                    self._infer_line(result)))
        ready = []
        while len(self._queue) > self.delay_frames:
            ready.append(self._finish(self._queue.popleft()))
        return ready

    def flush(self) -> list[np.ndarray]:
        """The frames still held back, at the end of the run."""
        ready = [self._finish(p) for p in self._queue]
        self._queue.clear()
        return ready

    def _draw_scene(self, result: FrameResult, image: np.ndarray) -> np.ndarray:
        canvas = image.copy()

        # Region under consideration, so it is obvious what the detector is even looking at.
        self.pipeline.roi.draw(canvas)
        for zone in self.pipeline.zones:
            zone.draw(canvas)

        detections = result.detections
        if len(detections):
            canvas = self.trace_annotator.annotate(canvas, detections=detections)
            canvas = self.box_annotator.annotate(canvas, detections=detections)
            # Speed is shown because it, not confidence, decides whether a track can count.
            # Labelled now rather than when the frame is written, so it is this frame's speed.
            # The track's class, and in brackets what the model called the box this frame
            # when that differs: a relabelled object keeps its track and its count.
            raw = detections.data.get(
                "seen_class_id", detections.data.get("raw_class_id", detections.class_id)
            )
            names = self.pipeline.names
            labels = []
            for tid, cid, rid, conf in zip(detections.tracker_id, detections.class_id, raw,
                                           detections.confidence):
                name = names.get(int(cid), str(cid))
                seen = names.get(int(rid), str(rid))
                labels.append(
                    f"#{tid} {name}{f' ({seen})' if seen != name else ''} {conf:.2f} "
                    f"{self.pipeline.tracker.speed_for(int(tid)):.0f}px/s"
                )
            canvas = self.label_annotator.annotate(canvas, detections=detections, labels=labels)
        return canvas

    def _infer_line(self, result: FrameResult) -> str:
        cfg = self.pipeline.cfg
        self._recent_ms.append(result.infer_ms)
        mean_ms = sum(self._recent_ms) / len(self._recent_ms)
        if not mean_ms:
            return ""
        return (f"stride={cfg.runtime.stride}  eff_fps={self.source_fps / cfg.runtime.stride:.1f}"
                f"  infer={mean_ms:.0f}ms ({1000 / mean_ms:.1f}/s)")

    def _finish(self, pending: _Pending) -> np.ndarray:
        result, canvas = pending.result, pending.canvas
        self._count_until(result.frame_index)

        # Drawn by hand rather than with LineZoneAnnotator: the zone's own in/out counters
        # include candidates that were later rejected by the class and speed gates, so only
        # the pipeline's verdicts are meaningful.
        if self.pipeline.line_zone is not None:
            zone = self.pipeline.line_zone
            start = (int(zone.vector.start.x), int(zone.vector.start.y))
            end = (int(zone.vector.end.x), int(zone.vector.end.y))
            cv2.arrowedLine(canvas, start, end, (255, 255, 255), 2, tipLength=0.04)
            cv2.putText(
                canvas, f"IN {self.in_total} / OUT {self.out_total}",
                (end[0] + 8, end[1] - 8), _HUD_FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
            )

        self._draw_hud(canvas, result, pending.infer_line)
        return canvas

    def _count_until(self, frame_index: int) -> None:
        """Add every known crossing up to this frame to the totals."""
        now = [s for s in self._uncounted if s.from_index <= frame_index]
        if not now:
            return
        self._uncounted = [s for s in self._uncounted if s.from_index > frame_index]
        for shown in now:
            ev = shown.event
            totals = self.counts.setdefault(ev.class_name, {"in": 0, "out": 0})
            totals[ev.direction] += 1
            if ev.direction == "in":
                self.in_total += 1
            else:
                self.out_total += 1
            self._shown.append(shown)

    # ------------------------------------------------------------------ HUD

    def _draw_hud(self, canvas: np.ndarray, result: FrameResult, infer_line: str) -> None:
        by_class: dict[str, int] = {}
        for cid in result.detections.data.get("raw_class_id", result.detections.class_id):
            name = self.pipeline.names.get(int(cid), str(cid))
            by_class[name] = by_class.get(name, 0) + 1
        breakdown = "  ".join(f"{k}={v}" for k, v in sorted(by_class.items())) or "none"

        lines = [
            f"t={_hms(result.timestamp)}  frame={result.frame_index}",
            infer_line,
            f"detections: {len(result.detections)}  [{breakdown}]",
            f"active tracks: {result.active_tracks}",
            f"IN={self.in_total}  OUT={self.out_total}",
            "  ".join(f"{k}={c['in']}/{c['out']}" for k, c in self.counts.items()),
        ]
        lines = [ln for ln in lines if ln]

        # Start below the camera's own burnt-in timestamp, which sits in the top-left corner.
        pad, lh, top = 8, 20, 46
        box_h = lh * len(lines) + pad
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, top), (430, top + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, dst=canvas)

        for i, text in enumerate(lines):
            cv2.putText(
                canvas, text, (pad, top + pad + lh * (i + 1) - 6),
                _HUD_FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
            )

        # Flash each crossing from the frame it happened on, so it is visible when scrubbing.
        while self._shown and result.timestamp - self._shown[0].from_ts > self.FLASH_S:
            self._shown.popleft()
        for i, shown in enumerate(reversed(self._shown)):
            ev = shown.event
            text = f"{ev.direction.upper()}  #{ev.track_id} {ev.class_name}"
            if shown.from_index > ev.frame_index:
                text += f"  (crossed {shown.from_ts - ev.timestamp:.1f}s ago)"
            cv2.putText(
                canvas, text, (pad, top + box_h + 30 + 28 * i),
                _HUD_FONT, 0.8, (0, 255, 255), 2, cv2.LINE_AA,
            )


def _hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}.{int((seconds % 1) * 10)}"
