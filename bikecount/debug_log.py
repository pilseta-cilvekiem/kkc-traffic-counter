"""A JSONL sidecar for the annotated debug video.

The annotated video shows what the detector saw; this records *where in that video* each
thing happened, which is what makes the recording navigable instead of something you scrub
through hoping to catch a crossing.

Two things are worth knowing about the mapping it maintains:

* **Video frame, not source frame.** The debug video holds one frame per *processed* frame,
  so at `--stride 3` its frame 100 is source frame 300 — and on a live stream, where frames
  are dropped whenever inference falls behind, there is no formula at all. The log is the
  only place that correspondence exists.
* **A crossing is logged where it happened, not where it was reported.** An event surfaces
  `resolve_delay_frames` frames after the fact, because the speed gate needs the track to
  mature first. Seeking to the frame the event was *emitted* on shows the rider already past
  the line, so each event also carries the video frame of the crossing itself and the frame
  its track was first seen on — that last one is what you actually want to watch from.

Written by `detect_bikes.py --debug-out`, read by `debug_viewer.py`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, TextIO

from .pipeline import FrameResult

# Bumped when the record shape changes in a way a reader must notice.
FORMAT_VERSION = 1


class DebugLog:
    """Writes one JSON object per processed frame, plus one per crossing."""

    def __init__(self, path: str | Path, meta: dict[str, Any] | None = None):
        self.path = Path(path)
        self.handle: TextIO = open(self.path, "w")
        self.frames = 0
        self.events = 0
        self._started = time.time()

        # source frame index -> video frame, so a crossing reported later can be placed back
        # on the frame it actually happened on.
        self._video_frame_of: dict[int, int] = {}
        # track id -> the video frame it was first seen on.
        self._first_seen: dict[int, int] = {}

        self._write(
            {
                "t": "meta",
                "format": FORMAT_VERSION,
                "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                **(meta or {}),
            }
        )

    # -- writing ----------------------------------------------------------------------

    def frame(self, video_frame: int, result: FrameResult, tracker: Any = None) -> None:
        """Record one processed frame, and any crossings it produced."""
        self._video_frame_of[result.frame_index] = video_frame

        detections = []
        raw = result.detections.data.get("raw_class_id")
        det_class = result.detections.data.get("det_class_id")
        # What the model said before a zone relabelled it, so that replaying the log applies
        # the zones as they are configured then, not as they were on the day.
        seen_raw = result.detections.data.get("seen_class_id")
        seen_merged = result.detections.data.get("seen_merged_id")
        if seen_raw is not None:
            raw, det_class = seen_raw, seen_merged
        for i in range(len(result.detections)):
            track_id = int(result.detections.tracker_id[i])
            if track_id not in self._first_seen:
                self._first_seen[track_id] = video_frame
            box = result.detections.xyxy[i]
            # `c` is what the model said this frame (after merging), so a replay feeds the
            # tracker the same detections; `tc` is the track's voted class where it differs.
            c = int(det_class[i]) if det_class is not None else int(result.detections.class_id[i])
            record = {
                "id": track_id,
                "c": c,
                "p": round(float(result.detections.confidence[i]), 3),
                "b": [int(v) for v in box],
            }
            # The class before `merge_classes`, only where merging changed it: replaying a
            # log needs it to name a track `motorcycle` or `truck` rather than its merged class.
            if raw is not None and int(raw[i]) != record["c"]:
                record["rc"] = int(raw[i])
            if int(result.detections.class_id[i]) != c:
                record["tc"] = int(result.detections.class_id[i])

            if tracker is not None:
                record["s"] = round(tracker.speed_for(track_id), 1)
                record["h"] = tracker.hits_for(track_id)
            detections.append(record)

        self._write(
            {
                "t": "f",
                "v": video_frame,
                "i": result.frame_index,
                "ts": round(result.timestamp, 3),
                "in": result.in_total,
                "out": result.out_total,
                "ms": round(result.infer_ms, 1),
                "det": detections,
            }
        )
        self.frames += 1

        for event in result.events:
            self.events += 1
            crossing_v = self._video_frame_of.get(event.frame_index, video_frame)
            self._write(
                {
                    "t": "e",
                    "n": self.events,
                    # where the crossing happened, where its track first appeared, and where
                    # it was finally judged — the viewer plays from the second of those.
                    "v": crossing_v,
                    "v_start": self._first_seen.get(event.track_id, crossing_v),
                    "v_emitted": video_frame,
                    **event.as_dict(),
                }
            )

    def close(self, **summary: Any) -> None:
        if self.handle.closed:
            return
        self._write(
            {
                "t": "summary",
                "frames": self.frames,
                "events": self.events,
                "elapsed_s": round(time.time() - self._started, 1),
                **summary,
            }
        )
        self.handle.close()

    def _write(self, record: dict[str, Any]) -> None:
        self.handle.write(json.dumps(record) + "\n")
        # Flushed per record: a run that is killed — which is most of them, on a live stream —
        # must still leave a log that opens.
        self.handle.flush()

    def __enter__(self) -> "DebugLog":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class DebugLogData:
    """A parsed log: metadata, per-frame records, and the events."""

    def __init__(
        self,
        meta: dict[str, Any],
        frames: list[dict[str, Any]],
        events: list[dict[str, Any]],
        summary: dict[str, Any] | None,
    ):
        self.meta = meta
        self.frames = frames
        self.events = events
        self.summary = summary
        self.by_video_frame = {f["v"]: f for f in frames}

    @property
    def video_fps(self) -> float:
        return float(self.meta.get("video_fps") or 15.0)

    def frame_at(self, video_frame: int) -> dict[str, Any] | None:
        return self.by_video_frame.get(video_frame)

    def event_start(
        self,
        event: dict[str, Any],
        max_lead_frames: int | None = None,
        min_lead_frames: int = 0,
    ) -> int:
        """The video frame to start watching an event from.

        The track's first sighting, because the approach is what tells you whether a count is
        real — a mislabelled pedestrian and a cyclist look much the same in the single frame
        they cross on. Bounded at both ends: a track held for a minute before crossing would
        otherwise mean sitting through it, and one that appeared half a second before would
        drop you effectively on top of the crossing.
        """
        crossing = int(event["v"])
        start = min(int(event.get("v_start", crossing)), crossing - min_lead_frames)
        if max_lead_frames is not None:
            start = max(start, crossing - max_lead_frames)
        return max(0, start)


def load_debug_log(path: str | Path) -> DebugLogData:
    meta: dict[str, Any] = {}
    frames: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    summary: dict[str, Any] | None = None

    with open(path) as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A killed run can leave a half-written last line. Everything before it is
                # still good, and refusing to open the log over one truncated record would
                # be the least useful possible behaviour.
                print(f"[warn] {path}:{line_no}: truncated record ignored")
                continue
            kind = record.get("t")
            if kind == "meta":
                meta = record
            elif kind == "f":
                frames.append(record)
            elif kind == "e":
                events.append(record)
            elif kind == "summary":
                summary = record

    return DebugLogData(meta, frames, events, summary)
