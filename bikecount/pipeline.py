"""Detect -> restrict to ROI -> track -> count line crossings."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import supervision as sv

from .config import COCO_PERSON, Config
from .detectors import Detector
from .roi import Roi, Zone
from .tracker import SimpleTracker

# What a counted `person` is called: walking pace, or riding pace on something the model has
# no class for (an e-scooter, or a bicycle it never boxed).
PEDESTRIAN = "pedestrian"
RIDER_PERSON = "rider(person)"
# The labels that make up the original bicycle count, which `groundtruth.yaml` labels.
RIDER_LABELS = ("bicycle", "motorcycle", RIDER_PERSON)


@dataclass
class CrossingEvent:
    timestamp: float  # seconds into the source video
    frame_index: int
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    direction: str  # "in" | "out"
    in_total: int
    out_total: int
    speed_px_s: float = 0.0
    # Running totals for this event's own label, alongside the all-classes ones above.
    class_in: int = 0
    class_out: int = 0
    # What the model called the track most often, before `merge_classes`: `motorcycle` for a
    # `bicycle`, `truck` or `bus` for a `car`. A hint, not a count — see `_raw_name`.
    raw_class: str = ""

    def as_dict(self) -> dict:
        return {
            "ts": round(self.timestamp, 3),
            "frame": self.frame_index,
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class": self.class_name,
            "raw_class": self.raw_class or self.class_name,
            "conf": round(self.confidence, 3),
            "speed_px_s": round(self.speed_px_s, 1),
            "direction": self.direction,
            "in_total": self.in_total,
            "out_total": self.out_total,
            "class_in": self.class_in,
            "class_out": self.class_out,
        }


@dataclass
class _PendingCrossing:
    """A line crossing awaiting judgement.

    Held for a few frames so the track can accumulate enough speed samples for the gate to
    be meaningful, then either emitted as a real event or dropped.
    """

    track: object  # tracker.Track | None — keeps accumulating speeds after the crossing
    track_id: int
    class_id: int
    confidence: float
    direction: str
    timestamp: float
    frame_index: int
    point: tuple[float, float]
    due_frame: int
    track_frame: int = 0  # the tracker's frame counter at the crossing
    # Set once a vehicle's crossing has failed the moving gate and is being held to see
    # whether it drives on (see `_may_still_move`); motion is then judged on the latest window.
    held: bool = False


@dataclass
class Timings:
    """Rolling per-stage millisecond costs, for --bench."""

    stages: dict[str, list[float]] = field(default_factory=dict)

    def add(self, stage: str, ms: float) -> None:
        self.stages.setdefault(stage, []).append(ms)

    def mean(self, stage: str) -> float:
        vals = self.stages.get(stage)
        return sum(vals) / len(vals) if vals else 0.0

    def summary(self) -> dict[str, float]:
        return {k: round(self.mean(k), 2) for k in self.stages}


@dataclass
class FrameResult:
    frame_index: int
    timestamp: float
    detections: sv.Detections  # full-frame coords, tracker_id populated
    events: list[CrossingEvent]
    in_total: int
    out_total: int
    active_tracks: int
    infer_ms: float


def _coexisted(a, b) -> bool:
    """Were these two tracks ever matched in the same frame? Unknown counts as yes."""
    if a is None or b is None or not hasattr(a, "born") or not hasattr(b, "born"):
        return True
    return a.born <= b.last_seen and b.born <= a.last_seen


class _TrackedLineZone(sv.LineZone):
    """`sv.LineZone` that remembers a track's side for as long as the tracker holds the track.

    supervision 0.30 forgets which side a tracker id was on after `crossing_history_length`
    frames without a detection — two, here — and takes the side it reappears on as where it
    started. A car hidden behind the parked cars for a few frames as it reaches the line
    therefore reappears past it, having crossed nothing, while our tracker, which bridges
    gaps of up to `lost_track_buffer` frames, still has it as the same track.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alive: set[int] = set()  # track ids the tracker still holds, set before trigger()

    def _evict_stale_crossing_history(self, current_keys: set[int]) -> None:
        super()._evict_stale_crossing_history(set(current_keys) | self.alive)


class Pipeline:
    def __init__(
        self,
        cfg: Config,
        detector: Detector,
        width: int,
        height: int,
        source_fps: float = 15.0,
        adaptive_dt: bool = False,
    ):
        self.cfg = cfg
        self.detector = detector
        self.roi = Roi(cfg.roi, width, height)
        self.zones = [Zone(z, width, height) for z in cfg.roi.zones if len(z.polygon) >= 3]
        self.timings = Timings()

        # Speed is measured in px per *real* second, so it stays comparable across strides.
        self.source_fps = source_fps or 15.0
        dt = cfg.runtime.stride / self.source_fps
        self.tracker = SimpleTracker(
            iou_threshold=cfg.track.minimum_matching_threshold * 0.25,
            # Scale the association radius with the frame interval, so a higher stride widens
            # the search rather than breaking tracks on fast riders.
            max_distance=cfg.track.max_match_speed_px_s * dt,
            max_age=max(2, cfg.track.lost_track_buffer // max(1, cfg.runtime.stride)),
            min_confidence=cfg.track.track_activation_threshold,
            dt=dt,
            cross_class_iou=cfg.track.cross_class_iou,
        )
        # On a live camera the interval between processed frames is not stride/fps: the
        # network drops frames, and so do we whenever inference falls behind. `dt` is then
        # measured from the frame timestamps instead — every speed downstream is px per real
        # second, and the counting gate is a speed threshold, so getting this wrong silently
        # changes what counts.
        self.adaptive_dt = bool(adaptive_dt)
        self._nominal_dt = dt
        self._max_match_speed = cfg.track.max_match_speed_px_s
        self._last_ts: float | None = None

        line = Roi.line_points(cfg.roi, width, height)
        self.line_zone: sv.LineZone | None = None
        if line is not None:
            (x1, y1), (x2, y2) = line
            self.line_zone = _TrackedLineZone(
                start=sv.Point(x1, y1),
                end=sv.Point(x2, y2),
                # The box centre, the point the debug trail is drawn from, so a count lands
                # where the trail visibly crosses the line. (The bottom edge was used before;
                # behind the parked cars it is often hidden, and using all four corners makes
                # wide boxes trigger early on one corner.)
                triggering_anchors=(sv.Position.CENTER,),
            )

        self.names = getattr(detector, "names", {})

        # Counting state. We keep our own totals rather than using LineZone's, because the
        # line sees every candidate track while only some of them survive the gates.
        self.in_total = 0
        self.out_total = 0
        # label -> {"in": n, "out": n}, in the order labels were first counted.
        self.counts: dict[str, dict[str, int]] = {}
        self._pending: list[_PendingCrossing] = []
        # (timestamp, point, direction, class id, label, track) of every counted crossing.
        self._emitted: list[tuple[float, tuple[float, float], str, int, str, object]] = []
        # track id -> timestamp of the crossing it was last counted for.
        self._last_count_ts: dict[int, float] = {}
        # track id -> direction it was last counted in.
        self._last_count_dir: dict[int, str] = {}
        self._frame_counter = 0
        self._now = 0.0  # timestamp of the frame being counted

    def _apply_zones(self, detections: sv.Detections) -> sv.Detections:
        """Relabel detections whose class cannot be where they touch the ground.

        Done here rather than in `process()` so a replayed log is relabelled too. The label
        the model gave is kept as `seen_class_id`, so the debug video shows `person (car)`.
        """
        if not self.zones or not len(detections):
            return detections
        raw = np.asarray(detections.data.get("raw_class_id", detections.class_id), dtype=int)
        # The box centre, as the counting line uses: the zones are drawn to where the boxes
        # of objects on the pavement sit, not only to the pavement itself.
        centres = detections.get_anchors_coordinates(sv.Position.CENTER)
        heights = detections.xyxy[:, 3] - detections.xyxy[:, 1]
        new_raw = raw.copy()
        for zone in self.zones:
            if not zone.relabel:
                continue
            # Too tall to be a walker: a van in the far lane whose centre sits over the pavement.
            big = heights >= zone.keep_vehicle_px if zone.keep_vehicle_px > 0 else np.zeros(
                len(heights), dtype=bool)
            candidates = np.array([
                int(c) in zone.relabel and not (b and zone.relabel[int(c)] == COCO_PERSON)
                for c, b in zip(new_raw, big)
            ])
            if not candidates.any():
                continue
            inside = candidates & zone.contains_points(centres)
            for i in np.flatnonzero(inside):
                new_raw[i] = zone.relabel[int(new_raw[i])]
        changed = new_raw != raw
        if not changed.any():
            return detections
        merge = self.cfg.model.merge_classes
        detections.data["seen_class_id"] = raw
        detections.data["seen_merged_id"] = np.asarray(detections.class_id, dtype=int).copy()
        detections.data["raw_class_id"] = new_raw
        detections.class_id = np.array(
            [merge.get(int(r), int(r)) if ch else int(c)
             for r, c, ch in zip(new_raw, detections.class_id, changed)], dtype=int
        )
        return detections

    def _retune_dt(self, timestamp: float) -> None:
        """Track the real interval between processed frames, smoothed.

        Exponential smoothing (not the raw gap) because one slow frame — a reconnect, a GC
        pause — should not throw every speed sample that follows it. Clamped to a factor of
        eight around the nominal interval so a stall cannot make the tracker match anything
        to anything.
        """
        last, self._last_ts = self._last_ts, timestamp
        if last is None:
            return
        gap = timestamp - last
        if not (0.0 < gap < 10.0):
            return
        lo, hi = self._nominal_dt / 8.0, self._nominal_dt * 8.0
        dt = min(max(gap, lo), hi)
        smoothed = 0.7 * self.tracker.dt + 0.3 * dt
        self.tracker.dt = smoothed
        self.tracker.max_distance = self._max_match_speed * smoothed

    @property
    def rider_total(self) -> int:
        """Crossings counted as bicycles, motorcycles or riders — the original count."""
        return sum(c["in"] + c["out"] for k, c in self.counts.items() if k in RIDER_LABELS)

    def _judge(self, pending: _PendingCrossing) -> str | None:
        """Apply the class, motion and coherence gates to a crossing, once its track matured.

        Returns the label to count it under, or None to drop it. The gates beyond "was it
        tracked at all":
          * coherence-> removes what the speed gates cannot see: a box flickering between two
                        nearby objects travels hundreds of px/s without going anywhere, and
                        the bollard is close enough to the counting line to drag real tracks
                        onto it and back. Straightness tells the two apart; speed does not.
          * riding   -> a `bicycle` only counts when ridden (a pushed one is a pedestrian's
                        luggage), and a `person` moving at riding pace is promoted to a rider,
                        because scooter riders never register as anything but `person`,
          * moving   -> everything else — pedestrians, cars — just has to be going
                        somewhere, which removes static boxes that jitter over the line (a
                        striped bollard YOLO labels `bicycle`, a parked car).
        """
        motion = self.cfg.motion
        seen = self._motion_at_crossing(pending)

        speeds = seen.speeds if seen is not None else []
        median = seen.median_speed if seen is not None else 0.0
        pace = seen.heights_per_s if seen is not None else 0.0
        mature = len(speeds) >= motion.min_speed_samples
        # A held crossing already failed on real evidence; only more of it can overturn that,
        # not a window left empty by the track coasting through missed detections.
        if pending.held and not mature:
            return None

        # Same "not enough evidence yet" rule as the speed gates: judged only once there is
        # a path worth measuring.
        if mature and motion.min_straightness > 0:
            if seen.straightness < motion.min_straightness:
                return None

        # A track with too few samples to judge is accepted on class alone rather than
        # dropped — a genuine crossing that ends immediately is better counted than lost.
        fast = median >= motion.min_speed_px_s if mature else True
        # Riding, not walking, measured against the box's own height: a px/s threshold alone
        # cannot separate the two, because near the camera a pushed bicycle can outrun it.
        class_id = pending.class_id
        countable = self.cfg.model.count_classes
        listed = not countable or class_id in countable

        riding_bar = motion.min_rider_heights_per_s
        if class_id in motion.rider_classes and motion.min_bike_heights_per_s is not None:
            riding_bar = motion.min_bike_heights_per_s
        rider = True
        if mature and riding_bar > 0:
            rider = pace >= riding_bar
        moving = pace >= motion.min_moving_heights_per_s if mature else True

        if class_id == COCO_PERSON:
            # A pedestrian must clear both riding bars outright, so require real evidence.
            if motion.promote_fast_person and mature and rider and (
                median >= motion.min_speed_px_s
            ):
                if self._rides_with_bike(pending.track):
                    return self.names.get(motion.rider_classes[0], str(motion.rider_classes[0]))
                return RIDER_PERSON
            return PEDESTRIAN if listed and moving else None

        name = self.names.get(class_id, str(class_id))
        if not listed:
            return None
        if class_id in motion.rider_classes:
            if motion.min_speed_px_s > 0 and not fast:
                return None
            return name if rider else None
        return name if moving else None

    def _rides_with_bike(self, person) -> bool:
        """Did a two-wheeler track move alongside this person track?

        A rider yields a `bicycle` box and a `person` box. When the bicycle's own crossing is
        missed or rejected, the person reaches the line alone at riding pace and would count
        as a scooter. Scooters themselves draw a stray `bicycle` box at most once or twice;
        a cyclist's bicycle is tracked for most of the way, within about a body height of
        the person — measured on vid.mp4, 0-3 such frames for scooters, 8-49 for cyclists.
        """
        need = self.cfg.motion.bike_companion_frames
        if need <= 0 or person is None or not hasattr(person, "frames"):
            return False
        at = {f: (p, h) for f, p, h in zip(person.frames, person.path, person.heights)}
        riders = self.cfg.motion.rider_classes
        for bike in self.tracker.tracks:
            if bike is person or bike.class_id not in riders or not _coexisted(bike, person):
                continue
            together = 0
            for f, (bx, by) in zip(bike.frames, bike.path):
                seen = at.get(f)
                if seen and np.hypot(seen[0][0] - bx, seen[0][1] - by) <= 0.75 * seen[1]:
                    together += 1
                    if together >= need:
                        return True
        return False

    def _motion_at_crossing(self, pending: _PendingCrossing):
        """The track's motion from `motion.window_seconds` before the crossing until now.

        Not its whole life: a car that queued behind the parked cars for a minute before
        driving across has a lifetime median speed of ~0, and was rejected as parked. What
        the gates need to know is whether it was moving when it crossed — or, for a crossing
        being held, whether it is moving now.
        """
        track = pending.track
        if track is None or not hasattr(track, "motion"):
            return None
        window = self.cfg.motion.window_seconds
        if window <= 0:
            return track.motion()
        frames = int(round(window / max(self.tracker.dt, 1e-6)))
        anchor = self.tracker.frame if pending.held else pending.track_frame
        return track.motion(since=anchor - frames)

    def _may_still_move(self, pending: _PendingCrossing) -> bool:
        """Should a vehicle crossing that failed the gates wait to see if it drives on?

        In a queue a car stops with its bottom edge on the line and the anchor jitters across
        it while it stands there: each of those crossings is rightly judged not moving, and
        when the car finally drives off it is already on the far side, so the line never sees
        it cross again. Holding the crossing while its track lives lets it count then. A
        parked car never drives on, and is dropped when its track ends or the hold runs out;
        one that goes back across the line cancels the held crossing (see `_cancels_held`).
        """
        hold = self.cfg.motion.hold_seconds
        track = pending.track
        return (
            hold > 0
            and self._holds(pending.class_id)
            and track is not None
            and getattr(track, "age", 0) <= self.tracker.max_age  # still tracked
            and pending.timestamp + hold >= self._now
        )

    def _holds(self, class_id: int) -> bool:
        """Vehicles judged on the moving gate alone: not people, not ridden classes."""
        return class_id != COCO_PERSON and class_id not in self.cfg.motion.rider_classes

    def _cancels_held(self, track_id: int, class_id: int, direction: str) -> bool:
        """Drop a held crossing of this track the other way; True if one was dropped.

        A car jittering on the line crosses in, out, in: only the side it drives off from
        matters, and every back-and-forth pair cancels out.
        """
        if not self._holds(class_id):
            return False
        for i, other in enumerate(self._pending):
            if other.track_id == track_id and other.direction != direction:
                del self._pending[i]
                return True
        return False

    def _raw_name(self, pending: _PendingCrossing) -> str:
        """The class a track was seen as most often, before merging.

        Reported alongside the count rather than counted under, because on this camera it is
        not trustworthy for two-wheelers: the uint8 NPU model calls almost every distant
        cyclist a `motorcycle` in daylight, while `car`/`truck`/`bus` hold up much better.
        """
        track = pending.track
        class_id = getattr(track, "raw_class_id", pending.class_id) if track else pending.class_id
        return self.names.get(class_id, str(class_id))

    def _waits_for_vehicle(self, pending: _PendingCrossing, waiting: list[_PendingCrossing]) -> bool:
        """Is this `person` crossing beside a two-wheeler crossing that is not judged yet?

        If so it is probably that vehicle's rider, and whether it counts depends on the
        vehicle's verdict: dedup against it if the vehicle counts, count as a pedestrian if the
        vehicle turns out to be a bicycle being pushed.
        """
        if pending.class_id != COCO_PERSON:
            return False
        riders = self.cfg.motion.rider_classes
        return any(
            other.class_id in riders and self._near(pending, other.timestamp, other.point,
                                                    self.cfg.track.dedup_seconds)
            for other in waiting
        )

    def _near(self, pending: _PendingCrossing, ts: float, point: tuple[float, float],
              window: float) -> bool:
        px, py = pending.point
        ex, ey = point
        return (
            abs(pending.timestamp - ts) <= window
            and (px - ex) ** 2 + (py - ey) ** 2 <= self.cfg.track.dedup_px**2
        )

    def _resolve_pending(self) -> list[CrossingEvent]:
        """Emit crossings whose hold period has elapsed."""
        events: list[CrossingEvent] = []
        still_waiting: list[_PendingCrossing] = []

        # Judged as whatever the track has been seen as most by now, not the label it happened
        # to carry on the crossing frame.
        for pending in self._pending:
            pending.class_id = getattr(pending.track, "class_id", pending.class_id)

        # Vehicles before people, so a rider's person box finds its bicycle already judged.
        # The sort is stable, so each group keeps its chronological order.
        for pending in sorted(self._pending, key=lambda p: p.class_id == COCO_PERSON):
            if self._frame_counter < pending.due_frame or self._waits_for_vehicle(
                pending, still_waiting
            ):
                still_waiting.append(pending)
                continue
            label = self._judge(pending)
            if label is None:
                if self._may_still_move(pending):
                    pending.held = True
                    still_waiting.append(pending)
                continue
            if self._is_duplicate(pending, label):
                continue

            totals = self.counts.setdefault(label, {"in": 0, "out": 0})
            totals[pending.direction] += 1
            if pending.direction == "in":
                self.in_total += 1
            else:
                self.out_total += 1
            self._emitted.append(
                (pending.timestamp, pending.point, pending.direction, pending.class_id, label,
                 pending.track)
            )
            self._last_count_ts[pending.track_id] = pending.timestamp
            self._last_count_dir[pending.track_id] = pending.direction

            events.append(
                CrossingEvent(
                    timestamp=pending.timestamp,
                    frame_index=pending.frame_index,
                    track_id=pending.track_id,
                    class_id=pending.class_id,
                    class_name=label,
                    confidence=pending.confidence,
                    speed_px_s=getattr(self._motion_at_crossing(pending), "median_speed", 0.0),
                    direction=pending.direction,
                    in_total=self.in_total,
                    out_total=self.out_total,
                    class_in=totals["in"],
                    class_out=totals["out"],
                    raw_class=label if pending.class_id == COCO_PERSON else self._raw_name(pending),
                )
            )

        self._pending = still_waiting
        return events

    def _forget_before(self, now: float) -> None:
        """Drop dedup history no future crossing can match.

        Counting every pedestrian and car on a live stream adds thousands of crossings a day,
        and each new one is compared against all of them.
        """
        if len(self._emitted) < 64:
            return
        cfg = self.cfg.track
        horizon = now - max(
            cfg.dedup_seconds, cfg.reverse_dedup_seconds, cfg.vehicle_dedup_seconds,
            cfg.track_cooldown_seconds,
        ) - 60.0  # slack: pending crossings are judged well after they happened
        self._emitted = [e for e in self._emitted if e[0] >= horizon]
        self._last_count_ts = {k: v for k, v in self._last_count_ts.items() if v >= horizon}
        self._last_count_dir = {k: v for k, v in self._last_count_dir.items()
                                if k in self._last_count_ts}

    def _dedup_window(
        self, a: tuple[int, str, object], b: tuple[int, str, object], same_way: bool
    ) -> float:
        """How close in time two crossings must be to count as one object, 0 if never.

        Only pairs that one object genuinely produces are candidates, so that a pedestrian
        beside a car, or a group walking together, are all counted:
          * a two-wheeler and a person   -> the vehicle and its rider,
          * two two-wheelers, two riders -> one rider seen twice (a label flicker the merge
                                            did not catch, a track broken at the line),
          * two people, one after another-> one person whose track broke at the line. Two
                                            people tracked *at the same time* are two
                                            people, however close — a couple walking.
          * two of any other vehicle     -> two boxes on one car (`car` + `truck`, merged).
                                            These cross together, and cars in one lane
                                            cross a second or so apart, hence the short
                                            window.
        """
        cfg = self.cfg.track
        riders = self.cfg.motion.rider_classes
        (ca, la, ta), (cb, lb, tb) = a, b
        window = cfg.dedup_seconds if same_way else cfg.reverse_dedup_seconds
        if (
            (ca in riders and cb in riders)
            or (ca == COCO_PERSON and cb in riders)
            or (cb == COCO_PERSON and ca in riders)
            or (la == RIDER_PERSON and lb == RIDER_PERSON)
        ):
            return window
        if ca == cb == COCO_PERSON:
            return 0.0 if _coexisted(ta, tb) else window
        if ca == cb:
            return cfg.vehicle_dedup_seconds
        return 0.0

    def _is_duplicate(self, pending: _PendingCrossing, label: str) -> bool:
        """One object, several boxes: count it once.

        A rider yields both a `bicycle` and a `person` box. Direction does not distinguish the
        two: the person box and the bicycle box of one rider sit either side of the counting
        line, so the pair frequently registers as one `in` and one `out` a fraction of a
        second apart. An opposite-direction repeat is therefore also a duplicate, within the
        shorter `reverse_dedup_seconds` window — long enough for any jitter, short enough to
        still count two riders genuinely passing each other.
        """
        cfg = self.cfg.track

        # A single track crossing twice in quick succession is oscillation, not two objects,
        # however far the second crossing lands from the first.
        # Nor can one track cross the same way twice without crossing back in between — and a
        # crossing back that was itself dropped as jitter means this one is jitter too. A
        # walker's box centre wobbling on the line otherwise counts again as soon as the
        # cooldown runs out.
        if self._last_count_dir.get(pending.track_id) == pending.direction:
            return True

        last = self._last_count_ts.get(pending.track_id)
        if last is not None and pending.timestamp - last < cfg.track_cooldown_seconds:
            return True

        for ts, point, direction, class_id, other_label, track in self._emitted:
            window = self._dedup_window(
                (pending.class_id, label, pending.track), (class_id, other_label, track),
                same_way=direction == pending.direction,
            )
            if window > 0 and self._near(pending, ts, point, window):
                return True
        return False

    def process(self, frame_index: int, timestamp: float, image: np.ndarray) -> FrameResult:
        # --- inference on the ROI crop -------------------------------------------------
        t0 = time.perf_counter()
        sub, (dx, dy) = self.roi.crop(image)
        t1 = time.perf_counter()
        detections = self.detector.infer(sub)
        t2 = time.perf_counter()

        # Crop coords -> full-frame coords, so tracking and the counting line live in one space.
        if len(detections) and (dx or dy):
            detections.xyxy = detections.xyxy + np.array([dx, dy, dx, dy], dtype=np.float32)

        # Collapse interchangeable classes (motorcycle -> bicycle) before tracking, so a
        # cyclist whose label flickers between frames stays on one track. The original label
        # is kept alongside: at night `motorcycle` is almost always a misread cyclist, but in
        # daylight there are real mopeds, and only the raw label can tell them apart later.
        if len(detections) and self.cfg.model.merge_classes:
            detections.data["raw_class_id"] = np.array(
                [int(c) for c in detections.class_id]
            )
            detections.class_id = np.array(
                [self.cfg.model.merge_classes.get(int(c), int(c)) for c in detections.class_id]
            )

        # --- drop anything outside the polygon -----------------------------------------
        if len(detections):
            anchors = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            detections = detections[self.roi.contains_points(anchors)]

        return self.count(frame_index, timestamp, detections, crop_ms=(t1 - t0) * 1000,
                          infer_ms=(t2 - t1) * 1000)

    def count(
        self,
        frame_index: int,
        timestamp: float,
        detections: sv.Detections,
        crop_ms: float = 0.0,
        infer_ms: float = 0.0,
    ) -> FrameResult:
        """Track, gate and count detections that are already in full-frame ROI coordinates.

        Split out of `process()` so a run recorded by `--debug-log` can be replayed through
        the real tracker and the real gates without the detector — which is the only way to
        re-score a change on the NPU build from a development machine, and the only way to
        test the gates against a recording of the exact failure they were written for.
        """
        if self.adaptive_dt:
            self._retune_dt(timestamp)
        t2 = time.perf_counter()
        detections = self._apply_zones(detections)

        # --- track ----------------------------------------------------------------------
        if len(detections):
            ids = self.tracker.update(
                detections.xyxy,
                detections.class_id,
                detections.confidence,
                detections.data.get("raw_class_id"),
            )
            detections.tracker_id = ids
            detections = detections[ids >= 0]
            # From here on a detection carries its track's class — what the object has been
            # seen as most — so a label flip does not change what it is counted as. What the
            # model said this frame is kept, for the debug log and for replaying it.
            detections.data["det_class_id"] = np.asarray(detections.class_id, dtype=int).copy()
            detections.class_id = np.array(
                [self.tracker.class_for(int(t), int(c))
                 for t, c in zip(detections.tracker_id, detections.class_id)], dtype=int
            )
        else:
            self.tracker.update(
                np.zeros((0, 4), dtype=np.float32),
                np.zeros(0, dtype=int),
                np.zeros(0, dtype=np.float32),
            )
            detections.tracker_id = np.zeros(0, dtype=int)
        t3 = time.perf_counter()

        # --- count line crossings --------------------------------------------------------
        # Every sufficiently-established track is shown to the line regardless of class or
        # speed, so the line's per-track side state is correct from the first sighting. The
        # class and speed gates are applied afterwards, in _resolve_pending(): a fast rider
        # may have only one or two speed samples at the instant it crosses, and gating here
        # would discard that crossing forever.
        self._frame_counter += 1
        self._now = timestamp
        if self.line_zone is not None and len(detections):
            established = np.array(
                [
                    self.tracker.hits_for(int(tid)) >= self.cfg.track.min_hits_to_count
                    for tid in detections.tracker_id
                ],
                dtype=bool,
            )
            candidates = detections[established]
            self.line_zone.alive = {t.track_id for t in self.tracker.tracks}
            # One class for the line: supervision 0.29 keeps a track's side per (track, class),
            # so a track whose voted class changes would otherwise start over on the far side.
            crossed_in, crossed_out = self.line_zone.trigger(sv.Detections(
                xyxy=candidates.xyxy,
                tracker_id=candidates.tracker_id,
                class_id=np.zeros(len(candidates), dtype=int),
            ))

            for i in range(len(candidates)):
                if crossed_in[i]:
                    direction = "in"
                elif crossed_out[i]:
                    direction = "out"
                else:
                    continue
                track_id = int(candidates.tracker_id[i])
                box = candidates.xyxy[i]
                if self._cancels_held(track_id, int(candidates.class_id[i]), direction):
                    continue
                self._pending.append(
                    _PendingCrossing(
                        # Hold the Track object, not just its id: it keeps accumulating speed
                        # samples after this frame, and survives removal from the tracker.
                        track=self.tracker.get(track_id),
                        track_id=track_id,
                        class_id=int(candidates.class_id[i]),
                        confidence=float(candidates.confidence[i]),
                        direction=direction,
                        timestamp=timestamp,
                        frame_index=frame_index,
                        point=(float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)),
                        track_frame=self.tracker.frame,
                        due_frame=self._frame_counter + (
                            self.cfg.track.person_resolve_delay_frames
                            if int(candidates.class_id[i]) == COCO_PERSON
                            else self.cfg.track.resolve_delay_frames
                        ),
                    )
                )

        events = self._resolve_pending()
        self._forget_before(timestamp)

        self.timings.add("crop", crop_ms)
        self.timings.add("infer", infer_ms)
        self.timings.add("track", (t3 - t2) * 1000)

        return FrameResult(
            frame_index=frame_index,
            timestamp=timestamp,
            detections=detections,
            events=events,
            in_total=self.in_total,
            out_total=self.out_total,
            active_tracks=self.tracker.active_count,
            infer_ms=infer_ms,
        )
