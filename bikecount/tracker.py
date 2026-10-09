"""A small, self-contained multi-object tracker.

Why not ByteTrack? `supervision` 0.30 dropped its `ByteTrack` from the public API, and
Ultralytics' built-in `.track()` only exists on the Ultralytics model object — using it would
tie tracking to one detector backend, which defeats the point of the `Detector` protocol when
we later swap in RKNN on the SBC.

What this does instead: greedy association by IoU, falling back to centroid distance for the
small/fast boxes where IoU goes to zero between decimated frames. That is weaker than ByteTrack
in crowds, but the scene here is a quiet street at night with a handful of objects, and the
tracker is deliberately easy to replace behind `update()`.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field

import numpy as np


# Path points between the samples `Motion.path_length` sums over.
PATH_STEP = 5


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two (N, 4) / (M, 4) xyxy arrays -> (N, M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)

    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]

    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)


def _centroids(boxes: np.ndarray) -> np.ndarray:
    return np.stack(
        [(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1
    )


@dataclass
class Motion:
    """How a track moved over some stretch of its life: speeds, centroid path, box heights.

    Split out of `Track` so the gates can ask about a window rather than the whole life. A car
    that queued for a minute behind the parked cars and then drove across has a lifetime
    median speed of ~0 — measured over its whole life it is indistinguishable from the parked
    cars, measured around the moment it crossed it plainly is not.
    """

    speeds: list[float]  # px/s, one per step between consecutive `path` points
    path: list[tuple[float, float]]
    heights: list[float]

    @property
    def median_speed(self) -> float:
        """Median px/s.

        Median rather than mean because detection jitter produces occasional huge
        single-frame jumps that would drag a mean well above the true pace.
        """
        return float(np.median(self.speeds)) if self.speeds else 0.0

    @property
    def median_height(self) -> float:
        """Median box height in px — a stand-in for distance from the camera."""
        return float(np.median(self.heights)) if self.heights else 0.0

    @property
    def heights_per_s(self) -> float:
        """Speed in box-heights per second: px/s made perspective-invariant.

        A pedestrian near the camera and a cyclist at the far end of the frame can share a
        px/s figure; measured against their own size they do not. Walking is roughly one body
        height per second, riding two to four, whatever the depth.
        """
        h = self.median_height
        return self.median_speed / h if h > 0 else 0.0

    @property
    def net_displacement(self) -> float:
        """Straight-line distance from the first point to the last."""
        if len(self.path) < 2:
            return 0.0
        (x0, y0), (x1, y1) = self.path[0], self.path[-1]
        return float(np.hypot(x1 - x0, y1 - y0))

    @property
    def path_length(self) -> float:
        """Distance actually travelled, summed over every `PATH_STEP`th point.

        Not frame to frame: near the camera a low-confidence box changes size every frame, and
        its centre jitters a few pixels either way. Summed at 15 fps over a few seconds that
        buries a walker's real progress — a pedestrian at 60 px/s measured 0.44 straightness
        and was dropped as a flicker. Over a third of a second the jitter averages out, while
        a box flickering between two objects still goes nowhere.
        """
        if len(self.path) < 2:
            return 0.0
        pts = np.asarray(self.path, dtype=np.float32)
        idx = list(range(0, len(pts), PATH_STEP))
        if idx[-1] != len(pts) - 1:
            idx.append(len(pts) - 1)
        return float(np.sum(np.linalg.norm(np.diff(pts[idx], axis=0), axis=1)))

    @property
    def straightness(self) -> float:
        """net_displacement / path_length, in [0, 1]. 1.0 is a straight line.

        This is what separates something that moved from something that only *looked* like it
        moved. A box flickering between a bollard and a pedestrian beside it racks up a large
        path length while going nowhere, and its median speed reads like a cyclist's; its
        straightness reads 0.06. Anything genuinely crossing this frame measures above 0.8.
        """
        path = self.path_length
        return self.net_displacement / path if path > 0 else 0.0


@dataclass
class Track:
    track_id: int
    box: np.ndarray  # xyxy
    class_id: int
    confidence: float
    hits: int = 1  # total frames matched
    age: int = 0  # frames since last match
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    # Per-frame velocity smoothed over matches, for predicting where a missed track is now.
    # `velocity` is the raw latest step, which the speed samples use; one jittery box would
    # otherwise throw the prediction off for the whole of a gap.
    drift: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    speeds: list[float] = field(default_factory=list)  # px/s, one per matched frame
    # Centroid and box height at every matched frame. Speed alone cannot tell a rider from a
    # detection flickering between two nearby objects — both register as fast. The shape of
    # the path can (see `straightness`), and the height is what makes a speed comparable
    # between the near and far ends of the frame (see `heights_per_s`).
    path: list[tuple[float, float]] = field(default_factory=list)
    heights: list[float] = field(default_factory=list)
    # Pre-merge class id -> frames it was seen as. `class_id` is the merged class the track
    # is matched on; this is what it actually looked like, for naming it (see `raw_class_id`).
    raw_votes: dict[int, int] = field(default_factory=dict)
    raw_class: int | None = None  # the first sighting's raw class, seeded into raw_votes
    # Tracker frame counter at the first and latest match. Two tracks whose spans overlap were
    # two objects; two that follow each other may be one object whose track broke.
    born: int = 0
    last_seen: int = 0
    # Tracker frame counter of each `path` point, so motion can be measured over a window.
    frames: list[int] = field(default_factory=list)
    # Merged class id -> summed confidence it was seen with. A track is matched across a label
    # flip (a near-camera walker read as `car` for a few frames), so `class_id` is whatever
    # it has been seen as most, by confidence, rather than what it was first.
    class_votes: dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.frames.append(self.born)
        self.path.append((float(self.centroid[0]), float(self.centroid[1])))
        self.heights.append(float(self.box[3] - self.box[1]))
        self.vote(self.class_id if self.raw_class is None else self.raw_class,
                  self.class_id, self.confidence)

    def vote(self, raw_class: int, class_id: int | None = None, confidence: float = 1.0) -> None:
        self.raw_votes[raw_class] = self.raw_votes.get(raw_class, 0) + 1
        if class_id is not None:
            self.class_votes[class_id] = self.class_votes.get(class_id, 0.0) + confidence
            self.class_id = max(self.class_votes, key=self.class_votes.get)

    @property
    def raw_class_id(self) -> int:
        """The raw class this track was seen as most often.

        A merged `bicycle` track is a motorcycle if most of its frames said so. Ties go to the
        merged class itself, so a cyclist that flickers evenly stays a bicycle.
        """
        if not self.raw_votes:
            return self.class_id
        return max(self.raw_votes, key=lambda c: (self.raw_votes[c], c == self.class_id))

    @property
    def centroid(self) -> np.ndarray:
        return np.array(
            [(self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2],
            dtype=np.float32,
        )

    def motion(self, since: int | None = None) -> Motion:
        """Motion from tracker frame `since` onwards, or over the whole life if None."""
        i = 0 if since is None else bisect_left(self.frames, since)
        return Motion(speeds=self.speeds[i:], path=self.path[i:], heights=self.heights[i:])

    # Whole-life figures, for display. The counting gates use a window (see `motion`).
    @property
    def median_speed(self) -> float:
        return self.motion().median_speed

    @property
    def median_height(self) -> float:
        return self.motion().median_height

    @property
    def heights_per_s(self) -> float:
        return self.motion().heights_per_s

    @property
    def straightness(self) -> float:
        return self.motion().straightness

    def predicted_box(self, frames: int = 1) -> np.ndarray:
        """Where the box should be `frames` frames after its last match."""
        return self.box + np.tile(self.drift * frames, 2)


class SimpleTracker:
    """Greedy IoU + centroid-distance tracker.

    Args:
        iou_threshold: minimum IoU to accept a match outright.
        max_distance: fallback association radius in pixels, applied when IoU is 0 but the
            boxes are plausibly the same object that moved between processed frames.
        max_age: how many processed frames a track survives unmatched before being dropped.
        min_confidence: detections below this never start a track (they can still extend one).
        cross_class_iou: minimum IoU for a detection to extend a track of another class; 0
            never matches across classes.
    """

    def __init__(
        self,
        iou_threshold: float = 0.2,
        max_distance: float = 80.0,
        max_age: int = 6,
        min_confidence: float = 0.25,
        dt: float = 1.0,
        cross_class_iou: float = 0.3,
    ):
        self.iou_threshold = iou_threshold
        self.cross_class_iou = cross_class_iou
        self.max_distance = max_distance
        self.max_age = max_age
        self.min_confidence = min_confidence
        self.dt = dt  # seconds between processed frames, for px/s
        self.tracks: list[Track] = []
        self._next_id = 1
        self._frame = 0

    def update(
        self,
        boxes: np.ndarray,
        class_ids: np.ndarray,
        confidences: np.ndarray,
        raw_class_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        """Feed one frame of detections; return the track id assigned to each detection.

        `raw_class_ids` are the classes before `merge_classes`, if any merging was done; they
        only affect how a track is named, never what it is matched with.

        A detection that is too weak to start a new track gets id -1.
        """
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        n_det = len(boxes)
        raw = class_ids if raw_class_ids is None else raw_class_ids
        self._frame += 1
        assigned = np.full(n_det, -1, dtype=int)

        # Frames since each track's last match. A missed detection is common — a low-confidence
        # box near the camera, a cyclist behind a parked car — and the object keeps moving, so
        # both the prediction and the search radius grow with the gap. Looking one frame ahead
        # after five missed ones put a cyclist ~50 px outside the radius, and it restarted as a
        # new track.
        gaps = np.array([max(1, self._frame - t.last_seen) for t in self.tracks], dtype=np.float32)
        predicted = (
            np.stack([t.predicted_box(int(g)) for t, g in zip(self.tracks, gaps)])
            if self.tracks
            else np.zeros((0, 4), dtype=np.float32)
        )

        # --- cost: IoU first, distance as a fallback ---------------------------------
        ious = iou_matrix(predicted, boxes)
        score = ious.copy()

        if n_det and len(self.tracks):
            det_c = _centroids(boxes)
            trk_c = _centroids(predicted)
            dist = np.linalg.norm(trk_c[:, None, :] - det_c[None, :, :], axis=2)
            # The fastest expected object's reach over the gap, and never less than half the
            # box's height: near the camera a box's centre jitters by more than one frame's
            # travel at any speed.
            heights = predicted[:, 3] - predicted[:, 1]
            radius = np.maximum(self.max_distance * np.sqrt(gaps), 0.5 * heights)[:, None]
            # Map distance into (0, iou_threshold) so a real IoU match always outranks it.
            close = (ious <= 0) & (dist < radius)
            score = np.where(
                close,
                self.iou_threshold * (1 - dist / radius) * 0.99,
                score,
            )
            # Across classes only on a strong overlap, and at half weight so that any
            # same-class match is taken first. That keeps a rider's `bicycle` and `person`
            # boxes on their own tracks, while a walker the model reads as `car` for a few
            # frames stays one track instead of a new one starting at every flip — which,
            # at the line, is a crossing lost.
            trk_cls = np.array([t.class_id for t in self.tracks])
            same = trk_cls[:, None] == class_ids[None, :]
            cross = (
                (ious >= self.cross_class_iou) * ious * 0.5
                if self.cross_class_iou > 0 else np.zeros_like(ious)
            )
            score = np.where(same, score, cross)

        # --- greedy matching, highest score first -------------------------------------
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        if score.size:
            order = np.dstack(np.unravel_index(np.argsort(-score, axis=None), score.shape))[0]
            for ti, di in order:
                if score[ti, di] <= 0:
                    break
                if ti in matched_tracks or di in matched_dets:
                    continue
                matched_tracks.add(int(ti))
                matched_dets.add(int(di))

                track = self.tracks[ti]
                new_box = boxes[di]
                new_centroid = _centroids(new_box[None, :])[0]
                # Per frame, over however many frames the track went unmatched: a walker
                # missed for a second otherwise reads as covering that second's ground in
                # one frame interval, and is promoted to a rider.
                gap = max(1, self._frame - track.last_seen)
                track.velocity = (new_centroid - track.centroid) / gap
                track.drift = (
                    track.velocity if track.hits == 1 else 0.5 * track.drift + 0.5 * track.velocity
                ).astype(np.float32)
                track.speeds.append(float(np.linalg.norm(track.velocity)) / self.dt)
                track.path.append((float(new_centroid[0]), float(new_centroid[1])))
                track.frames.append(self._frame)
                track.heights.append(float(new_box[3] - new_box[1]))
                track.box = new_box
                track.confidence = float(confidences[di])
                track.hits += 1
                track.age = 0
                track.last_seen = self._frame
                track.vote(int(raw[di]), int(class_ids[di]), float(confidences[di]))
                assigned[di] = track.track_id

        # --- unmatched detections become new tracks ------------------------------------
        for di in range(n_det):
            if di in matched_dets:
                continue
            if confidences[di] < self.min_confidence:
                continue
            track = Track(
                track_id=self._next_id,
                box=boxes[di],
                class_id=int(class_ids[di]),
                confidence=float(confidences[di]),
                raw_class=int(raw[di]),
                born=self._frame,
                last_seen=self._frame,
            )
            self._next_id += 1
            self.tracks.append(track)
            assigned[di] = track.track_id

        # --- age out the unmatched -----------------------------------------------------
        for ti, track in enumerate(self.tracks):
            if ti not in matched_tracks:
                track.age += 1
        self.tracks = [t for t in self.tracks if t.age <= self.max_age]

        return assigned

    @property
    def frame(self) -> int:
        """Frames seen so far — the clock `Track.born`, `last_seen` and `frames` use."""
        return self._frame

    def get(self, track_id: int) -> Track | None:
        for t in self.tracks:
            if t.track_id == track_id:
                return t
        return None

    def hits_for(self, track_id: int) -> int:
        track = self.get(track_id)
        return track.hits if track else 0

    def class_for(self, track_id: int, default: int) -> int:
        track = self.get(track_id)
        return track.class_id if track else default

    def speed_for(self, track_id: int) -> float:
        track = self.get(track_id)
        return track.median_speed if track else 0.0

    @property
    def active_count(self) -> int:
        return sum(1 for t in self.tracks if t.age == 0)
