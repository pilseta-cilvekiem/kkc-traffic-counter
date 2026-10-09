"""Configuration loading/saving.

ROI polygon and counting line are stored as *normalised* (0..1) coordinates so a config
survives a change of camera resolution.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml

# COCO class ids we care about: everything that moves along a street. A cyclist seen from a
# street camera usually produces both a `person` and a `bicycle` box, and YOLO confuses
# `bicycle` with `motorcycle` at small scales, and `car` with `truck` and `bus` at any scale.
# E-scooters have no COCO class at all — they register as a fast-moving `person`.
COCO_PERSON = 0
COCO_BICYCLE = 1
COCO_CAR = 2
COCO_MOTORCYCLE = 3
COCO_BUS = 5
COCO_TRUCK = 7


@dataclass
class ZoneConfig:
    """A part of the view where some classes cannot physically be, e.g. a pavement.

    The camera is fixed, so where a box touches the ground says a lot about what it can be:
    nothing on the pavement is a car, whatever the model thinks of a walker with a suitcase
    near the camera. A detection whose box centre falls in the zone and whose class is a key of
    `relabel` is treated as the value's class instead, before tracking. The centre sits above
    the ground, so a zone reaches above its pavement: into the building fronts, or up to the
    kerb-side markings of the road.
    """

    name: str = ""
    # Normalised polygon vertices [[x, y], ...].
    polygon: list[list[float]] = field(default_factory=list)
    # Raw COCO class id -> the class it is taken to be in this zone.
    relabel: dict[int, int] = field(default_factory=dict)
    # A vehicle box at least this tall (fraction of frame height) keeps its label here. A van
    # or bus in the far lane is tall enough for its box centre to sit over the far pavement,
    # but no walker there is anywhere near that size (~60 px of 720). 0 relabels every size.
    keep_vehicle_height: float = 0.0

    def __post_init__(self) -> None:
        self.relabel = {int(k): int(v) for k, v in (self.relabel or {}).items()}


@dataclass
class RoiConfig:
    # Normalised polygon vertices [[x, y], ...]. Empty means "whole frame".
    polygon: list[list[float]] = field(default_factory=list)
    # Normalised counting line [[x1, y1], [x2, y2]]. Empty means "no counting".
    line: list[list[float]] = field(default_factory=list)
    # Crop inference to the polygon's bounding box (big speed + small-object accuracy win).
    crop: bool = True
    # Blank out pixels inside the crop but outside the polygon.
    mask_outside: bool = False
    # Pixels of slack added around the crop box, so objects straddling the edge stay whole.
    crop_padding: int = 16
    # Regions that constrain what a detection can be (see `ZoneConfig`).
    zones: list[ZoneConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.zones = [z if isinstance(z, ZoneConfig) else ZoneConfig(**z) for z in self.zones]


@dataclass
class ModelConfig:
    weights: str = "yolo11n.pt"
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.5
    classes: list[int] = field(
        default_factory=lambda: [
            COCO_PERSON, COCO_BICYCLE, COCO_CAR, COCO_MOTORCYCLE, COCO_BUS, COCO_TRUCK
        ]
    )
    device: str | None = None  # None -> let the backend choose ("CPU" for OpenVINO)
    half: bool = False

    # Which runtime executes the weights. "auto" picks OpenVINO for a `*_openvino_model/`
    # directory and Ultralytics for everything else.
    backend: str = "auto"

    # OpenVINO only. `f16` uses the ARM cores' native half-precision kernels (~1.3x over f32
    # on this board, and the accuracy difference is invisible at these box sizes). INT8 was
    # measured here and is *slower*, so there is deliberately no int8 option.
    precision: str = "f16"

    # CPUs the inference runtime may use, `taskset` syntax ("6,7", "0-5"). "auto" picks the
    # fastest cluster on a big.LITTLE part — on the A733 that is the two Cortex-A76 cores,
    # which beat all eight together because OpenVINO splits work evenly and then waits for
    # the slow cores. Empty string means "no pinning".
    cpu_affinity: str = "auto"
    # 0 -> one thread per pinned CPU.
    num_threads: int = 0

    # NPU only (a `.nb` model). Directory holding libVIPhal.so and libNBGlinker.so; empty
    # means "wherever the loader finds them". Which head decoder the `.nb` needs — its export
    # strips the detection head, so the shape of what comes back is model-family specific.
    npu_libs: str = ""
    # "auto" reads the model family off the network's output signature; see _sniff_decoder.
    npu_decoder: str = "auto"

    # Relabel one class as another *for tracking*. On this camera a cyclist at night flips
    # between `bicycle` and `motorcycle` depending on model size and input resolution, and a
    # van between `car` and `truck`, and the tracker never associates across classes — so
    # unmerged, one vehicle whose label flickers becomes several tracks. Merging keeps one
    # track per vehicle, and it is counted under the merged class; the raw label the track
    # saw most is reported beside it as `raw_class` (`motorcycle`, `bus`, `truck`).
    merge_classes: dict[int, int] = field(
        default_factory=lambda: {
            COCO_MOTORCYCLE: COCO_BICYCLE, COCO_BUS: COCO_CAR, COCO_TRUCK: COCO_CAR
        }
    )

    # Which classes (after merging) may trigger the counting line. A `person` is counted as
    # a `pedestrian`, or as `rider(person)` when it moves at riding pace (a scooter); drop it
    # from this list to count vehicles only. A person listed in `classes` but not here is
    # still promoted to a rider when `motion.promote_fast_person` is on.
    count_classes: list[int] = field(default_factory=lambda: [COCO_PERSON, COCO_BICYCLE, COCO_CAR])


@dataclass
class TrackConfig:
    # ByteTrack knobs. `track_activation_threshold` gates which detections start a track.
    track_activation_threshold: float = 0.25
    lost_track_buffer: int = 30
    minimum_matching_threshold: float = 0.8

    # Fastest object we expect to associate across a gap, in px/s. The tracker's association
    # radius is derived from this and the frame interval, so raising --stride automatically
    # widens the search instead of silently breaking tracks: at stride 12 a scooter covers
    # ~130 px between processed frames, far beyond any sensible fixed radius.
    max_match_speed_px_s: float = 400.0
    # A track must be seen this many times before its line crossings are considered at all.
    min_hits_to_count: int = 2
    # Minimum IoU for a detection to continue a track of another class. The model relabels
    # one object between frames — a walker near the camera as `car`, a cyclist as
    # `motorcycle`, then `car`, then `person` — and a new track at each flip loses the crossing
    # if the flip happens near the line. 0 never matches across classes.
    cross_class_iou: float = 0.3

    # A rider usually yields two overlapping tracks (a `bicycle` box and a `person` box) that
    # cross within a few frames of each other. Crossings this close in time and space are
    # treated as one vehicle.
    dedup_seconds: float = 2.0
    dedup_px: float = 150.0
    # The same, for a pair of crossings pointing *opposite* ways. Two boxes on one rider sit
    # either side of the line, so one reads `in` while the other jitters back `out` a few
    # frames later — direction is no protection against a duplicate, it is a symptom. The
    # window is shorter than `dedup_seconds` because a genuine pair of riders passing each
    # other at the same spot is possible, while a real rider reversing over the line in under
    # a second is not.
    reverse_dedup_seconds: float = 1.0
    # The same, for two boxes on one car, van or bus (a `car` and a `truck`, merged). Those
    # cross the line together, while two cars following in one lane cross a second or so
    # apart, so the window is kept well short of that.
    vehicle_dedup_seconds: float = 0.4
    # One track cannot be counted twice this close together, wherever the second crossing
    # lands. Catches a box oscillating across the line further out than `dedup_px`.
    track_cooldown_seconds: float = 1.0

    # A crossing is held this many processed frames before being judged, so a track that only
    # has one or two speed samples at the moment it crosses still gets a fair decision.
    resolve_delay_frames: int = 6
    # A `person` crossing is held longer, so that when it is the rider of a bicycle, the
    # bicycle is judged first and the person dedups against it — otherwise every cyclist
    # whose person box happens to cross first would be counted as a pedestrian.
    person_resolve_delay_frames: int = 12


@dataclass
class MotionConfig:
    """Speed gating.

    Measured on this camera (all within the lower half of the frame, so perspective scaling
    is roughly constant): scooter ~153 px/s, cyclist riding ~102 px/s, pedestrians and people
    pushing bikes 32-55 px/s, and stationary false positives (a striped bollard that YOLO
    labels `bicycle`) 0-5 px/s. A gate at 80 px/s separates riders from walkers and removes
    the static false positives at the same time.

    Note this threshold is in pixels, so it is specific to this camera's mounting and zoom.
    Re-measure with `calibrate.py --speeds` if the camera moves.
    """

    # Minimum median speed for a track to be counted at the line.
    min_speed_px_s: float = 80.0
    # Count a fast-moving `person` as a rider. Scooter and e-scooter riders never register as
    # `bicycle` in COCO, so without this they are missed entirely.
    promote_fast_person: bool = True
    # A track needs at least this many speed samples before the gate is trusted.
    min_speed_samples: int = 3

    # Minimum net-displacement / path-length for a track to be counted, once it has
    # `min_speed_samples` to judge from. Speed alone cannot see the difference between a
    # rider and a box flickering between a bollard and the pedestrian next to it: both read
    # 200+ px/s. Measured on this camera, every real crossing scores above 0.83 and the
    # flickering ones score 0.01-0.06, so the gate sits far from anything real.
    min_straightness: float = 0.5

    # Riding, not walking, in box-heights per second — a px/s threshold cannot separate the
    # two because a walker near the camera covers as many pixels as a cyclist at the far end.
    # Measured on one live run: a person walking a dog 1.19, the slowest real rider counted
    # 1.85, cyclists and scooters 2.2-6.3; a person pushing a bike sits lower still. Applies
    # to every counted track, because a `bicycle` box is a bicycle pushed as well as ridden.
    # The margin either side is ~25%, so re-measure this before trusting it on another camera
    # — and measure it in the regime it will run in, since like px/s it scales with `dt`.
    min_rider_heights_per_s: float = 1.5
    # The same gate for a `bicycle` box (`rider_classes`); None uses `min_rider_heights_per_s`.
    # A bicycle box measures lower than its rider's person box: near the camera, where most
    # cyclists cross this line, real riders read 1.0-1.5 and were rejected at 1.5. A pushed
    # bike reads ~0.8. Promoting a fast `person` keeps the stricter figure, because a brisk
    # walker is a person box too.
    min_bike_heights_per_s: float | None = None
    # A `person` at riding pace is counted as `rider(person)` — a scooter. If a bicycle track
    # moved alongside it for at least this many frames it is a cyclist whose bicycle crossing
    # was missed, and counts as `bicycle` instead. 0 disables the check.
    bike_companion_frames: int = 0

    # Classes that only count when *ridden*: the two gates above (`min_speed_px_s`,
    # `min_rider_heights_per_s`) apply to these and to promoted pedestrians. A bicycle pushed
    # along the pavement is not counted as a bicycle — its pusher counts as a pedestrian.
    rider_classes: list[int] = field(default_factory=lambda: [COCO_BICYCLE])
    # Every other counted class — pedestrians, cars — only has to be *moving*, in box-heights
    # per second. Walking is ~1; a person standing at the kerb, a parked car whose box jitters
    # over the line, or a static false positive sits near 0. A floor, not a pace test.
    min_moving_heights_per_s: float = 0.3

    # The gates above judge a track's motion from this many seconds before it crossed the line
    # until it is judged, not over its whole life. A car that waited a minute in the queue
    # behind the parked cars reads ~0 px/s over its life, and was rejected as parked itself.
    # Riders cross the frame in a few seconds, so for them the two are nearly the same. 0 means
    # the whole life (the old behaviour).
    window_seconds: float = 3.0
    # How long a vehicle crossing that fails the gates is held, in case the vehicle is only
    # stopped on the line and drives on. In a queue the anchor of a stopped car jitters over the
    # line, and the car is on the far side by the time it moves. 0 drops such crossings at once.
    hold_seconds: float = 30.0


@dataclass
class RuntimeConfig:
    # Process every Nth frame. Source is 15 fps, so stride 5 -> 3 inferences/sec.
    stride: int = 5
    start: float = 0.0  # seconds into the video
    duration: float | None = None  # seconds, None = to the end
    # Frames decoded ahead on a background thread. Keep it shallow — on a live camera a deep
    # queue converts "can't keep up" into growing latency instead of dropped frames.
    prefetch: int = 2

    # --- live sources (RTSP/HTTP camera, /dev/videoN) ------------------------------------
    # "auto" decides from the source: a URL or a device node is live, a file is not. Force it
    # with true/false if the auto-detection guesses wrong (an HTTP URL serving a plain .mp4,
    # say). On a live source `start` is ignored and `duration` becomes wall-clock seconds.
    live: str | bool = "auto"
    # RTSP transport. TCP by default: UDP on a long or congested link drops packets, and a
    # torn frame looks exactly like a detector failure while being nothing of the kind.
    rtsp_transport: str = "tcp"
    # Reopen the stream when it stalls or the camera reboots, instead of exiting.
    reconnect: bool = True
    reconnect_delay: float = 2.0
    # Frame rate to assume when the camera does not advertise a usable one.
    stream_fps: float = 15.0
    # On a live source, derive the tracker's frame interval from real timestamps instead of
    # stride/fps. Frames get dropped by the network and by us whenever inference falls behind,
    # so a fixed dt would silently mis-scale every px/s speed — and the counting gate is a
    # speed threshold.
    adaptive_dt: bool = True

    def live_for(self, path: str | int) -> bool | None:
        """Resolve `live` against a source path. None means "auto-detect"."""
        if isinstance(self.live, bool):
            return self.live
        value = str(self.live).strip().lower()
        if value in ("auto", ""):
            return None
        return value in ("1", "true", "yes", "on")


@dataclass
class Config:
    roi: RoiConfig = field(default_factory=RoiConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        if path is None:
            return cls()
        p = Path(path)
        if not p.exists():
            return cls()
        raw = yaml.safe_load(p.read_text()) or {}
        return cls(
            roi=RoiConfig(**raw.get("roi", {})),
            model=ModelConfig(**raw.get("model", {})),
            track=TrackConfig(**raw.get("track", {})),
            motion=MotionConfig(**raw.get("motion", {})),
            runtime=RuntimeConfig(**raw.get("runtime", {})),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(asdict(self), sort_keys=False))

    def with_overrides(self, **kv) -> "Config":
        """Apply flat CLI overrides like ``stride=3`` / ``conf=0.15`` onto a copy."""
        out = copy.deepcopy(self)
        targets = (out.roi, out.model, out.track, out.motion, out.runtime)
        for key, value in kv.items():
            if value is None:
                continue
            for target in targets:
                if hasattr(target, key):
                    setattr(target, key, value)
                    break
            else:
                raise KeyError(f"unknown config override: {key}")
        return out
