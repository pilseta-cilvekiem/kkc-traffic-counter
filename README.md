# Street traffic detection & counting — proof of concept

Detects traffic in a defined region of a fixed CCTV view — bicycles, cars, vans and buses,
pedestrians, and scooter riders — tracks it between frames, and counts crossings of a
counting line, per class and per direction. It started as a bicycle counter, and the bicycle
count is still the one with ground truth behind it. Built to be extended, and to move onto a
low-power ARM board later without rewriting the pipeline.

## Install

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt

# only if you will export or calibrate models — pulls in torch, and the board does not need it
.venv/bin/pip install ultralytics
```

## Use

```bash
# 1. Define the region and the counting line by clicking on a still frame
python roi_tool.py video.mp4 --at 900

# 2. Look at what it does — ROI, boxes, track ids, speeds, counters
python detect_bikes.py video.mp4 --debug --debug-out annotated.mp4 --start 585 --duration 42

# 2b. Check every crossing it counted, without scrubbing (annotated.log is written alongside)
python debug_viewer.py annotated.mp4

# 3. Deployment shape — no rendering, one JSON line per crossing
python detect_bikes.py video.mp4 --events events.jsonl

# 4. Check accuracy against hand-labelled clips
python calibrate.py video.mp4 --models yolo11n.pt yolo11s.pt --imgsz 640 960 1280

# 5. See where the time goes (run this on the target board)
python detect_bikes.py video.mp4 --duration 60 --bench

# 5b. Read a live camera instead of a file — same command, a URL instead of a path
python detect_bikes.py rtsp://user:pass@192.168.1.3:554/Streaming/Channels/101 --events events.jsonl

# 5c. No camera yet? Serve a recording as one, from the laptop (see NETWORK.md)
python stream_test_video.py video.mp4          # -> rtsp://192.168.1.1:8554/cam

# 6. Re-export the model for the current ROI shape (workstation only, needs ultralytics)
python export_openvino.py --weights yolo11n.pt --long 960
```

Each crossing is one JSON line:

```json
{"ts": 117.0, "frame": 1755, "track_id": 19, "class_id": 1, "class": "bicycle",
 "raw_class": "motorcycle", "conf": 0.34, "speed_px_s": 149.4, "direction": "in",
 "in_total": 3, "out_total": 1, "class_in": 1, "class_out": 0}
```

`in_total`/`out_total` are across every class; `class_in`/`class_out` are for this event's
`class` alone. The run summary prints the per-class table as well.

### What gets counted, and under which name

| `class` | from | counts when |
|---|---|---|
| `bicycle` | COCO `bicycle` + `motorcycle` | ridden — clears both riding gates |
| `car` | COCO `car` + `truck` + `bus` | moving |
| `pedestrian` | COCO `person` | moving, at less than riding pace |
| `rider(person)` | COCO `person` | moving at riding pace — e-scooters, which have no COCO class |

Similar classes are merged (`model.merge_classes`) because the tracker never matches across
classes, and the model's label for one object flickers: a cyclist between `bicycle` and
`motorcycle`, a van between `car` and `truck`. What the model called the track most often is
kept as `raw_class`. For vehicles it is a usable hint (a `bus` is a bus). For two-wheelers it
is not: the uint8 NPU model calls nearly every daytime cyclist a `motorcycle` (282 of 284
two-wheeler boxes in one 5-minute run), so mopeds cannot be told apart from bicycles yet.

A rider produces a `bicycle` box and a `person` box. The person crossing is held longer than
the bicycle crossing (`track.person_resolve_delay_frames`), so the bicycle is judged first and
the person dedups against it. If the bicycle is rejected as not ridden, the person is judged
on its own and counts as a pedestrian pushing a bike. Only pairs that one object can actually
produce are deduplicated (a two-wheeler with a person, two boxes on one car), so a group
walking together, or a pedestrian next to a car, are all counted.

### Sending counts to Supabase

```bash
python detect_bikes.py rtsp://... --events events.jsonl --report
```

`--report` also posts each crossing to the `crossings` table, as it is counted. Credentials
come from `.env` (copy `.env.example`): the project URL, the *publishable* key, and the email
and password of a Supabase Auth user made for the board. `supabase/schema.sql` creates the
table and the row-level security that lets only users flagged as uploaders insert, so the
board's credentials can add counts and nothing else.

Sending runs on its own thread and never holds up counting. While the uplink is down events
wait in memory and are retried with backoff; each carries its own `event_id`, so a retried
batch is not stored twice. The queue is not written to disk: a restart during an outage
loses what was waiting, and the run summary says how many.

### The public viewer

`viewer/` is a static page that reads `crossings` with the publishable key and shows totals
per hour, day, week or month for any interval, split by type (bicycles, scooters & riders,
pedestrians, cars) and direction (`in` is labelled *From center*, `out` *To center*). It also
downloads the table or the raw events of the interval as CSV. `.github/workflows/pages.yml`
publishes it to GitHub Pages on every push to `main` that touches it (one-time setup:
Settings → Pages → Source: GitHub Actions).

PostgREST aggregates are off, so the page fetches the interval's rows in pages of 1000 and
buckets them in the browser, in Riga time. To try it locally:
`python -m http.server -d viewer`.

## The ROI tool, and what the two shapes do

```bash
python roi_tool.py video.mp4 --at 900          # opens a window on the frame at 900s
python roi_tool.py video.mp4 --dump-frame f.png # headless: just export a still
```

| key / click | what it does |
|---|---|
| **left click** | add a point — a polygon vertex, or a line endpoint in line mode |
| **right click** | undo the last point of whichever shape you are editing |
| **L** | toggle between polygon mode and line mode (the header shows which) |
| **C** | clear both shapes and start over |
| **S** | save to `config.yaml` and quit (needs at least 3 polygon points) |
| **Q** or **Esc** | quit *without* saving |

The polygon needs 3+ points; the line takes exactly 2 (a third click starts a new line).
Both are written to `config.yaml` as fractions of frame width/height, so they survive a
resolution change. There is no undo after saving — the old values are overwritten.

Working headless over SSH? `--dump-frame` writes the still, you read pixel coordinates off it
in any image editor, and you divide by 1280 and 720 to get the normalised numbers to paste
into `config.yaml`.

### Polygon = *where* to look. Line = *when* to count.

They answer different questions, and you need both.

The **polygon** restricts attention: inference runs only on its bounding box, and detections
whose bottom-centre falls outside it are discarded. It buys speed and removes whole categories
of false positive — parked bikes on the far pavement, the building, the sky.

The **line** is what turns detections into a number. Without it you get boxes and tracks but
no count at all (`detect_bikes.py` warns and only detects).

The reason a count needs a line is that "how many bicycles are in the polygon" is not the
quantity you want. One cyclist present for 40 processed frames would be 40 sightings of the
same person; you cannot sum per-frame counts, and a maximum only tells you the busiest instant.
What you want is **traffic through a place over time**, so the pipeline counts *events*: a
tracked object whose anchor point is on one side of the line in one frame and the other side
in the next scores exactly one crossing. One rider, one count, however long they were visible.

That also gives you **direction for free**, which is why the counts are split IN/OUT rather
than being a single total:

- **Crossing to the right of the arrow counts as OUT; crossing to the left counts as IN.**
- Reversing the two points swaps IN and OUT — that is the only thing point order affects.
- Verified empirically against `supervision`'s `LineZone`, not assumed.

So on this camera IN and OUT are just "riders heading up the path" versus "riders heading
down it". Which is which is arbitrary until you watch one crossing in `--debug` and label it.

Practical consequences worth knowing:

- **The line must sit inside the polygon.** Detections are dropped outside the polygon, so a
  line reaching beyond it can never be crossed there.
- **A rider must be tracked on both sides.** If they are only detected after passing the line,
  or the track breaks at the line (here: behind the scaffolding pole), there is no crossing.
  This, not detection, is why line placement was tuned from trajectory data — see below.
- **Riders who never cross it are never counted.** Cyclists on the main road do not cross the
  bike-path line. Counting them needs a second line, not a bigger polygon.

## Checking a run: the debug log and the viewer

`--debug-out live.mp4` also writes **`live.log`** beside the video — one JSON line per
processed frame, plus one per crossing. `debug_viewer.py` plays the video against that log,
so verifying a run is stepping through its events rather than scrubbing an hour of pavement:

```bash
python detect_bikes.py video.mp4 --debug-out live.mp4 --events events.jsonl
python debug_viewer.py live.mp4          # finds live.log automatically
python debug_viewer.py live.mp4 --list   # headless: just print the events
```

| | |
|---|---|
| `n` / `p` | next / previous crossing |
| `r` | replay this one |
| `space`, `,` `.` | play/pause, step a frame |
| `←` `→`, `[` `]` | seek 1 s, 10 s |
| `-` `=` | slower / faster |
| `a` | auto-pause after each event on/off |
| click the timeline | seek there |

### Scoring that run, and re-scoring it without the model

Watching events one by one says whether each is real. `validate_log.py` says whether the *set*
is right, by matching a recorded log against the labelled windows in `groundtruth.yaml` —
anything counted outside a window is a false positive, because the ground truth lists every
rider in the stretch a human watched:

```bash
python validate_log.py live.log --video 134146 --auto-offset
```

Log timestamps start when the pipeline connected, while ground-truth seconds are positions in
the source file, so the two are a constant apart; `--auto-offset` fits it, `--offset` sets it.
Windows past the end of the run are excluded rather than scored as misses.

`replay_log.py` then re-counts that same log under a different `config.yaml`. The log holds
every detection box, which is enough to drive the tracker and the gates again with no model and
no video — the only way to score a tuning change against the NPU build's own footage from a
development machine:

```bash
python replay_log.py live.log -o replayed.log && \
  python validate_log.py replayed.log --video 134146 --auto-offset
```

It does not re-run inference, and the log keeps only detections that were assigned a track, so
it reproduces the same crossings but not necessarily the same count to the last event. It is
exact about the thing it is for: comparing two configs over one recording.

**Jumping to an event starts at the beginning of its track, not at the line.** A rider and a
mislabelled pedestrian look much the same in the one frame they cross on; the approach is
what tells them apart, so playback starts where the detector first saw the track (at least
2 s and at most 6 s of run-up, `--min-lead` / `--lead`), plays through the crossing, and
pauses 1.5 s after it. The bottom strip is a timeline with a tick per crossing, green IN and
orange OUT.

The log exists because **the two clocks are not the same**. The debug video holds one frame
per *processed* frame, so at `--stride 3` its frame 100 is source frame 300 — and on a live
stream, where frames are dropped whenever inference falls behind, there is no formula at all;
the run above logged 892 frames over 60 s of a 15 fps stream. The log is the only place that
correspondence exists, and it also records where each crossing *happened* rather than where
it was reported: an event surfaces `resolve_delay_frames` frames late, because the speed gate
needs the track to mature first, so seeking to the reported frame shows the rider already
past the line.

Each frame line carries the boxes, track ids, confidences, speeds and hit counts that frame
saw, so the log answers "was it detected at all, and how fast did we think it was going"
without re-running anything:

```bash
grep '"t": "e"' live.log | jq -r '"\(.ts) \(.direction) \(.class) \(.speed_px_s)px/s"'
```

`--debug-log PATH` puts it somewhere else, `--no-debug-log` skips it.

## Reading a live camera

`detect_bikes.py` takes a stream URL wherever it takes a file — `rtsp://`, `http(s)://`
(MJPEG), `udp://`, `tcp://`, `/dev/video0`, or a bare camera index — and switches to live
handling automatically.

```bash
python detect_bikes.py rtsp://user:pass@192.168.1.3:554/Streaming/Channels/101 --events events.jsonl
```

A camera is not a file, and four things change accordingly:

* **Stale frames are dropped.** The camera does not wait for us, so the prefetch queue keeps
  only the newest frame. Falling behind costs frames, never latency — a deep buffer would
  turn "cannot keep up" into an ever-growing delay, which is the worse failure.
* **Timestamps come from the wall clock.** With frames dropped by both the network and us,
  `frame index / fps` is a fiction, and every speed here is px per *real* second.
* **The tracker's frame interval is measured rather than assumed** (`runtime.adaptive_dt`).
  The counting gate is an 80 px/s speed threshold, so a `dt` taken from `stride / fps` when
  frames are being dropped would quietly change what counts.
* **A dropped stream reconnects** rather than ending the run — a camera reboot or a pulled
  cable is an event to ride out, not the end of the input. A camera that is not answering
  yet at startup is waited for the same way. `--no-reconnect` to opt out.

`--start` is ignored (there is no past to seek to) and `--duration` is wall-clock seconds.
The progress line gains `fps in`, the rate the camera actually delivered: if `fps proc` sits
well below it, the board is behind and `--stride` should go up.

On the board it runs as a systemd service, `tools/bikecount.service`, with the camera URL
set as `CAMERA_URL` in `.env` — see NETWORK.md, "Running it as a service".

**Testing it without a camera.** `stream_test_video.py` loops one of the recordings out as a
real stream, so the whole path — RTSP, dropped frames, reconnects — is exercised on the
board before any hardware is mounted:

```bash
# on the laptop
python stream_test_video.py video.mp4                 # rtsp://192.168.1.1:8554/cam
python stream_test_video.py video.mp4 --mode mjpeg    # http://192.168.1.1:8080/stream.mjpg — pure Python

# on the board
python detect_bikes.py rtsp://192.168.1.1:8554/cam --duration 45 --bench
```

Measured over the direct cable, RTSP H.264 720p15 into yolo11n on the NPU at stride 1:
**14.8 of the camera's 15 fps processed, 23.5 ms a frame**, and killing the publisher
mid-run reconnected and carried on counting.

**The cable is statically addressed** — laptop `192.168.1.1`, board `192.168.1.2`, camera on
the same `/24` — because there is no DHCP server on it. The board's ethernet ships on DHCP
and will otherwise sit in *"connecting (getting IP configuration)"* forever, which looks
exactly like a dead cable. The settings, the `nmcli` commands that applied them, how to find
a camera's address, a systemd unit, and the failure modes are in **[NETWORK.md](NETWORK.md)**.

## How it works

```
VideoSource      grab() skips frames without decoding them; only every Nth frame is decoded
   -> Roi        crop to the polygon's bounding box before inference
   -> Detector   YOLO11 (swappable: .pt / .onnx / OpenVINO / later RKNN)
   -> merge      motorcycle -> bicycle, truck/bus -> car (see below)
   -> Roi        drop detections whose bottom-centre falls outside the polygon
   -> Tracker    greedy IoU + centroid matching, records px/s per track
   -> LineZone   raw crossings, then class + speed gates decide what actually counts
```

Two knobs carry the performance story, and both are in from the start:

- **`--stride N`** — decode and infer on every Nth frame. Skipped frames use `cap.grab()`,
  which demuxes but does not decode, so a high stride genuinely saves CPU rather than just
  saving inference.
- **ROI cropping** — inference runs on the polygon's bounding box (here 1271×420 instead of
  1280×720). Fewer pixels, and small distant objects survive the resize better.

### Why counting is deferred, not filtered inline

The line sees *every* established track regardless of class or speed, and the gates are applied
a few frames later in `_resolve_pending()`. This matters: a scooter rider crosses the line about
two frames after first being seen, at which point there aren't enough samples to judge its
speed. Filtering before the line silently discarded that crossing forever. Holding the crossing
for `resolve_delay_frames` and judging it once the track has matured fixed it.

A rider also usually produces two overlapping tracks — a `bicycle` box and a `person` box — that
cross within a few frames of each other, so crossings close in time and space are deduplicated
into one vehicle. Deduplication ignores direction: the two boxes sit either side of the line, so
the pair routinely registers as one `in` and one `out` a fraction of a second apart, and reading
that as two riders was the single largest source of over-counting on the live stream. An
opposite-direction repeat is judged over the shorter `reverse_dedup_seconds` window, so two
riders genuinely passing each other are still two. A track that has just been counted is also
held off for `track_cooldown_seconds`, whatever the geometry.

Both gates apply to every counted track, not only to pedestrians promoted by
`promote_fast_person`: a `bicycle` box is a bicycle whether it is ridden or pushed alongside, and
near the camera a pushed one can outrun the flat px/s gate.

Finally, a crossing must be *coherent*: `motion.min_straightness` requires net displacement to
be at least half the path actually travelled. Speed alone cannot see the difference between a
rider and a detection flickering between a bollard and the pedestrian beside it — both read
200+ px/s. Straightness can: everything real on this camera measures above 0.83, the flickering
ones 0.01–0.06.

### Queued traffic

Cars behind the parked cars at the left edge queue for a minute or more before driving across,
which broke counting three ways (on the first 10 minutes of `vid.mp4`, 14 of ~32 cars counted):

- **Gates judged the whole life.** A car that waited 60 s has a lifetime median speed of ~0 and
  was rejected as parked. The gates now look at `motion.window_seconds` before the crossing
  until it is judged.
- **Cars stop on the line.** A stopped car's box centre jitters over the line; each of those
  crossings is rightly "not moving", and when the car drives off it is already past. A vehicle
  crossing that fails the gates is held for up to `motion.hold_seconds` while its track lives,
  re-judged on its latest motion, and cancelled if the car goes back across.
- **The line forgot hidden cars.** `supervision` 0.30's `LineZone` drops a track's side after
  two frames without a detection, so a car hidden behind the parked cars for a few frames as it
  reaches the line reappears past it having crossed nothing. The line now keeps the side for as
  long as the tracker keeps the track.

Speeds are also per frame over however long a track went unmatched: a walker missed for a
second used to read as covering that ground in one frame interval, and was promoted to a rider.

## What the footage actually showed

Measured on the two supplied recordings from this camera. These findings drove most of the
design, so they are worth keeping.

**1. The class label is unreliable; the detection is not.** A cyclist at night is found
consistently, but is labelled `bicycle` or `motorcycle` depending on model size and input
resolution — larger models were *more* confidently wrong:

| model | imgsz | label for the same rider | conf |
|---|---|---|---|
| yolo11n | 640 | motorcycle | 0.42 |
| yolo11n | 1280 | **bicycle** | 0.42 |
| yolo11s | 1280 | motorcycle | 0.72 |
| yolo11m | 1280 | motorcycle | 0.77 |

So `motorcycle` is merged into `bicycle` (`model.merge_classes`). Across the whole 113-minute
night recording there were only 10 motorcycle detections total, so this costs almost nothing.

**2. Scooter riders never register as a bicycle at all** — only as a fast-moving `person`.

**3. Speed separates riders from walkers cleanly**, and kills static false positives at the
same time:

| | median speed |
|---|---|
| scooter | 153 px/s |
| cyclist riding | 102 px/s |
| pedestrian / person pushing a bike | 32–55 px/s |
| striped bollard that YOLO labels `bicycle` | 0–5 px/s |

Hence the `motion.min_speed_px_s: 80` gate. It promotes fast `person` tracks to riders (catching
scooters) and rejects the stationary scenery that YOLO insists is a bicycle.

**These are pixel speeds, so they are specific to this camera's mounting and zoom.** Re-measure
if the camera moves.

**4. A pixel speed cannot separate a walker from a rider on its own.** A pedestrian near the
camera covers as many px/s as a cyclist at the far end of the ROI, and on the live stream a
person walking a dog cleared the 80 px/s gate at 107 px/s and was counted. Measured against
their own box height the two separate, wherever they are in the frame — every crossing from one
live run, scored the way the gate scores it:

| | px/s | box h | heights/s |
|---|---|---|---|
| pedestrian walking a dog | 107 | 90 | **1.19** |
| slowest real rider counted | 150 | 81 | **1.85** |
| scooter rider | 205 | 92 | 2.23 |
| cyclists | 169–591 | 48–94 | 2.41–6.29 |

Hence `motion.min_rider_heights_per_s: 1.5`, applied to every counted track. Note the margin is
about 25% either side, not an order of magnitude — this is a real threshold on real data, and it
is the number to re-measure first if walkers start counting or riders stop.

Pushing a bike sits further down the same scale than walking a dog does: the `expect: 0` night
window measures **0.83 heights/s** for the person pushing, against 3.58 for the cyclist riding
in the next window.

**Heights/s is regime-dependent, exactly like px/s.** Both are derived from the tracker's `dt`,
so stride, the model, and whether the run is live (measured `dt`) or a file (`stride/fps`) all
shift the absolute numbers — the same rider measures 3.96 heights/s on the live stream and 5.79
replayed from file at stride 3. The separation between walking and riding holds in every regime;
the threshold has to come from data taken in the one it will run in. The value above is from the
live stream, which is what production is.

**5. A static false positive does damage even when the speed gate rejects it.** A striped
barrier beside the counting line fires as `bicycle`/`person` at conf 0.20–0.42 in 93 of 643
logged detections. Its own speed (~5 px/s) is rejected, as intended — but the tracker snaps
passing riders onto it and back, which fabricates 200–400 px/s and phantom crossings out of
nothing. It sits ~30 px from the line, so it cannot be cropped out of the ROI without cutting
the line in half; `min_straightness` is what removes it.

## Known limitations

- **All-traffic ground truth is one daytime hour.** `groundtruth.yaml` labels riders only
  (`calibrate.py`, `validate_log.py`); `vid.csv` covers every class, but for one hour of one
  day (see "All traffic, one full hour"). Night and bad weather are unmeasured for cars and
  pedestrians. Walkers (pushing a bike ~0.83 heights/s, the dog walker 1.19) are kept out of
  the *rider* count by the `expect: 0` windows; they count as `pedestrian` instead.
- **Small pedestrians drop out.** On the far pavement a walker is ~55 px tall, and the model
  can lose them for a couple of seconds — which, when it happens at the line, splits the track
  in two and neither half is counted. Higher network resolution helps (see the NPU section);
  the tracker does not yet bridge gaps longer than `lost_track_buffer`.
- **Mopeds count as bicycles.** `merge_classes` folds `motorcycle` into `bicycle`, because a
  cyclist is frequently labelled `motorcycle` — by the NPU model, almost always. In daylight
  there are real mopeds. The label the model gave is kept in each event's `raw_class`, but
  on the current model it cannot separate them.
- **`promote_fast_person` is a recall/precision trade.** It is what catches scooters, but a
  brisk pedestrian also clears 80 px/s — one walking a dog was counted on the live stream, which
  is what `min_rider_heights_per_s` now guards against, with about 25% of margin. Set
  `promote_fast_person: false` for strictly-bicycles counting.
- **The px/s gate is a flat pixel threshold.** It ignores perspective. That was tolerable
  while the ROI covered only the lower half of the frame; the current ROI spans the street to
  the far pavement, so the gate is lowered to 40 px/s and `min_rider_heights_per_s` carries the
  walker/rider distinction on its own. Replaying both logged runs, 0–80 px/s made no
  difference to what was counted.
- The tracker is a simple greedy IoU/centroid matcher, not ByteTrack (`supervision` 0.30
  removed `ByteTrack` from its public API, and Ultralytics' `.track()` would tie tracking to
  one detector backend). It is adequate for a quiet corner with a handful of objects and is
  isolated behind `SimpleTracker.update()` if it needs replacing.

## Where accuracy actually stands

Against all 14 labelled windows in `groundtruth.yaml` (21 riders across both recordings),
with the shipped config:

```
yolo11s.pt  imgsz=640  conf=0.20  stride=3   ->  13 / 21 counted, 0 false positives
```

**~62% recall, and no over-counting.** Every crossing it reports was checked by eye and was a
genuine rider — cyclists, kick scooters, and one moped. The `expect: 0` window (two people
walking bikes) correctly stays at zero.

More model capacity does not help, which is the useful finding:

| model | imgsz | conf | counted / 21 | ms/frame | |
|---|---|---|---|---|---|
| **yolo11s** | **640** | **0.20** | **13** | **31** | shipped |
| yolo11s | 960 | 0.15 | 11 | 54 | |
| yolo11m | 960 | 0.15 | 11 | 137 | also over-counts pushed bikes |
| yolo11n | 640/960/1280 | 0.20 | erratic | 15–50 | on the detection threshold |

yolo11m is both slower *and* less accurate here: at conf 0.15 it detects the *pushed* bikes
well enough to count two riders that should not count. Small and decisive beats big and
hesitant on this footage.

The remaining 8 misses are **detection intermittency, not geometry** — in most missed windows
the rider is never detected in a single frame (`hits = 0`). Closing that gap means fine-tuning
on this camera, not more tuning of thresholds.

### All traffic, one full hour (`vid.csv`)

`vid.csv` is a human tally of *everything* that passed during `vid.mp4` — 712 objects, class,
time in view and direction. The burnt-in clock runs at 0.6× file time from 11:59:59.
`validate_csv.py` scores a run against it (maximum matching of events to rows, same class,
preferring the same direction, plus counts per 10 minutes):

```bash
python detect_bikes.py vid.mp4 --events vid_events.jsonl     # on the board, production shape
python validate_csv.py vid_events.jsonl
```

On the board (yolo11s uint8 480×960, stride 1, file mode, 19.5 fps), 2026-09-27:

| | truth | counted | precision | recall | direction right | \|count error\| per 10 min |
|---|---|---|---|---|---|---|
| car | 285 | 284 | 98.9% | 98.6% | 100% | 5.3% |
| pedestrian | 333 | 320 | 90.3% | 86.8% | 95.5% | 8.1% |
| two-wheeler (bicycle + scooter) | 94 | 90 | 94.4% | 90.4% | 98.8% | 8.5% |
| **all** | **712** | **694** | **94.4%** | **92.0%** | | |

Before the four changes below it was 93.8% / 89.7% (cars 94.4% recall, two-wheelers 81.9%).
Each was found by replaying the board's own detections for the hour through the pipeline, and
each also improves a replay of the laptop's f32 model (F1 91.0% → 93.1%), with both `expect: 0`
windows in `groundtruth.yaml` still at zero:

- **`motion.min_bike_heights_per_s: 1.1`.** Most cyclists cross this line near the camera,
  where a *bicycle* box moves 1.0–1.5 heights/s; the shared 1.5 gate rejected 11 real
  riders. A fast `person` is still promoted only at 1.5 (the dog walker is 1.19 and 1.28).
- **`motion.bike_companion_frames: 2`.** A person at riding pace with a bicycle track beside it
  is a cyclist whose bicycle crossing was missed, not a scooter: `rider(person)` went from 31
  counted for 10 real scooters to 18, bicycle recall from 58% to 82%.
- **`motion.hold_seconds: 120`.** Queued cars stand on the line for up to ~75 s; at 30 s their
  held crossing had expired by the time they drove off (10 cars).
- **`keep_vehicle_height` on the far-pavement zone.** A van in the far lane is tall enough for
  its box centre to sit over the pavement, and was relabelled a pedestrian.

What is left: pedestrians are the weakest class, and undercount when it is busy (57 of 68 in
the busiest 10 minutes) — groups boxed as one, and tracks that break at the line. Scooters are
still over-counted (18 for 10), from cyclists with no bicycle box at all.

### The counting line was chosen from data

Placing it by eye, perpendicular to the painted lane markings, looked obviously right and
scored 18/24 on extracted rider trajectories. Riders are lost behind the scaffolding pole in
the lower left, so the line was instead swept across all 24 trajectories extracted from the
labelled windows and placed where it intercepts the most of them. If the camera or the scene
furniture changes, redo this rather than re-eyeballing it — the reproduction steps are in
`calibrate.py --speeds` plus the trajectory dump.

### Stride

Stride is less forgiving than expected: at stride 5+ crossings start being lost, because a
rider is present for only a handful of processed frames and cannot establish a track on both
sides of the line. **Stride 3 (5 inferences/sec) is the floor for reliable counting.** It is
the first thing to re-verify with `calibrate.py` if the SBC cannot keep up.

## On the Radxa Cubie A7Z (Allwinner A733) — the deployment target

This is the board it runs on, and it now runs **real-time**. The starting point was 650 ms a
frame (1.5 fps); it is 126 ms (7.9 fps), against a 133 ms budget at stride 2.

| step | ms/frame | note |
|---|---|---|
| yolo11s `.pt`, PyTorch, square 640 | 650 | where this started |
| → OpenVINO instead of PyTorch | 560 | same model, same size |
| → **rectangular input** (192×640, not 640×640) | 175 | biggest single win |
| → yolo11n instead of yolo11s | 68 | |
| → **`f16` inference precision** | 47 | native `asimdhp` kernels |
| → back up to 288×960 for accuracy | 104 | spends the win where it counts |
| → +letterbox, NMS, in-graph preprocessing | **126** | end to end, `--bench` |

Accuracy went *up* at the same time — 15/21 on the ground-truth clips against 9/21 for the
original yolo11s config, because the speed buys a lower stride and a higher input resolution,
and both matter far more here than model size.

### The four things that mattered

**Rectangular input, derived from the ROI.** The crop is a 1144×309 strip, 3.7:1. Fed to a
square network, ~70% of the multiply-accumulates land on grey letterbox padding. Exporting at
288×960 does identical work on identical pixels for a third of the cost. `export_openvino.py`
computes that shape from the polygon in `config.yaml`, so redrawing the ROI just means
re-running it — there is no hand-tuned constant to forget.

**`f16` inference.** These cores carry `asimdhp` (native half-precision), and OpenVINO's ARM
backend has kernels for it: ~1.3× for free, with no measurable accuracy change.

**INT8 is a trap here — it is *slower*.** 55 ms against 47 ms for f16 on yolo11n, despite the
CPU advertising `asimddp`. OpenVINO's ARM plugin has no tuned int8 path for these shapes, so
quantisation buys nothing and costs accuracy. Measured, not assumed; do not re-try it without
re-measuring.

**Big cores only.** This is 6× Cortex-A55 + 2× Cortex-A76. OpenVINO splits work evenly across
whatever cores it is given, so adding the six slow cores makes the two fast ones *wait*:

| cores | threads | ms |
|---|---|---|
| all 8 | 8 | 55.7 |
| **6,7 (A76 only)** | **2** | **47.0** |
| 0-5 (A55 only) | 6 | 64.5 |

`model.cpu_affinity: auto` picks the fastest cluster by reading `cpuinfo_max_freq`. The
detector builds its OpenVINO runtime inside a worker thread that has already set its own
affinity — the thread pool inherits it — so the *caller* stays unpinned and video decode runs
on the little cores in parallel.

One thing to set outside this repo: the CPU governor is `ondemand`, which leaves the A76 pair
idling at 416 MHz. For a deployment, pin it — `tools/cpu-performance.service` does that at
every boot (installed and enabled on the a7s board):

```bash
sudo cp tools/cpu-performance.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now cpu-performance.service

# or once, until the next reboot
echo performance | sudo tee /sys/devices/system/cpu/cpufreq/policy*/scaling_governor
```

### The NPU: this is where the performance is

**The NPU runs YOLO11 at 59 fps.** `bikecount/npu.py` drives it from Python through `ctypes`,
and `--model something.nb` uses it:

| | input | ms/frame | fps | counted /21 |
|---|---|---|---|---|
| **yolo11n uint8 on the NPU** | **288×960** | **17.0** | **59** | **18** |
| the same model, f16 on 2× A76 | 288×960 | 126 | 8 | 15 |

Same weights, same input size, **7.4× faster** — and counting slightly better, because the
speed buys stride 1 where the CPU could only afford stride 2. (Of those 18, two windows
overcount, so it is ~16 true positives against 15; call the accuracy equal and the speed free.)

It is a VeriSilicon VIP9000 behind the VIPLite runtime — *not* Rockchip, so RKNN tooling is
irrelevant. Radxa's apt repo carries no userspace for it (`allwinner-prebuilt-extra` is an
empty placeholder), which is what makes it look unavailable; the libraries are in Allwinner's
model zoo tarball instead. **See [NPU.md](NPU.md)** — where the libraries come from, how the
`ctypes` binding works, the full conversion route, and the four silent failure modes in
Allwinner's own scripts.

The conversion needs Allwinner's ACUITY toolkit, which is a browser download (a Synology
share) and cannot be scripted. Everything after that is `export_npu.py` plus three commands.

Two things about the model itself are worth keeping:

* **The head is cut off and decoded on the CPU.** 8-bit quantisation of the head's box
  arithmetic costs far more accuracy than the time it saves, so the `.nb` emits raw per-scale
  grid tensors and `_DECODERS` in `detectors.py` finishes the job. This is Allwinner's own
  recipe, not a shortcut.
* **Calibrate on this camera.** The int8 scales are fitted to whatever frames you feed the
  quantiser; `export_npu.py` writes them from the deployment footage rather than from COCO.

### The GPU: usable, but not worth it

**Imagination PowerVR BXM-4-64.** OpenCL 3.0 and Vulkan both work (`clinfo` sees the device,
`cl_khr_fp16` is present). But it reports **1 compute unit at 600 MHz** — on the order of
100–150 GFLOPS fp16, roughly what the two A76 cores already deliver on this model, and an
order of magnitude below the NPU. The obvious route in, OpenCV's DNN OpenCL target, is a dead
end on the installed OpenCV 5.0: the new graph engine silently ignores `setPreferableTarget`
("Targets are not supported by the new graph engine for now") and runs on the CPU regardless —
the three "backends" benchmark to within 4% of each other, which is how that was caught. With
the NPU working there is no reason to pursue it.

**Hardware video decode was measured and rejected as pointless.** The Cedar VPU is available
through GStreamer (`omxh264dec`), but software decode of this 720p stream costs 5.7 ms a
frame against 104 ms of inference. There is nothing to win.

### Deploying

The board does not need torch or Ultralytics — only `openvino`, `opencv-python`,
`supervision`, `numpy` and `PyYAML`. Export on a workstation, copy the directory over:

```bash
# on a workstation, with ultralytics installed
python export_openvino.py --weights yolo11n.pt --long 960   # -> yolo11n_rect_openvino_model/
scp -r yolo11n_rect_openvino_model/ radxa:~/kkc_bicycles/

# on the board
python detect_bikes.py video.mp4 --events events.jsonl
python detect_bikes.py video.mp4 --duration 40 --bench      # confirm the budget
```

`--bench` prints the per-stage breakdown and compares it against the real-time budget for the
configured stride. Run it on the board after any config change.

### Choosing model and stride

Everything measured on the board at stride 1 unless noted, with the accuracy each config
scores on the ground-truth clips. **`counted` is not the same as `correct`** — read the
per-window `OVERCOUNT` rows too, which is what the last column is for.

| backend | model | input | ms | counted /21 | false positives |
|---|---|---|---|---|---|
| **NPU** | **yolo11n uint8** | **288×960** | **17** | **18** | 2 |
| NPU | yolo11s uint8 | 288×960 | 32 | 20 | 4, incl. the "pushing bikes" guard |
| NPU | yolo11n uint8 | 320×1152 | 18 | 15 | 2 |
| NPU | yolov5s uint8 (Allwinner's prebuilt) | 640×640 | 31 | 9 | 0 |
| CPU | yolo11n f16 | 288×960 | 126 | 15 | 0 |
| CPU | yolo11n f16 | 192×640 | 47 | 12 | 0 |
| CPU | yolo11s f16 | 192×640 | 104 | 12 | 0 |
| CPU | yolo11s f16 | 288×960 | 231 | 12 | 0 |

This table predates the riding gate. **The deployed model is now
`models/yolo11s_6_uint8_a733_480x960.nb`** — see the 2026-09-26 entry in [NPU.md](NPU.md),
where yolo11s was retested and won once pushing bikes stopped counting.

yolo11s is the trap worth knowing about: on the raw total it looks better at 20/21, but four
of those come from the *"2 people pushing bikes, expect 0"* window — the regression test that
exists because only people **riding** should count. Net of false positives both models land on
~16 true positives, and yolo11n does it in half the time.

Going the other way is no better: 320×1152 is above the crop's native 1144 px, and upscaling
past native just costs false positives — it finds nothing extra and counts 15 with two
overcount windows. 288×960 is the sweet spot on this camera.

Two useful fallbacks: a yolo11n OpenVINO export at 288×960 with `--stride 2` is the fastest
CPU-only configuration (no NPU libraries needed; `export_openvino.py --long 960`), and `192×640` at stride 1 trades accuracy
for latency and thermal headroom.

## Speed on the dev laptop (Framework 13, Ryzen 7 7840U + Radeon 780M)

For reference; the board numbers above are the ones that matter.

All measured on this machine, yolo11s at imgsz 640 on the real ROI crop, 8 physical cores:

| backend | ms/frame | vs PyTorch |
|---|---|---|
| **OpenVINO, CPU** | **16.5** | **1.7× faster** |
| PyTorch, CPU (default) | 27.7 | baseline |
| NCNN + **Vulkan on the 780M iGPU** | 40.7 | 1.5× *slower* |
| NCNN, CPU | 45.2 | 1.6× slower |

### Vulkan / iGPU: supported, but not worth it here

Ultralytics and PyTorch have no Vulkan backend. The route that exists is NCNN, which Ultralytics
exports to (`format=ncnn`) and which has a real Vulkan compute backend — and the pip `ncnn`
wheel is Vulkan-enabled and does pick up the 780M (RADV, fp16 cooperative matrix).

It just doesn't pay off: **40.7 ms on the iGPU versus 16.5 ms on the CPU with OpenVINO.** Vulkan
buys only ~10% over NCNN's own CPU path. Two reasons — the iGPU shares LPDDR5 bandwidth with
the CPU, so there is no memory advantage to move work there, and per-layer dispatch overhead is
significant for a model this small. Meanwhile OpenVINO's x86 kernels exploit the AVX-512 this
Ryzen actually has. Discrete-GPU intuitions do not carry over to an APU.

(ROCm would be the "proper" AMD GPU path, but the 780M / gfx1103 is not officially supported and
needs `HSA_OVERRIDE_GFX_VERSION` hacks. Given Vulkan already lost to the CPU by 2.5×, it is not
worth chasing.)

**Use OpenVINO instead** — identical accuracy (13/21, verified with the full calibration run),
1.5× faster end-to-end:

```bash
yolo export model=yolo11s.pt format=openvino imgsz=640     # once, creates yolo11s_openvino_model/
python detect_bikes.py video.mp4 --model yolo11s_openvino_model/
```

### CPU cores

Cores matter, and the script already uses all of them — there is nothing to turn on.

| threads | OpenVINO ms | |
|---|---|---|
| 1 | 90.0 | |
| 2 | 47.1 | |
| 4 | 26.3 | |
| **8** | **15.8** | all physical cores — the default |
| 16 | 15.8 | SMT adds nothing |

Scaling is near-linear to 8 threads, then flat: hyperthreading gives no benefit because the
kernels are already saturating each core's vector units. To *limit* cores (say, to keep a
laptop responsive), set `OMP_NUM_THREADS` and pin with `taskset`:

```bash
OMP_NUM_THREADS=4 taskset -c 0-3 python detect_bikes.py video.mp4
```

Note that `torch.set_num_threads()` alone does **not** restrict PyTorch here — it appears to
work but the underlying oneDNN parallelism keeps using every core, which makes naive thread
benchmarks read as flat. Pin with `taskset` if you want a real measurement.

One caveat that matters more than any of this: **a single stream cannot use the machine well.**
Inference is one sequential 16 ms step per frame. If you want throughput — sweeping
configurations, or processing several recordings — run multiple processes over different time
ranges or files. That scales far better than any per-frame tuning.

## If accuracy needs to improve

In the order likely to pay off:

1. **Fine-tune on this camera.** The misses are detection failures on small, motion-blurred
   riders at this specific angle and lighting. A few hundred labelled crops from these two
   recordings would address that far more effectively than any threshold. Everything else here
   is already at its ceiling — bigger stock models measurably made it worse.
2. **A second counting line.** Riders using the main road rather than the bike path never
   cross the current line. If they should be counted, add a line and sum them.
3. **Better occlusion handling in the tracker.** Riders passing behind the scaffolding pole
   get a new track id, and a track that changes identity mid-crossing cannot be counted.
   Re-associating tracks across a short gap by predicted position would recover some.

## Files

| | |
|---|---|
| `detect_bikes.py` | main CLI |
| `roi_tool.py` | click out the ROI polygon and counting line |
| `debug_viewer.py` | play a debug video, jumping between the crossings it counted |
| `calibrate.py` | score configurations against `groundtruth.yaml` |
| `validate_log.py` | score a *recorded* run's log against `groundtruth.yaml` |
| `replay_log.py` | re-count a recorded log under a different config, without the model |
| `export_openvino.py` | export weights at the ROI's aspect ratio — the main speed win |
| `NPU.md` | how to run on the A733 NPU, and how to compile a model for it |
| `NETWORK.md` | the camera cable: static addressing on both ends, and the test stream |
| `stream_test_video.py` | serve a recording as an RTSP/MJPEG camera, to test the stream path |
| `export_npu.py` | ONNX export + head surgery + calibration set, for the ACUITY toolchain |
| `bikecount/npu.py` | VIPLite `ctypes` binding — the NPU runtime, no build step |
| `config.yaml` | ROI, model, tracking, speed gate, stride |
| `groundtruth.yaml` | hand-labelled clips — the only real accuracy measure we have |
| `bikecount/source.py` | frame decimation via `grab()`, threaded prefetch, live-stream handling |
| `bikecount/roi.py` | polygon, cropping, containment |
| `bikecount/detectors.py` | detector protocol; OpenVINO, NPU and Ultralytics backends |
| `bikecount/tracker.py` | greedy IoU/centroid tracker with per-track speed |
| `bikecount/pipeline.py` | detect → filter → track → count, and the deferred gates |
| `bikecount/debug_view.py` | all rendering, kept out of the deployment path |
| `bikecount/debug_log.py` | the JSONL sidecar: video-frame ↔ source-frame, and every crossing |

### Weights kept in the repo

Only the deployed model is kept: `models/yolo11s_6_uint8_a733_480x960.nb` (see the 2026-09-26
entry in [NPU.md](NPU.md)). Everything else that was benchmarked — the other `.nb` builds, the
OpenVINO exports, the `.pt` sources — was deleted. All of it is regenerable: `yolo11*.pt`
re-download on first use, OpenVINO exports are one command (`python export_openvino.py
--weights yolo11n.pt --long 960`), and `.nb` builds follow [NPU.md](NPU.md).

Add every clip you verify by eye to `groundtruth.yaml`. It is what stops a tuning change from
quietly trading one cyclist for another.
