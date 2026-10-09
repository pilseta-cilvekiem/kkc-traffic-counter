# The camera link — static addressing, and streaming a test video over it

The counter reads its video from a camera over an ethernet cable. There is no DHCP server on
that cable, so **both ends must be given fixed addresses** or nothing on the link comes up.
This file is the settings, the commands that applied them, and how to verify the result.

Two situations use the same subnet, which is the point — the board's configuration does not
change between them:

```
bench / development                          deployment
  laptop 192.168.1.1  ──cable──  a7s .2        camera 192.168.1.3 ──cable──  a7s .2
  (serves a recorded video as RTSP)            (real RTSP stream)
  a7s also on wifi 192.168.88.18               a7s on wifi for ssh, or headless
```

## What is configured, right now

| | laptop (Framework 13) | Radxa Cubie A7Z (`a7s`) |
|---|---|---|
| interface | `enx207bd2af38fc` (USB-C ethernet) | `eth0` |
| NetworkManager profile | `Wired connection 1` | `cam-link` |
| address | `192.168.1.1/24` | `192.168.1.2/24` |
| gateway | `192.168.1.2` | *none* (`ipv4.never-default yes`) |
| default route | wifi — see below | wifi |
| other interface | wifi `192.168.88.167` | wifi `192.168.88.18` |

Both machines run NetworkManager, so `nmcli` is the tool on both.

### Laptop

```bash
nmcli connection modify "Wired connection 1" \
    ipv4.method manual \
    ipv4.addresses 192.168.1.1/24 \
    ipv4.gateway 192.168.1.2
nmcli connection up "Wired connection 1"
```

The gateway pointing at the board is deliberate but *not* load-bearing. NetworkManager gives
a static profile a route metric of 20100 against the wifi's DHCP metric of 600, so the wifi
default route keeps winning and the laptop's internet is unaffected:

```
default via 192.168.88.1 dev wlp1s0            metric 600     <- internet goes here
default via 192.168.1.2  dev enx207bd2af38fc   metric 20100
192.168.1.0/24           dev enx207bd2af38fc   metric 100      <- the camera link
```

Drop it with `nmcli connection modify "Wired connection 1" ipv4.gateway ""` if you would
rather not have a second default route at all; the link works either way, because everything
on it is in the same `/24` and needs no gateway to be reached.

### Board

```bash
# the stock profile is DHCP, and on a cable with no DHCP server it sits forever in
# "connecting (getting IP configuration)" — which looks like a dead cable and is not
sudo nmcli connection modify "Wired connection 1" connection.autoconnect no

sudo nmcli connection add type ethernet ifname eth0 con-name cam-link \
    ipv4.method manual \
    ipv4.addresses 192.168.1.2/24 \
    ipv4.never-default yes \
    ipv6.method disabled \
    connection.autoconnect yes \
    connection.autoconnect-priority 10
sudo nmcli connection up cam-link
```

`never-default` is the setting that matters: without it a gateway on this profile would take
over the board's default route, and ssh over wifi would drop the moment the cable came up.
`autoconnect yes` is what makes it survive a reboot — check with `nmcli -f NAME,AUTOCONNECT
connection show`.

### Verifying

```bash
ip -br addr show eth0                    # on the board: 192.168.1.2/24, state UP
ping -c3 192.168.1.2                     # from the laptop: ~3.4 ms over the cable
ssh a7s-cable                            # ssh over the cable rather than the wifi
```

`~/.ssh/config` on the laptop has both routes to the same board, which is worth keeping —
when the cable configuration is what you are debugging, you still want a way in:

```
Host a7s      HostName 192.168.88.18   User radxa   # over the office wifi
Host a7s-cable HostName 192.168.1.2    User radxa   # over the camera cable
```

### Changing the subnet for a real camera

Most IP cameras ship on a fixed factory address — `192.168.1.10`, `192.168.1.64` (Hikvision),
`192.168.1.108` (Dahua) are the common ones — so `192.168.1.0/24` is usually already the
right subnet and only the camera's own address has to be found. If the camera insists on a
different one, move the *board* to the camera's subnet rather than the other way round:

```bash
sudo nmcli connection modify cam-link ipv4.addresses 192.168.0.2/24
sudo nmcli connection up cam-link
```

To find a camera whose address you do not know, scan the link from the board:

```bash
sudo nmap -sn 192.168.1.0/24            # or: ip neigh, after pinging the broadcast address
sudo nmap -p 554,80,8000 192.168.1.0/24 # 554 = RTSP, and the web UI usually confirms the model
```

## Streaming a test video from the laptop

`stream_test_video.py` publishes one of the recordings as a live stream, looping it in real
time. This is how the board's stream path gets exercised before there is a camera.

```bash
# on the laptop — RTSP, which is what a real camera speaks
python stream_test_video.py "210235...134146....mp4"
# [info] RTSP  rtsp://192.168.1.1:8554/cam

# on the board
python detect_bikes.py rtsp://192.168.1.1:8554/cam --events events.jsonl
```

Three modes, in the order you should reach for them:

| mode | URL | needs | when |
|---|---|---|---|
| `--mode rtsp` (default) | `rtsp://192.168.1.1:8554/cam` | ffmpeg + mediamtx | the realistic one — same protocol as the camera |
| `--mode mjpeg` | `http://192.168.1.1:8080/stream.mjpg` | nothing but Python | when RTSP is the thing that is broken, or as a second camera type |
| `--mode mpegts` | `tcp://192.168.1.1:8554` | ffmpeg | one client, no server at all — isolates "is it the network or the RTSP server" |

`mediamtx` is a single static binary and is downloaded into `tools/` on first use
(`--no-download` refuses to). The RTSP mode transcodes to H.264 with a two-second GOP by
default: the recordings are HEVC with a long GOP, and a client joining mid-GOP then waits
~2.3 s and logs a screenful of `Could not find ref with POC` first. With the transcode a
client is showing frames **1.2 s** after connecting. `--codec copy` skips the transcode if
you specifically want to test HEVC.

Useful flags: `--start 585` (skip to an interesting part of the recording), `--fps`,
`--no-loop`, `--port`, `--path`, `-v` to see what ffmpeg and mediamtx are actually saying.

### What this was measured to do

Laptop → cable → board, RTSP H.264 720p at 15 fps, yolo11n on the NPU at stride 1:

```
[info] live stream: 1280x720 @ 15.0fps, stride=1 -> 15.0 inferences/s
[done] processed 667 frames in 45.0s (14.8 fps) | IN=1 OUT=0
[bench] per-frame mean ms: {'crop': 0.04, 'infer': 23.24, 'track': 0.24}
[bench] 23.5ms/frame vs 66.7ms budget at stride 1 -> REAL-TIME OK
```

14.8 of the camera's 15 fps processed, and the counting works on a stream exactly as it does
on a file. Pulling the plug on the publisher mid-run reconnected on its own and carried on
counting — that is the behaviour the deployment needs, and it is tested rather than assumed.

Add `--debug-out live.mp4` to record what the detector saw and step through the crossings
afterwards with `debug_viewer.py`. On a stream the sidecar `live.log` is the only thing tying
video frames back to real time: the run above wrote 892 frames for 60 s of a 15 fps camera,
so the video's own clock and the source's have already drifted apart by the end.

## Reading a live stream in the counter

`detect_bikes.py` takes a URL wherever it takes a file, and switches to live handling
automatically for `rtsp://`, `http(s)://`, `udp://`, `tcp://`, `/dev/videoN` and a bare
camera index. Live handling differs from file handling in four ways that all matter:

* **Stale frames are dropped.** The camera does not wait for us. If inference falls behind,
  the prefetch queue throws away everything but the newest frame, so latency stays flat and
  "too slow" shows up as a lower frame rate instead of an ever-growing backlog.
* **Timestamps come from the wall clock.** With frames being dropped by both the network and
  us, `frame index / fps` is a fiction, and every speed in the pipeline is px per *real*
  second.
* **The tracker's frame interval is measured, not assumed** (`runtime.adaptive_dt`). The
  counting gate is a speed threshold at 80 px/s, so a wrong `dt` silently changes what counts.
* **A dropped stream reconnects** instead of ending the run (`--no-reconnect` to opt out).

`--start` is ignored on a live source — there is no past to seek to — and `--duration`
becomes wall-clock seconds. The progress line gains a second number, `fps in`, which is what
the camera actually delivered: if `fps proc` sits well below it, the board is not keeping up
and `--stride` should go up.

Relevant `config.yaml` keys (all under `runtime:`):

| key | default | |
|---|---|---|
| `live` | `auto` | `true`/`false` forces it — e.g. an `http://` URL that serves a finite `.mp4` |
| `rtsp_transport` | `tcp` | UDP loses packets on a long link, and a torn frame looks exactly like a detector failure |
| `reconnect` / `reconnect_delay` | `true` / `2.0` | |
| `stream_fps` | `15.0` | assumed rate when the camera advertises none |
| `adaptive_dt` | `true` | measure the frame interval from timestamps on live sources |

### Pointing it at a real camera

Camera RTSP URLs are vendor-specific; the two common shapes are

```
rtsp://user:password@192.168.1.64:554/Streaming/Channels/101      # Hikvision
rtsp://user:password@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0   # Dahua
```

Check the URL with ffprobe from the board before involving the detector — it separates
"camera or network problem" from "counter problem" in one command:

```bash
ffprobe -rtsp_transport tcp -v error -show_entries stream=codec_name,width,height,r_frame_rate \
    -of default=nw=1 rtsp://user:pass@192.168.1.64:554/Streaming/Channels/101
```

Then draw the ROI against the real camera — the polygon and counting line in `config.yaml`
are specific to a camera's mounting and zoom, so they must be redrawn when the view changes:

```bash
python roi_tool.py rtsp://... --dump-frame frame.png    # headless: grabs one live frame
```

Prefer the camera's **sub-stream** if its main stream is 4K: this pipeline crops to the ROI
and infers at 288×960, so extra source resolution costs decode time and buys nothing.

### Running it as a service

`tools/bikecount.service` starts the counter with the board. It reads the stream URL from
`CAMERA_URL` in `.env` (falling back to the laptop's test rig, `rtsp://192.168.88.28:8554/cam`),
and sends crossings to Supabase (`--report`) and nowhere else — no events file, nothing
per-crossing in the journal — to keep writes to the SD card down.

```bash
sudo cp tools/bikecount.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now bikecount.service
journalctl -u bikecount -f
```

To move to the real camera, change `CAMERA_URL` in `.env` and
`sudo systemctl restart bikecount`.

A camera that is not up yet, or drops later, is waited for by the source itself — it retries
every `reconnect_delay` and logs once when the stream comes back. `Restart=always` is a
backstop for crashes, not the reconnect mechanism, and a service restart should be rare
enough to be worth investigating.

## When it does not work

| symptom | cause |
|---|---|
| `eth0` stuck in *"connecting (getting IP configuration)"* | it is on DHCP and there is no DHCP server on the cable. That is what `cam-link` fixes. |
| ping works, RTSP does not | the stream, not the link. `ffprobe` the URL; check the camera's port 554 is open, and try `--rtsp-transport udp`. |
| `mediamtx exited: listen tcp :8554: bind: address already in use` | an orphaned mediamtx from a killed run. `ss -ltnp \| grep 8554`, then kill it. |
| RTSP client takes seconds to show anything, `Could not find ref with POC` | joining mid-GOP on a long-GOP stream. Normal; the H.264 transcode in the test rig avoids it, a real camera's I-frame interval decides it. |
| `fps proc` well below `fps in` | the board is not keeping up. Raise `--stride`, or check the CPU governor (`performance`, see README). |
| ssh over wifi dies when the cable comes up | a default route escaped onto the cable profile. `ipv4.never-default yes` on the board's profile. |
| board is unreachable at `192.168.1.2` after a reboot | `nmcli -f NAME,AUTOCONNECT connection show` — `cam-link` must be `yes` and the DHCP profile `no`. |
