# Running on the NPU (Allwinner A733 / VeriSilicon VIP9000)

The A733's NPU is reachable from Python, and it is fast: **yolo11n at 288×960 runs
end to end in 17.0 ms — 59 fps.** The same model at the same input size takes 126 ms on the
tuned CPU path, so this is 7.4× faster, and it counts slightly better as well.

Everything below was verified on the board, not taken from documentation.

## What the stack actually is

It is **not** Rockchip. `rknn-toolkit2` and anything RKNN is irrelevant here. The chain is:

```
your .pt  →  ONNX (head stripped)  →  ACUITY/pegasus quantise+compile  →  .nb  →  VIPLite  →  /dev/vipcore
             (workstation)            (x86 Docker, Allwinner)            (board)
```

* **`.nb`** — an ACUITY "Network Binary Graph". Not ONNX; no ONNX runtime will load it.
* **VIPLite** — the on-board runtime, a 40-function C API. On A733 that is `libNBGlinker.so`
  (the API) plus `libVIPhal.so` (the kernel shim). Version here: `2.0.3.2-AW-2024-08-30`.
* `bikecount/npu.py` drives that C API through `ctypes` — no compiler, no extension module,
  nothing to build on the board.

Note the A733 differs from Allwinner's other parts: T527 and friends ship
`libVIPlite.so` + `libVIPuser.so`, A733 ships `libNBGlinker.so` + `libVIPhal.so`. Picking the
wrong directory out of the model zoo gets you a load error, not a wrong answer.

## Getting the runtime libraries

The kernel driver is already loaded on a stock Radxa image (`/dev/vipcore` exists, and
`vipcore` shows in `lsmod`). Only userspace is missing, and Radxa's apt repo does not carry
it — `allwinner-prebuilt-extra` is an explicitly empty placeholder. Take it from Allwinner's
model zoo instead:

```bash
wget https://dl.radxa.com/cubie/allwinner-model-zoo.tar.gz
tar xzf allwinner-model-zoo.tar.gz
Z=$(ls -d awnpu_model_zoo-*)

# the two libraries the board needs
scp "$Z"/common/npuruntime/lib_linux_aarch64/A733/*.so radxa:~/kkc_bicycles/npu_lib/

# a prebuilt A733 model, handy for checking the NPU works before converting anything
scp "$Z"/examples/yolov5/model/yolov5s_rt_uint8_a733.nb radxa:~/kkc_bicycles/models/
```

Then, on the board:

```bash
python detect_bikes.py video.mp4 --model models/yolov5s_rt_uint8_a733.nb
```

`model.npu_libs` in `config.yaml` points at the library directory; setting `LD_LIBRARY_PATH`
works too.

## Converting your own model

`export_npu.py` does the workstation side; the quantise/compile happens in Allwinner's ACUITY
container. The whole route has been run end to end and produces a working `.nb`.

**1. Get the ACUITY Docker image** — from Allwinner's netdisk,
<https://netstorage.allwinnertech.com:5001/sharing/Mh23BhPHq>, take `docker_images_v2.0.x.zip`
(2.9 GB) → `ubuntu-npu_v2.0.10.2.tar` (7.9 GB unpacked). **A733 needs the v2.0.x image**; T527
uses v1.8.13. It is a Synology share and needs a browser.

```bash
docker load -i ubuntu-npu_v2.0.10.2.tar
mkdir -p ~/docker_data
docker run --ipc=host -itd -v ~/docker_data:/workspace --name acuity ubuntu-npu:v2.0.10.2 /bin/bash
```

**2. Prepare the model and calibration set** (on the host):

```bash
python export_npu.py --weights yolo11n.pt --long 960 --zoo path/to/awnpu_model_zoo-* --out npu_build
cp -r npu_build ~/docker_data/
```

That exports ONNX at the ROI's aspect ratio, cuts the detection head off at the six convolutions
Allwinner's recipe cuts at, and writes 64 letterboxed crops from your own footage as the
calibration set. Calibrating on this camera rather than COCO is what keeps the int8 model
usable — it fits the quantisation scales to the activation ranges this angle and lighting
actually produce.

**3. Quantise and compile** (inside the container):

```bash
export ACUITY_PATH=$(echo ~/acuity-toolkit-whl-*)/bin
export VIV_SDK=/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools
cd /workspace/npu_build
./pegasus_import.sh         yolo11n_6
./pegasus_quantize.sh       yolo11n_6 uint8 64
./pegasus_export_ovx_nbg.sh yolo11n_6 uint8 a733   # <- a733, not t527
```

**4. Copy the `.nb` to the board** and run it — the decoder is detected automatically:

```bash
scp npu_build/model/yolo11n_6_uint8_a733.nb radxa:~/kkc_bicycles/models/
python detect_bikes.py video.mp4 --model models/yolo11n_6_uint8_a733.nb
```

### Four things that will waste an afternoon

All four are handled by `export_npu.py`, but they are silent failures if you drive the scripts
by hand:

* **`ACUITY_PATH` must end in `/bin`.** The readme in the image says so; the model zoo's own
  examples do not.
* **`VIV_SDK` must be exported.** The export script passes `--viv-sdk ${VIV_SDK}` and never
  sets it. It is defined in the container's `/root/.bashrc`, which a non-interactive
  `docker exec` does not source, so it arrives empty and pegasus fails with
  `argument --viv-sdk: expected one argument`.
* **`../model/` must exist before compiling.** `pegasus_export_ovx_nbg.sh` copies the finished
  `.nb` there and then unconditionally `rm -rf`s its workspace. If the directory is missing,
  the compile *succeeds* and the output is deleted one line later — with no error that looks
  like the cause.
* **Layout matters.** The scripts `pushd` into a directory named after the model and expect
  `<name>/<name>.onnx`. `config_yml.py` is resolved as `$(dirname $0)/config_yml.py` *after*
  that `pushd`, so with a relative invocation it is looked for inside the model directory.

## Two things that cost real time in Python

Both are already handled, but they are easy to reintroduce:

* **Threshold in logit space.** `sigmoid(obj) * sigmoid(cls)` can only clear the confidence
  threshold if both factors do, so comparing raw logits against `logit(conf)` is exact — and
  it avoids ~1.5M exponentials a frame. Doing the naive thing costs 43 ms, more than twice the
  entire NPU inference.
* **Write as little as possible into NPU memory.** `Tensor.array` maps uncached device memory.
  The grey letterbox border never changes, so it is painted once at startup and only the image
  rectangle is written per frame.

## Known-good numbers on this board

End to end through the pipeline — letterbox, NPU, head decode and NMS — at stride 1:

| model | input | ms | fps | counted /21 |
|---|---|---|---|---|
| **yolo11n uint8, our conversion** | **288×960** | **17.0** | **59** | **18** |
| yolov5s uint8, Allwinner's prebuilt | 640×640 | 31.3 | 32 | 9 |
| *the CPU path*, yolo11n f16 on 2× A76 | 288×960 | 126 | 8 | 15 |

`vip_run_network` alone is 20.3 ms for yolov5s at 640×640; the rest is letterboxing and the
CPU-side head decode.

### After the ROI grew to the whole street (2026-09-25)

The ROI was redrawn to cover the street and both pavements, and counting widened from riders to
all traffic. The crop went from a ~3.6:1 strip to 1237×619 (2:1), so the 288×960 model now
letterboxes it down to 0.47× — a far-pavement pedestrian ends up ~25 px tall, drops out for a
couple of seconds at a time, and the track breaks exactly where it matters. The fix is the same
as before: re-export at the ROI's aspect (`export_npu.py --long 960` / `--long 1152`, compiled
exactly as above). On the busy daytime clip, 5 minutes, same config:

| model | input | scale | ms/frame | person boxes | two-wheeler boxes | riders /22 (calibrate) |
|---|---|---|---|---|---|---|
| yolo11n uint8, old export | 288×960 | 0.47 | 32.6 | 1324 | 284 | 14, 1 false (dog walker) |
| yolo11n uint8 | 480×960 | 0.78 | 38.2 | 1356 | 307 | 11 |
| **yolo11n uint8** | **576×1152** | **0.93** | **46.2** | **1937** | **620** | **11** |

`576x1152` was the deployed one until yolo11s replaced it (below): the rider score is a wash on 22 labelled riders, but it
sees 46% more people and twice the two-wheeler boxes, which is what counting all traffic needs.
In production shape (no rendering) it runs 20 fps end to end against the camera's 15, so
~33% headroom. `480x960` is kept in `models/` as the fallback if the board gets busier.

For reference, the original code with the original ROI scores 12/22 on the same windows
from file; the 18/21 in the table above was measured against an earlier, 14-window ground truth.

### Bigger is not better — check the guard window

The NPU has enough headroom to run yolo11**s** at 288×960 (31.9 ms, still comfortably
real-time), and on the raw total it looks better: 20 counted against 18. It is not better.
Four of those came from the *"2 people pushing bikes, expect 0"* window — the regression test
that exists precisely because only people **riding** should count. yolo11s calls them riders.

Netting the false positives out, both models land on ~16 true positives, and yolo11n does it
in half the time without breaking the guard. Read the per-window rows, not the total:
`calibrate.py` flags `OVERCOUNT` for a reason, and a model that finds more of everything is
not the same as a model that counts better.

### yolo11s at 480×960 — deployed (2026-09-26)

Retested once the riding gate measured pace in box heights per second, which is what now keeps
people pushing bikes out — the reason yolo11s was turned down above. Built exactly as yolo11n
(`export_npu.py --weights yolo11s.pt --long 960`, then the same three `pegasus_*` steps, uint8,
the same 64 calibration crops as `roi2_n960`). On the board, production shape, same code:

| model | input | detector ms | end to end | riders /22 (calibrate) | guards |
|---|---|---|---|---|---|
| yolo11n uint8 | 576×1152 | 44.0 | 19.5 fps | 10 | 0 / 0 |
| yolo11s uint8 | 576×1152 | 72.9 | 12.1 fps | 13 | 0 / 0 |
| **yolo11s uint8** | **480×960** | **54.5** | **14.6 fps** | **15** | **0 / 0** |

`guards` are the two windows that must count 0 (people pushing bikes, a dog walker). yolo11s
also leaves fewer holes in a track: frames detected within a track's life went from 73% to
76–82% over two 3-minute clips, and tracks restarted after a gap from 7 to 1.

480×960 runs at ~68 ms a frame against the camera's 67, so on the live stream about one frame
in 40 is skipped. `adaptive_dt` measures the real interval and the tracker bridges gaps, so
this is harmless; the extra ~14 ms beyond the NPU is decode and Python, if headroom is ever
needed. 576×1152 is too slow at 12 fps.
