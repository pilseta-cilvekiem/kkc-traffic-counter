"""Detector backends.

Everything downstream talks to the `Detector` protocol only, so swapping the Ultralytics
backend for ONNX / OpenVINO / RKNN on the target SBC touches this file and nothing else.
"""

from __future__ import annotations

import os
import queue
import threading
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import supervision as sv
import yaml

from .config import ModelConfig


class Detector(Protocol):
    """Detects objects in a BGR image and returns full-image-coordinate detections."""

    names: dict[int, str]

    def infer(self, image_bgr: np.ndarray) -> sv.Detections: ...


class UltralyticsDetector:
    """YOLO via Ultralytics.

    Accepts `.pt`, `.onnx`, `.engine` or an exported `*_openvino_model/` directory —
    Ultralytics dispatches on the path, so the whole export/optimise path costs us nothing
    here beyond passing a different `weights` value.

    Convenient, but it drags in torch and runs NMS through it. On the ARM board that is dead
    weight; `OpenVinoDetector` is the deployment path. This one stays for training-time work
    and for checking that the fast path agrees with the reference.
    """

    def __init__(self, cfg: ModelConfig):
        from ultralytics import YOLO  # imported lazily: ~2s and pulls in torch

        self.cfg = cfg
        self.model = YOLO(cfg.weights)
        self.names = dict(self.model.names)

        # `half` was renamed to `quantize` in ultralytics 8.4; only pass it when actually
        # enabled so we do not emit a deprecation warning on every single frame.
        self._extra = {"half": True} if cfg.half else {}

    def infer(self, image_bgr: np.ndarray) -> sv.Detections:
        result = self.model.predict(
            image_bgr,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            classes=self.cfg.classes or None,
            device=self.cfg.device,
            verbose=False,
            **self._extra,
        )[0]
        return sv.Detections.from_ultralytics(result)


class OpenVinoDetector:
    """YOLO via the OpenVINO runtime directly — no torch, no Ultralytics.

    Three things make this ~13x faster than the stock Ultralytics/PyTorch path on the
    Allwinner A733, and each is worth stating because none of them is the obvious one:

    * **Rectangular input.** The ROI crop is 1144x310 — a 3.7:1 letterbox strip. A square
      640x640 model spends 70% of its multiplies on grey padding. A 192x640 export does the
      same work on the same pixels for a third of the cost. `export_openvino.py` derives that
      shape from the ROI in `config.yaml`, so it is not a magic number.

    * **f16 inference.** The A55/A76 cores here carry `asimdhp` (native half-precision
      arithmetic), and OpenVINO's ARM backend has fp16 kernels for it. Roughly 1.3x, free.
      Note INT8 is *slower* on this chip despite `asimddp` being present — the ARM plugin
      has no tuned int8 path for these shapes, so it is measured and rejected, not skipped.

    * **Big cores only.** This is a 6x Cortex-A55 + 2x Cortex-A76 part. OpenVINO splits work
      evenly across whatever cores it is given, so adding the six slow cores makes the two
      fast ones *wait*: all 8 cores run at 56ms, the 2 A76s alone at 47ms. Pinning to the big
      cluster is both faster and leaves the little cluster free to decode video.

    The pinning is done by building the runtime inside a worker thread that has already set
    its own affinity: OpenVINO's thread pool inherits the affinity of the thread that creates
    it, and every inference then runs on that same thread. That keeps the *caller* — video
    decode, tracking, drawing — unpinned and on the little cores.
    """

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self._feeds_u8 = False
        self._in_q: queue.Queue = queue.Queue(maxsize=1)
        self._out_q: queue.Queue = queue.Queue(maxsize=1)

        path = Path(cfg.weights)
        xmls = sorted(path.glob("*.xml")) if path.is_dir() else [path]
        if not xmls:
            raise RuntimeError(f"no OpenVINO .xml found in {path}")
        self._xml = xmls[0]

        meta_path = (path if path.is_dir() else path.parent) / "metadata.yaml"
        meta = yaml.safe_load(meta_path.read_text()) if meta_path.exists() else {}
        self.names = {int(k): v for k, v in (meta.get("names") or {}).items()}

        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._worker = threading.Thread(target=self._serve, daemon=True)
        self._worker.start()
        self._ready.wait()
        if self._error is not None:
            raise self._error

    # ------------------------------------------------------------------ worker

    def _serve(self) -> None:
        """Own the OpenVINO runtime end to end, on cores of our choosing."""
        try:
            affinity = self._resolve_affinity()
            if affinity:
                os.sched_setaffinity(0, affinity)

            import openvino as ov

            core = ov.Core()
            model = core.read_model(self._xml)
            shape = model.inputs[0].shape
            self.net_h, self.net_w = int(shape[2]), int(shape[3])
            model = self._fold_in_preprocessing(model)

            config = {
                "PERFORMANCE_HINT": "LATENCY",
                "INFERENCE_PRECISION_HINT": self.cfg.precision,
            }
            threads = self.cfg.num_threads or (len(affinity) if affinity else 0)
            if threads:
                config["INFERENCE_NUM_THREADS"] = threads

            compiled = core.compile_model(model, self.cfg.device or "CPU", config)
            req = compiled.create_infer_request()
            # Warm up *here*, inside the pinned thread, so any lazily-created worker threads
            # in the runtime's pool inherit this affinity too.
            req.infer(
                np.zeros((1, self.net_h, self.net_w, 3), dtype=np.uint8)
                if self._feeds_u8
                else np.zeros((1, 3, self.net_h, self.net_w), dtype=np.float32)
            )
        except BaseException as exc:  # surfaced to the constructor's caller
            self._error = exc
            self._ready.set()
            return

        self._ready.set()
        while True:
            blob = self._in_q.get()
            if blob is None:
                return
            try:
                self._out_q.put(req.infer(blob)[0])
            except BaseException as exc:
                self._out_q.put(exc)

    def _fold_in_preprocessing(self, model):
        """Move BGR->RGB, uint8->f32 and the /255 scale inside the graph.

        Done in numpy (via `blobFromImage`) this costs ~6ms a frame on the caller's thread —
        against a ~104ms inference that is 6% of the budget spent on a memory shuffle, and it
        lands on whichever little core the caller happens to be on. OpenVINO fuses these steps
        into the network's own first layer, so they run vectorised on the big cores instead,
        and the detector hands the runtime a plain uint8 HWC image.

        Best-effort: if the API shape ever changes, fall back to doing it in numpy.
        """
        try:
            from openvino import Layout, Type
            from openvino.preprocess import ColorFormat, PrePostProcessor

            ppp = PrePostProcessor(model)
            ppp.input().tensor().set_element_type(Type.u8).set_layout(
                Layout("NHWC")
            ).set_color_format(ColorFormat.BGR)
            ppp.input().model().set_layout(Layout("NCHW"))
            ppp.input().preprocess().convert_element_type(Type.f32).convert_color(
                ColorFormat.RGB
            ).scale(255.0)
            built = ppp.build()
            self._feeds_u8 = True
            return built
        except Exception:
            self._feeds_u8 = False
            return model

    def _resolve_affinity(self) -> set[int]:
        """Which CPUs the runtime may use. `auto` means "the biggest cluster"."""
        want = self.cfg.cpu_affinity
        available = os.sched_getaffinity(0)
        if not want:
            return set()
        if want != "auto":
            return {c for c in _parse_cpu_list(want) if c in available}

        # Pick the cluster with the highest max frequency. On a uniform CPU every core lands
        # in one group and we fall through to "use everything", which is the right answer.
        by_freq: dict[int, set[int]] = {}
        for cpu in available:
            f = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/cpuinfo_max_freq")
            try:
                by_freq.setdefault(int(f.read_text().strip()), set()).add(cpu)
            except OSError:
                return set()
        if len(by_freq) < 2:
            return set()
        return by_freq[max(by_freq)]

    def close(self) -> None:
        self._in_q.put(None)

    # ------------------------------------------------------------------ inference

    def _letterbox(self, image_bgr: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        h, w = image_bgr.shape[:2]
        r = min(self.net_h / h, self.net_w / w)
        nw, nh = round(w * r), round(h * r)
        left, top = (self.net_w - nw) // 2, (self.net_h - nh) // 2

        canvas = np.full((self.net_h, self.net_w, 3), 114, dtype=np.uint8)
        # INTER_AREA, deliberately, even though it costs 5.0ms against 0.4ms for INTER_LINEAR
        # at this 0.84x scale. It is worth it: averaging rather than sampling preserves the
        # few-pixel detail that a distant cyclist is made of, and swapping it for INTER_LINEAR
        # measurably loses a rider on the ground-truth clips (14/21 vs 15/21).
        canvas[top : top + nh, left : left + nw] = cv2.resize(
            image_bgr, (nw, nh), interpolation=cv2.INTER_AREA
        )

        if self._feeds_u8:
            return canvas[None], r, left, top
        return cv2.dnn.blobFromImage(canvas, 1 / 255.0, swapRB=True), r, left, top

    def infer(self, image_bgr: np.ndarray) -> sv.Detections:
        blob, r, left, top = self._letterbox(image_bgr)
        self._in_q.put(blob)
        out = self._out_q.get()
        if isinstance(out, BaseException):
            raise out
        return self._decode(out, r, left, top)

    def _decode(self, out: np.ndarray, r: float, left: int, top: int) -> sv.Detections:
        """(1, 4 + nc, N) raw head output -> NMS'd detections in crop coordinates."""
        pred = out[0]  # (4 + nc, N)
        boxes, scores = pred[:4], pred[4:]

        wanted = self.cfg.classes
        if wanted:
            # Score only the classes we care about, so a high-scoring `car` cannot mask the
            # `bicycle` underneath it and cannot survive to eat an NMS slot.
            keep_rows = np.array(wanted, dtype=int)
            scores = scores[keep_rows]
        else:
            keep_rows = np.arange(scores.shape[0])

        best = scores.argmax(axis=0)
        conf = scores[best, np.arange(scores.shape[1])]
        keep = conf >= self.cfg.conf
        if not keep.any():
            return _empty_detections()

        conf = conf[keep].astype(np.float32)
        class_id = keep_rows[best[keep]].astype(int)
        cx, cy, bw, bh = boxes[:, keep]

        # Undo letterbox: network pixels -> crop pixels.
        x = (cx - bw / 2 - left) / r
        y = (cy - bh / 2 - top) / r
        w = bw / r
        h = bh / r

        idx = cv2.dnn.NMSBoxesBatched(
            np.stack([x, y, w, h], axis=1).tolist(),
            conf.tolist(),
            class_id.tolist(),
            float(self.cfg.conf),
            float(self.cfg.iou),
        )
        idx = np.asarray(idx, dtype=int).reshape(-1)
        if idx.size == 0:
            return _empty_detections()

        xyxy = np.stack([x, y, x + w, y + h], axis=1)[idx].astype(np.float32)
        return sv.Detections(
            xyxy=xyxy,
            confidence=conf[idx],
            class_id=class_id[idx],
            data={"class_name": np.array([self.names.get(int(c), str(c)) for c in class_id[idx]])},
        )


class NpuDetector:
    """YOLO on the A733's VeriSilicon NPU, through the VIPLite runtime.

    The NPU runs the convolutional body; the detection head's decode and NMS stay on the CPU.
    That split is not a shortcut — it is what Allwinner's own model zoo does, because 8-bit
    quantisation of the head's box arithmetic costs far more accuracy than it saves time. The
    `.nb` therefore emits raw per-scale grid tensors, and `_DECODERS` turns them into boxes.

    Preprocessing must match what the model was quantised against exactly: letterbox to the
    network size with 114-grey, BGR->RGB, and hand the NPU raw uint8 — the /255 scaling is a
    preprocessing node compiled *into* the graph, so doing it here too would halve every pixel
    value twice over.
    """

    def __init__(self, cfg: ModelConfig):
        from .npu import VipNetwork

        self.cfg = cfg
        self.net = VipNetwork(cfg.weights, library_path=cfg.npu_libs or None)

        tensor = self.net.inputs[0]
        if len(tensor.shape) != 4:
            raise RuntimeError(f"expected a 4D image input, got {tensor.shape}")
        # ACUITY emits NHWC inputs for image graphs, but the layout is a property of the
        # export rather than a guarantee, so infer it from where the channel axis sits.
        if tensor.shape[3] in (1, 3):
            self._nhwc = True
            _, self.net_h, self.net_w, _ = tensor.shape
        elif tensor.shape[1] in (1, 3):
            self._nhwc = False
            _, _, self.net_h, self.net_w = tensor.shape
        else:
            raise RuntimeError(f"cannot find the channel axis in input shape {tensor.shape}")
        self._input = tensor

        self._geom: tuple | None = None
        self.names = _coco_names(cfg)
        decoder = cfg.npu_decoder or "auto"
        if decoder == "auto":
            decoder = _sniff_decoder(self.net.outputs)
        if decoder not in _DECODERS:
            raise RuntimeError(
                f"unknown npu_decoder {decoder!r}; known: auto, {', '.join(sorted(_DECODERS))}"
            )
        self.decoder = decoder
        self._decode_head = _DECODERS[decoder]

    def close(self) -> None:
        self.net.close()

    def _geometry(self, shape: tuple[int, int]) -> tuple[float, int, int, int, int]:
        """Letterbox placement for a given crop size, computed once and cached.

        The ROI crop is a fixed size for the life of the run, so the grey border never changes.
        Painting it once at startup and then writing only the image rectangle each frame halves
        the bytes pushed into NPU memory, which is uncached and slow to write.
        """
        if self._geom is not None and self._geom[0] == shape:
            return self._geom[1]

        h, w = shape
        r = min(self.net_h / h, self.net_w / w)
        nw, nh = int(w * r), int(h * r)
        left, top = (self.net_w - nw) // 2, (self.net_h - nh) // 2

        self._input.array.fill(114)
        self._geom = (shape, (r, left, top, nw, nh))
        return self._geom[1]

    def _letterbox(self, image_bgr: np.ndarray) -> tuple[float, int, int]:
        """Fill the NPU input tensor in place. Returns the scale and padding to undo later."""
        r, left, top, nw, nh = self._geometry(image_bgr.shape[:2])

        resized = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        # The graph's preprocessing node consumes RGB. cvtColor beats a `[:, :, ::-1]` view:
        # the negative stride forces numpy into a slow element-wise copy on the way out.
        cv2.cvtColor(resized, cv2.COLOR_BGR2RGB, dst=resized)

        # `array` is a live mapping of NPU memory, so this write *is* the upload.
        if self._nhwc:
            self._input.array[0, top : top + nh, left : left + nw] = resized
        else:
            self._input.array[0, :, top : top + nh, left : left + nw] = resized.transpose(2, 0, 1)
        return r, left, top

    def infer(self, image_bgr: np.ndarray) -> sv.Detections:
        r, left, top = self._letterbox(image_bgr)
        outputs = self.net.run()

        boxes, conf, class_id = self._decode_head(
            outputs, self.net_h, self.net_w, float(self.cfg.conf), self.cfg.classes
        )
        if len(conf) == 0:
            return _empty_detections()

        # Letterbox pixels -> crop pixels.
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / r
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / r

        wh = np.stack([boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], axis=1)
        idx = cv2.dnn.NMSBoxesBatched(
            wh.tolist(), conf.tolist(), class_id.tolist(),
            float(self.cfg.conf), float(self.cfg.iou),
        )
        idx = np.asarray(idx, dtype=int).reshape(-1)
        if idx.size == 0:
            return _empty_detections()

        return sv.Detections(
            xyxy=boxes[idx].astype(np.float32),
            confidence=conf[idx].astype(np.float32),
            class_id=class_id[idx].astype(int),
            data={"class_name": np.array([self.names.get(int(c), str(c)) for c in class_id[idx]])},
        )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# YOLOv5's COCO anchor boxes, per stride 8 / 16 / 32.
_V5_ANCHORS = np.array(
    [[[10, 13], [16, 30], [33, 23]],
     [[30, 61], [62, 45], [59, 119]],
     [[116, 90], [156, 198], [373, 326]]], dtype=np.float32,
)


def _decode_yolov5(outputs, net_h, net_w, conf_threshold, wanted):
    """Anchor-based YOLOv5 head, matching `yolov5_post.cpp::generate_proposals_rt`.

    Each output is one scale, shaped (1, anchors, 5 + nc, gy, gx) with *no* sigmoid applied —
    the export strips the Detect layer so the NPU never has to quantise it.
    """
    boxes, confs, classes = [], [], []
    keep_rows = np.array(wanted, dtype=int) if wanted else None

    # Threshold in logit space. Both factors of `sigmoid(obj) * sigmoid(cls)` are below 1, so
    # their product can only clear `conf_threshold` if each does — which means the cheap test
    # `logit >= logit(conf_threshold)` is exact, not an approximation. It matters: applying
    # sigmoid to every grid cell first costs ~1.5M exponentials a frame, several times the
    # entire NPU inference. This is what the reference `generate_proposals_rt` does too.
    logit_threshold = -np.log(1.0 / max(conf_threshold, 1e-6) - 1.0)

    for out in outputs:
        arr = np.asarray(out)
        _, _, _, gy, gx = arr.shape
        stride = net_h // gy
        anchors = _V5_ANCHORS[{8: 0, 16: 1, 32: 2}[stride]]

        raw = arr[0]  # (anchors, 5 + nc, gy, gx), still logits
        a_i, y_i, x_i = np.nonzero(raw[:, 4] >= logit_threshold)
        if a_i.size == 0:
            continue

        # Advanced indices either side of a slice: numpy puts the gathered axis first, giving
        # (n_candidates, nc).
        cls_logits = raw[a_i, 5:, y_i, x_i]
        if keep_rows is not None:
            cls_logits = cls_logits[:, keep_rows]

        best = cls_logits.argmax(axis=1)
        best_logit = cls_logits[np.arange(len(best)), best]
        hit = best_logit >= logit_threshold
        if not hit.any():
            continue

        a_i, y_i, x_i, best, best_logit = a_i[hit], y_i[hit], x_i[hit], best[hit], best_logit[hit]
        score = _sigmoid(raw[a_i, 4, y_i, x_i]) * _sigmoid(best_logit)

        d = _sigmoid(raw[a_i, :4, y_i, x_i].astype(np.float32))
        cx = (d[:, 0] * 2.0 - 0.5 + x_i) * stride
        cy = (d[:, 1] * 2.0 - 0.5 + y_i) * stride
        bw = (d[:, 2] * 2.0) ** 2 * anchors[a_i, 0]
        bh = (d[:, 3] * 2.0) ** 2 * anchors[a_i, 1]

        boxes.append(np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1))
        confs.append(score.astype(np.float32))
        classes.append(keep_rows[best] if keep_rows is not None else best)

    if not boxes:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)
    return np.concatenate(boxes), np.concatenate(confs), np.concatenate(classes)


def _decode_yolo11(outputs, net_h, net_w, conf_threshold, wanted):
    """Anchor-free YOLO11/YOLOv8 head with DFL box regression.

    The `.nb` is cut at the six head convolutions, so what comes back per scale is a pair:
    a 64-channel box branch (4 sides x 16 DFL bins) and an `nc`-channel class branch, both
    still raw logits. Mirrors `examples/yolo11/yolo11_6_post.cpp::generate_proposals_6`.

    Outputs are matched by shape rather than by index: the box branch is the one with 64
    channels, and its partner at the same grid size is the classes. ACUITY does not promise
    to preserve the order the ONNX declared them in.
    """
    arrays = [np.asarray(o)[0] for o in outputs]
    if len(arrays) % 2:
        raise RuntimeError(f"expected head outputs in pairs, got {len(arrays)}")

    # 64 is unambiguous: no feature-map side is 64 wide for any sane input size here.
    nhwc = any(a.shape[-1] == 64 for a in arrays)
    if not nhwc and not any(a.shape[0] == 64 for a in arrays):
        raise RuntimeError(f"no 64-channel DFL branch among {[a.shape for a in arrays]}")

    # -> (channels, gy, gx) regardless of how the graph laid them out.
    chw = [a.transpose(2, 0, 1) if nhwc else a for a in arrays]

    grids: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for a in chw:
        key = (a.shape[1], a.shape[2])
        grids.setdefault(key, {})["box" if a.shape[0] == 64 else "cls"] = a

    logit_threshold = -np.log(1.0 / max(conf_threshold, 1e-6) - 1.0)
    keep_rows = np.array(wanted, dtype=int) if wanted else None
    bins = np.arange(16, dtype=np.float32)

    boxes, confs, classes = [], [], []
    for (gy, gx), pair in grids.items():
        if "box" not in pair or "cls" not in pair:
            raise RuntimeError(f"unpaired head output at grid {gy}x{gx}")
        stride = net_h / gy
        cls, box = pair["cls"], pair["box"]

        scores = cls[keep_rows] if keep_rows is not None else cls
        # Threshold on raw logits: sigmoid is monotonic, so this is exact, and it keeps the
        # expensive part off the ~99.9% of cells that hold nothing.
        best = scores.argmax(axis=0)
        best_logit = np.take_along_axis(scores, best[None], axis=0)[0]
        y_i, x_i = np.nonzero(best_logit >= logit_threshold)
        if y_i.size == 0:
            continue

        picked = best[y_i, x_i]
        conf = _sigmoid(best_logit[y_i, x_i].astype(np.float32))

        # DFL: each side is a softmax over 16 bins, read out as its expected value.
        dist = box[:, y_i, x_i].astype(np.float32).reshape(4, 16, -1)
        dist -= dist.max(axis=1, keepdims=True)
        np.exp(dist, out=dist)
        dist /= dist.sum(axis=1, keepdims=True)
        ltrb = np.tensordot(bins, dist, axes=(0, 1))  # (4, n)

        cx, cy = x_i + 0.5, y_i + 0.5
        boxes.append(np.stack([
            (cx - ltrb[0]) * stride, (cy - ltrb[1]) * stride,
            (cx + ltrb[2]) * stride, (cy + ltrb[3]) * stride,
        ], axis=1))
        confs.append(conf)
        classes.append(keep_rows[picked] if keep_rows is not None else picked)

    if not boxes:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)
    return np.concatenate(boxes), np.concatenate(confs), np.concatenate(classes)


_DECODERS = {"yolov5": _decode_yolov5, "yolo11": _decode_yolo11}


def _sniff_decoder(outputs) -> str:
    """Pick the head decoder from the network's output signature.

    A `.nb` carries no hint about which model family produced it, and choosing wrong does not
    degrade gracefully — it crashes, or worse, silently decodes nonsense. The two families are
    unambiguous in shape, so infer it rather than making it a config knob to get wrong:

      * YOLO11/v8 — anchor-free, six 4D outputs, three of them 64-channel DFL box branches;
      * YOLOv5    — anchor-based, three 5D outputs of (1, anchors, 5 + nc, gy, gx).
    """
    shapes = [tuple(t.shape) for t in outputs]
    if len(shapes) == 6 and all(len(s) == 4 for s in shapes):
        if any(64 in (s[1], s[3]) for s in shapes):
            return "yolo11"
    if all(len(s) == 5 for s in shapes):
        return "yolov5"
    raise RuntimeError(
        f"cannot tell which head these outputs belong to: {shapes}. "
        f"Set model.npu_decoder explicitly to one of: {', '.join(sorted(_DECODERS))}."
    )


def _coco_names(cfg: ModelConfig) -> dict[int, str]:
    """Class names from a sibling metadata.yaml, else the COCO ids we actually name."""
    meta = Path(cfg.weights).parent / "metadata.yaml"
    if meta.exists():
        loaded = (yaml.safe_load(meta.read_text()) or {}).get("names") or {}
        if loaded:
            return {int(k): v for k, v in loaded.items()}
    return {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def _empty_detections() -> sv.Detections:
    return sv.Detections(
        xyxy=np.zeros((0, 4), dtype=np.float32),
        confidence=np.zeros(0, dtype=np.float32),
        class_id=np.zeros(0, dtype=int),
        data={"class_name": np.array([], dtype=object)},
    )


def _parse_cpu_list(spec: str) -> list[int]:
    """`taskset`-style CPU list: "6,7" or "0-5" or "0-3,6,7"."""
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def build_detector(cfg: ModelConfig) -> Detector:
    """Factory. Extend here with `if cfg.weights.endswith('.rknn'): return RknnDetector(cfg)`."""
    weights = str(cfg.weights)
    if cfg.backend == "npu" or (cfg.backend == "auto" and weights.endswith(".nb")):
        return NpuDetector(cfg)
    if cfg.backend == "openvino" or (
        cfg.backend == "auto" and weights.rstrip("/").endswith("_openvino_model")
    ):
        return OpenVinoDetector(cfg)
    return UltralyticsDetector(cfg)
