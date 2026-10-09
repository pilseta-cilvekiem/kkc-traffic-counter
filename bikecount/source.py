"""Video input with cheap frame decimation.

Handles two quite different things behind one iterator:

* **A file.** Finite, seekable, and every frame matters — timestamps come from the container,
  `--start`/`--duration` are positions in the recording, and a slow consumer simply takes
  longer to finish.
* **A live stream** (RTSP/HTTP camera, or a V4L2 device). Infinite, not seekable, and *late
  frames are worthless* — the camera keeps producing at its own rate whether or not we keep
  up, so the only sensible policy is to drop what we could not process and always work on the
  newest frame. Timestamps come from the wall clock, because dropped frames make
  "frame index / fps" a lie.

`is_live` selects between the two and is auto-detected from the path.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np

# URL schemes OpenCV's FFMPEG backend can open as a live stream. `http(s)` covers the MJPEG
# and HLS endpoints most IP cameras expose alongside RTSP.
STREAM_SCHEMES = (
    "rtsp://",
    "rtsps://",
    "rtmp://",
    "http://",
    "https://",
    "udp://",
    "tcp://",
    "rtp://",
    "srt://",
)


def is_stream_url(path: str | int) -> bool:
    """True for a network stream URL."""
    return isinstance(path, str) and path.lower().startswith(STREAM_SCHEMES)


def is_live_source(path: str | int) -> bool:
    """True for anything that produces frames in real time and cannot be rewound.

    Network streams, V4L2 device nodes (`/dev/video0`) and bare camera indices (`0`).
    """
    if isinstance(path, int):
        return True
    if is_stream_url(path):
        return True
    if isinstance(path, str):
        if path.startswith("/dev/video"):
            return True
        if path.isdigit():
            return True
    return False


@dataclass
class Frame:
    index: int  # index in the source video
    timestamp: float  # seconds from the start of the source (wall clock, on a live source)
    image: np.ndarray


class VideoSource:
    """Iterates a video file or stream, yielding only every Nth frame.

    Skipped frames go through ``cap.grab()``, which demuxes but does not decode or
    colour-convert. On HEVC that is far cheaper than ``read()``-and-throw-away, which is what
    makes a high stride actually save CPU rather than just save inference time.
    """

    def __init__(
        self,
        path: str | int,
        stride: int = 1,
        start: float = 0.0,
        duration: float | None = None,
        prefetch: int = 0,
        *,
        live: bool | None = None,
        fps: float | None = None,
        rtsp_transport: str = "tcp",
        reconnect: bool = True,
        reconnect_delay: float = 2.0,
        quiet: bool = False,
    ):
        self.path = path
        self.stride = max(1, int(stride))
        self.start = start
        self.duration = duration
        self.prefetch = max(0, int(prefetch))
        self.reconnect = reconnect
        self.reconnect_delay = max(0.1, float(reconnect_delay))
        self.rtsp_transport = rtsp_transport
        self.quiet = quiet

        self.is_live = is_live_source(path) if live is None else bool(live)
        self._delivered = 0

        try:
            self.cap = self._open()
        except RuntimeError as exc:
            # A camera that is not up yet (the board booted first, a power cut took both) is
            # the same situation as one that dropped: wait for it rather than exit.
            if not (self.is_live and self.reconnect):
                raise
            self._log(f"[warn] {exc}; retrying every {self.reconnect_delay:.0f}s")
            self.cap = self._retry_open()
        # Set once, not per reconnect: `measured_fps` is a session average, and resetting the
        # clock on every reconnect while keeping the frame count would inflate it.
        self._started_at = time.monotonic()

        raw_fps = self.cap.get(cv2.CAP_PROP_FPS)
        # A camera's advertised rate is frequently 0, or nonsense like 90000 (the RTP clock).
        # `fps` is only used to size the tracker's association window, so a sane fallback
        # beats a wrong number; on a live source `adaptive_dt` corrects it from real
        # timestamps anyway.
        if not (0.5 < raw_fps < 240.0):
            raw_fps = fps or (15.0 if self.is_live else 25.0)
        self.fps = float(fps) if (fps and not self.is_live) else float(raw_fps)

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = 0 if self.is_live else int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if start > 0 and not self.is_live:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)

    # -- opening / reconnecting -------------------------------------------------------

    def _open(self) -> cv2.VideoCapture:
        target: str | int = self.path
        if isinstance(target, str) and target.isdigit():
            target = int(target)

        if self.is_live and is_stream_url(str(self.path)):
            # These are read by the FFMPEG backend when the capture is *constructed*, so they
            # have to be in the environment before the call. TCP transport is the default
            # because UDP on a long or busy link produces torn frames that look like detector
            # failures; `stimeout` (microseconds) stops a dead camera hanging the process
            # forever in the middle of a read.
            opts = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
            if opts is None:
                parts = ["stimeout;5000000", "max_delay;500000", "fflags;nobuffer"]
                if str(self.path).lower().startswith(("rtsp://", "rtsps://")):
                    parts.insert(0, f"rtsp_transport;{self.rtsp_transport}")
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "|".join(parts)

        cap = cv2.VideoCapture(target)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {self.path}")

        if self.is_live:
            # Ask the backend to hold one frame, not a queue of them. Honoured by V4L2 and by
            # some FFMPEG builds; harmless where it is not, since we drop stale frames
            # ourselves in `_pump`.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        return cap

    def _reopen(self) -> bool:
        """Reconnect to a live source after a read failure. False if we should give up."""
        if not (self.is_live and self.reconnect):
            return False
        self._log(f"[warn] stream ended or stalled; reconnecting to {self.path}")
        try:
            self.cap.release()
        except Exception:
            pass
        self.cap = self._retry_open()
        return True

    def _retry_open(self) -> cv2.VideoCapture:
        """Open the live source, retrying until it answers.

        Logs once when it comes back rather than on every attempt: a camera that is off
        overnight would otherwise put a line in the journal every `reconnect_delay` seconds.
        """
        started = time.monotonic()
        while True:
            time.sleep(self.reconnect_delay)
            try:
                cap = self._open()
            except RuntimeError:
                continue
            self._log(f"[info] stream connected after {time.monotonic() - started:.0f}s")
            return cap

    def _log(self, message: str) -> None:
        if not self.quiet:
            print(message, file=sys.stderr, flush=True)

    # -- iteration --------------------------------------------------------------------

    @property
    def effective_fps(self) -> float:
        """Frames per second actually handed to the detector."""
        return self.fps / self.stride

    @property
    def measured_fps(self) -> float:
        """Frames per second actually delivered so far — the honest number on a live source."""
        elapsed = time.monotonic() - self._started_at
        return self._delivered / elapsed if elapsed > 0 else 0.0

    def __iter__(self) -> Iterator[Frame]:
        """Frames, decoded ahead on a worker thread when `prefetch` is set.

        Decoding costs ~6ms a frame here against ~47ms of inference, so it is not the
        bottleneck — but the two run on different clusters (the detector pins itself to the
        big cores, leaving the little ones idle), so overlapping them is close to free time.
        The queue is deliberately shallow: on a live camera a deep buffer converts "cannot
        keep up" into growing latency rather than absorbing anything, so there the worker
        drops the oldest frame instead of blocking.
        """
        if self.prefetch <= 0 and not self.is_live:
            return self._decode()
        return self._prefetched()

    def _prefetched(self) -> Iterator[Frame]:
        depth = max(1, self.prefetch)
        q: queue.Queue = queue.Queue(maxsize=depth)
        sentinel = object()

        def pump() -> None:
            try:
                for frame in self._decode():
                    if self.is_live:
                        # Newest frame wins. Anything still queued is already stale by the
                        # time we get here, and processing it only adds latency.
                        while True:
                            try:
                                q.put_nowait(frame)
                                break
                            except queue.Full:
                                try:
                                    q.get_nowait()
                                except queue.Empty:
                                    pass
                    else:
                        q.put(frame)
            except BaseException as exc:  # re-raised on the consumer side
                q.put(exc)
            finally:
                q.put(sentinel)

        worker = threading.Thread(target=pump, daemon=True)
        worker.start()
        while True:
            item = q.get()
            if item is sentinel:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    def _decode(self) -> Iterator[Frame]:
        if self.is_live:
            yield from self._decode_live()
        else:
            yield from self._decode_file()

    def _decode_file(self) -> Iterator[Frame]:
        index = int(self.start * self.fps)
        end = None if self.duration is None else self.start + self.duration
        # Emit the first available frame, then every `stride` frames after it.
        countdown = 0
        while True:
            ok = self.cap.grab()
            if not ok:
                break
            if countdown > 0:
                countdown -= 1
                index += 1
                continue
            countdown = self.stride - 1

            ok, image = self.cap.retrieve()
            if not ok:
                break

            # POS_MSEC is unreliable on some containers; fall back to index/fps.
            ts = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if ts <= 0:
                ts = index / self.fps

            if end is not None and ts > end:
                break

            self._delivered += 1
            yield Frame(index=index, timestamp=ts, image=image)
            index += 1

    def _decode_live(self) -> Iterator[Frame]:
        """Same shape as `_decode_file`, but nothing about a camera can be trusted.

        `start` is meaningless (there is no past to seek to), `duration` is wall-clock
        seconds rather than a position in a recording, and a failed read is a network event
        to recover from rather than the end of the input.
        """
        index = 0
        countdown = 0
        t0 = time.monotonic()
        end = None if self.duration is None else self.duration

        while True:
            if end is not None and (time.monotonic() - t0) > end:
                return

            if not self.cap.grab():
                if self._reopen():
                    countdown = 0
                    continue
                return

            index += 1
            if countdown > 0:
                countdown -= 1
                continue
            countdown = self.stride - 1

            ok, image = self.cap.retrieve()
            if not ok:
                if self._reopen():
                    countdown = 0
                    continue
                return

            # Wall clock, not frame index: frames get dropped both by the network and by us,
            # so index/fps would drift away from real time — and every speed in the pipeline
            # is measured in px per *real* second.
            self._delivered += 1
            yield Frame(index=index, timestamp=time.monotonic() - t0, image=image)

    def release(self) -> None:
        self.cap.release()

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def grab_frame(path: str, at: float = 0.0) -> np.ndarray:
    """Pull a single frame at `at` seconds. Used by the ROI tool.

    On a live source `at` is ignored — there is nothing to seek to — and the next frame the
    camera sends is returned instead, which is what you want for drawing an ROI against a
    camera that is already mounted.
    """
    cap = cv2.VideoCapture(int(path) if isinstance(path, str) and path.isdigit() else path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    try:
        if at > 0 and not is_live_source(path):
            cap.set(cv2.CAP_PROP_POS_MSEC, at * 1000.0)
        ok, image = cap.read()
        if not ok:
            raise RuntimeError(f"cannot read a frame at {at}s from {path}")
        return image
    finally:
        cap.release()
