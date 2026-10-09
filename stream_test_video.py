#!/usr/bin/env python3
"""Serve one of the test recordings as a live camera stream.

The point is to exercise the *stream* path of `detect_bikes.py` — reconnects, dropped
frames, wall-clock timestamps — before there is a real camera on the other end of the cable.
It loops the file in real time and publishes it exactly the way an IP camera would.

    # RTSP, what most cameras actually speak (needs the mediamtx helper, fetched on demand)
    python stream_test_video.py video.mp4
    # -> rtsp://192.168.1.1:8554/cam

    # MJPEG over HTTP — no helper, no encoder, pure Python; also what cheap cameras offer
    python stream_test_video.py video.mp4 --mode mjpeg
    # -> http://192.168.1.1:8080/stream.mjpg

Then on the board:

    python detect_bikes.py rtsp://192.168.1.1:8554/cam --events events.jsonl

Ctrl-C stops the stream. See NETWORK.md for the cabling and the static addresses.
"""

from __future__ import annotations

import argparse
import http.server
import io
import os
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from pathlib import Path


def _die_with_parent() -> None:
    """Make a child receive SIGTERM if this process dies.

    Without it, a `kill -9` on the test rig leaves mediamtx holding port 8554, and the next
    run fails with "address already in use" — pointing at the port rather than at the orphan
    that is actually the problem.
    """
    try:
        import ctypes

        PR_SET_PDEATHSIG = 1
        ctypes.CDLL("libc.so.6").prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
    except Exception:
        pass

MEDIAMTX_VERSION = "v1.20.1"
MEDIAMTX_URL = (
    "https://github.com/bluenviron/mediamtx/releases/download/"
    f"{MEDIAMTX_VERSION}/mediamtx_{MEDIAMTX_VERSION}_linux_amd64.tar.gz"
)
TOOLS_DIR = Path(__file__).resolve().parent / "tools"

# Only RTSP is enabled: the other servers mediamtx ships would bind further ports for no
# reason here, and a test rig that quietly listens on six ports is a bad habit on a machine
# that is also on the office wifi.
#
# TCP-only transport, for two reasons. The client uses TCP anyway (see `rtsp_transport` in
# config.yaml), and UDP mode makes mediamtx bind the fixed RTP/RTCP ports 8000/8001, which on
# a developer laptop are frequently already taken — it then exits with nothing but
# "bind: address already in use", which reads as if the RTSP port were the problem.
MEDIAMTX_CONFIG = """\
logLevel: {loglevel}
rtsp: true
rtspTransports: [tcp]
rtspAddress: {bind}:{port}
rtmp: false
hls: false
webrtc: false
srt: false
api: false
metrics: false
playback: false
paths:
  all_others:
"""


# ---------------------------------------------------------------------------------------
# RTSP: mediamtx as the server, ffmpeg as the camera
# ---------------------------------------------------------------------------------------


def ensure_mediamtx(allow_download: bool = True) -> str:
    """Path to a mediamtx binary, downloading it into `tools/` on first use.

    mediamtx is the server; ffmpeg cannot be one. It is a single static binary with no
    install step, which is why this is a download rather than a packaging problem.
    """
    local = TOOLS_DIR / "mediamtx"
    if local.exists() and os.access(local, os.X_OK):
        return str(local)
    found = shutil.which("mediamtx")
    if found:
        return found
    if not allow_download:
        raise RuntimeError(
            "mediamtx not found. Install it, put the binary in tools/, or drop --no-download."
        )

    TOOLS_DIR.mkdir(exist_ok=True)
    print(f"[info] fetching mediamtx {MEDIAMTX_VERSION} -> {local}", file=sys.stderr)
    with urllib.request.urlopen(MEDIAMTX_URL, timeout=120) as response:
        payload = response.read()
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
        member = tar.extractfile("mediamtx")
        if member is None:
            raise RuntimeError("mediamtx missing from the release tarball")
        local.write_bytes(member.read())
    local.chmod(0o755)
    return str(local)


def serve_rtsp(args: argparse.Namespace) -> int:
    mediamtx = ensure_mediamtx(allow_download=not args.no_download)
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found — needed to publish the file into mediamtx")

    TOOLS_DIR.mkdir(exist_ok=True)
    config = TOOLS_DIR / "mediamtx-test.yml"
    config.write_text(
        MEDIAMTX_CONFIG.format(
            bind="" if args.bind in ("0.0.0.0", "") else args.bind,
            port=args.port,
            loglevel="info" if args.verbose else "error",
        )
    )

    url = f"rtsp://{advertised_host(args.bind)}:{args.port}/{args.path}"
    publish_to = f"rtsp://127.0.0.1:{args.port}/{args.path}"

    log = TOOLS_DIR / "mediamtx-test.log"
    with open(log, "wb") as handle:
        # Run it from tools/: mediamtx drops a self-signed auto.crt/auto.key into its
        # working directory on startup, and they do not belong in the repo root.
        server = subprocess.Popen(
            [mediamtx, str(config)], stdout=handle, stderr=handle,
            cwd=str(TOOLS_DIR), preexec_fn=_die_with_parent,
        )
    # mediamtx binds in well under a second, but ffmpeg exits immediately if the port is not
    # listening yet, so wait for the socket rather than guessing at a sleep.
    if not wait_for_port("127.0.0.1", args.port, timeout=10.0):
        server.terminate()
        raise RuntimeError(
            f"mediamtx did not start listening on port {args.port}:\n"
            + log.read_text().strip()
        )
    if args.verbose:
        _tail(log)

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "info" if args.verbose else "warning"]
    if args.loop:
        cmd += ["-stream_loop", "-1"]
    if args.start:
        cmd += ["-ss", str(args.start)]
    # `-re` paces the file at its own frame rate — without it ffmpeg pushes the whole
    # recording in a few seconds and nothing about the test resembles a camera.
    cmd += ["-re", "-i", str(args.video), "-an"]
    if args.codec == "copy":
        cmd += ["-c:v", "copy"]
    else:
        # Re-encoding costs CPU on this laptop but produces the H.264 that most cameras and
        # every decoder speak; the recordings here are HEVC.
        cmd += [
            "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
            "-pix_fmt", "yuv420p", "-g", "30",
        ]
    if args.fps:
        cmd += ["-r", str(args.fps)]
    cmd += ["-f", "rtsp", "-rtsp_transport", "tcp", publish_to]

    print(f"[info] serving {args.video}", file=sys.stderr)
    print(f"[info] RTSP  {url}", file=sys.stderr)
    print(f"[info] on the board:  python detect_bikes.py {url} --debug --debug-out live.mp4", file=sys.stderr)

    publisher = subprocess.Popen(cmd, preexec_fn=_die_with_parent)
    try:
        while True:
            if publisher.poll() is not None:
                if not args.loop:
                    print("[info] source finished", file=sys.stderr)
                    break
                print("[warn] publisher exited, restarting", file=sys.stderr)
                publisher = subprocess.Popen(cmd, preexec_fn=_die_with_parent)
            if server.poll() is not None:
                print(f"[error] mediamtx exited:\n{log.read_text().strip()}", file=sys.stderr)
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[info] stopping", file=sys.stderr)
    finally:
        for proc in (publisher, server):
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
    return 0


# ---------------------------------------------------------------------------------------
# MPEG-TS over TCP: no helper binary at all
# ---------------------------------------------------------------------------------------


def serve_mpegts(args: argparse.Namespace) -> int:
    """ffmpeg's own TCP listener. One client at a time, but zero moving parts.

    Useful when you want to test the stream path without RTSP in the picture at all — if
    this works and RTSP does not, the problem is the RTSP server, not the network.
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found")

    listen = f"tcp://{args.bind}:{args.port}?listen=1"
    url = f"tcp://{advertised_host(args.bind)}:{args.port}"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "info" if args.verbose else "warning"]
    if args.loop:
        cmd += ["-stream_loop", "-1"]
    if args.start:
        cmd += ["-ss", str(args.start)]
    cmd += ["-re", "-i", str(args.video), "-an", "-c:v", "copy", "-f", "mpegts", listen]

    print(f"[info] MPEG-TS {url}  (waiting for a client to connect)", file=sys.stderr)
    print(f"[info] on the board:  python detect_bikes.py {url} --live", file=sys.stderr)
    try:
        while True:
            subprocess.run(cmd)
            if not args.loop:
                return 0
            print("[info] client gone — listening again", file=sys.stderr)
    except KeyboardInterrupt:
        print("\n[info] stopping", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------------------
# MJPEG over HTTP: pure Python, no ffmpeg, no mediamtx
# ---------------------------------------------------------------------------------------


class _MjpegBroadcaster:
    """Decodes the file in real time and hands the newest JPEG to every client.

    Deliberately last-frame-only rather than a per-client queue: a slow client should see
    fewer frames, exactly as it would from a camera, not make everyone else wait.
    """

    def __init__(self, video: str, fps: float | None, quality: int, loop: bool, start: float):
        import cv2  # local: this mode is the one that must work with nothing else installed

        self.cv2 = cv2
        self.video = video
        self.quality = quality
        self.loop = loop
        self.start = start
        self.frame: bytes | None = None
        self.frame_no = 0
        self.condition = threading.Condition()
        self.stop_flag = threading.Event()

        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {video}")
        self.fps = fps or cap.get(cv2.CAP_PROP_FPS) or 15.0
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

    def run(self) -> None:
        cv2 = self.cv2
        interval = 1.0 / self.fps
        params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        while not self.stop_flag.is_set():
            cap = cv2.VideoCapture(self.video)
            if self.start:
                cap.set(cv2.CAP_PROP_POS_MSEC, self.start * 1000.0)
            next_at = time.monotonic()
            while not self.stop_flag.is_set():
                ok, image = cap.read()
                if not ok:
                    break
                ok, buf = cv2.imencode(".jpg", image, params)
                if ok:
                    with self.condition:
                        self.frame = buf.tobytes()
                        self.frame_no += 1
                        self.condition.notify_all()
                # Pace against an absolute schedule, so encoding time does not make the
                # stream drift slower and slower than real time.
                next_at += interval
                sleep = next_at - time.monotonic()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_at = time.monotonic()
            cap.release()
            if not self.loop:
                break

    def next_frame(self, seen: int, timeout: float = 5.0) -> tuple[bytes | None, int]:
        with self.condition:
            if self.frame_no == seen:
                self.condition.wait(timeout)
            return self.frame, self.frame_no


def serve_mjpeg(args: argparse.Namespace) -> int:
    broadcaster = _MjpegBroadcaster(
        str(args.video), args.fps, args.quality, args.loop, args.start
    )
    worker = threading.Thread(target=broadcaster.run, daemon=True)
    worker.start()

    boundary = "bikecountframe"

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, fmt, *fmt_args):  # noqa: D102 - quieter than the default
            if args.verbose:
                super().log_message(fmt, *fmt_args)

        def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler's naming
            if self.path.rstrip("/") in ("", "/index.html"):
                body = (
                    f"<h3>bikecount test stream</h3>"
                    f"<p>{broadcaster.width}x{broadcaster.height} @ {broadcaster.fps:.1f} fps</p>"
                    f'<img src="/stream.mjpg">'
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if not self.path.startswith("/stream"):
                self.send_error(404)
                return

            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
            self.end_headers()
            seen = 0
            try:
                while True:
                    frame, seen = broadcaster.next_frame(seen)
                    if frame is None:
                        break
                    self.wfile.write(f"--{boundary}\r\n".encode())
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass  # client went away; normal

    class Server(http.server.ThreadingHTTPServer):
        allow_reuse_address = True
        daemon_threads = True

    url = f"http://{advertised_host(args.bind)}:{args.port}/stream.mjpg"
    httpd = Server((args.bind, args.port), Handler)
    print(f"[info] serving {args.video}", file=sys.stderr)
    print(f"[info] MJPEG {url}", file=sys.stderr)
    print(f"[info] on the board:  python detect_bikes.py {url} --live", file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[info] stopping", file=sys.stderr)
    finally:
        broadcaster.stop_flag.set()
        httpd.server_close()
    return 0


# ---------------------------------------------------------------------------------------


def _tail(path: Path) -> None:
    """Mirror a log file to stderr for as long as we live. Only used with --verbose."""

    def follow() -> None:
        with open(path) as handle:
            while True:
                line = handle.readline()
                if line:
                    sys.stderr.write(line)
                else:
                    time.sleep(0.2)

    threading.Thread(target=follow, daemon=True).start()


def advertised_host(bind: str) -> str:
    """The address to print in the URLs — what the *board* should connect to.

    Binding to 0.0.0.0 is right, but telling someone to open `rtsp://0.0.0.0:8554` is not,
    so prefer this machine's address on the camera link when it exists.
    """
    if bind not in ("0.0.0.0", "", "::"):
        return bind
    for candidate in _local_addresses():
        if candidate.startswith("192.168.1."):
            return candidate
    return next(iter(_local_addresses()), "127.0.0.1")


def _local_addresses() -> list[str]:
    addresses: list[str] = []
    try:
        out = subprocess.run(
            ["ip", "-4", "-o", "addr", "show", "scope", "global"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        for line in out.splitlines():
            parts = line.split()
            if len(parts) > 3:
                addresses.append(parts[3].split("/")[0])
    except Exception:
        pass
    return addresses


def wait_for_port(host: str, port: int, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket() as probe:
            probe.settimeout(0.5)
            if probe.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.1)
    return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Serve a recording as a live camera stream, for testing the board.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("video", help="video file to stream")
    p.add_argument(
        "--mode", choices=["rtsp", "mjpeg", "mpegts"], default="rtsp",
        help="rtsp: what most IP cameras speak. mjpeg: pure Python, no helpers. "
             "mpegts: ffmpeg's own TCP listener, one client",
    )
    p.add_argument("--bind", default="0.0.0.0", help="address to listen on")
    p.add_argument("--port", type=int, default=0, help="0 -> 8554 for rtsp/mpegts, 8080 for mjpeg")
    p.add_argument("--path", default="cam", help="RTSP path name (rtsp://host:port/<path>)")
    p.add_argument("--start", type=float, default=0.0, help="seconds into the file to start at")
    p.add_argument("--fps", type=float, help="override the source frame rate")
    p.add_argument("--codec", choices=["h264", "copy"], default="h264",
                   help="rtsp: transcode to H.264 with a short GOP (a client joins in <1s, and "
                        "H.264 is what cameras send), or copy the file's HEVC through untouched")
    p.add_argument("--quality", type=int, default=80, help="mjpeg: JPEG quality")
    p.add_argument("--no-loop", dest="loop", action="store_false", help="play once instead of looping")
    p.add_argument("--no-download", action="store_true", help="never fetch mediamtx")
    p.add_argument("-v", "--verbose", action="store_true", help="show ffmpeg/mediamtx output")
    args = p.parse_args(argv)
    if args.port == 0:
        args.port = 8080 if args.mode == "mjpeg" else 8554
    if not Path(args.video).exists():
        p.error(f"no such file: {args.video}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "rtsp":
        return serve_rtsp(args)
    if args.mode == "mpegts":
        return serve_mpegts(args)
    return serve_mjpeg(args)


if __name__ == "__main__":
    raise SystemExit(main())
