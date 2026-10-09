#!/usr/bin/env python3
"""Detect and count traffic crossing a line inside a region of a fixed-camera video.

Bicycles, motorcycles, cars, buses, trucks, pedestrians, and riders the model has no class
for (e-scooters, counted as a `person` moving at riding pace). `model.count_classes` in
`config.yaml` picks which of them count.

Examples
--------
Calibration pass — wide class net, low threshold, annotated video out:

    python detect_bikes.py video.mp4 --debug --debug-out cal.mp4 \
        --start 600 --duration 180 --conf 0.15 --classes 0 1 2 3 5 7

Deployment-shaped run — no rendering, JSONL events only:

    python detect_bikes.py video.mp4 --events events.jsonl --stride 5

Where the time goes on the target hardware:

    python detect_bikes.py video.mp4 --duration 60 --bench

A live camera, reviewed afterwards — `--debug-out` also writes `live.log` beside the video,
which `debug_viewer.py` uses to jump between the crossings:

    python detect_bikes.py rtsp://192.168.1.3:554/... --debug-out live.mp4
    python debug_viewer.py live.mp4
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import cv2

from bikecount.config import Config
from bikecount.debug_log import DebugLog
from bikecount.detectors import build_detector
from bikecount.pipeline import Pipeline
from bikecount.report import SupabaseReporter, load_env_file
from bikecount.source import VideoSource


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detect and count traffic in a region of a video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("video", help="video file or stream URL")
    p.add_argument("--config", default="config.yaml", help="YAML config with ROI and defaults")

    g = p.add_argument_group("model")
    g.add_argument("--model", dest="weights", help="weights: .pt, .onnx, *_openvino_model/")
    g.add_argument("--imgsz", type=int, help="inference size")
    g.add_argument("--conf", type=float, help="confidence threshold")
    g.add_argument("--classes", type=int, nargs="+", help="COCO class ids (0=person 1=bicycle 2=car 3=motorcycle 5=bus 7=truck)")
    g.add_argument("--device", help="cpu, cuda, cuda:0, ...")

    g = p.add_argument_group("live stream")
    g.add_argument(
        "--live", dest="live", action="store_const", const=True,
        help="force live-stream handling (no seeking, wall-clock timestamps, drop stale frames)",
    )
    g.add_argument(
        "--no-live", dest="live", action="store_const", const=False,
        help="force file handling for a URL that actually serves a finite recording",
    )
    g.add_argument("--rtsp-transport", choices=["tcp", "udp"], help="RTSP transport (default tcp)")
    g.add_argument("--stream-fps", type=float, help="frame rate to assume if the camera does not report one")
    g.add_argument("--no-reconnect", action="store_true", help="exit when the stream drops instead of reconnecting")

    g = p.add_argument_group("sampling")
    g.add_argument("--stride", type=int, help="process every Nth frame")
    g.add_argument("--start", type=float, help="start offset in seconds")
    g.add_argument("--duration", type=float, help="seconds to process")
    g.add_argument("--no-roi-crop", action="store_true", help="infer on the full frame")

    g = p.add_argument_group("output")
    g.add_argument("--events", help="write JSONL crossing events here (default: stdout)")
    g.add_argument(
        "--report", action="store_true",
        help="also send each crossing to Supabase (credentials from --env-file / the environment)",
    )
    g.add_argument("--env-file", default=".env", help="KEY=VALUE file read for --report")
    g.add_argument("--debug", action="store_true", help="render ROI, boxes, labels, counters")
    g.add_argument("--debug-out", help="write the annotated video here (implies --debug)")
    g.add_argument("--debug-window", action="store_true", help="live preview (implies --debug)")
    g.add_argument(
        "--debug-log",
        help="JSONL sidecar for the annotated video (default: alongside --debug-out, .log). "
             "Read by debug_viewer.py",
    )
    g.add_argument("--no-debug-log", action="store_true", help="skip the sidecar log")
    g.add_argument("--bench", action="store_true", help="print per-stage timing at the end")
    g.add_argument("--quiet", action="store_true", help="suppress the progress line")
    p.set_defaults(live=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    debug = args.debug or bool(args.debug_out) or args.debug_window

    cfg = Config.load(args.config).with_overrides(
        weights=args.weights,
        imgsz=args.imgsz,
        conf=args.conf,
        classes=args.classes,
        device=args.device,
        stride=args.stride,
        start=args.start,
        duration=args.duration,
        crop=False if args.no_roi_crop else None,
        rtsp_transport=args.rtsp_transport,
        stream_fps=args.stream_fps,
        reconnect=False if args.no_reconnect else None,
        live=args.live,
    )

    if not Path(args.config).exists():
        print(
            f"[warn] {args.config} not found — using the whole frame and no counting line. "
            f"Run roi_tool.py to define them.",
            file=sys.stderr,
        )

    source = VideoSource(
        args.video,
        stride=cfg.runtime.stride,
        start=cfg.runtime.start,
        duration=cfg.runtime.duration,
        prefetch=cfg.runtime.prefetch,
        live=cfg.runtime.live_for(args.video),
        fps=cfg.runtime.stream_fps,
        rtsp_transport=cfg.runtime.rtsp_transport,
        reconnect=cfg.runtime.reconnect,
        reconnect_delay=cfg.runtime.reconnect_delay,
        quiet=args.quiet,
    )
    print(
        f"[info] {'live stream' if source.is_live else 'file'}: "
        f"{source.width}x{source.height} @ {source.fps:.1f}fps, "
        f"stride={cfg.runtime.stride} -> {source.effective_fps:.1f} inferences/s",
        file=sys.stderr,
    )
    if source.is_live:
        # Nothing waits for us on a camera: frames we cannot process in time are dropped, so
        # "too slow" shows up as a lower effective frame rate rather than as a backlog.
        print(
            "[info] live source — start/duration are wall-clock, stale frames are dropped, "
            f"reconnect={'on' if source.reconnect else 'off'}",
            file=sys.stderr,
        )

    detector = build_detector(cfg.model)
    pipeline = Pipeline(
        cfg,
        detector,
        source.width,
        source.height,
        source.fps,
        adaptive_dt=cfg.runtime.adaptive_dt and source.is_live,
    )

    if pipeline.line_zone is None:
        print("[warn] no counting line configured — detecting only, no counts.", file=sys.stderr)

    renderer = None
    writer = None
    if debug:
        from bikecount.debug_view import DebugRenderer

        renderer = DebugRenderer(pipeline, source.width, source.height, source.fps)
        if args.debug_out:
            # Play the annotated video back at the effective rate so it matches wall time.
            writer = cv2.VideoWriter(
                args.debug_out,
                cv2.VideoWriter_fourcc(*"mp4v"),
                max(1.0, source.effective_fps),
                (source.width, source.height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"cannot open video writer for {args.debug_out}")

    debug_log = None
    log_path = args.debug_log or (
        str(Path(args.debug_out).with_suffix(".log")) if args.debug_out else None
    )
    if log_path and not args.no_debug_log:
        # The annotated video on its own is unnavigable — you scrub it hoping to land on a
        # crossing. The sidecar is what debug_viewer.py jumps around by.
        debug_log = DebugLog(
            log_path,
            meta={
                "source": args.video,
                "live": source.is_live,
                "video": args.debug_out,
                "video_fps": max(1.0, source.effective_fps),
                "source_fps": source.fps,
                "width": source.width,
                "height": source.height,
                "stride": cfg.runtime.stride,
                "start": cfg.runtime.start,
                "model": cfg.model.weights,
                "imgsz": cfg.model.imgsz,
                "conf": cfg.model.conf,
                "min_speed_px_s": cfg.motion.min_speed_px_s,
            },
        )
        print(f"[info] debug log: {log_path}", file=sys.stderr)

    def show_frames(canvases) -> bool:
        """Write/show rendered frames; False if the preview window asked to quit."""
        for canvas in canvases:
            if writer is not None:
                writer.write(canvas)
            if args.debug_window:
                cv2.imshow("bikecount", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    return False
                if key == ord(" "):
                    while (cv2.waitKey(50) & 0xFF) not in (ord(" "), ord("q")):
                        pass
        return True

    reporter = None
    if args.report:
        load_env_file(args.env_file)
        reporter = SupabaseReporter.from_env()
        print(
            f"[info] reporting to {reporter.url}",
            file=sys.stderr,
        )

    # systemd stops a service with SIGTERM; unwind like Ctrl-C so the finally block below
    # still gets to send what is queued.
    def _terminate(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _terminate)

    events_fh = open(args.events, "w") if args.events else None
    processed = 0
    started = time.perf_counter()
    interrupted = False

    try:
        with source:
            for frame in source:
                result = pipeline.process(frame.index, frame.timestamp, frame.image)
                processed += 1

                for event in result.events:
                    data = event.as_dict()
                    line = json.dumps(data)
                    if reporter is not None:
                        # The event surfaces a few frames after the crossing; back-date it.
                        reporter.submit(
                            data, time.time() - max(0.0, frame.timestamp - event.timestamp)
                        )
                    if events_fh:
                        events_fh.write(line + "\n")
                        events_fh.flush()
                    else:
                        print(line, flush=True)

                if debug_log is not None:
                    # `processed - 1` is this frame's index in the annotated video: the
                    # writer takes exactly one frame per processed frame.
                    debug_log.frame(processed - 1, result, pipeline.tracker)

                if renderer is not None:
                    # Frames come out `renderer.delay_frames` behind, once the crossings on
                    # them have been judged, so each count shows on its crossing frame.
                    if not show_frames(renderer.push(result, frame.image)):
                        break

                # Only on a terminal: under systemd every redraw would be a journal line.
                if not args.quiet and processed % 50 == 0 and sys.stderr.isatty():
                    elapsed = time.perf_counter() - started
                    print(
                        f"\r[{_hms(frame.timestamp)}] frames={processed} "
                        f"in={result.in_total} out={result.out_total} "
                        f"({processed / elapsed:.1f} fps proc"
                        + (f", {source.measured_fps:.1f} fps in" if source.is_live else "")
                        + ")",
                        end="", file=sys.stderr, flush=True,
                    )
    except KeyboardInterrupt:
        interrupted = True
    finally:
        if renderer is not None and writer is not None:
            # The last `delay_frames` frames are still held back; without them the video
            # would come out shorter than the log that indexes it.
            for canvas in renderer.flush():
                writer.write(canvas)
        if writer is not None:
            writer.release()
        if events_fh is not None:
            events_fh.close()
        if reporter is not None:
            reporter.close()
            print(
                f"\n[report] sent {reporter.sent} event(s)"
                + (f", {reporter.pending} UNSENT (lost)" if reporter.pending else "")
                + (f", {reporter.dropped} dropped from a full queue" if reporter.dropped else ""),
                file=sys.stderr,
            )
        if debug_log is not None:
            debug_log.close(
                in_total=pipeline.in_total,
                out_total=pipeline.out_total,
                counts=pipeline.counts,
                interrupted=interrupted,
            )
        if args.debug_window:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started
    if not args.quiet:
        print(file=sys.stderr)
    if interrupted:
        print("[info] interrupted", file=sys.stderr)

    print(
        f"[done] processed {processed} frames in {elapsed:.1f}s "
        f"({processed / max(elapsed, 1e-9):.1f} fps) | "
        f"IN={pipeline.in_total} OUT={pipeline.out_total}",
        file=sys.stderr,
    )
    for label, c in pipeline.counts.items():
        print(f"[done]   {label:<14} IN={c['in']:<5} OUT={c['out']}", file=sys.stderr)
    if args.debug_out:
        print(f"[done] annotated video: {args.debug_out}", file=sys.stderr)
    if debug_log is not None:
        print(
            f"[done] debug log: {debug_log.path} "
            f"({debug_log.frames} frames, {debug_log.events} events)",
            file=sys.stderr,
        )
        if args.debug_out:
            print(
                f"[done] review it:  python debug_viewer.py {args.debug_out}",
                file=sys.stderr,
            )

    if args.bench:
        print(f"[bench] per-frame mean ms: {pipeline.timings.summary()}", file=sys.stderr)
        realtime = source.effective_fps
        budget = 1000.0 / realtime if realtime else 0
        total = sum(pipeline.timings.mean(s) for s in pipeline.timings.stages)
        print(
            f"[bench] {total:.1f}ms/frame vs {budget:.1f}ms budget at stride "
            f"{cfg.runtime.stride} -> {'REAL-TIME OK' if total < budget else 'TOO SLOW, raise --stride or shrink --imgsz'}",
            file=sys.stderr,
        )

    return 0


def _hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


if __name__ == "__main__":
    raise SystemExit(main())
