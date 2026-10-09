"""Bicycle detection and line-crossing counting for fixed CCTV cameras."""

from .config import Config
from .detectors import build_detector
from .pipeline import Pipeline, CrossingEvent
from .roi import Roi
from .source import VideoSource, grab_frame
from .tracker import SimpleTracker

__all__ = [
    "Config",
    "CrossingEvent",
    "Pipeline",
    "Roi",
    "SimpleTracker",
    "VideoSource",
    "build_detector",
    "grab_frame",
]
