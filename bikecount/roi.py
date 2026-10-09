"""Region of interest: polygon in normalised coords, cropping, and containment tests."""

from __future__ import annotations

import cv2
import numpy as np

from .config import RoiConfig, ZoneConfig


class Roi:
    """A polygon restricting where detections count, resolved against a frame size.

    Two independent jobs:
      * `crop()` narrows what the detector even looks at (speed + effective resolution),
      * `contains_points()` rejects detections that landed inside the crop box but outside
        the polygon itself.
    """

    def __init__(self, cfg: RoiConfig, width: int, height: int):
        self.cfg = cfg
        self.width = width
        self.height = height

        if cfg.polygon and len(cfg.polygon) >= 3:
            self.polygon = np.array(
                [[x * width, y * height] for x, y in cfg.polygon], dtype=np.float32
            )
            self.is_full_frame = False
        else:
            # Degenerate case: whole frame. Keeps every downstream call branch-free.
            self.polygon = np.array(
                [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
            )
            self.is_full_frame = True

        pad = cfg.crop_padding
        xs, ys = self.polygon[:, 0], self.polygon[:, 1]
        self.x0 = int(max(0, np.floor(xs.min()) - pad))
        self.y0 = int(max(0, np.floor(ys.min()) - pad))
        self.x1 = int(min(width, np.ceil(xs.max()) + pad))
        self.y1 = int(min(height, np.ceil(ys.max()) + pad))

        self._poly_i32 = self.polygon.astype(np.int32)
        self._mask: np.ndarray | None = None

    # ---------------------------------------------------------------- line

    @staticmethod
    def line_points(cfg: RoiConfig, width: int, height: int) -> tuple[tuple[int, int], tuple[int, int]] | None:
        if not cfg.line or len(cfg.line) != 2:
            return None
        (x1, y1), (x2, y2) = cfg.line
        return (int(x1 * width), int(y1 * height)), (int(x2 * width), int(y2 * height))

    # ---------------------------------------------------------------- crop

    @property
    def crop_offset(self) -> tuple[int, int]:
        return (self.x0, self.y0)

    def crop(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        """Return (sub-image, (dx, dy)) where (dx, dy) maps sub-image coords to full frame."""
        if not self.cfg.crop or self.is_full_frame:
            return image, (0, 0)

        sub = image[self.y0 : self.y1, self.x0 : self.x1]

        if self.cfg.mask_outside:
            if self._mask is None or self._mask.shape[:2] != sub.shape[:2]:
                mask = np.zeros(sub.shape[:2], dtype=np.uint8)
                shifted = self._poly_i32 - np.array([self.x0, self.y0], dtype=np.int32)
                cv2.fillPoly(mask, [shifted], 255)
                self._mask = mask
            sub = cv2.bitwise_and(sub, sub, mask=self._mask)

        return sub, (self.x0, self.y0)

    # ----------------------------------------------------------- containment

    def contains_points(self, points: np.ndarray) -> np.ndarray:
        """Boolean mask for an (N, 2) array of full-frame points."""
        if len(points) == 0:
            return np.zeros(0, dtype=bool)
        if self.is_full_frame:
            return np.ones(len(points), dtype=bool)
        return np.array(
            [
                cv2.pointPolygonTest(self._poly_i32, (float(x), float(y)), False) >= 0
                for x, y in points
            ],
            dtype=bool,
        )

    def draw(self, image: np.ndarray, colour=(0, 200, 255), alpha: float = 0.15) -> np.ndarray:
        """Translucent fill + outline. Returns the same array (drawn in place)."""
        if self.is_full_frame:
            return image
        overlay = image.copy()
        cv2.fillPoly(overlay, [self._poly_i32], colour)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, dst=image)
        cv2.polylines(image, [self._poly_i32], isClosed=True, color=colour, thickness=2)
        return image


class Zone:
    """A `ZoneConfig` resolved against a frame size."""

    COLOUR = (255, 160, 60)

    def __init__(self, cfg: ZoneConfig, width: int, height: int):
        self.cfg = cfg
        self.name = cfg.name
        self.relabel = cfg.relabel
        self.keep_vehicle_px = cfg.keep_vehicle_height * height
        self._poly_i32 = np.array(
            [[x * width, y * height] for x, y in cfg.polygon], dtype=np.float32
        ).astype(np.int32)

    def contains_points(self, points: np.ndarray) -> np.ndarray:
        return np.array(
            [cv2.pointPolygonTest(self._poly_i32, (float(x), float(y)), False) >= 0
             for x, y in points],
            dtype=bool,
        )

    def draw(self, image: np.ndarray) -> np.ndarray:
        cv2.polylines(image, [self._poly_i32], isClosed=True, color=self.COLOUR, thickness=1,
                      lineType=cv2.LINE_AA)
        x, y = self._poly_i32[:, 0].min(), self._poly_i32[:, 1].min()
        cv2.putText(image, self.name, (int(x) + 6, int(y) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    self.COLOUR, 1, cv2.LINE_AA)
        return image
