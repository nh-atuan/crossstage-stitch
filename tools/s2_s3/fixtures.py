"""Deterministic grayscale test images for S2/S3 (no dataset needed)."""
from __future__ import annotations

import cv2
import numpy as np


def textured_scene(seed: int = 17, height: int = 480, width: int = 640) -> np.ndarray:
    """Smoothed noise with filled circles and digits, like the S4/S5 T4 fixture."""
    rng = np.random.default_rng(seed)
    image = cv2.GaussianBlur(rng.integers(0, 256, (height, width), dtype=np.uint8), (3, 3), 0.7)
    margin = 24
    for i in range(max(1, height * width // 2200)):
        x, y = map(int, rng.integers([margin, margin], [width - margin, height - margin]))
        colour = 255 if i % 2 else 0
        cv2.circle(image, (x, y), int(rng.integers(6, 19)), colour, -1)
        cv2.putText(image, str(i), (x - 9, y + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                    255 - colour, 1, cv2.LINE_AA)
    return image


def noise(seed: int, height: int, width: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (height, width), dtype=np.uint8)
