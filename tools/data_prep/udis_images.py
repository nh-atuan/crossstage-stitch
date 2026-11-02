"""Bounded uint8 RGB decoding and exact decoded-pixel PPM serialization."""
from __future__ import annotations

import re
import struct

import cv2
import numpy as np
from numpy.typing import NDArray

from tools.repro.io import digest


def source_dimensions(data: bytes) -> tuple[int, int]:
    """Read dimensions before OpenCV allocation; accept PNG, JPEG or P6 PPM."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        if len(data) < 33 or data[12:16] != b"IHDR" or data[8:12] != b"\x00\x00\x00\r":
            raise ValueError("invalid PNG header")
        return struct.unpack(">II", data[16:24])
    if data.startswith(b"\xff\xd8"):
        offset = 2
        while offset < len(data):
            if data[offset] != 255:
                raise ValueError("invalid JPEG marker")
            while offset < len(data) and data[offset] == 255:
                offset += 1
            if offset >= len(data):
                break
            marker = data[offset]
            offset += 1
            if marker in (0xD9, 0xDA):
                break
            if marker == 0x01 or 0xD0 <= marker <= 0xD7:
                continue
            if offset + 2 > len(data):
                break
            length = int.from_bytes(data[offset:offset + 2], "big")
            if length < 2 or offset + length > len(data):
                raise ValueError("invalid JPEG segment length")
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                if length < 8:
                    raise ValueError("truncated JPEG frame")
                height, width = struct.unpack(">HH", data[offset + 3:offset + 7])
                return width, height
            offset += length
        raise ValueError("JPEG dimensions not found")
    if data.startswith(b"P6"):
        header = re.match(rb"P6\s+(?:(?:#[^\r\n]*[\r\n]+)|\s)*(\d+)\s+(\d+)\s+(\d+)\s", data)
        if header and int(header[3]) == 255:
            return int(header[1]), int(header[2])
        raise ValueError("only P6 uint8 PPM is supported")
    raise ValueError("unsupported image signature")


def decode_rgb(data: bytes, max_pixels: int) -> NDArray[np.uint8]:
    width, height = source_dimensions(data)
    if min(width, height) <= 0 or width * height > max_pixels:
        raise ValueError("image exceeds pixel admission limit")
    # UNCHANGED preserves bit depth/channels and ignores EXIF orientation.
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint8 or image.shape != (height, width, 3):
        raise ValueError("require decodable H x W x 3 uint8 image; no alpha/depth/channel coercion")
    return np.ascontiguousarray(image[:, :, ::-1])


def ppm_bytes(rgb: NDArray[np.uint8]) -> bytes:
    height, width, channels = rgb.shape
    if rgb.dtype != np.uint8 or channels != 3:
        raise ValueError("require RGB uint8")
    return f"P6\n{width} {height}\n255\n".encode("ascii") + rgb.tobytes()


def pixel_hash(rgb: NDArray[np.uint8]) -> str:
    height, width = rgb.shape[:2]
    return digest(f"RGB8:{width}:{height}:".encode("ascii") + rgb.tobytes())
