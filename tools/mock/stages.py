"""OpenCV/numpy stand-ins for pipeline stages S1..S6.

Parameters here are for mock data only; they are not the stage owners' method decisions.
Conventions follow docs/conventions.md: RGB images, pixel centres at integer coordinates,
H is 3x3 float64 row-major, column vectors, mapping B -> A.
"""
import math

import cv2
import numpy as np

# S1 defaults (docs/s1-preprocessing.md)
S1_SIGMA = 1.0
# S2/S3 defaults for the mock bundle
ORB_FEATURES = 1000
RATIO = 0.8
RANSAC_THRESH_PX = 3.0
RANSAC_MAX_ITERS = 2000
RANSAC_CONFIDENCE = 0.995


def scale_info(w, h, s):
    """Output size and D matrix of S1. Mirrors make_scale_info() in src/common/geometry.hpp."""
    if not (0.0 < s <= 1.0):
        raise ValueError("S must satisfy 0 < S <= 1")
    ww = max(1, int(math.floor(w * s + 0.5)))  # lround: half away from zero (values are positive)
    wh = max(1, int(math.floor(h * s + 0.5)))
    sx, sy = ww / w, wh / h
    d = np.array([[sx, 0, 0.5 * (sx - 1)], [0, sy, 0.5 * (sy - 1)], [0, 0, 1]], dtype=np.float64)
    return {"orig_w": w, "orig_h": h, "work_w": ww, "work_h": wh, "sx": sx, "sy": sy, "D": d}


def s1_gray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def s1_resize(gray, work_w, work_h):
    """Always pass dsize (never fx/fy): OpenCV would otherwise use fx as the scale."""
    if (gray.shape[1], gray.shape[0]) == (work_w, work_h):
        return gray.copy()
    return cv2.resize(gray, (work_w, work_h), interpolation=cv2.INTER_AREA)


def s1_blur(gray, sigma=S1_SIGMA):
    if sigma == 0:
        return gray.copy()
    ksize = 2 * math.ceil(3 * sigma) + 1
    return cv2.GaussianBlur(gray, (ksize, ksize), sigma, borderType=cv2.BORDER_REFLECT_101)


def s1_preprocess(rgb, s, sigma=S1_SIGMA):
    """Returns (gray_uint8 (H_s, W_s), scale_info dict)."""
    info = scale_info(rgb.shape[1], rgb.shape[0], s)
    gray = s1_resize(s1_gray(rgb), info["work_w"], info["work_h"])
    return s1_blur(gray, sigma), info


def s2s3_features(gray, kind="orb"):
    """Returns (keypoints float32 (N,3) [x, y, response], descriptors, descriptor type)."""
    if kind == "orb":
        det = cv2.ORB_create(nfeatures=ORB_FEATURES)
        dtype = "BINARY_U8"
    elif kind == "sift":
        det = cv2.SIFT_create(nfeatures=ORB_FEATURES)
        dtype = "FLOAT32"
    else:
        raise ValueError(kind)
    kps, desc = det.detectAndCompute(gray, None)
    if desc is None or len(kps) == 0:
        width = 32 if kind == "orb" else 128
        return (np.zeros((0, 3), np.float32),
                np.zeros((0, width), np.uint8 if kind == "orb" else np.float32), dtype)
    pts = np.array([[k.pt[0], k.pt[1], k.response] for k in kps], dtype=np.float32)
    return pts, desc, dtype


def s4_match(desc_a, desc_b, dtype, ratio=RATIO):
    """kNN (k=2) ratio test; A is the query. Returns (idx (M,2) int32, dist (M,) float32)."""
    norm = cv2.NORM_HAMMING if dtype == "BINARY_U8" else cv2.NORM_L2
    if len(desc_a) < 1 or len(desc_b) < 2:
        return np.zeros((0, 2), np.int32), np.zeros((0,), np.float32)
    pairs = cv2.BFMatcher(norm).knnMatch(desc_a, desc_b, k=2)
    idx, dist = [], []
    for p in pairs:
        if len(p) == 2 and p[0].distance < ratio * p[1].distance:
            idx.append((p[0].queryIdx, p[0].trainIdx))
            dist.append(p[0].distance)
    return np.array(idx, dtype=np.int32).reshape(-1, 2), np.array(dist, dtype=np.float32)


def to_original(h_s, d_a, d_b):
    """H = D_A^-1 H_s D_B. Mirrors to_original() in src/common/geometry.hpp."""
    return np.linalg.inv(d_a) @ h_s @ d_b


def s5_homography(kp_a, kp_b, idx, d_a, d_b, seed=0):
    """RANSAC homography B -> A. Returns dict with H_s, H, inlier mask (M,) uint8 and counts.

    H_s is None when fewer than 4 matches or RANSAC fails.
    """
    out = {"H_s": None, "H": None, "mask": np.zeros((len(idx),), np.uint8), "num_matches": int(len(idx))}
    if len(idx) < 4:
        return out
    cv2.setRNGSeed(int(seed))
    src = kp_b[idx[:, 1], :2].astype(np.float64)
    dst = kp_a[idx[:, 0], :2].astype(np.float64)
    h_s, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THRESH_PX,
                                   maxIters=RANSAC_MAX_ITERS, confidence=RANSAC_CONFIDENCE)
    if h_s is None:
        return out
    out["H_s"] = h_s.astype(np.float64)
    out["H"] = to_original(out["H_s"], d_a, d_b)
    out["mask"] = mask.reshape(-1).astype(np.uint8)
    return out


def _feather_weight(x, y, w, h):
    """Project-defined relative distance to the source border (docs/s6-warp-blend.md 5.3)."""
    return np.minimum.reduce([(x + 0.5) / w, (w - 0.5 - x) / w, (y + 0.5) / h, (h - 0.5 - y) / h])


def s6_warp_feather(img_a, img_b, h):
    """Inverse warp + feather following docs/s6-warp-blend.md 5.2-5.3 (float64 geometry).

    Returns dict(panorama uint8 (Hc, Wc, 3), mask uint8, offset (ox, oy), canvas (Wc, Hc)).
    """
    ha, wa = img_a.shape[:2]
    hb, wb = img_b.shape[:2]
    corners_b = np.array([[0, 0, 1], [wb - 1, 0, 1], [0, hb - 1, 1], [wb - 1, hb - 1, 1]], np.float64).T
    proj = h @ corners_b
    if np.any(proj[2] == 0) or np.any(np.sign(proj[2]) != np.sign(proj[2][0])):
        raise ValueError("projective denominator changes sign: horizon crosses image B")
    proj = proj[:2] / proj[2]
    xs = np.concatenate([proj[0], [0, wa - 1]])
    ys = np.concatenate([proj[1], [0, ha - 1]])
    x0, x1 = math.floor(xs.min()), math.ceil(xs.max())
    y0, y1 = math.floor(ys.min()), math.ceil(ys.max())
    wc, hc = x1 - x0 + 1, y1 - y0 + 1
    ox, oy = -x0, -y0

    gx, gy = np.meshgrid(np.arange(wc, dtype=np.float64), np.arange(hc, dtype=np.float64))
    xa, ya = gx - ox, gy - oy                                  # p_A = T^-1 p_C
    q = np.linalg.inv(h) @ np.stack([xa.ravel(), ya.ravel(), np.ones(xa.size)])
    with np.errstate(divide="ignore", invalid="ignore"):
        xb, yb = (q[0] / q[2]).reshape(hc, wc), (q[1] / q[2]).reshape(hc, wc)

    def sample(img, x, y):
        w, hh = img.shape[1], img.shape[0]
        valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= hh - 1)
        xs_, ys_ = np.where(valid, x, 0), np.where(valid, y, 0)
        col = cv2.remap(img.astype(np.float32), xs_.astype(np.float32), ys_.astype(np.float32),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        wgt = np.where(valid, _feather_weight(xs_, ys_, w, hh), 0.0)
        return col, valid, wgt

    ca, va, wa_ = sample(img_a, xa, ya)
    cb, vb, wb_ = sample(img_b, xb, yb)
    both = va & vb
    tot = np.where(both, wa_ + wb_, 1.0)
    blended = (wa_[..., None] * ca + wb_[..., None] * cb) / tot[..., None]
    pano = np.where(both[..., None], blended, np.where(va[..., None], ca, np.where(vb[..., None], cb, 0.0)))
    pano = np.clip(np.rint(pano), 0, 255).astype(np.uint8)
    return {"panorama": pano, "mask": (va | vb).astype(np.uint8), "overlap": both.astype(np.uint8),
            "offset": (int(ox), int(oy)), "canvas": (int(wc), int(hc))}


def corner_error(h_est, h_true, w, h):
    """Mean distance (px) between the 4 image corners mapped by two homographies."""
    c = np.array([[0, 0, 1], [w - 1, 0, 1], [0, h - 1, 1], [w - 1, h - 1, 1]], np.float64).T
    a, b = h_est @ c, h_true @ c
    return float(np.mean(np.linalg.norm(a[:2] / a[2] - b[:2] / b[2], axis=0)))
