#!/usr/bin/env python3
"""
AllSky Video Processor
Ubuntu desktop app — compile AllSky timelapse MP4 from .jpg/.png/.fit/.fits images.

Dependencies:
    pip install customtkinter opencv-python numpy scipy astropy
"""

import os
import glob
import threading
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from tkinter import filedialog, messagebox

import base64
import io
import json
import tkinter as _tk

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image as _PILImage

try:
    from PIL import ImageTk as _PILImageTk
    _HAS_PIL_TK = True
except ImportError:
    _HAS_PIL_TK = False

_SETTINGS_FILE  = os.path.expanduser("~/.config/allsky_app/settings.json")

_N_IO_WORKERS   = min(os.cpu_count() or 4, 8)          # parallel image readers
_PREFETCH       = _N_IO_WORKERS * 2                    # sliding-window buffer size
_N_DET_WORKERS  = max(1, (os.cpu_count() or 4) // 2)  # CPU workers for Dg (median)
# GPU morpho runs in a single dedicated thread — no contention, continuous GPU load

# OpenCV names Bayer conversions after the 2×2 block starting at pixel (1, 1),
# not (0, 0): a FITS BAYERPAT "BGGR" sensor needs COLOR_BAYER_RG2BGR (checked
# against a Siril debayer of an IMX477 frame — the same-named constants swap
# red and blue).
_BAYER_MAP = {
    "RGGB": cv2.COLOR_BAYER_BG2BGR,
    "BGGR": cv2.COLOR_BAYER_RG2BGR,
    "GRBG": cv2.COLOR_BAYER_GB2BGR,
    "GBRG": cv2.COLOR_BAYER_GR2BGR,
}

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ─── GHS — Generalized Hyperbolic Stretch ─────────────────────────────────────

def _ghs_on_array(x: np.ndarray, SP: float, D: float, b: float,
                  LP: float, HP: float) -> np.ndarray:
    """
    GHS on a 1-D float64 array in [0, 1].  Used to build the LUT (256 pts).

    Formulas (continuous at SP, T(0)=0, T(1)=1):
      v > SP  : T(v) = SP + (1-SP) * arcsinh(D*(v-SP)/(1-SP)) / arcsinh(D)
      v <= SP : T(v) = SP - SP     * arcsinh(D*(SP-v)/SP)      / arcsinh(D)

    arcsinh is concave → T(v) ≥ v above SP (brightening stretch).
    Lowering SP widens the stretched zone → brighter result, as expected.
    D=0 → identity.  b compresses highlights after the stretch.
    LP/HP: linear extensions to protect shadows / highlights.
    """
    eps = 1e-12
    x = np.clip(x, 0.0, 1.0)

    if D < eps:
        return x.copy()

    asinh_D   = float(np.arcsinh(D))
    range_hi  = max(1.0 - SP, eps)
    range_lo  = max(SP, eps)

    def _core(v):
        out = np.empty_like(v)
        hi = v > SP
        lo = ~hi
        if hi.any():
            t = np.clip((v[hi] - SP) / range_hi, 0.0, 1.0)
            out[hi] = SP + (1.0 - SP) * np.arcsinh(D * t) / asinh_D
        if lo.any():
            t = np.clip((SP - v[lo]) / range_lo, 0.0, 1.0)
            out[lo] = SP - SP * np.arcsinh(D * t) / asinh_D
        if b > 0.0:
            above = out > SP
            exc = out[above] - SP
            out[above] = SP + exc / (1.0 + b * exc)
        return out

    m_shadow = x <= LP
    m_high   = x >= HP
    m_core   = (~m_shadow) & (~m_high)
    result   = np.empty_like(x)

    if m_core.any():
        result[m_core] = _core(x[m_core])

    if m_shadow.any():
        if LP > eps:
            t_LP = float(_core(np.array([LP]))[0])
            result[m_shadow] = x[m_shadow] * (t_LP / LP)
        else:
            result[m_shadow] = x[m_shadow]

    if m_high.any():
        if HP < 1.0 - eps:
            t_HP = float(_core(np.array([HP]))[0])
            result[m_high] = t_HP + (x[m_high] - HP) / (1.0 - HP) * (1.0 - t_HP)
        else:
            result[m_high] = x[m_high]

    return result


def build_ghs_lut(SP: float, D: float, b: float,
                  LP: float, HP: float,
                  gamma: float = 1.0) -> np.ndarray:
    """
    Precompute a 256-entry uint8 LUT for GHS (called once per video).
    cv2.LUT() then applies it in SIMD to every frame: ~0.5 ms vs ~40 ms
    for per-frame sinh() over millions of pixels.

    gamma > 1  (typiquement 2.2 pour JPEG ISP sRGB) :
      1. Délinéarise l'entrée  x_lin = x ^ gamma     (ISP gamma → espace linéaire)
      2. Applique le GHS sur données linéaires
      3. Ré-encode la sortie   y_enc = y ^ (1/gamma)  (retour espace affichage)
    gamma = 1.0 → aucune correction (données déjà linéaires ou RAW).
    """
    x = np.arange(256, dtype=np.float64) / 255.0
    if gamma > 1.01:
        np.power(x, gamma, out=x)
    y = _ghs_on_array(x, SP, D, b, LP, HP)
    if gamma > 1.01:
        np.power(np.clip(y, 0.0, 1.0), 1.0 / gamma, out=y)
    return (np.clip(y, 0.0, 1.0) * 255.0).astype(np.uint8)


# ─── FITS loader ──────────────────────────────────────────────────────────────

def _load_fits(path: str, bayer_code=None) -> np.ndarray | None:
    """
    Load a FITS file → float32 BGR array normalised to [0, 255].
    bayer_code: cv2.COLOR_BAYER_* for raw Bayer FITS, or None (grayscale → 3ch).
    Supports uint16 (ZWO/ASI standard), uint8, int16/int32, float32/float64.
    """
    try:
        from astropy.io import fits as _fits
        with _fits.open(path, memmap=False) as hdul:
            data = hdul[0].data
            if data is None:
                for hdu in hdul[1:]:
                    if hdu.data is not None and hdu.data.ndim >= 2:
                        data = hdu.data
                        break
            if data is None:
                return None

            arr = np.array(data, dtype=np.float64)

            # Percentile stretch: map p0.5–p99.5 to [0,255].
            # More robust than dtype-range normalization for AllSky FITS where
            # the actual sky signal typically uses only a small fraction of the
            # ADU range (e.g. 300–8000 out of 65535 for a cooled ASI camera).
            p_lo = float(np.percentile(arr, 0.5))
            p_hi = float(np.percentile(arr, 99.5))
            if p_hi > p_lo:
                arr = (arr - p_lo) * 255.0 / (p_hi - p_lo)

            arr = np.clip(arr, 0.0, 255.0).astype(np.float32)

            if arr.ndim == 2:
                # Grayscale or raw Bayer
                if bayer_code is not None:
                    bgr = cv2.cvtColor(arr.astype(np.uint8), bayer_code).astype(np.float32)
                else:
                    bgr = np.stack([arr, arr, arr], axis=2)
            elif arr.ndim == 3:
                # Detect (C, H, W) vs (H, W, C)
                if arr.shape[0] in (1, 3) and arr.shape[0] < arr.shape[1] // 2:
                    arr = arr.transpose(1, 2, 0)
                if arr.shape[2] == 1:
                    g = arr[:, :, 0]
                    bgr = np.stack([g, g, g], axis=2)
                elif arr.shape[2] == 3:
                    bgr = arr[:, :, ::-1].copy()   # RGB → BGR
                else:
                    g = arr[:, :, 0]
                    bgr = np.stack([g, g, g], axis=2)
            else:
                return None

            return bgr

    except ImportError:
        return None
    except Exception:
        return None


# ─── I/O helper (top-level → picklable, used by ThreadPoolExecutor) ──────────

def _load_frame_group(args: tuple) -> np.ndarray:
    """
    Load one group of image paths and return a float32 frame.
    Group size = 1 (no stacking) or N (rolling stack → mean).
    fallback: pre-read ndarray used when a file is unreadable.
    bayer_code: cv2.COLOR_BAYER_* for raw FITS, or None.
    """
    paths_group, fallback, bayer_code = args
    imgs: list[np.ndarray] = []
    for p in paths_group:
        ext = os.path.splitext(p)[1].lower()
        if ext in ('.fit', '.fits'):
            img = _load_fits(p, bayer_code)
        else:
            img = cv2.imread(p)
        imgs.append(img if img is not None else (imgs[-1] if imgs else fallback))
    if len(imgs) == 1:
        return imgs[0].astype(np.float32)
    acc = imgs[0].astype(np.float32)
    for img in imgs[1:]:
        acc += img.astype(np.float32)
    return acc / len(imgs)


# ─── Color correction helper ──────────────────────────────────────────────────

def _apply_color_correction(frame_u8: np.ndarray,
                             saturation: float,
                             rgb_r: float, rgb_g: float, rgb_b: float) -> np.ndarray:
    """Apply saturation then per-channel RGB balance to a uint8 BGR frame."""
    if abs(saturation - 1.0) > 0.01:
        hsv = cv2.cvtColor(frame_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        frame_u8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if abs(rgb_r - 1.0) > 0.01 or abs(rgb_g - 1.0) > 0.01 or abs(rgb_b - 1.0) > 0.01:
        f = frame_u8.astype(np.float32)
        f[:, :, 2] *= rgb_r   # BGR: ch2=R, ch1=G, ch0=B
        f[:, :, 1] *= rgb_g
        f[:, :, 0] *= rgb_b
        frame_u8 = np.clip(f, 0, 255).astype(np.uint8)
    return frame_u8


# ─── Satellite enhancement helpers ───────────────────────────────────────────

def compute_global_scale(paths: list, sample: int = 30,
                          bayer_code=None) -> tuple:
    """
    Sample up to `sample` FITS images to estimate a single (p_lo, p_hi)
    pair for radiometrically consistent linear loading across a sequence.
    For JPEG/PNG sequences returns (0.0, 255.0) — no-op.
    """
    try:
        from astropy.io import fits as _fits
        has_astropy = True
    except ImportError:
        has_astropy = False

    fits_paths = [p for p in paths
                  if os.path.splitext(p)[1].lower() in ('.fit', '.fits')]
    if not fits_paths or not has_astropy:
        return 0.0, 255.0

    step = max(1, len(fits_paths) // sample)
    sampled = fits_paths[::step][:sample]
    lows, highs = [], []
    for p in sampled:
        try:
            from astropy.io import fits as _fits
            with _fits.open(p, memmap=False) as hdul:
                data = hdul[0].data
                if data is None:
                    continue
                arr = np.array(data, dtype=np.float64).ravel()
                lows.append(float(np.percentile(arr, 0.5)))
                highs.append(float(np.percentile(arr, 99.5)))
        except Exception:
            continue

    if not lows:
        return 0.0, 255.0
    p_lo = float(np.median(lows))
    p_hi = float(np.median(highs))
    return p_lo, max(p_hi, p_lo + 1.0)


def _load_fits_linear(path: str, bayer_code,
                       p_lo: float, p_hi: float) -> "np.ndarray | None":
    """Load FITS with a globally pre-computed scale (no per-frame stretch)."""
    try:
        from astropy.io import fits as _fits
        with _fits.open(path, memmap=False) as hdul:
            data = hdul[0].data
            if data is None:
                for hdu in hdul[1:]:
                    if hdu.data is not None and hdu.data.ndim >= 2:
                        data = hdu.data
                        break
            if data is None:
                return None
            arr = (np.array(data, dtype=np.float32) - p_lo) * 255.0 / (p_hi - p_lo)
            arr = np.clip(arr, 0.0, 255.0)
            if arr.ndim == 2:
                if bayer_code is not None:
                    bgr = cv2.cvtColor(arr.astype(np.uint8), bayer_code).astype(np.float32)
                else:
                    bgr = np.stack([arr, arr, arr], axis=2)
            elif arr.ndim == 3:
                if arr.shape[0] in (1, 3) and arr.shape[0] < arr.shape[1] // 2:
                    arr = arr.transpose(1, 2, 0)
                if arr.shape[2] == 1:
                    g = arr[:, :, 0]
                    bgr = np.stack([g, g, g], axis=2)
                elif arr.shape[2] == 3:
                    bgr = arr[:, :, ::-1].copy()
                else:
                    g = arr[:, :, 0]
                    bgr = np.stack([g, g, g], axis=2)
            else:
                return None
            return bgr
    except Exception:
        return None


def _load_frame_linear(args: tuple) -> np.ndarray:
    """
    ThreadPoolExecutor-compatible loader with global linear scale.
    args = (path, fallback_array, bayer_code, p_lo, p_hi)
    JPEG/PNG: p_lo/p_hi ignored — returned as-is in float32.
    """
    path, fallback, bayer_code, p_lo, p_hi = args
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.fit', '.fits'):
        img = _load_fits_linear(path, bayer_code, p_lo, p_hi)
    else:
        img = cv2.imread(path)
    if img is None:
        return fallback.astype(np.float32)
    return img.astype(np.float32)


# ─── Video processing (background thread) ────────────────────────────────────

def process_video(params: dict,
                  progress_cb,   # (done: int, total: int) -> None
                  done_cb,        # (out_path: str) -> None
                  error_cb):      # (message: str) -> None
    """
    Full pipeline: load → stack → boost satellites → EMA →
                   contraste/luminosité → saturation → balance RVB → GHS → encode.

    Optimisations :
      • GHS via LUT 256 pts précompilé une seule fois → cv2.LUT() SIMD (~0.5 ms/frame)
      • ThreadPoolExecutor : _PREFETCH lectures disque en parallèle en avance de phase
    """
    try:
        src_dir    = params["src_dir"]
        out_path   = params["out_path"]
        fps        = params["fps"]
        use_stack        = params["use_stack"]
        stack_n          = params["stack_n"]
        use_ema          = params["use_ema"]
        alpha_ema        = params["alpha_ema"]
        sat_boost        = params.get("sat_boost", 1.0)
        sat_thresh       = params.get("sat_thresh", 0.0)
        contrast         = params["contrast"]
        brightness       = params["brightness"]
        saturation       = params.get("saturation", 1.0)
        rgb_r            = params.get("rgb_r", 1.0)
        rgb_g            = params.get("rgb_g", 1.0)
        rgb_b            = params.get("rgb_b", 1.0)
        use_ghs    = params["use_ghs"]
        ghs_params = params["ghs"]
        bayer_code = params.get("bayer_code", None)

        # ── Collect images ─────────────────────────────────────────────────
        exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG",
                "*.fit", "*.fits", "*.FIT", "*.FITS")
        paths: list[str] = []
        for e in exts:
            paths.extend(glob.glob(os.path.join(src_dir, e)))
        paths.sort()

        if not paths:
            error_cb("Aucune image (.jpg/.jpeg/.png/.fit/.fits) trouvée dans le dossier sélectionné.")
            return

        n = len(paths)
        if use_stack and n < stack_n:
            error_cb(f"Pas assez d'images pour le stacking "
                     f"(N={stack_n} demandé, {n} trouvé{'s' if n > 1 else ''}).")
            return

        total_frames = (n - stack_n + 1) if use_stack else n

        # ── Probe first readable image ─────────────────────────────────────
        first_img = None
        for p in paths:
            ext = os.path.splitext(p)[1].lower()
            if ext in ('.fit', '.fits'):
                first_img = _load_fits(p, bayer_code)
            else:
                first_img = cv2.imread(p)
            if first_img is not None:
                break
        if first_img is None:
            error_cb("Impossible de lire la première image. Vérifiez le format.")
            return
        h, w = first_img.shape[:2]

        # ── Init VideoWriter ───────────────────────────────────────────────
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            error_cb(f"Impossible d'ouvrir le fichier de sortie :\n{out_path}")
            return

        # ── Precompute GHS LUT once (256 valeurs, ~1 ms) ──────────────────
        ghs_lut: np.ndarray | None = None
        if use_ghs:
            ghs_lut = build_ghs_lut(**ghs_params)

        # ── Prepare load-argument list ─────────────────────────────────────
        if use_stack:
            load_args = [([paths[i + j] for j in range(stack_n)], first_img, bayer_code)
                         for i in range(total_frames)]
        else:
            load_args = [([paths[i]], first_img, bayer_code) for i in range(total_frames)]

        # ── Main loop with I/O prefetch ────────────────────────────────────
        ema_state: np.ndarray | None = None
        ema_bg:    np.ndarray | None = None

        with ThreadPoolExecutor(max_workers=_N_IO_WORKERS) as io_pool:
            pending: dict[int, object] = {}
            next_submit = 0

            def _fill_prefetch(up_to: int) -> None:
                nonlocal next_submit
                while next_submit < min(up_to, total_frames):
                    pending[next_submit] = io_pool.submit(
                        _load_frame_group, load_args[next_submit])
                    next_submit += 1

            _fill_prefetch(_PREFETCH)

            for i in range(total_frames):
                frame_f: np.ndarray = pending.pop(i).result()
                _fill_prefetch(i + 1 + _PREFETCH)

                # ── 2. Boost satellites ───────────────────────────────────
                if sat_boost > 1.01:
                    if ema_bg is None:
                        ema_bg = frame_f.copy()
                    else:
                        cv2.addWeighted(frame_f, 0.02, ema_bg, 0.98, 0.0, dst=ema_bg)
                        excess = np.maximum(frame_f - ema_bg - sat_thresh, 0.0)
                        frame_f = frame_f + (sat_boost - 1.0) * excess

                # ── 3. EMA ────────────────────────────────────────────────
                if use_ema:
                    if ema_state is None:
                        ema_state = frame_f.copy()
                    else:
                        cv2.addWeighted(frame_f, alpha_ema,
                                        ema_state, 1.0 - alpha_ema,
                                        0.0, dst=ema_state)
                    frame_f = ema_state.copy()

                # ── 4. Contraste centré sur 128 + décalage luminosité ─────
                # beta_c centres the multiplication on 128 so contrast=1.0
                # always means "no change" regardless of brightness offset.
                beta_c = brightness - 128.0 * (contrast - 1.0)
                frame_u8 = cv2.convertScaleAbs(frame_f, alpha=contrast, beta=beta_c)

                # ── 5. Saturation + Balance RVB ───────────────────────────
                frame_u8 = _apply_color_correction(frame_u8, saturation, rgb_r, rgb_g, rgb_b)

                # ── 6. GHS via precomputed LUT (SIMD, ~0.5 ms/frame) ─────
                if ghs_lut is not None:
                    frame_u8 = cv2.LUT(frame_u8, ghs_lut)

                # ── 7. Encode ─────────────────────────────────────────────
                if frame_u8.shape[:2] != (h, w):
                    frame_u8 = cv2.resize(frame_u8, (w, h))
                writer.write(frame_u8)

                if i % 20 == 0 or i == total_frames - 1:
                    progress_cb(i + 1, total_frames)

        writer.release()
        done_cb(out_path)

    except Exception as exc:
        error_cb(f"{exc}\n\n{traceback.format_exc()}")


# ─── Star Trail processing ────────────────────────────────────────────────────

def process_star_trail(params: dict, progress_cb, done_cb, error_cb):
    """
    Star Trail : accumulation pixel-par-pixel par maximum (lighten blend).

    Progressive → MP4 animé où les traînées se construisent image par image.
    Finale      → PNG unique = max de toutes les images.

    Le paramètre decay (0.90–1.00) applique une atténuation à chaque étape :
      1.00 = traînées permanentes  |  < 1.00 = effet comète (les vieilles traînées
      s'effacent progressivement).
    """
    try:
        src_dir    = params["src_dir"]
        out_path   = params["out_path"]
        fps        = params["fps"]
        star_mode  = params["star_mode"]    # "progressive" | "final"
        decay      = params["star_decay"]
        sat_boost   = params["star_boost"]
        sat_thresh  = params["star_thresh"]
        contrast   = params["contrast"]
        brightness = params["brightness"]
        saturation = params.get("saturation", 1.0)
        rgb_r      = params.get("rgb_r", 1.0)
        rgb_g      = params.get("rgb_g", 1.0)
        rgb_b      = params.get("rgb_b", 1.0)
        use_ghs    = params["use_ghs"]
        ghs_params = params["ghs"]
        bayer_code = params.get("bayer_code", None)

        # ── Collect images ─────────────────────────────────────────────────
        exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG",
                "*.fit", "*.fits", "*.FIT", "*.FITS")
        paths: list[str] = []
        for e in exts:
            paths.extend(glob.glob(os.path.join(src_dir, e)))
        paths.sort()

        if not paths:
            error_cb("Aucune image trouvée dans le répertoire sélectionné.")
            return

        first_img = None
        for p in paths:
            ext = os.path.splitext(p)[1].lower()
            if ext in ('.fit', '.fits'):
                first_img = _load_fits(p, bayer_code)
            else:
                first_img = cv2.imread(p)
            if first_img is not None:
                break
        if first_img is None:
            error_cb("Impossible de lire la première image.")
            return
        h, w = first_img.shape[:2]

        ghs_lut = build_ghs_lut(**ghs_params) if use_ghs else None
        total   = len(paths)
        load_args = [([p], first_img, bayer_code) for p in paths]

        # ── Init VideoWriter (progressive only) ────────────────────────────
        writer = None
        if star_mode == "progressive":
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            if not writer.isOpened():
                error_cb(f"Impossible d'ouvrir le fichier de sortie :\n{out_path}")
                return

        acc:    np.ndarray | None = None
        ema_bg: np.ndarray | None = None

        with ThreadPoolExecutor(max_workers=_N_IO_WORKERS) as pool:
            pending: dict[int, object] = {}
            next_submit = 0

            def fill(up_to: int) -> None:
                nonlocal next_submit
                while next_submit < min(up_to, total):
                    pending[next_submit] = pool.submit(
                        _load_frame_group, load_args[next_submit])
                    next_submit += 1

            fill(_PREFETCH)

            for i in range(total):
                frame_f = pending.pop(i).result()
                fill(i + 1 + _PREFETCH)

                if ema_bg is None:
                    ema_bg = frame_f.copy()
                    frame_proc = frame_f
                else:
                    cv2.addWeighted(frame_f, 0.02, ema_bg, 0.98, 0.0, dst=ema_bg)
                    if sat_boost > 1.01:
                        excess = np.maximum(frame_f - ema_bg - sat_thresh, 0.0)
                        frame_proc = frame_f + (sat_boost - 1.0) * excess
                    else:
                        frame_proc = frame_f

                if acc is None:
                    acc = frame_proc.copy()
                else:
                    if decay < 1.0:
                        acc *= decay
                    np.maximum(acc, frame_proc, out=acc)

                if writer is not None:
                    beta_c = brightness - 128.0 * (contrast - 1.0)
                    frame_u8 = cv2.convertScaleAbs(acc, alpha=contrast, beta=beta_c)
                    frame_u8 = _apply_color_correction(frame_u8, saturation, rgb_r, rgb_g, rgb_b)
                    if ghs_lut is not None:
                        frame_u8 = cv2.LUT(frame_u8, ghs_lut)
                    writer.write(frame_u8)

                if i % 20 == 0 or i == total - 1:
                    progress_cb(i + 1, total)

        if writer is not None:
            writer.release()

        # ── Mode final : sauvegarde PNG ────────────────────────────────────
        if star_mode == "final":
            beta_c = brightness - 128.0 * (contrast - 1.0)
            final_u8 = cv2.convertScaleAbs(acc, alpha=contrast, beta=beta_c)
            final_u8 = _apply_color_correction(final_u8, saturation, rgb_r, rgb_g, rgb_b)
            if ghs_lut is not None:
                final_u8 = cv2.LUT(final_u8, ghs_lut)
            png_path = os.path.splitext(out_path)[0] + "_startrail.png"
            cv2.imwrite(png_path, final_u8)
            done_cb(png_path)
        else:
            done_cb(out_path)

    except Exception as exc:
        error_cb(f"{exc}\n\n{traceback.format_exc()}")


# ─── Satellite / aircraft trail enhancement ──────────────────────────────────

def _compute_Dg(args: tuple) -> np.ndarray:
    """
    CPU worker — loads detection-window frames, computes half-resolution median
    background, returns the raw difference map Dg (float32, full resolution).

    Runs in parallel across _N_DET_WORKERS threads; each worker is fully
    independent (no shared mutable state).  Median is computed at ÷2 linear
    resolution (÷4 data volume) because the background is smooth — no visible
    loss on the difference map.

    args = (paths_window, center_idx, bayer_code, p_lo, p_hi, sat_thresh, mask_f32)
    """
    paths_window, center_idx, bayer_code, p_lo, p_hi, sat_thresh, mask_f32 = args

    fallback: "np.ndarray | None" = None
    frames: list = []
    for p in paths_window:
        ext = os.path.splitext(p)[1].lower()
        if ext in ('.fit', '.fits'):
            img = _load_fits_linear(p, bayer_code, p_lo, p_hi)
        else:
            img = cv2.imread(p)
        if img is not None:
            img = img.astype(np.float32)
            if fallback is None:
                fallback = img
        else:
            img = fallback
        frames.append(img)

    frames = [f for f in frames if f is not None]
    if not frames:
        return np.zeros((1, 1), np.float32)

    det_stack = np.stack(frames, axis=0)              # (N, H, W, 3) float32
    ci = min(center_idx, len(det_stack) - 1)

    # Grayscale luminance — 3× less data than RGB
    gray = (0.114 * det_stack[:, :, :, 0]
          + 0.587 * det_stack[:, :, :, 1]
          + 0.299 * det_stack[:, :, :, 2])            # (N, H, W)

    # Median at half resolution — background is spatially smooth
    h_full, w_full = gray.shape[1], gray.shape[2]
    gray_half = np.stack([
        cv2.resize(gray[i], (w_full // 2, h_full // 2), interpolation=cv2.INTER_AREA)
        for i in range(len(gray))
    ])
    fond_half = np.median(gray_half, axis=0)
    fond = cv2.resize(fond_half, (w_full, h_full), interpolation=cv2.INTER_LINEAR)

    Dg = np.maximum(gray[ci] - fond - sat_thresh, 0.0)
    if mask_f32 is not None:
        Dg *= mask_f32

    return Dg


def _hough_E(dg_futures_list: list, current_idx: int,
             hough_thresh: int, min_line_len: int,
             max_line_gap: int, tunnel_px: int) -> np.ndarray:
    """
    Worker — waits for one or more CPU Dg futures, then isolates linear
    streaks via cv2.HoughLinesP.

    When dg_futures_list contains multiple futures, their Dg maps are
    combined with a per-pixel maximum before binarising and running
    HoughLinesP.  Faint trails that fall below the detection threshold on
    any single frame accumulate enough combined signal to form a detectable
    line.  The tunnel mask is then applied only to the current frame's Dg
    (dg_futures_list[current_idx]) so rendered pixel values keep their
    natural per-frame texture (intensity flicker, motion blur).
    """
    dg_maps = [f.result() for f in dg_futures_list]
    Dg_current = dg_maps[current_idx]

    Dg_detect = (np.max(np.stack(dg_maps), axis=0)
                 if len(dg_maps) > 1 else dg_maps[0])

    binary = (Dg_detect > 0).astype(np.uint8) * 255
    lines = cv2.HoughLinesP(binary, rho=1, theta=np.pi / 180,
                            threshold=hough_thresh,
                            minLineLength=min_line_len,
                            maxLineGap=max_line_gap)

    if lines is None:
        return np.zeros_like(Dg_current)

    tunnel = np.zeros_like(binary)
    for x1, y1, x2, y2 in lines[:, 0]:
        cv2.line(tunnel, (x1, y1), (x2, y2), 255, thickness=tunnel_px)

    return Dg_current * (tunnel > 0)


def process_satellites(params: dict, progress_cb, done_cb, error_cb):
    """
    Two-track satellite / aircraft trail enhancement pipeline.

    Track A (detection, short window):
      global-scale load → centered temporal median → subtract → threshold →
      Hough line isolation (tunnel mask on raw pixels) → persistence accumulator

    Track B (background, long EMA):
      EMA smoothed sky background → contrast/GHS stretch

    Output = stretch_doux(B) + β · stretch_dur(A)
    """
    try:
        src_dir    = params["src_dir"]
        out_path   = params["out_path"]
        fps        = params["fps"]
        bayer_code = params.get("bayer_code", None)

        win_det    = params["sat_win_det"] | 1          # force odd
        sat_thresh = float(params["sat_thresh2"])
        hough_thresh    = int(params.get("sat_hough_thresh", 15))
        hough_min_len   = int(params.get("sat_hough_min_len", 12))
        hough_max_gap   = int(params.get("sat_hough_max_gap", 4))
        hough_tunnel_px = int(params.get("sat_hough_tunnel", 5))
        sat_temporal_win = max(1, int(params.get("sat_temporal_win", 3)))
        sat_hough_dg_win = max(1, int(params.get("sat_hough_dg_win", 3)))
        sat_decay  = float(params["sat_decay"])
        win_bg     = int(params["sat_win_bg"])
        bg_src     = params["sat_bg_src"]               # "mean" | "current"
        beta       = float(params["sat_beta"])
        color_mode = params["sat_color"]                # "white" | "tinted"
        use_mask   = bool(params["sat_use_mask"])
        mask_rad   = float(params.get("sat_mask_radius", 0.45))

        contrast   = params["contrast"]
        brightness = params["brightness"]
        saturation = params.get("saturation", 1.0)
        rgb_r      = params.get("rgb_r", 1.0)
        rgb_g      = params.get("rgb_g", 1.0)
        rgb_b      = params.get("rgb_b", 1.0)
        use_ghs    = params["use_ghs"]
        ghs_params = params["ghs"]

        sat_sp = float(params.get("sat_stretch_SP", 0.02))
        sat_d  = float(params.get("sat_stretch_D",  12.0))

        # ── Collect images ──────────────────────────────────────────────────
        exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG",
                "*.fit", "*.fits", "*.FIT", "*.FITS")
        paths: list = []
        for e in exts:
            paths.extend(glob.glob(os.path.join(src_dir, e)))
        paths.sort()

        if not paths:
            error_cb("Aucune image trouvée dans le répertoire sélectionné.")
            return

        n = len(paths)

        # ── Probe first readable image ──────────────────────────────────────
        first_img = None
        for p in paths:
            ext = os.path.splitext(p)[1].lower()
            if ext in ('.fit', '.fits'):
                img = _load_fits_linear(p, bayer_code, 0.0, 65535.0)
            else:
                img = cv2.imread(p)
            if img is not None:
                first_img = img.astype(np.float32)
                break
        if first_img is None:
            error_cb("Impossible de lire la première image.")
            return
        h, w = first_img.shape[:2]

        # ── Global radiometric scale (FITS only) ────────────────────────────
        p_lo, p_hi = compute_global_scale(paths, bayer_code=bayer_code)

        # ── Horizon mask ────────────────────────────────────────────────────
        mask_f32 = None
        if use_mask:
            mask_f32 = np.zeros((h, w), np.float32)
            r = int(mask_rad * min(h, w))
            cv2.circle(mask_f32, (w // 2, h // 2), r, 1.0, -1)

        # ── GHS LUTs ────────────────────────────────────────────────────────
        ghs_lut = build_ghs_lut(**ghs_params) if use_ghs else None
        sat_lut = build_ghs_lut(SP=sat_sp, D=sat_d, b=0.0, LP=0.0, HP=1.0)

        # ── VideoWriter ──────────────────────────────────────────────────────
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            error_cb(f"Impossible d'ouvrir le fichier de sortie :\n{out_path}")
            return

        # ── Build detection task args ────────────────────────────────────────
        #
        #   Pipeline architecture:
        #     cpu_pool   (_N_DET_WORKERS threads): load frames + half-res median → Dg
        #     hough_pool (_N_DET_WORKERS threads): HoughLinesP + tunnel mask    → E
        #
        #   HoughLinesP is CPU-only (no OpenCL UMat support), so unlike the old
        #   morphology stage it parallelizes freely across threads instead of
        #   being pinned to a single GPU command queue.
        k = win_det // 2
        cpu_task_args = []
        for i in range(n):
            det_lo = max(0, i - k)
            det_hi = min(n - 1, i + k)
            cpu_task_args.append((
                [paths[j] for j in range(det_lo, det_hi + 1)],
                i - det_lo,           # center_idx inside window
                bayer_code, p_lo, p_hi, sat_thresh, mask_f32,
            ))

        # EMA background state
        bg_ema:        "np.ndarray | None" = None
        alpha_bg = 2.0 / (win_bg + 1)

        # Rolling mean background state
        bg_roll_sum:   "np.ndarray | None" = None
        bg_roll_deque: deque = deque()

        # How many detection frames to keep in-flight simultaneously.
        # Bounds peak memory: LOOK_AHEAD × 8 MB (Dg float32 1920×1080).
        # Keep enough to saturate cpu_pool + hough_pool without buffering
        # thousands of frames for a full night's sequence.
        _LOOK_AHEAD = max(_N_DET_WORKERS * 4, 32)

        # Three pools: I/O (render frames), CPU detection (median), Hough (line isolation)
        with ThreadPoolExecutor(max_workers=_N_IO_WORKERS)  as io_pool,    \
             ThreadPoolExecutor(max_workers=_N_DET_WORKERS) as cpu_pool,   \
             ThreadPoolExecutor(max_workers=_N_DET_WORKERS) as hough_pool:

            dg_futures:  dict = {}   # i → Future[Dg]
            det_futures: dict = {}   # i → Future[E]

            def _submit_det(i: int):
                dg_futures[i] = cpu_pool.submit(_compute_Dg, cpu_task_args[i])
                # Pass a sliding window of Dg futures so HoughLinesP detects
                # on their per-pixel max — faint trails missed on a single frame
                # gain enough combined signal to cross the line threshold.
                # Popping dg_futures[i] later is safe: the Future objects remain
                # alive as long as the hough task closures that reference them.
                lo = max(0, i - sat_hough_dg_win + 1)
                dg_list = [dg_futures[j] for j in range(lo, i + 1)]
                det_futures[i] = hough_pool.submit(
                    _hough_E, dg_list, i - lo,
                    hough_thresh, hough_min_len, hough_max_gap, hough_tunnel_px)

            # Pre-fill the look-ahead window
            for i in range(min(_LOOK_AHEAD, n)):
                _submit_det(i)

            # Pre-fetch center frames (for background + render) via io_pool
            frame_pending: dict = {}
            next_frame_submit = [0]

            def _submit_frames_up_to(idx: int):
                while next_frame_submit[0] <= min(idx, n - 1):
                    j = next_frame_submit[0]
                    frame_pending[j] = io_pool.submit(
                        _load_frame_linear,
                        (paths[j], first_img, bayer_code, p_lo, p_hi))
                    next_frame_submit[0] += 1

            _submit_frames_up_to(_PREFETCH - 1)

            # Persistence accumulator (mono, float32)
            acc: "np.ndarray | None" = None
            # Temporal max-pool buffer: bridges frames where HoughLinesP misses
            # the trail (signal near threshold). max over the last sat_temporal_win
            # E maps so a gap of up to (sat_temporal_win-1) frames is filled.
            e_buf: deque = deque(maxlen=sat_temporal_win)

            for i in range(n):
                # Keep the look-ahead window full
                next_det = i + _LOOK_AHEAD
                if next_det < n:
                    _submit_det(next_det)

                _submit_frames_up_to(min(i + _PREFETCH, n - 1))

                # Collect detection result and release Dg memory immediately
                E = det_futures.pop(i).result()
                dg_futures.pop(i, None)

                # Collect center frame for background/render
                frame = frame_pending.pop(i).result()   # float32 BGR [0,255]

                # Resize E if detection returned a stub (bad file)
                if E.shape != (h, w):
                    E = np.zeros((h, w), np.float32)

                # Temporal max-pool: keeps signal alive across missed frames
                e_buf.append(E)
                E_smooth = (np.max(np.stack(e_buf), axis=0)
                            if len(e_buf) > 1 else E)

                # Persistence accumulator (sequential — depends on previous acc)
                if acc is None:
                    acc = E_smooth.copy()
                else:
                    acc = np.maximum(E_smooth, acc * sat_decay)

                # ── Track B: background ─────────────────────────────────────
                if bg_src == "current":
                    bg = frame
                elif bg_src == "rolling":
                    if bg_roll_sum is None:
                        bg_roll_sum = frame.astype(np.float64)
                    else:
                        bg_roll_sum += frame.astype(np.float64)
                    bg_roll_deque.append(frame)
                    if len(bg_roll_deque) > win_bg:
                        old = bg_roll_deque.popleft()
                        bg_roll_sum -= old.astype(np.float64)
                    bg = (bg_roll_sum / len(bg_roll_deque)).astype(np.float32)
                else:   # "mean" — EMA
                    if bg_ema is None:
                        bg_ema = frame.copy()
                    else:
                        cv2.addWeighted(frame, alpha_bg,
                                        bg_ema, 1.0 - alpha_bg,
                                        0.0, dst=bg_ema)
                    bg = bg_ema

                # ── Render background (Track B stretch) ─────────────────────
                beta_c = brightness - 128.0 * (contrast - 1.0)
                bg_u8 = cv2.convertScaleAbs(bg, alpha=contrast, beta=beta_c)
                bg_u8 = _apply_color_correction(bg_u8, saturation, rgb_r, rgb_g, rgb_b)
                if ghs_lut is not None:
                    bg_u8 = cv2.LUT(bg_u8, ghs_lut)

                # ── Render satellite trail (Track A aggressive stretch) ───────
                acc_u8 = np.clip(acc, 0, 255).astype(np.uint8)
                acc_s  = cv2.LUT(acc_u8, sat_lut).astype(np.float32)

                out_f = bg_u8.astype(np.float32)
                if color_mode == "white":
                    out_f[:, :, 0] += beta * acc_s
                    out_f[:, :, 1] += beta * acc_s
                    out_f[:, :, 2] += beta * acc_s
                else:   # tinted: frame chrominance scaled by detection intensity
                    lum = (0.114 * frame[:, :, 0]
                         + 0.587 * frame[:, :, 1]
                         + 0.299 * frame[:, :, 2]) + 1.0
                    for c in range(3):
                        out_f[:, :, c] += beta * acc_s * (frame[:, :, c] / lum)

                out_u8 = np.clip(out_f, 0, 255).astype(np.uint8)
                if out_u8.shape[:2] != (h, w):
                    out_u8 = cv2.resize(out_u8, (w, h))
                writer.write(out_u8)

                if i % 20 == 0 or i == n - 1:
                    progress_cb(i + 1, n)

        writer.release()
        done_cb(out_path)

    except Exception as exc:
        error_cb(f"{exc}\n\n{traceback.format_exc()}")


# ─── Stack & Align — "freeze the ground, align the sky" (Sequator-style) ─────

def _detect_stars(gray_f32: np.ndarray, mask_u8: "np.ndarray | None",
                   max_stars: int = 150, k_sigma: float = 6.0,
                   min_area: int = 2, max_area: int = 60,
                   max_aspect: float = 3.0) -> np.ndarray:
    """
    Detect point-like bright sources (stars) in a grayscale frame.

    Flattens the sky gradient/vignetting with a large-sigma Gaussian blur,
    thresholds the residual robustly (mean + k_sigma*std inside the sky
    mask), then filters connected components by area (rejects hot pixels
    and large blobs = clouds/moon) and bounding-box aspect ratio (rejects
    elongated streaks = satellites/planes, not stars).

    Returns an (K, 2) float32 array of (x, y) centroids, brightest first,
    K <= max_stars. Empty (0, 2) array if none found.
    """
    h, w = gray_f32.shape
    sigma_bg = max(4.0, 0.02 * min(h, w))
    bg = cv2.GaussianBlur(gray_f32, (0, 0), sigmaX=sigma_bg)
    resid = gray_f32 - bg

    if mask_u8 is not None:
        m = mask_u8 > 0
    else:
        m = np.ones((h, w), bool)

    vals = resid[m]
    if vals.size == 0:
        return np.zeros((0, 2), np.float32)
    mu, sigma = float(vals.mean()), float(vals.std())
    thresh = mu + k_sigma * max(sigma, 1e-6)

    binary = ((resid > thresh) & m).astype(np.uint8)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8)
    if n_labels <= 1:
        return np.zeros((0, 2), np.float32)

    # Peak flux per component in one pass over the (sparse) detected pixels —
    # a per-label np.where(labels == lbl) is O(K·H·W), seconds on 1080p.
    on = binary > 0
    peak = np.full(n_labels, -np.inf, np.float32)
    np.maximum.at(peak, labels[on], resid[on])

    area = stats[1:, cv2.CC_STAT_AREA]
    bw = stats[1:, cv2.CC_STAT_WIDTH]
    bh = stats[1:, cv2.CC_STAT_HEIGHT]
    aspect = np.maximum(bw, bh) / np.maximum(1, np.minimum(bw, bh))
    # elongated -> satellite/plane streak, not a star
    keep = (area >= min_area) & (area <= max_area) & (aspect <= max_aspect)
    lbls = np.nonzero(keep)[0] + 1
    if lbls.size == 0:
        return np.zeros((0, 2), np.float32)
    order = np.argsort(-peak[lbls], kind="stable")[:max_stars]
    return centroids[lbls[order]].astype(np.float32)


def _star_signatures(stars: np.ndarray, k: int = 4) -> np.ndarray:
    """
    Rotation/translation-invariant signature per star: sorted distances to
    its k nearest neighbours. Brute-force O(K^2) distance matrix — fine for
    K <= ~150 (no scipy dependency in this project).
    """
    n = len(stars)
    if n == 0:
        return np.zeros((0, k), np.float32)
    diff = stars[:, None, :] - stars[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)
    kk = min(k, n - 1)
    if kk <= 0:
        return np.full((n, k), np.nan, np.float32)
    part = np.sort(dist, axis=1)[:, :kk].astype(np.float32)
    if kk < k:
        part = np.pad(part, ((0, 0), (0, k - kk)), constant_values=np.nan)
    return part


def _match_stars(ref_sig: np.ndarray, tgt_sig: np.ndarray) -> np.ndarray:
    """
    For each reference star, find the index of the target-frame star with
    the closest signature (nearest neighbour in signature space). Returns
    an (R,) int array: ref index i -> tgt index match_idx[i]. Some of these
    candidate correspondences will be wrong — RANSAC in
    cv2.estimateAffinePartial2D rejects them.
    """
    a = np.nan_to_num(ref_sig, nan=1e6)
    b = np.nan_to_num(tgt_sig, nan=1e6)
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)   # (R, T)
    return np.argmin(d, axis=1)


def _estimate_registration(ref_stars: np.ndarray, ref_sig: np.ndarray,
                            tgt_stars: np.ndarray, tgt_sig: np.ndarray,
                            min_inliers: int = 6):
    """
    Candidate-match ref/target star lists via nearest-signature
    correspondence, then robustly fit a similarity transform
    (rotation + translation, scale ~= 1 since the camera never moves) with
    cv2.estimateAffinePartial2D's internal RANSAC.

    Returns (M, n_inliers). M is None (n_inliers=0) if either frame has
    fewer than min_inliers stars, or fewer than min_inliers RANSAC inliers
    are found — caller should skip the frame from the sky stack, not abort
    the whole run.
    """
    if len(ref_stars) < min_inliers or len(tgt_stars) < min_inliers:
        return None, 0
    match_idx = _match_stars(ref_sig, tgt_sig)
    src_pts = tgt_stars[match_idx].reshape(-1, 1, 2)   # warps target -> ref space
    dst_pts = ref_stars.reshape(-1, 1, 2)
    M, inliers = cv2.estimateAffinePartial2D(
        src_pts, dst_pts, method=cv2.RANSAC,
        ransacReprojThreshold=2.0, confidence=0.99)
    if M is None or inliers is None:
        return None, 0
    n_inl = int(inliers.sum())
    if n_inl < min_inliers:
        return None, 0
    return M, n_inl


def process_stack_align(params: dict, progress_cb, done_cb, error_cb):
    """
    Sequator-style "Freeze the Ground, Align the Sky":
      1. Pass 1 (detection): stream-load N consecutive frames, detect stars
         per frame (sky-disk masked), discard the frame after detection —
         keeps peak RAM to O(1) frames, not O(N).
      2. Pass 2 (warp + accumulate): stream-load the same N frames again,
         register each against the reference frame's stars (RANSAC similarity
         transform), warp+accumulate the sky (weighted by a warped coverage
         mask to avoid dark rotated-in borders), and separately accumulate
         the raw (unwarped) ground — the camera never moves, so a plain
         average is optimal.
      3. Feather-blend sky/ground at the mask boundary, apply the shared
         color pipeline (contrast/brightness -> saturation/RGB -> GHS),
         and save a single PNG.
    """
    try:
        src_dir    = params["src_dir"]
        out_path   = params["out_path"]
        bayer_code = params.get("bayer_code", None)

        align_n     = max(2, int(params["align_n"]))
        mask_rad    = float(params["align_mask_radius"])
        ground_cutoff = float(params.get("align_ground_cutoff", 0.0))
        k_sigma     = float(params["align_k_sigma"])
        ref_path    = params.get("align_ref_path", "") or ""
        min_inliers = 6
        max_stars   = 150

        contrast   = params["contrast"]
        brightness = params["brightness"]
        saturation = params.get("saturation", 1.0)
        rgb_r = params.get("rgb_r", 1.0)
        rgb_g = params.get("rgb_g", 1.0)
        rgb_b = params.get("rgb_b", 1.0)
        use_ghs    = params["use_ghs"]
        ghs_params = params["ghs"]

        # ── Collect images ──────────────────────────────────────────────
        exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG",
                "*.fit", "*.fits", "*.FIT", "*.FITS")
        paths: list = []
        for e in exts:
            paths.extend(glob.glob(os.path.join(src_dir, e)))
        paths.sort()
        if not paths:
            error_cb("Aucune image trouvée dans le répertoire sélectionné.")
            return

        # ── Resolve start index from the optional reference image ──────
        start_idx = 0
        if ref_path:
            target = os.path.normcase(os.path.abspath(ref_path))
            for i, p in enumerate(paths):
                if os.path.normcase(os.path.abspath(p)) == target:
                    start_idx = i
                    break

        available = len(paths) - start_idx
        if available < 2:
            error_cb("Pas assez d'images après l'image de référence pour l'empilement "
                      "(minimum 2 requis).")
            return
        n_use = min(align_n, available)
        seq_paths = paths[start_idx:start_idx + n_use]

        # ── Global radiometric scale (FITS only) + probe first frame ───
        p_lo, p_hi = compute_global_scale(seq_paths, bayer_code=bayer_code)
        first_img = None
        for p in seq_paths:
            ext = os.path.splitext(p)[1].lower()
            img = (_load_fits_linear(p, bayer_code, 0.0, 65535.0)
                    if ext in ('.fit', '.fits') else cv2.imread(p))
            if img is not None:
                first_img = img.astype(np.float32)
                break
        if first_img is None:
            error_cb("Impossible de lire la première image de la séquence.")
            return
        h, w = first_img.shape[:2]

        # ── Sky/ground disk mask + feathered alpha ──────────────────────
        # Radius is a fraction of the half-diagonal, so mask_rad=1.0 reaches
        # the frame corners consistently regardless of aspect ratio (a plain
        # fraction of min(h,w) silently covers 100% of some wide/short
        # frames well before 1.0, silently disabling ground exclusion).
        half_diag = 0.5 * float(np.hypot(h, w))
        sky_mask_u8 = np.zeros((h, w), np.uint8)
        r = int(mask_rad * half_diag)
        cv2.circle(sky_mask_u8, (w // 2, h // 2), r, 255, -1)

        # Optional bottom cutoff: fixed AllSky rigs often have their
        # obstruction (roof/trees) concentrated in a band along the bottom
        # of the frame rather than a uniform outer ring — a circle alone
        # can't exclude that band without also cutting good sky near the
        # corners. ground_cutoff (fraction of H, measured from the bottom)
        # unconditionally forces that band to the raw/unwarped ground.
        if ground_cutoff > 0.0:
            cutoff_row = int(h * (1.0 - min(ground_cutoff, 0.9)))
            sky_mask_u8[cutoff_row:, :] = 0

        mask_f32 = sky_mask_u8.astype(np.float32) / 255.0
        feather_sigma = max(3.0, r * 0.02)
        alpha = cv2.GaussianBlur(mask_f32, (0, 0), feather_sigma)[:, :, None]

        load_args = [(seq_paths[i], first_img, bayer_code, p_lo, p_hi)
                     for i in range(n_use)]

        # ── Pass 1: detect stars per frame (bounded RAM: 1 frame at a time) ──
        stars_list: list = [None] * n_use
        with ThreadPoolExecutor(max_workers=_N_IO_WORKERS) as pool:
            pending: dict = {}
            next_submit = 0

            def fill(up_to: int) -> None:
                nonlocal next_submit
                while next_submit < min(up_to, n_use):
                    pending[next_submit] = pool.submit(_load_frame_linear, load_args[next_submit])
                    next_submit += 1

            fill(_PREFETCH)
            for i in range(n_use):
                frame = pending.pop(i).result()
                fill(i + 1 + _PREFETCH)
                gray = (0.114 * frame[:, :, 0] + 0.587 * frame[:, :, 1]
                        + 0.299 * frame[:, :, 2])
                stars_list[i] = _detect_stars(gray, sky_mask_u8,
                                               max_stars=max_stars, k_sigma=k_sigma)
                # First half of the bar: detection pass (both passes read the
                # same N images — reported as N total, not 2N, to match what
                # the user actually configured).
                progress_cb((i + 1) // 2, n_use)

        ref_stars = stars_list[0]
        if len(ref_stars) < min_inliers:
            error_cb(f"Pas assez d'étoiles détectées dans l'image de référence "
                      f"({len(ref_stars)} trouvées, {min_inliers} requises). "
                      "Choisissez une autre référence ou baissez la sensibilité (k·σ).")
            return
        ref_sig = _star_signatures(ref_stars)

        # ── Pass 2: warp + accumulate (bounded RAM: 1 frame + accumulators) ──
        sky_sum      = np.zeros((h, w, 3), np.float32)
        sky_weight   = np.zeros((h, w, 1), np.float32)
        ground_sum   = np.zeros((h, w, 3), np.float32)
        used = 0

        with ThreadPoolExecutor(max_workers=_N_IO_WORKERS) as pool:
            pending = {}
            next_submit = 0

            def fill(up_to: int) -> None:
                nonlocal next_submit
                while next_submit < min(up_to, n_use):
                    pending[next_submit] = pool.submit(_load_frame_linear, load_args[next_submit])
                    next_submit += 1

            fill(_PREFETCH)
            for i in range(n_use):
                frame = pending.pop(i).result()
                fill(i + 1 + _PREFETCH)

                ground_sum += frame

                if i == 0:
                    warped, coverage = frame, np.ones((h, w, 1), np.float32)
                    used += 1
                else:
                    tgt_stars = stars_list[i]
                    if len(tgt_stars) < min_inliers:
                        progress_cb(n_use // 2 + (i + 1) // 2, n_use)
                        continue
                    tgt_sig = _star_signatures(tgt_stars)
                    M, n_inl = _estimate_registration(ref_stars, ref_sig,
                                                       tgt_stars, tgt_sig, min_inliers)
                    if M is None:
                        progress_cb(n_use // 2 + (i + 1) // 2, n_use)
                        continue
                    warped = cv2.warpAffine(frame, M, (w, h),
                                             flags=cv2.INTER_LANCZOS4, borderValue=0)
                    coverage = cv2.warpAffine(np.ones((h, w), np.float32), M, (w, h),
                                               flags=cv2.INTER_LANCZOS4, borderValue=0)[:, :, None]
                    used += 1

                sky_sum    += warped * mask_f32[:, :, None] * coverage
                sky_weight += mask_f32[:, :, None] * coverage
                progress_cb(n_use // 2 + (i + 1) // 2, n_use)

        if used < 2:
            error_cb(f"Trop peu d'images alignées avec succès ({used}/{n_use}). "
                      "Essayez de baisser la sensibilité de détection ou d'augmenter "
                      "le rayon du masque.")
            return

        eps = 1e-6
        sky_stack    = sky_sum / np.maximum(sky_weight, eps)
        ground_stack = ground_sum / n_use
        final_f = alpha * sky_stack + (1.0 - alpha) * ground_stack

        # ── Shared color pipeline (mirrors process_star_trail's "final") ──
        beta_c = brightness - 128.0 * (contrast - 1.0)
        final_u8 = cv2.convertScaleAbs(final_f, alpha=contrast, beta=beta_c)
        final_u8 = _apply_color_correction(final_u8, saturation, rgb_r, rgb_g, rgb_b)
        if use_ghs:
            final_u8 = cv2.LUT(final_u8, build_ghs_lut(**ghs_params))

        png_path = os.path.splitext(out_path)[0] + "_stackalign.png"
        cv2.imwrite(png_path, final_u8)
        done_cb(png_path, f"{used} / {n_use} images alignées avec succès.")

    except Exception as exc:
        error_cb(f"{exc}\n\n{traceback.format_exc()}")


# ─── Stack Video — progressive live stack (port of Multicam's Live Stack) ────
#
# Video frame 0 = raw image, frame 1 = 1 image through the pipeline, frame k =
# k images stacked + pipeline … up to N. The per-frame pipeline mirrors
# Multicam's dso_stacker.js (hot pixels, star registration, background
# normalisation, noise weighting, running σ-clipping) and auto_stretch.js
# (colour-preserving arcsinh auto-stretch, local contrast, green removal),
# adapted to a fixed AllSky rig: aligned sky + frozen ground, as in
# process_stack_align.

_AS_SHADOWS_CLIP = -2.8    # black point = median − 2.8 σ
_AS_TARGET_MIN   = 0.10    # background level at the first (noisy) render
_AS_TARGET_MAX   = 0.18    # background level once noise is divided by 5 (~25 images)
_AS_RAMP_IMAGES  = 25
_AS_LCE_RADIUS   = 1 / 40  # local-contrast radius / largest dimension
_AS_SMOOTHING    = 0.5     # weight of the new render in the stats smoothing
_AS_LUT_N        = 65536   # entries of the stretch-curve table


def _srgb_to_linear_lut() -> np.ndarray:
    x = np.arange(256, dtype=np.float64) / 255.0
    lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    return (lin * 255.0).astype(np.float32)


_SRGB_LIN_LUT = _srgb_to_linear_lut()

# Reductions over a size-3 last axis (img.max(axis=2), .mean(axis=2)) are ~10×
# slower than elementwise ops in NumPy — these helpers stay elementwise.
_LUMA_MEAN = np.float32([[1 / 3, 1 / 3, 1 / 3]])


def _max3(img: np.ndarray) -> np.ndarray:
    return np.maximum(np.maximum(img[:, :, 0], img[:, :, 1]), img[:, :, 2])


def _min3(img: np.ndarray) -> np.ndarray:
    return np.minimum(np.minimum(img[:, :, 0], img[:, :, 1]), img[:, :, 2])


def _remove_hot_pixels(frame: np.ndarray, k: float = 6.0) -> np.ndarray:
    """
    Replace isolated hot/dead pixels by their 3×3 median. A pixel is hot when
    it sticks out of its median by k·σ AND that median itself sits on the
    local background: a star core's neighbours stay bright, which raises its
    3×3 median, so stars are spared (same idea as Multicam's hotPixelsCPU).
    """
    med = cv2.medianBlur(frame, 3)
    resid = cv2.absdiff(frame, med)
    s = max(1.4826 * float(np.median(resid[::7, ::7])), 0.05)
    bad = resid > k * s
    del resid
    local = cv2.blur(med, (15, 15))
    np.subtract(med, local, out=local)              # height above local background
    bad &= local < 2.0 * s
    del local
    np.copyto(frame, med, where=bad)               # in place: frames are large
    return frame


def _load_fits_float(path: str, bayer_code, p_lo: float, p_hi: float):
    """
    FITS → float32 BGR on the global linear scale (p_lo → 0, p_hi → 255)
    WITHOUT clipping, debayered at full 16-bit precision. The other modes go
    through uint8 before debayering and clip at p_hi, which saturates every
    star and galaxy core of deep-sky data (e.g. IMX477 12-bit: sky ~280 ADU,
    p99.5 ~330 ADU, stars up to 4095). With no Bayer pattern chosen in the
    UI, the header's BAYERPAT is used for 2-D frames.
    """
    try:
        from astropy.io import fits as _fits
        with _fits.open(path, memmap=False) as hdul:
            hdu = hdul[0]
            if hdu.data is None:
                hdu = next((x for x in hdul[1:] if x.data is not None and x.data.ndim >= 2), None)
                if hdu is None:
                    return None
            data = np.asarray(hdu.data)
            pat = str(hdu.header.get("BAYERPAT", "")).strip().upper()
        scale = np.float32(255.0 / (p_hi - p_lo))
        if data.ndim == 2:
            code = bayer_code if bayer_code is not None else _BAYER_MAP.get(pat)
            if code is not None:
                raw = np.clip(data, 0, 65535).astype(np.uint16)
                bgr = cv2.cvtColor(raw, code).astype(np.float32)
            else:
                g = data.astype(np.float32)
                bgr = np.stack([g, g, g], axis=2)
        elif data.ndim == 3:
            arr = data.astype(np.float32)
            if arr.shape[0] in (1, 3) and arr.shape[0] < arr.shape[1] // 2:
                arr = arr.transpose(1, 2, 0)
            if arr.shape[2] == 3:
                bgr = np.ascontiguousarray(arr[:, :, ::-1])
            else:
                g = arr[:, :, 0]
                bgr = np.stack([g, g, g], axis=2)
        else:
            return None
        bgr -= np.float32(p_lo)
        bgr *= scale
        return bgr
    except Exception:
        return None


def _load_stack_frame(args: tuple):
    """
    Worker-side loading for process_stack_video: linear float32 BGR frame
    (sRGB JPEG/PNG optionally linearised), hot pixels removed (at full
    resolution), downscaled to the working size if smaller, stars detected.
    Returns (None, None) for an unreadable image or one whose size differs
    from the reference — it is skipped, not replaced.
    Also returns the sensor-clipping map (None if no clipping level): it must
    be found at full resolution — downscaling averages a clipped star core
    below any threshold.
    args = (path, (full_w, full_h), (work_w, work_h), bayer_code, p_lo, p_hi,
            linearize, hot, mask_u8, k_sigma, sat_level)
    """
    (path, full_size, work_size, bayer_code, p_lo, p_hi,
     linearize, hot, mask_u8, k_sigma, sat_level) = args
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.fit', '.fits'):
        frame = _load_fits_float(path, bayer_code, p_lo, p_hi)
    else:
        img = cv2.imread(path)
        if img is None:
            frame = None
        elif linearize:
            frame = _SRGB_LIN_LUT[img]
        else:
            frame = img.astype(np.float32)
    if frame is None or (frame.shape[1], frame.shape[0]) != tuple(full_size):
        return None, None, None
    if hot:
        frame = _remove_hot_pixels(frame)
    clip = None
    if sat_level:
        # clipped in any channel, grown 2 px (debayering spreads it)
        clip = (_max3(frame) >= 0.98 * sat_level).astype(np.float32)
        clip = cv2.dilate(clip, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    if tuple(work_size) != tuple(full_size):
        frame = cv2.resize(frame, tuple(work_size), interpolation=cv2.INTER_AREA)
        if clip is not None:
            clip = cv2.resize(clip, tuple(work_size), interpolation=cv2.INTER_AREA)
    gray = cv2.transform(frame, np.float32([[0.114, 0.587, 0.299]]))
    stars = _detect_stars(gray, mask_u8, max_stars=150, k_sigma=k_sigma)
    return frame, stars, clip


def _background_map(frame: np.ndarray, mask_f32: "np.ndarray | None" = None,
                    scale: int = 16) -> np.ndarray:
    """
    Large-scale background (stars removed by the median), full size. With a
    mask, only masked pixels contribute (normalised convolution) and the map
    is smoothly extrapolated outside: the bright ground must not leak into the
    sky background, which is rotated with the sky afterwards.
    """
    h, w = frame.shape[:2]
    size = (max(4, w // scale), max(4, h // scale))
    if mask_f32 is None:
        small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    else:
        num = cv2.resize(frame * mask_f32[:, :, None], size, interpolation=cv2.INTER_AREA)
        den = cv2.resize(mask_f32, size, interpolation=cv2.INTER_AREA)
        good = (den > 0.5).astype(np.float32)
        small = num / np.maximum(den, 1e-6)[:, :, None] * good[:, :, None]
        sig = max(size) / 8.0
        fill = (cv2.GaussianBlur(small, (0, 0), sig)
                / np.maximum(cv2.GaussianBlur(good, (0, 0), sig), 1e-6)[:, :, None])
        small = np.where(good[:, :, None] > 0, small, fill).astype(np.float32)
    small = cv2.medianBlur(small, 5)
    small = cv2.GaussianBlur(small, (0, 0), 2.0)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _frame_noise(frame: np.ndarray, bg: np.ndarray, mask: np.ndarray) -> float:
    """Robust σ (1.4826·MAD) of the background-subtracted luminance in the sky."""
    resid = cv2.transform(frame[::5, ::5] - bg[::5, ::5], _LUMA_MEAN)
    vals = resid[mask[::5, ::5]]
    if vals.size == 0:
        return 1.0
    med = float(np.median(vals))
    return max(1.4826 * float(np.median(np.abs(vals - med))), 1e-4)


class _ClipStack:
    """
    Running weighted mean + variance per pixel and per channel (weighted
    Welford), with streaming σ-rejection once `min_frames` samples are in:
    a value more than κ·σ above (satellite, plane, cosmic) or κ_low·σ below
    the running mean is not stacked. The tolerance includes a share of the
    signal, σ² + (β·signal)², so star wings of a softer frame survive.
    The tolerance moves slowly, so it is only refreshed every `tol_every`
    images (this runs on full-resolution 3-channel arrays at every image).
    """

    def __init__(self, shape, kappa: float, kappa_low: float = 5.0,
                 min_frames: int = 10, signal_tol: float = 0.15, tol_every: int = 4):
        self.mean = np.zeros(shape, np.float32)
        self.S    = np.zeros(shape, np.float32)
        self.W    = np.zeros(shape, np.float32)
        self.n    = np.zeros(shape, np.float32)
        self.kappa, self.kappa_low = kappa, kappa_low
        self.min_frames, self.signal_tol = min_frames, signal_tol
        self.tol_every = tol_every
        self.frames = 0          # upper bound of n
        self._hi = self._lo = self._young = None
        self.track_rejects = False   # add() returns the rejected fraction (preview)

    def _refresh_tolerance(self, bg: np.ndarray) -> None:
        sig2 = self.S / np.maximum(self.W, 1e-12)
        sig2 *= self.n / np.maximum(self.n - 1.0, 1.0)
        sgn = np.maximum(self.mean - bg, 0.0)
        sgn *= self.signal_tol
        sgn *= sgn
        sig2 += sgn
        np.sqrt(sig2, out=sig2)
        self._hi = sig2 * self.kappa
        sig2 *= -self.kappa_low
        self._lo = sig2
        self._young = self.n < self.min_frames     # too few samples to judge

    def add(self, x: np.ndarray, weight: float, valid: "np.ndarray | None",
            bg: np.ndarray) -> "float | None":
        self.frames += 1
        delta = x - self.mean
        acc = None
        rejected = None
        if self.kappa > 0 and self.frames > self.min_frames:
            if self._hi is None or (self.frames - self.min_frames - 1) % self.tol_every == 0:
                self._refresh_tolerance(bg)
            acc = (delta <= self._hi) & (delta >= self._lo)
            acc |= self._young
            if valid is not None:
                acc &= valid
            if self.track_rejects:
                n_cand = (3 * int(valid.sum())) if valid is not None else acc.size
                rejected = 1.0 - int(acc.sum()) / max(1, n_cand)
        elif valid is not None:
            acc = np.broadcast_to(valid, x.shape)
        if acc is None:
            w = np.full(x.shape, weight, np.float32)
            self.n += 1.0
        else:
            w = acc.astype(np.float32)
            self.n += w
            w *= weight
        self.W += w
        r = w / np.maximum(self.W, 1e-12)              # w = 0 wherever W = 0
        r *= delta
        self.mean += r
        w *= delta
        w *= x - self.mean
        self.S += w
        return rejected


class _MeanStack:
    """Plain weighted running mean — the frozen ground (camera never moves)."""

    def __init__(self, shape):
        self.sum = np.zeros(shape, np.float32)
        self.W = 0.0

    def add(self, x: np.ndarray, weight: float) -> None:
        self.sum += x * np.float32(weight)
        self.W += weight

    @property
    def mean(self) -> np.ndarray:
        return self.sum * np.float32(1.0 / max(self.W, 1e-12))


def _measure_auto_stretch(img: np.ndarray, valid: np.ndarray) -> "dict | None":
    """Port of measureAutoStretch (auto_stretch.js). img: float BGR."""
    pix = img[::2, ::2][valid[::2, ::2]][:, ::-1].astype(np.float64)   # (n, 3) RGB, 1/4
    if pix.shape[0] < 64:
        return None
    smp = pix[::4]                                                       # 1/16
    med = np.median(smp, axis=0)
    sigma = 1.4826 * np.median(np.abs(smp - med), axis=0)
    mx = smp.max(axis=0)

    gains = np.ones(3)
    s4 = pix
    d = s4 - med
    sel = (np.all(d >= 10 * sigma, axis=1)
           & (s4.max(axis=1) <= 0.9 * float(mx.max())))
    if int(sel.sum()) > 100:
        sm = d[sel].sum(axis=0)
        if sm[0] > 0 and sm[2] > 0:
            gains = np.clip(np.array([sm[1] / sm[0], 1.0, sm[1] / sm[2]]), 0.25, 4.0)

    sig_mean = float((sigma * gains).mean())
    c0 = med + _AS_SHADOWS_CLIP * sig_mean
    bg = -_AS_SHADOWS_CLIP * sig_mean
    lum = smp.mean(axis=1)
    top = float(np.quantile(lum, 0.99995))
    white = max((top - float(med.mean())) * float(gains.mean()) + bg, bg * 4, 1e-6)
    return dict(med=med, gains=gains, c0=c0, bg=bg, white=white, sigma=sig_mean)


def _beta_for(x: float, t: float) -> float:
    """β such that asinh(β·x)/asinh(β) = t (log-scale bisection)."""
    if not (x > 0) or x >= t:
        return 1e-3
    a, b = 1e-3, 1e8
    for _ in range(70):
        c = float(np.sqrt(a * b))
        if np.arcsinh(c * x) / np.arcsinh(c) < t:
            a = c
        else:
            b = c
    return float(np.sqrt(a * b))


def _auto_stretch_params(img: np.ndarray, valid: np.ndarray, state: dict,
                         target: "float | None", count: int) -> "dict | None":
    """Port of autoStretchParams: stats smoothed from one render to the next."""
    m = _measure_auto_stretch(img, valid)
    if m is None:
        return state.get("stats")
    p = state.get("stats")
    if p is not None:
        a = _AS_SMOOTHING
        for k in ("bg", "white", "sigma", "med", "gains", "c0"):
            m[k] = a * m[k] + (1 - a) * p[k]
    if state.get("sigma_ref") is None:
        state["sigma_ref"] = m["sigma"]
        state["n_ref"] = max(1, count)
    n_eff = state["n_ref"] * (state["sigma_ref"] / max(m["sigma"], 1e-12)) ** 2
    state["stats"] = m
    t = target
    if t is None:
        r = min(1.0, max(0.0, np.log(max(1.0, n_eff)) / np.log(_AS_RAMP_IMAGES)))
        t = _AS_TARGET_MIN + (_AS_TARGET_MAX - _AS_TARGET_MIN) * r
    return dict(m, beta=_beta_for(m["bg"] / m["white"], t), target=t, n_eff=n_eff)


def _stretch_input(img: np.ndarray, p: dict) -> np.ndarray:
    """
    Channels after black point, white balance and white normalisation —
    the [0, 1] domain the stretch curve is applied to (black = 0, white = 1).
    """
    med   = p["med"][::-1].astype(np.float32)          # RGB → BGR
    gains = p["gains"][::-1].astype(np.float32)
    c0    = p["c0"][::-1].astype(np.float32)
    inv = np.float32(1.0 / p["white"])
    # ((img − med)·gains + med − c0) / white, as one affine per channel
    return img * (gains * inv) + (med - c0 - med * gains) * inv


_GHS_LUT_CACHE: dict = {}


def _ghs_curve_lut(ghs: dict) -> np.ndarray:
    """GHS curve (_ghs_on_array, the app's formula) tabulated on [0, 1]."""
    key = tuple(round(float(ghs[k]), 6) for k in ("SP", "D", "b", "LP", "HP"))
    lut = _GHS_LUT_CACHE.get(key)
    if lut is None:
        x = np.linspace(0.0, 1.0, _AS_LUT_N)
        lut = _ghs_on_array(x, *key).astype(np.float32)
        # With HP = 1, _ghs_on_array returns the identity at exactly x = 1
        # (T(1) = 1 even when b has compressed the highlights to ~0.2): a
        # jump at the white point. Continue the curve smoothly instead.
        lut[-1] = 2.0 * lut[-2] - lut[-3]
        if len(_GHS_LUT_CACHE) > 32:
            _GHS_LUT_CACHE.clear()
        _GHS_LUT_CACHE[key] = lut
    return lut


def _auto_stretch_render(img: np.ndarray, valid: np.ndarray, p: dict,
                         local_contrast: float = 0.0,
                         clip: "np.ndarray | None" = None,
                         curve: "np.ndarray | None" = None) -> np.ndarray:
    """
    Port of autoStretchRender: colour-preserving arcsinh (Lupton 2004) — the
    curve is computed on the luminance and R, G, B are scaled by the same
    factor, so hues survive. Optional large-scale local contrast. → uint8 BGR.
    clip (0–1 map, see _LiveStacker): where the sensor clipped, the white
    balance is faded to a neutral gain — otherwise the equal clipped channels
    times the WB gains (×1.8 on R and B for a raw IMX477) turn star cores
    magenta/blue.
    curve: a manual tone curve tabulated on [0, 1] (GHS) replacing the auto
    arcsinh; black point, white balance and white stay the measured ones.
    """
    h, w = img.shape[:2]
    gains = p["gains"][::-1].astype(np.float32)
    beta  = p["beta"]
    v = _stretch_input(img, p)
    if clip is not None:
        t = np.minimum(clip * 3.0, 1.0)
        t = cv2.GaussianBlur(t, (0, 0), 1.0)[:, :, None]
        g_n = float(gains.mean())
        v_n = _stretch_input(img, dict(p, gains=np.full(3, g_n)))
        v_n -= v
        v_n *= t
        v += v_n
    I = cv2.transform(v, _LUMA_MEAN)
    Ig = cv2.blur(I, (3, 3), borderType=cv2.BORDER_REPLICATE)   # no chroma noise
    ab = float(np.arcsinh(beta))
    if curve is None:
        # f(x) = asinh(βx)/asinh(β): table on [0, 1] (as in auto_stretch.js), exact above
        lut = (np.arcsinh(beta * np.linspace(0.0, 1.0, _AS_LUT_N)) / ab).astype(np.float32)
        slope0 = beta / ab                       # f(x)/x when x → 0
    else:
        lut = curve
        slope0 = float(lut[1]) * (_AS_LUT_N - 1)
    idx = np.clip(Ig * (_AS_LUT_N - 1) + 0.5, 0, _AS_LUT_N - 1).astype(np.int32)
    fI = lut[idx]
    fI[Ig <= 0] = 0.0
    over = Ig >= 1.0
    if over.any():
        if curve is None:
            fI[over] = np.arcsinh(beta * Ig[over]) / ab
        else:
            # beyond the white: continue with the curve's end slope
            end_slope = max(0.0, float(lut[-1] - lut[-2]) * (_AS_LUT_N - 1))
            fI[over] = float(lut[-1]) + (Ig[over] - 1.0) * end_slope
    if local_contrast > 0:
        F = 4
        sw, sh = -(-w // F), -(-h // F)
        small = cv2.resize(fI, (sw, sh), interpolation=cv2.INTER_AREA)
        r = max(1, round(max(w, h) * _AS_LCE_RADIUS / F))
        for _ in range(3):                                         # ≈ gaussian
            small = cv2.blur(small, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REPLICATE)
        blur = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        fI = np.maximum(0.0, fI + local_contrast * (fI - blur) * (1.0 - np.minimum(fI, 1.0)))
    k = np.where(Ig > 1e-7, fI / np.maximum(Ig, 1e-7), slope0).astype(np.float32)
    out = v * k[:, :, None]
    mx = _max3(out)
    out *= (255.0 / np.maximum(mx, 1.0))[:, :, None]   # out of gamut: keep hue
    np.clip(out, 0.0, 255.0, out=out)
    out += 0.5
    out[~valid] = 0
    return out.astype(np.uint8)


def _remove_green(img_u8: np.ndarray) -> np.ndarray:
    """
    Port of removeGreen8: average-neutral SCNR (G ≤ (R+B)/2) computed on local
    means (image ÷4, blurred) and applied as a smooth factor, brightness
    restored with the same smoothed factor.
    """
    h, w = img_u8.shape[:2]
    small = cv2.resize(img_u8.astype(np.float32), (-(-w // 4), -(-h // 4)),
                       interpolation=cv2.INTER_AREA)
    small = cv2.blur(small, (3, 3), borderType=cv2.BORDER_REPLICATE)
    b, g, r = small[:, :, 0], small[:, :, 1], small[:, :, 2]
    fG = np.where(g > 1e-3, np.minimum(1.0, (r + b) / np.maximum(2 * g, 1e-6)), 1.0)
    den = r + fG * g + b
    fL = np.where(den > 1e-3, (r + g + b) / np.maximum(den, 1e-6), 1.0)
    fG = cv2.resize(fG.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    fL = cv2.resize(fL.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    out = img_u8.astype(np.float32)
    out[:, :, 0] *= fL
    out[:, :, 1] *= fG * fL
    out[:, :, 2] *= fL
    out *= (255.0 / np.maximum(_max3(out), 255.0))[:, :, None]   # keep hue
    out = np.where((fG >= 0.999)[:, :, None], img_u8.astype(np.float32), out)
    return (out + 0.5).astype(np.uint8)


def _apply_saturation_clamped(img_u8: np.ndarray, sat: float) -> np.ndarray:
    """Port of applySaturation (stacker.js): luminance kept, gain limited per pixel."""
    if abs(sat - 1.0) < 1e-3:
        return img_u8
    f = img_u8.astype(np.float32)
    y = cv2.transform(f, np.float32([[0.114, 0.587, 0.299]]))
    d = f - y[:, :, None]
    # largest gain keeping every channel inside [0, 255]
    k = np.full(y.shape, sat, np.float32)
    dmax, dmin = _max3(d), _min3(d)
    np.minimum(k, np.where(dmax > 1e-6, (255.0 - y) / np.maximum(dmax, 1e-6), np.inf), out=k)
    np.minimum(k, np.where(dmin < -1e-6, y / np.maximum(-dmin, 1e-6), np.inf), out=k)
    d *= k[:, :, None]
    d += y[:, :, None]
    d += 0.5
    return np.clip(d, 0, 255).astype(np.uint8)


def _draw_counter(img_u8: np.ndarray, text: str) -> None:
    h = img_u8.shape[0]
    scale = max(0.5, h / 1080.0 * 1.1)
    thick = max(1, int(round(scale * 2)))
    org = (int(20 * scale), h - int(24 * scale))
    shadow = (org[0] + thick, org[1] + thick)
    cv2.putText(img_u8, text, shadow, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img_u8, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (235, 235, 235), thick, cv2.LINE_AA)


def _stack_video_schedule(n_imgs: int, n_frames: int, progression: str) -> list:
    """
    Stack count shown by each video frame: frame 0 = raw (0), then 1 … n_imgs,
    spread over n_frames. Fewer images than frames → each state is held
    several frames; more → intermediate states are skipped (still stacked).
    'log' gives more time to the first images, where the image changes most.
    """
    n_frames = max(3, n_frames)
    counts = [0]
    for j in range(1, n_frames):
        t = (j - 1) / (n_frames - 2)
        if progression == "log":
            c = int(round(n_imgs ** t))
        else:
            c = 1 + int(round(t * (n_imgs - 1)))
        counts.append(min(n_imgs, max(counts[-1], c, 1)))
    counts[-1] = n_imgs
    return counts


def _sensor_saturation_level(path: str, p_lo: float, p_hi: float,
                             linearize: bool) -> "float | None":
    """
    Sensor clipping level in stack units (same scale as _load_stack_frame),
    or None if the first frame shows no clipping. 8-bit images: 255. FITS:
    the frame's maximum ADU, if enough pixels sit on it (a 12-bit IMX477 in a
    uint16 FITS clips at 4095, not 65535).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in ('.fit', '.fits'):
        return float(_SRGB_LIN_LUT[255]) if linearize else 255.0
    try:
        from astropy.io import fits as _fits
        with _fits.open(path, memmap=False) as hdul:
            data = next((x.data for x in hdul if x.data is not None and x.data.ndim >= 2), None)
        if data is None:
            return None
        top = float(data.max())
        if int((data >= top).sum()) < 10:
            return None
        return (top - p_lo) * 255.0 / (p_hi - p_lo)
    except Exception:
        return None


def _stack_snapshot(sky: "_ClipStack", grnd: "_MeanStack | None",
                    alpha: np.ndarray) -> np.ndarray:
    """Current stack as a new array: aligned sky feathered into frozen ground."""
    if grnd is None:
        return sky.mean.copy()        # uncovered pixels: W = 0 → mean = 0 (black)
    g_mean = grnd.mean
    sky_img = np.where(sky.W > 0, sky.mean, g_mean)
    return alpha * sky_img + (1.0 - alpha) * g_mean


class _StackVideoError(Exception):
    """User-facing error of the stack-video pipeline (message shown as is)."""


def _stack_video_config(params: dict) -> dict:
    """Stack-video settings from the UI params (shared by the run and the preview)."""
    return dict(
        bayer_code    = params.get("bayer_code", None),
        mask_rad      = float(params["align_mask_radius"]),
        ground_cutoff = float(params.get("align_ground_cutoff", 0.0)),
        k_sigma       = float(params["align_k_sigma"]),
        target        = params["sv_target"],                # None = auto
        lce           = float(params["sv_local_contrast"]),
        rm_green      = bool(params["sv_remove_green"]),
        saturation    = float(params["sv_saturation"]),
        kappa         = float(params["sv_kappa"]),          # 0 = no σ-rejection
        linearize     = bool(params["sv_linearize"]),
        hot           = bool(params["sv_hot_pixels"]),
        bg_norm       = bool(params["sv_bg_norm"]),
        weighting     = bool(params["sv_weighting"]),
        stretch       = params.get("sv_stretch", "auto"),   # "auto" | "ghs"
        ghs           = params.get("sv_ghs", dict(SP=0.0, D=100.0, b=0.0, LP=0.0, HP=1.0)),
    )


def _stack_video_paths(params: dict, n_max: int) -> list:
    """Sorted images of src_dir from the reference image on, at most n_max."""
    src_dir = params["src_dir"]
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG",
            "*.fit", "*.fits", "*.FIT", "*.FITS")
    paths: list = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(src_dir, e)))
    paths.sort()
    if not paths:
        raise _StackVideoError("Aucune image trouvée dans le répertoire sélectionné.")
    start_idx = 0
    ref_path = params.get("align_ref_path", "") or ""
    if ref_path:
        target_p = os.path.normcase(os.path.abspath(ref_path))
        for i, p in enumerate(paths):
            if os.path.normcase(os.path.abspath(p)) == target_p:
                start_idx = i
                break
    if len(paths) - start_idx < 2:
        raise _StackVideoError("Pas assez d'images après l'image de référence pour "
                               "l'empilement (minimum 2 requis).")
    return paths[start_idx:start_idx + max(2, n_max)]


def _image_size(path: str) -> "tuple | None":
    """(w, h) from the file header only — no pixel decoding."""
    try:
        if os.path.splitext(path)[1].lower() in ('.fit', '.fits'):
            from astropy.io import fits as _fits
            hd = _fits.getheader(path)
            if hd.get("NAXIS", 0) < 2:
                hd = _fits.getheader(path, 1)
            n1, n2, n3 = hd.get("NAXIS1"), hd.get("NAXIS2"), hd.get("NAXIS3")
            # (C, H, W) colour FITS: NAXIS1 = W, NAXIS2 = H; (H, W, C): NAXIS1 = C
            if n1 in (1, 3) and n3 and n3 > 3:
                return (n2, n3)
            return (n1, n2)
        with _PILImage.open(path) as im:
            return im.size
    except Exception:
        return None


def _sequence_warnings(seq_paths: list, sample: int = 24) -> list:
    """Mixed formats / image sizes in the sequence (e.g. raw + processed FITS)."""
    warns = []
    exts = sorted({os.path.splitext(p)[1].lower() for p in seq_paths})
    if len(exts) > 1:
        warns.append("Formats mélangés dans la séquence : " + ", ".join(exts)
                     + " — vérifier que le dossier ne contient que les images brutes.")
    step = max(1, len(seq_paths) // sample)
    sizes: dict = {}
    for p in seq_paths[::step]:
        sizes.setdefault(_image_size(p), []).append(os.path.basename(p))
    sizes.pop(None, None)
    if len(sizes) > 1:
        desc = "; ".join(f"{s[0]}×{s[1]} ({len(v)}, ex. {v[0]})" for s, v in sizes.items())
        warns.append("Tailles d'image différentes : " + desc
                     + " — les images d'une autre taille que la référence seront ignorées.")
    return warns


class _StackGeometry:
    """Sky/ground mask + feathered alpha (same geometry as Stack & Align)."""

    def __init__(self, h: int, w: int, mask_rad: float, ground_cutoff: float):
        self.h, self.w = h, w
        half_diag = 0.5 * float(np.hypot(h, w))
        self.radius = int(mask_rad * half_diag)
        self.cutoff_row = (int(h * (1.0 - min(ground_cutoff, 0.9)))
                           if ground_cutoff > 0.0 else None)
        m = np.zeros((h, w), np.uint8)
        cv2.circle(m, (w // 2, h // 2), self.radius, 255, -1)
        if self.cutoff_row is not None:
            m[self.cutoff_row:, :] = 0
        self.sky_mask_u8 = m
        self.sky_mask = m > 0
        self.mask_f32 = m.astype(np.float32) / 255.0
        self.alpha = cv2.GaussianBlur(self.mask_f32, (0, 0),
                                      max(3.0, self.radius * 0.02))[:, :, None]
        # Sky covers the whole frame (deep sky, or mask ≥ the corners): no
        # frozen-ground track needed — saves a full-size accumulator.
        self.use_ground = bool(self.alpha.min() < 0.999)

    def sky_mask_at(self, w: int, h: int) -> np.ndarray:
        if (w, h) == (self.w, self.h):
            return self.sky_mask
        return cv2.resize(self.sky_mask_u8, (w, h), interpolation=cv2.INTER_NEAREST) > 0

    def draw_overlay(self, img_u8: np.ndarray) -> np.ndarray:
        """Sky/ground boundary drawn on a (possibly resized) copy."""
        out = img_u8.copy()
        ih, iw = out.shape[:2]
        sx, sy = iw / self.w, ih / self.h
        th = max(1, int(round(2 * max(iw, ih) / 1000)))
        cv2.ellipse(out, (int(iw / 2), int(ih / 2)),
                    (int(self.radius * sx), int(self.radius * sy)),
                    0, 0, 360, (0, 200, 255), th, cv2.LINE_AA)
        if self.cutoff_row is not None:
            y = int(self.cutoff_row * sy)
            cv2.line(out, (0, y), (iw, y), (0, 200, 255), th, cv2.LINE_AA)
        return out


def _prepare_stack_sequence(seq_paths: list, cfg: dict,
                            work_width: int = 0) -> dict:
    """
    Global FITS scale, raw first frame, processing size and geometry.
    work_width > 0: frames are processed downscaled to that width (preview);
    the mask geometry is relative, so it matches the full-size run.
    """
    bayer_code = cfg["bayer_code"]
    p_lo, p_hi = compute_global_scale(seq_paths, bayer_code=bayer_code)
    raw_u8 = None
    for p in seq_paths:
        ext = os.path.splitext(p)[1].lower()
        # FITS: the p0.5–p99.5 linear scale shown as is = the raw frame
        img = (_load_fits_float(p, bayer_code, p_lo, p_hi)
               if ext in ('.fit', '.fits') else cv2.imread(p))
        if img is not None:
            raw_u8 = np.clip(img, 0, 255).astype(np.uint8)
            break
    if raw_u8 is None:
        raise _StackVideoError("Impossible de lire la première image de la séquence.")
    full_h, full_w = raw_u8.shape[:2]
    if 0 < work_width < full_w:
        w = work_width
        h = max(2, int(round(full_h * w / full_w)))
        raw_u8 = cv2.resize(raw_u8, (w, h), interpolation=cv2.INTER_AREA)
    else:
        h, w = full_h, full_w
    return dict(
        p_lo=p_lo, p_hi=p_hi, raw_u8=raw_u8, h=h, w=w, full_h=full_h, full_w=full_w,
        sat_level=_sensor_saturation_level(seq_paths[0], p_lo, p_hi, cfg["linearize"]),
        geom=_StackGeometry(h, w, cfg["mask_rad"], cfg["ground_cutoff"]),
    )


def _stack_load_args(seq_paths: list, cfg: dict, prep: dict) -> list:
    full = (prep["full_w"], prep["full_h"])
    work = (prep["w"], prep["h"])
    return [(p, full, work, cfg["bayer_code"], prep["p_lo"], prep["p_hi"],
             cfg["linearize"], cfg["hot"], prep["geom"].sky_mask_u8, cfg["k_sigma"],
             prep["sat_level"])
            for p in seq_paths]


class _LiveStacker:
    """
    One image at a time: background normalisation, noise weighting, star
    registration on the reference (first image), σ-clipped sky stack and
    frozen-ground mean. add() returns per-image diagnostics.
    """
    MIN_INLIERS = 6

    def __init__(self, prep: dict, cfg: dict, track_rejects: bool = False):
        self.cfg, self.geom = cfg, prep["geom"]
        h, w = prep["h"], prep["w"]
        self.h, self.w = h, w
        self.sky = _ClipStack((h, w, 3), cfg["kappa"])
        self.sky.track_rejects = track_rejects
        self.grnd = _MeanStack((h, w, 3)) if self.geom.use_ground else None
        # Sensor clipping seen in any image, in the stack's geometry (sky
        # warped, ground fixed) — the renderer neutralises colour there.
        self.clip = np.zeros((h, w), np.float32) if prep["sat_level"] else None
        self.ref_stars = self.ref_sig = self.bg_ref = self.bg_sky_ref = None
        self.sigma_ref = 1.0
        self.count = self.aligned = self.skipped = 0

    def add(self, frame: "np.ndarray | None", stars,
            clip: "np.ndarray | None" = None) -> dict:
        self.count += 1
        if frame is None:
            self.skipped += 1
            return dict(status="skipped")
        geom, cfg, bg_norm = self.geom, self.cfg, self.cfg["bg_norm"]
        first = self.ref_stars is None
        bg = (_background_map(frame)
              if self.grnd is not None and (bg_norm or first) else None)
        bg_s = _background_map(frame, geom.mask_f32) if (bg_norm or first) else None
        if first:
            if len(stars) < self.MIN_INLIERS:
                raise _StackVideoError(
                    f"Pas assez d'étoiles détectées dans l'image de référence "
                    f"({len(stars)} trouvées, {self.MIN_INLIERS} requises). "
                    "Choisissez une autre référence ou baissez la sensibilité (k·σ).")
            self.ref_stars, self.ref_sig = stars, _star_signatures(stars)
            self.bg_ref, self.bg_sky_ref = bg, bg_s
            self.sigma_ref = _frame_noise(frame, bg_s, geom.sky_mask)

        weight = 1.0
        if cfg["weighting"] and not first:
            sig = _frame_noise(frame, bg_s if bg_s is not None else self.bg_sky_ref,
                               geom.sky_mask)
            weight = float(np.clip((self.sigma_ref / sig) ** 2, 0.1, 10.0))

        # Background brought back to the reference (sensor space for the
        # ground, then warped for the sky): a sky gradient that changes
        # during the night is not taken for an outlier.
        if self.grnd is not None:
            self.grnd.add(frame - bg + self.bg_ref if bg_norm else frame, weight)
        del bg

        M, n_inl = None, 0
        if first:
            M, n_inl = np.float32([[1, 0, 0], [0, 1, 0]]), len(stars)
        elif len(stars) >= self.MIN_INLIERS:
            M, n_inl = _estimate_registration(self.ref_stars, self.ref_sig, stars,
                                              _star_signatures(stars), self.MIN_INLIERS)
        info = dict(status="aligned" if M is not None else "not_aligned",
                    stars=len(stars), inliers=n_inl, weight=weight, rejected=None)
        if M is not None:
            h, w = self.h, self.w
            if bg_norm:
                frame -= bg_s                      # in place: not used after
            warped = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LANCZOS4,
                                    borderValue=0)
            del frame
            if bg_norm:
                warped += self.bg_sky_ref
            cov = cv2.warpAffine(np.ones((h, w), np.float32), M, (w, h),
                                 flags=cv2.INTER_LINEAR, borderValue=0)
            info["rejected"] = self.sky.add(warped, weight, (cov > 0.999)[:, :, None],
                                            self.bg_sky_ref)
            self.aligned += 1
            if clip is not None and self.clip is not None:
                wc = cv2.warpAffine(clip, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
                a = self.geom.alpha[:, :, 0]
                np.maximum(self.clip, a * wc + (1.0 - a) * clip, out=self.clip)
        return info

    def snapshot(self) -> tuple:
        """(stack image, clip map or None) — new arrays, safe to hand to a thread."""
        return (_stack_snapshot(self.sky, self.grnd, self.geom.alpha),
                None if self.clip is None else self.clip.copy())


class _StackRenderer:
    """Multicam Live Stack rendering of a stack snapshot → uint8 BGR."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.last_params: "dict | None" = None     # stretch of the last render

    def render(self, snap: tuple, sky_m: np.ndarray, count: int,
               state: dict) -> "np.ndarray | None":
        final_f, clip = snap
        cfg = self.cfg
        fmax = _max3(final_f)
        valid = fmax > 0.0
        stats_mask = valid & sky_m & (fmax > 0.5)
        if int(stats_mask.sum()) < 1000:
            stats_mask = valid
        p_st = _auto_stretch_params(final_f, stats_mask, state, cfg["target"], count)
        self.last_params = p_st
        if p_st is None:
            return None
        curve = _ghs_curve_lut(cfg["ghs"]) if cfg["stretch"] == "ghs" else None
        out = _auto_stretch_render(final_f, valid, p_st, cfg["lce"], clip, curve)
        if cfg["rm_green"]:
            out = _remove_green(out)
        return _apply_saturation_clamped(out, cfg["saturation"])


def _stack_workers(h: int, w: int) -> tuple:
    """
    Bounded loading: a full-size float frame of an 8.7 MP sensor is ~105 MB,
    and each loader thread holds ~5 such temporaries (hot pixels, star
    detection) — the default 8 threads / 16-frame window would need several
    GB. ~1 GB budget, at least 2 threads. → (n_workers, prefetch)
    """
    frame_bytes = h * w * 3 * 4
    n_workers = int(max(2, min(_N_IO_WORKERS, 2**30 // (6 * frame_bytes))))
    return n_workers, int(min(_PREFETCH, n_workers + 2))


def _iter_stack_frames(load_args: list, n_workers: int, prefetch: int,
                       cancelled=lambda: False):
    """Yield (frame, stars, clip) in order, loaded ahead by a bounded thread pool."""
    n = len(load_args)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        pending: dict = {}
        next_submit = 0

        def fill(up_to: int) -> None:
            nonlocal next_submit
            while next_submit < min(up_to, n):
                pending[next_submit] = pool.submit(_load_stack_frame,
                                                   load_args[next_submit])
                next_submit += 1

        fill(prefetch)
        for i in range(n):
            if cancelled():
                for f in pending.values():
                    f.cancel()
                return
            res = pending.pop(i).result()
            fill(i + 1 + prefetch)
            yield res


def _stack_video_estimate(n_use: int, full_w: int, full_h: int, n_video: int,
                          vw: int, vh: int, sec_per_mpx: float) -> dict:
    """
    Rough cost of a full run: stacking time scales with the sensor size
    (sec_per_mpx measured by the preview), memory ≈ 0.3 GB + ~23 full-size
    float frames (measured: 2.7 GB for 4056×2160, 0.9 GB for 1920×1080).
    """
    mpx = full_w * full_h / 1e6
    frame_bytes = full_w * full_h * 3 * 4
    return dict(seconds=n_use * mpx * sec_per_mpx,
                mem_gb=0.3 + 23 * frame_bytes / 1e9,
                n_video=n_video, video_size=(vw, vh))


def process_stack_video(params: dict, progress_cb, done_cb, error_cb):
    """
    Progressive stack video: the sky is star-aligned (as in Stack & Align)
    and stacked one image at a time; after each image the running stack goes
    through the Multicam Live Stack rendering pipeline and is written to the
    MP4 — the viewer watches the sky gain detail and lose noise.
    """
    try:
        out_path    = params["out_path"]
        fps         = int(params["fps"])
        duration    = float(params["sv_duration"])
        progression = params["sv_progression"]           # "linear" | "log"
        counter     = bool(params["sv_counter"])
        max_width   = int(params.get("sv_max_width", 1920))   # 0 = native

        cfg = _stack_video_config(params)
        seq_paths = _stack_video_paths(params, int(params["align_n"]))
        n_use = len(seq_paths)
        prep = _prepare_stack_sequence(seq_paths, cfg)
        h, w, geom = prep["h"], prep["w"], prep["geom"]

        # Video frame size (stacking and the final PNG stay at full size)
        if 0 < max_width < w:
            vw = max_width - max_width % 2
            vh = int(round(h * vw / w)) // 2 * 2
        else:
            vw, vh = w, h
        small_sky = geom.sky_mask_at(vw, vh)

        # ── Video schedule + writer ────────────────────────────────────
        n_frames = max(3, int(round(duration * fps)))
        counts = _stack_video_schedule(n_use, n_frames, progression)
        video_path = os.path.splitext(out_path)[0] + "_stackvideo.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, fps, (vw, vh))
        if not writer.isOpened():
            error_cb(f"Impossible de créer la vidéo : {video_path}")
            return

        j = 0
        first = (cv2.resize(prep["raw_u8"], (vw, vh), interpolation=cv2.INTER_AREA)
                 if (vw, vh) != (w, h) else prep["raw_u8"].copy())
        if counter:
            _draw_counter(first, "Image brute")   # Hershey fonts: ASCII only
        while j < len(counts) and counts[j] == 0:
            writer.write(first)
            j += 1

        stacker = _LiveStacker(prep, cfg)
        renderer = _StackRenderer(cfg)
        stretch_state: dict = {}
        last = {"u8": first}

        def render_and_write(snap: tuple, count: int, n_rep: int) -> None:
            # Runs on the single render thread: frames stay in order and
            # rendering/encoding overlaps the stacking of the next image.
            if (vw, vh) != (w, h):
                snap = tuple(None if a is None else
                             cv2.resize(a, (vw, vh), interpolation=cv2.INTER_AREA)
                             for a in snap)
            out = renderer.render(snap, small_sky, count, stretch_state)
            if out is not None:
                last["u8"] = out
            frame_u8 = last["u8"].copy()
            if counter:
                _draw_counter(frame_u8, f"Stack : {count} image{'s' if count > 1 else ''}")
            for _ in range(n_rep):
                writer.write(frame_u8)

        n_workers, prefetch = _stack_workers(h, w)
        load_args = _stack_load_args(seq_paths, cfg, prep)
        try:
            with ThreadPoolExecutor(max_workers=1) as render_pool:
                render_fut = None
                for frame, stars, clip in _iter_stack_frames(load_args, n_workers, prefetch):
                    stacker.add(frame, stars, clip)
                    del frame
                    count = stacker.count
                    if j < len(counts) and counts[j] == count:
                        n_rep = 0
                        while j < len(counts) and counts[j] == count:
                            n_rep += 1
                            j += 1
                        # Snapshot (new array) of the stack, rendered in the background
                        snap = stacker.snapshot()
                        if render_fut is not None:
                            render_fut.result()          # one render in flight at most
                        render_fut = render_pool.submit(render_and_write, snap,
                                                        count, n_rep)
                    progress_cb(count, n_use)
                if render_fut is not None:
                    render_fut.result()
        finally:
            writer.release()

        png_path = os.path.splitext(out_path)[0] + "_stackvideo_final.png"
        if (vw, vh) != (w, h):
            # Final image at full sensor resolution (the video is downscaled)
            out = renderer.render(stacker.snapshot(), geom.sky_mask, n_use, {})
            if out is not None:
                last["u8"] = out
        cv2.imwrite(png_path, last["u8"])
        note = (f"{stacker.aligned} / {n_use} images alignées sur le ciel.\n"
                f"{len(counts)} images vidéo à {fps} ips ({len(counts) / fps:.1f} s).\n"
                f"Image finale : {png_path}")
        if stacker.skipped:
            note += (f"\n{stacker.skipped} image(s) ignorée(s) : illisibles ou d'une autre "
                     "taille que la référence.")
        done_cb(video_path, note)

    except _StackVideoError as exc:
        error_cb(str(exc))
    except Exception as exc:
        error_cb(f"{exc}\n\n{traceback.format_exc()}")


# ─── Preview window — single-frame pipeline simulators ───────────────────────
#
# These mirror process_video / process_satellites closely enough to give a
# faithful settings preview, but only ever touch a small local window of
# files around the selected frame (never the full night's sequence). Any
# step that is normally cumulative over the whole video (EMA, satellite
# boost background) is therefore approximated using a bounded trailing
# window — close enough to judge a parameter's effect, not bit-exact with
# a full run.

_PREVIEW_TRAIL = 40   # frames used to locally approximate night-long EMA states


def _load_preview_frames(paths: list, bayer_code, p_lo: float, p_hi: float) -> list:
    """Load a small list of frames as float32 BGR, skipping unreadable files."""
    frames = []
    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        if ext in ('.fit', '.fits'):
            img = _load_fits_linear(p, bayer_code, p_lo, p_hi)
        else:
            img = cv2.imread(p)
        if img is not None:
            frames.append(img.astype(np.float32))
    return frames


def _preview_resolve_window(src_dir: str, picked_path: str) -> tuple:
    """
    Locates picked_path inside the sorted file list of src_dir (same glob
    pattern as the real pipelines), so the two dynamic preview panels can
    pull neighbouring frames exactly like the full processing run would.

    Returns (paths, idx). If picked_path isn't found in src_dir's listing
    (e.g. picked from another folder), returns ([picked_path], 0) — the
    dynamic panels then degrade to a single-frame window.
    """
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG",
            "*.fit", "*.fits", "*.FIT", "*.FITS")
    paths: list = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(src_dir, e)))
    paths.sort()
    target = os.path.normcase(os.path.abspath(picked_path))
    for i, p in enumerate(paths):
        if os.path.normcase(os.path.abspath(p)) == target:
            return paths, i
    return [picked_path], 0


def _preview_fond_de_ciel(paths: list, idx: int, params: dict,
                          bayer_code, p_lo: float, p_hi: float) -> np.ndarray:
    """
    Simulates the background that the *active* mode would actually render:
      - Renforcement des Traînées actif → Track B (EMA lissé / Moyenne
        glissante / Image courante, sat_win_bg) — same source as the real video.
      - Sinon → Stacking Glissant + Boost satellites + EMA (process_video).
    """
    if params.get("use_sat_enhanced"):
        win_bg = max(1, int(params["sat_win_bg"]))
        bg_src = params["sat_bg_src"]
        lo = max(0, idx - win_bg + 1)
        frames = _load_preview_frames(paths[lo:idx + 1], bayer_code, p_lo, p_hi)
        if bg_src == "current":
            bg = frames[-1]
        elif bg_src == "rolling":
            bg = np.mean(frames, axis=0)
        else:   # "mean" — EMA lissé, approximated locally
            alpha = 2.0 / (win_bg + 1)
            bg = frames[0].copy()
            for f in frames[1:]:
                bg = alpha * f + (1.0 - alpha) * bg
    else:
        stack_n = max(1, int(params["stack_n"])) if params.get("use_stack") else 1
        hi = min(len(paths), idx + stack_n)
        frames = _load_preview_frames(paths[idx:hi], bayer_code, p_lo, p_hi)
        bg = np.mean(frames, axis=0) if len(frames) > 1 else frames[0]

        sat_boost = float(params.get("sat_boost", 1.0))
        if sat_boost > 1.01:
            lo = max(0, idx - _PREVIEW_TRAIL)
            trail = _load_preview_frames(paths[lo:idx + 1], bayer_code, p_lo, p_hi)
            ema_bg = trail[0].copy()
            for f in trail[1:]:
                ema_bg = 0.02 * f + 0.98 * ema_bg
            sat_thresh = float(params.get("sat_thresh", 0.0))
            excess = np.maximum(bg - ema_bg - sat_thresh, 0.0)
            bg = bg + (sat_boost - 1.0) * excess

        if params.get("use_ema"):
            lo = max(0, idx - _PREVIEW_TRAIL)
            trail = _load_preview_frames(paths[lo:idx + 1], bayer_code, p_lo, p_hi)
            alpha = float(params.get("alpha_ema", 0.3))
            ema = trail[0].copy()
            for f in trail[1:]:
                ema = alpha * f + (1.0 - alpha) * ema
            bg = ema

    contrast   = params["contrast"]
    brightness = params["brightness"]
    beta_c = brightness - 128.0 * (contrast - 1.0)
    u8 = cv2.convertScaleAbs(bg, alpha=contrast, beta=beta_c)
    u8 = _apply_color_correction(u8, params.get("saturation", 1.0),
                                 params.get("rgb_r", 1.0), params.get("rgb_g", 1.0),
                                 params.get("rgb_b", 1.0))
    if params.get("use_ghs"):
        u8 = cv2.LUT(u8, build_ghs_lut(**params["ghs"]))
    return u8


def _preview_satellite_filter(paths: list, idx: int, params: dict,
                              bayer_code, p_lo: float, p_hi: float) -> np.ndarray:
    """
    Single-frame simulation of the Hough-based satellite/aircraft isolation
    (median diff → Hough tunnel mask → Track A render), so the 4 Hough
    sliders can be tuned by eye without re-processing a whole sequence.

    Shows the isolated trail signal alone on black — the background is
    subtracted out, exactly like the β·stretch_dur(A) term that gets added
    onto the Fond de ciel panel (Track B) in the real video. This makes
    Hough false positives/missed detections obvious, undiluted by the sky.
    """
    win_det = int(params["sat_win_det"]) | 1
    k = win_det // 2
    lo = max(0, idx - k)
    hi = min(len(paths) - 1, idx + k)
    frames = _load_preview_frames(paths[lo:hi + 1], bayer_code, p_lo, p_hi)
    ci = min(idx - lo, len(frames) - 1)

    gray = np.stack([0.114 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.299 * f[:, :, 2]
                      for f in frames])
    fond = np.median(gray, axis=0)
    sat_thresh = float(params["sat_thresh2"])
    Dg = np.maximum(gray[ci] - fond - sat_thresh, 0.0)

    binary = (Dg > 0).astype(np.uint8) * 255
    lines = cv2.HoughLinesP(binary, rho=1, theta=np.pi / 180,
                            threshold=int(params["sat_hough_thresh"]),
                            minLineLength=int(params["sat_hough_min_len"]),
                            maxLineGap=int(params["sat_hough_max_gap"]))
    tunnel = np.zeros_like(binary)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(tunnel, (x1, y1), (x2, y2), 255,
                     thickness=int(params["sat_hough_tunnel"]))
    E = Dg * (tunnel > 0)

    sat_lut = build_ghs_lut(SP=float(params.get("sat_stretch_SP", 0.02)),
                            D=float(params.get("sat_stretch_D", 12.0)),
                            b=0.0, LP=0.0, HP=1.0)
    acc_u8 = np.clip(E, 0, 255).astype(np.uint8)
    acc_s  = cv2.LUT(acc_u8, sat_lut).astype(np.float32)
    beta = float(params.get("sat_beta", 1.5))

    out_f = np.zeros_like(frames[ci], dtype=np.float32)
    if params.get("sat_color", "white") == "white":
        out_f[:, :, 0] = beta * acc_s
        out_f[:, :, 1] = beta * acc_s
        out_f[:, :, 2] = beta * acc_s
    else:   # tinted
        lum = gray[ci] + 1.0
        for c in range(3):
            out_f[:, :, c] = beta * acc_s * (frames[ci][:, :, c] / lum)

    return np.clip(out_f, 0, 255).astype(np.uint8)


class _ZoomWindow(ctk.CTkToplevel):
    """Full-size popup for a single preview panel, opened by clicking its thumbnail.
    Rescales on its own resize and can receive live updates from PreviewWindow."""

    def __init__(self, master: ctk.CTk, title: str, bgr: np.ndarray):
        super().__init__(master)
        self.title(title)
        self._bgr = bgr
        self._photo = None
        self._resize_job: str | None = None
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h = int(sw * 0.85), int(sh * 0.85)
        self.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")
        self._lbl = _tk.Label(self, background="#1a1a1a")
        self._lbl.pack(fill="both", expand=True)
        self._lbl.bind("<Button-1>", lambda e: self.destroy())
        self.bind("<Configure>", self._on_resize)
        self.bind("<Escape>", lambda e: self.destroy())
        self.lift()
        self.focus_force()
        self._render()

    def update_image(self, bgr: np.ndarray):
        self._bgr = bgr
        self._render()

    def _on_resize(self, event):
        if event.widget is not self:
            return
        if self._resize_job:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(80, self._render)

    def _render(self):
        self._resize_job = None
        if self._bgr is None:
            return
        w = max(50, self.winfo_width())
        h = max(50, self.winfo_height())
        rgb = cv2.cvtColor(self._bgr, cv2.COLOR_BGR2RGB)
        pil = _PILImage.fromarray(rgb)
        pil.thumbnail((w, h), _PILImage.LANCZOS)
        if _HAS_PIL_TK:
            self._photo = _PILImageTk.PhotoImage(pil)
        else:
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            self._photo = _tk.PhotoImage(data=base64.b64encode(buf.getvalue()).decode())
        self._lbl.configure(image=self._photo)


class PreviewWindow(ctk.CTkToplevel):
    """
    Three-panel live preview: Original | Fond de ciel | Filtre Satellites.
    Refreshes automatically (debounced 120 ms) whenever any param changes.
    Panels rescale when the window is resized, and clicking a panel opens
    a full-size, independently resizable view of that image.
    """

    _W = 380   # initial panel width  (px)
    _H = 320   # initial panel height (px)

    def __init__(self, master: ctk.CTk, get_params):
        super().__init__(master)
        self.title("Aperçu — Fond de ciel & Filtre Satellites")
        self.geometry(f"{self._W * 3 + 100}x{self._H + 110}")
        self.resizable(True, True)
        self._get_params  = get_params
        self._orig_bgr: np.ndarray | None = None
        self._last_bg_bgr: np.ndarray | None = None
        self._last_sat_bgr: np.ndarray | None = None
        self._dir_paths: list = []
        self._idx: int = 0
        self._bayer_code = None
        self._p_lo, self._p_hi = 0.0, 255.0
        self._refresh_job: str | None     = None
        self._resize_job: str | None      = None
        self._zoom_wins: dict             = {"orig": None, "bg": None, "sat": None}
        self._build()
        self.bind("<Configure>", self._on_window_resize)
        self.lift()
        self.focus_force()

    def _build(self):
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.pack(fill="x", padx=10, pady=(10, 4))
        ctk.CTkButton(bar, text="Choisir une image…", width=170,
                      command=self._pick).pack(side="left")
        self._lbl_name = ctk.CTkLabel(bar, text="—", anchor="w")
        self._lbl_name.pack(side="left", padx=10, fill="x", expand=True)

        panels = ctk.CTkFrame(self, fg_color="transparent")
        panels.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self._panels = panels

        for title, attr, key in (("Original", "_lbl_orig", "orig"),
                                  ("Fond de ciel", "_lbl_bg", "bg"),
                                  ("Filtre Satellites (Hough)", "_lbl_sat", "sat")):
            col = ctk.CTkFrame(panels)
            col.pack(side="left", fill="both", expand=True, padx=4)
            ctk.CTkLabel(col, text=title,
                         font=ctk.CTkFont(size=12, weight="bold")).pack(pady=(8, 0))
            ctk.CTkLabel(col, text="(cliquer pour agrandir)",
                         font=ctk.CTkFont(size=10), text_color="gray60").pack(pady=(0, 2))
            # tk.Label instead of CTkLabel: supports PIL.ImageTk and tk.PhotoImage
            # directly, no CTkImage wrapper needed → avoids PIL.ImageTk import check
            lbl = _tk.Label(col, background="#1a1a1a",
                            width=self._W, height=self._H, cursor="hand2")
            lbl.pack(fill="both", expand=True, padx=4, pady=(0, 6))
            lbl.bind("<Button-1>", lambda e, k=key: self._open_zoom(k))
            setattr(self, attr, lbl)

    # ── Public ────────────────────────────────────────────────────────────────

    def refresh(self, *_):
        """Debounced refresh — safe to call on every slider event."""
        if self._orig_bgr is None:
            return
        if self._refresh_job:
            self.after_cancel(self._refresh_job)
        self._refresh_job = self.after(120, self._do_refresh)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _on_window_resize(self, event):
        """Rescale the three thumbnails to fill the window as it's resized."""
        if event.widget is not self:
            return
        if self._resize_job:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(150, self._apply_resize)

    def _apply_resize(self):
        self._resize_job = None
        self.update_idletasks()
        total_w = self._panels.winfo_width()
        total_h = self._panels.winfo_height()
        if total_w < 60 or total_h < 60:
            return
        new_w = max(120, total_w // 3 - 20)
        new_h = max(100, total_h - 40)
        if abs(new_w - self._W) < 4 and abs(new_h - self._H) < 4:
            return
        self._W, self._H = new_w, new_h
        for lbl in (self._lbl_orig, self._lbl_bg, self._lbl_sat):
            lbl.configure(width=self._W, height=self._H)
        if self._orig_bgr is not None:
            bg  = self._last_bg_bgr  if self._last_bg_bgr  is not None else self._orig_bgr
            sat = self._last_sat_bgr if self._last_sat_bgr is not None else self._orig_bgr
            self._show(self._orig_bgr, bg, sat)

    def _open_zoom(self, key: str):
        bgr = {"orig": self._orig_bgr,
               "bg":   self._last_bg_bgr,
               "sat":  self._last_sat_bgr}.get(key)
        if bgr is None:
            return
        win = self._zoom_wins.get(key)
        if win is not None and win.winfo_exists():
            win.lift()
            win.focus_force()
            return
        title = {"orig": "Original",
                 "bg":   "Fond de ciel",
                 "sat":  "Filtre Satellites (Hough)"}[key]
        self._zoom_wins[key] = _ZoomWindow(self, title, bgr)

    def _pick(self):
        src_dir = getattr(self.master, "_src_dir", "")
        path = filedialog.askopenfilename(
            title="Image de référence pour le réglage",
            initialdir=src_dir or None,
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.fit *.fits *.JPG *.JPEG *.PNG *.FIT *.FITS"),
                ("JPEG",   "*.jpg *.jpeg"),
                ("PNG",    "*.png"),
                ("FITS",   "*.fit *.fits"),
            ])
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        bayer_code = self._get_params().get("bayer_code", None)
        if ext in ('.fit', '.fits'):
            # Use current bayer setting from main app params so preview matches processing
            img = _load_fits(path, bayer_code)
            if img is not None:
                img = np.clip(img, 0, 255).astype(np.uint8)
        else:
            img = cv2.imread(path)
        if img is None:
            messagebox.showerror(
                "Erreur",
                f"Impossible de lire :\n{path}\n\n"
                "Pour les FITS, vérifiez que astropy est installé\n"
                "(pip install astropy).",
                parent=self)
            return
        self._orig_bgr = img
        self._lbl_name.configure(text=os.path.basename(path))

        # Resolve the file's position in its directory's sorted sequence so
        # the Fond de ciel / Filtre Satellites panels can pull the same
        # neighbouring frames the real pipelines would use (stack/median
        # window). Falls back to a single-frame window if not found.
        self._dir_paths, self._idx = _preview_resolve_window(
            src_dir or os.path.dirname(path), path)
        self._bayer_code = bayer_code
        has_fits = any(os.path.splitext(p)[1].lower() in ('.fit', '.fits')
                       for p in self._dir_paths)
        self._p_lo, self._p_hi = (
            compute_global_scale(self._dir_paths, bayer_code=bayer_code)
            if has_fits else (0.0, 255.0))

        self.refresh()

    def _do_refresh(self):
        self._refresh_job = None
        orig   = self._orig_bgr.copy()
        params = self._get_params()
        # Re-read bayer_code from current params (not the pick-time snapshot)
        # so toggling "Débayeriser les FITS" after picking an image is honored.
        bayer_code = params.get("bayer_code", self._bayer_code)
        threading.Thread(target=self._compute,
                         args=(orig, params, list(self._dir_paths), self._idx,
                               bayer_code, self._p_lo, self._p_hi),
                         daemon=True).start()

    def _compute(self, orig_bgr: np.ndarray, params: dict, dir_paths: list,
                idx: int, bayer_code, p_lo: float, p_hi: float):
        try:
            bg_u8 = _preview_fond_de_ciel(dir_paths, idx, params,
                                          bayer_code, p_lo, p_hi)
        except Exception:
            bg_u8 = orig_bgr
        try:
            sat_u8 = _preview_satellite_filter(dir_paths, idx, params,
                                               bayer_code, p_lo, p_hi)
        except Exception:
            sat_u8 = orig_bgr
        self.after(0, lambda: self._show(orig_bgr, bg_u8, sat_u8))

    def _show(self, orig_bgr: np.ndarray, bg_bgr: np.ndarray, sat_bgr: np.ndarray):
        def _to_photo(bgr: np.ndarray):
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil = _PILImage.fromarray(rgb)
            pil.thumbnail((self._W, self._H), _PILImage.LANCZOS)
            if _HAS_PIL_TK:
                return _PILImageTk.PhotoImage(pil)
            # Fallback: PNG → base64 → tk.PhotoImage (no PIL.ImageTk needed)
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            return _tk.PhotoImage(data=base64.b64encode(buf.getvalue()).decode())

        # Keep references to prevent GC from destroying the photos
        self._photo_orig = _to_photo(orig_bgr)
        self._photo_bg    = _to_photo(bg_bgr)
        self._photo_sat   = _to_photo(sat_bgr)
        self._lbl_orig.configure(image=self._photo_orig)
        self._lbl_bg.configure(image=self._photo_bg)
        self._lbl_sat.configure(image=self._photo_sat)

        self._last_bg_bgr  = bg_bgr
        self._last_sat_bgr = sat_bgr
        for key, bgr in (("orig", orig_bgr), ("bg", bg_bgr), ("sat", sat_bgr)):
            win = self._zoom_wins.get(key)
            if win is not None and win.winfo_exists():
                win.update_image(bgr)


_SV_PREVIEW_WIDTH = 1000   # working width of the preview mini-stack (px)


def _mem_available_gb() -> "float | None":
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return None


def _draw_stretch_histogram(lum: np.ndarray, curve: np.ndarray, w: int, h: int,
                            bg_x: "float | None", sp: "float | None") -> np.ndarray:
    """
    Histogram of the stack luminance in the stretch domain (black = 0,
    white = 1) with the tone curve on top. √x horizontal axis: the sky
    background sits at 1–3 % of the white on linear data and would be a
    single column on a linear axis. Pure OpenCV drawing (no matplotlib).
    """
    img = np.full((h, w, 3), 26, np.uint8)
    m_l, m_r, m_t, m_b = 6, 6, 8, 18
    pw, ph = w - m_l - m_r, h - m_t - m_b
    X = lambda x: m_l + int(round(np.sqrt(np.clip(x, 0.0, 1.0)) * (pw - 1)))
    Y = lambda y: m_t + int(round((1.0 - np.clip(y, 0.0, 1.0)) * (ph - 1)))
    # histogram (log counts) in √x bins
    u = np.sqrt(np.clip(lum, 0.0, 1.0))
    counts, _ = np.histogram(u, bins=pw, range=(0.0, 1.0))
    c = np.log1p(counts.astype(np.float64))
    if c.max() > 0:
        c /= c.max()
    for i, v in enumerate(c):
        if v > 0:
            cv2.line(img, (m_l + i, m_t + ph - 1), (m_l + i, m_t + ph - 1 - int(v * (ph - 1))),
                     (95, 95, 95), 1)
    # ticks
    for t in (0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0):
        x = X(t)
        cv2.line(img, (x, m_t + ph), (x, m_t + ph + 3), (150, 150, 150), 1)
        lbl = f"{t:g}"
        cv2.putText(img, lbl, (min(x - 4, w - 6 * len(lbl) - 2), h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (170, 170, 170), 1, cv2.LINE_AA)
    # background / SP markers
    if bg_x is not None:
        cv2.line(img, (X(bg_x), m_t), (X(bg_x), m_t + ph), (90, 200, 90), 1)
        cv2.putText(img, "fond", (X(bg_x) + 3, m_t + 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (90, 200, 90), 1, cv2.LINE_AA)
    if sp is not None and sp > 0:
        cv2.line(img, (X(sp), m_t), (X(sp), m_t + ph), (230, 200, 60), 1)
        cv2.putText(img, "SP", (X(sp) + 3, m_t + 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (230, 200, 60), 1, cv2.LINE_AA)
    # tone curve (output 0–1 vertical, linear)
    xs = (np.arange(pw) / max(1, pw - 1)) ** 2
    ys = curve[np.clip((xs * (len(curve) - 1)).astype(np.int64), 0, len(curve) - 1)]
    pts = np.array([[m_l + i, Y(y)] for i, y in enumerate(ys)], np.int32)
    cv2.polylines(img, [pts], False, (40, 150, 255), 2, cv2.LINE_AA)
    return img


def _bgr_to_photo(bgr: np.ndarray, w: int, h: int):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = _PILImage.fromarray(rgb)
    pil.thumbnail((w, h), _PILImage.LANCZOS)
    if _HAS_PIL_TK:
        return _PILImageTk.PhotoImage(pil)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return _tk.PhotoImage(data=base64.b64encode(buf.getvalue()).decode())


class StackVideoPreview(ctk.CTkToplevel):
    """
    Preview of the stack video before a long run: a mini-stack of the first
    K images, downscaled to ~1000 px, through the very same components as
    process_stack_video (_LiveStacker / _StackRenderer). Three panels:
    raw | 1 image processed | K images stacked, plus diagnostics and a cost
    estimate. Rendering settings re-render instantly from the cached stack
    states; stacking settings recompute the mini-stack (debounced).
    """

    _W = 380
    _H = 230

    def __init__(self, master: ctk.CTk, get_params):
        super().__init__(master)
        self.title("Aperçu — Vidéo de progression du stack")
        self.geometry(f"{self._W * 3 + 100}x{self._H + 600}")
        self.resizable(True, True)
        self._get_params = get_params
        self._gen = 0                      # bumped at each recompute: stale threads stop
        self._data: "dict | None" = None
        self._imgs: dict = {"raw": None, "one": None, "k": None}
        self._stack_job = self._render_job = self._resize_job = None
        self._zoom_wins: dict = {"raw": None, "one": None, "k": None}
        self._k_var = ctk.IntVar(value=20)
        self._show_mask = ctk.BooleanVar(value=False)
        self._build()
        self.bind("<Configure>", self._on_window_resize)
        self.lift()
        self.focus_force()
        self.after(150, self._start_stack)

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build(self):
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.pack(fill="x", padx=10, pady=(10, 2))
        ctk.CTkLabel(bar, text="Images du mini-stack (K) :", anchor="w").pack(side="left")
        k_lbl = ctk.CTkLabel(bar, text=str(self._k_var.get()), width=34)
        ctk.CTkSlider(bar, from_=2, to=100, number_of_steps=98, width=200,
                      variable=self._k_var,
                      command=lambda v: (k_lbl.configure(text=str(int(round(v)))),
                                         self.refresh_stack())).pack(side="left", padx=6)
        k_lbl.pack(side="left")
        ctk.CTkCheckBox(bar, text="Masque ciel/sol", variable=self._show_mask,
                        command=self.refresh_render).pack(side="left", padx=14)
        ctk.CTkButton(bar, text="Recalculer", width=110,
                      command=self._start_stack).pack(side="right")
        ctk.CTkButton(bar, text="GHS : partir de l'auto", width=170,
                      command=self._ghs_from_auto).pack(side="right", padx=8)

        self._progress = ctk.CTkProgressBar(self)
        self._progress.set(0.0)
        self._progress.pack(fill="x", padx=12, pady=(4, 0))
        self._lbl_status = ctk.CTkLabel(self, text="—", anchor="w")
        self._lbl_status.pack(fill="x", padx=12)

        panels = ctk.CTkFrame(self, fg_color="transparent")
        panels.pack(fill="both", expand=True, padx=8, pady=(0, 4))
        self._panels = panels
        self._titles: dict = {}
        self._lbls: dict = {}
        for key, title in (("raw", "Image brute"), ("one", "1 image traitée"),
                           ("k", "K images empilées")):
            col = ctk.CTkFrame(panels)
            col.pack(side="left", fill="both", expand=True, padx=4)
            t = ctk.CTkLabel(col, text=title, font=ctk.CTkFont(size=12, weight="bold"))
            t.pack(pady=(8, 0))
            ctk.CTkLabel(col, text="(cliquer pour agrandir)",
                         font=ctk.CTkFont(size=10), text_color="gray60").pack(pady=(0, 2))
            lbl = _tk.Label(col, background="#1a1a1a",
                            width=self._W, height=self._H, cursor="hand2")
            lbl.pack(fill="both", expand=True, padx=4, pady=(0, 6))
            lbl.bind("<Button-1>", lambda e, k=key: self._open_zoom(k))
            self._titles[key], self._lbls[key] = t, lbl

        hist_row = ctk.CTkFrame(self, fg_color="transparent")
        hist_row.pack(fill="x", padx=12, pady=(0, 4))
        ctk.CTkLabel(hist_row, text="Histogramme du stack (K images, échelle √) et "
                                    "courbe d'étirement",
                     font=ctk.CTkFont(size=11), text_color="gray60").pack(anchor="w")
        self._lbl_hist = _tk.Label(hist_row, background="#1a1a1a", height=150)
        self._lbl_hist.pack(fill="x")

        self._diag = ctk.CTkTextbox(self, height=190, wrap="word",
                                    font=ctk.CTkFont(family="monospace", size=12))
        self._diag.pack(fill="x", padx=12, pady=(0, 10))
        self._diag.configure(state="disabled")

    # ── Public (called by the main window's variable traces) ─────────────────

    def refresh_stack(self, *_):
        """Stacking setting changed: recompute the mini-stack (debounced)."""
        if self._stack_job:
            self.after_cancel(self._stack_job)
        self._stack_job = self.after(800, self._start_stack)

    def refresh_render(self, *_):
        """Rendering setting changed: re-render from the cached stack states."""
        if self._data is None:
            return
        if self._render_job:
            self.after_cancel(self._render_job)
        self._render_job = self.after(120, self._render_panels)

    # ── Mini-stack (background thread) ────────────────────────────────────────

    def _start_stack(self):
        self._stack_job = None
        params = self._get_params()
        if not params.get("src_dir"):
            self._set_status("Sélectionnez d'abord un répertoire source.")
            return
        self._gen += 1
        gen, k = self._gen, int(self._k_var.get())
        self._progress.set(0.0)
        self._set_status("Calcul du mini-stack…")
        threading.Thread(target=self._compute, args=(params, gen, k), daemon=True).start()

    def _post(self, gen: int, fn):
        def run():
            if gen == self._gen and self.winfo_exists():
                fn()
        try:
            self.after(0, run)
        except Exception:
            pass          # window closed meanwhile

    def _compute(self, params: dict, gen: int, k: int):
        try:
            cfg = _stack_video_config(params)
            n_full = max(2, int(params["align_n"]))
            seq_full = _stack_video_paths(params, n_full)
            seq = seq_full[:max(2, min(k, len(seq_full)))]
            warns = _sequence_warnings(seq_full)
            prep = _prepare_stack_sequence(seq, cfg, work_width=_SV_PREVIEW_WIDTH)
            stacker = _LiveStacker(prep, cfg, track_rejects=True)
            n_workers, prefetch = _stack_workers(prep["full_h"], prep["full_w"])
            load_args = _stack_load_args(seq, cfg, prep)
            rows, snap1 = [], None
            t0, t_stack = time.perf_counter(), 0.0
            for frame, stars, clip in _iter_stack_frames(load_args, n_workers, prefetch,
                                                         cancelled=lambda: gen != self._gen):
                ta = time.perf_counter()
                info = stacker.add(frame, stars, clip)
                t_stack += time.perf_counter() - ta
                del frame
                info["name"] = os.path.basename(seq[stacker.count - 1])
                rows.append(info)
                if snap1 is None and info["status"] == "aligned":
                    snap1 = stacker.snapshot()
                c = stacker.count
                self._post(gen, lambda c=c: (
                    self._progress.set(c / len(seq)),
                    self._set_status(f"Calcul du mini-stack… {c} / {len(seq)}")))
            if gen != self._gen:
                return
            t_total = time.perf_counter() - t0
            data = dict(cfg=cfg, prep=prep, seq=seq, n_full=len(seq_full),
                        seq_first=seq_full[0], seq_last=seq_full[-1],
                        rows=rows, warns=warns, snap1=snap1, snapk=stacker.snapshot(),
                        k=stacker.count, aligned=stacker.aligned,
                        skipped=stacker.skipped, ref_stars=len(stacker.ref_stars),
                        t_total=t_total, t_stack=t_stack, n_workers=n_workers)
            self._post(gen, lambda: self._on_stack_done(data))
        except _StackVideoError as exc:
            msg = str(exc)
            self._post(gen, lambda: self._on_error(msg))
        except Exception as exc:
            msg = f"{exc}\n{traceback.format_exc(limit=3)}"
            self._post(gen, lambda: self._on_error(msg))

    def _on_error(self, msg: str):
        self._data = None
        self._progress.set(0.0)
        self._set_status("Erreur — voir le détail ci-dessous.")
        self._set_diag("⚠ " + msg)

    def _on_stack_done(self, data: dict):
        self._data = data
        self._progress.set(1.0)
        self._set_status(f"Mini-stack de {data['k']} images calculé en "
                         f"{data['t_total']:.1f} s.")
        self._render_panels()

    # ── Rendering (UI thread: small images, ~0.1 s) ──────────────────────────

    def _render_panels(self):
        self._render_job = None
        d = self._data
        if d is None:
            return
        params = self._get_params()
        cfg = dict(d["cfg"])
        cfg.update({k: v for k, v in _stack_video_config(params).items()
                    if k in ("target", "lce", "rm_green", "saturation", "stretch", "ghs")})
        prep, geom = d["prep"], d["prep"]["geom"]
        renderer = _StackRenderer(cfg)
        counter = bool(params.get("sv_counter", True))

        st1: dict = {}
        one = (renderer.render(d["snap1"], geom.sky_mask, 1, st1)
               if d["snap1"] is not None else None)
        # The K panel continues the noise ramp started at 1 image, like the video
        stk = {"sigma_ref": st1.get("sigma_ref"), "n_ref": st1.get("n_ref")}
        t_r = time.perf_counter()
        kimg = renderer.render(d["snapk"], geom.sky_mask, d["k"], stk)
        t_render = time.perf_counter() - t_r
        p_k = renderer.last_params

        raw = prep["raw_u8"].copy()
        if counter:
            _draw_counter(raw, "Image brute")
            if one is not None:
                _draw_counter(one, "Stack : 1 image")
            if kimg is not None:
                _draw_counter(kimg, f"Stack : {d['k']} images")
        if self._show_mask.get():
            raw = geom.draw_overlay(raw)
            one = geom.draw_overlay(one) if one is not None else None
            kimg = geom.draw_overlay(kimg) if kimg is not None else None
        self._imgs = {"raw": raw, "one": one, "k": kimg}
        self._titles["k"].configure(text=f"{d['k']} images empilées")
        self._show()
        self._p_k = p_k
        if p_k is not None:
            img_k = d["snapk"][0]
            lum = cv2.transform(_stretch_input(img_k, p_k), _LUMA_MEAN)
            lum = lum[geom.sky_mask & (_max3(img_k) > 0.0)][::4]
            ghs_on = cfg["stretch"] == "ghs"
            curve = (_ghs_curve_lut(cfg["ghs"]) if ghs_on else
                     (np.arcsinh(p_k["beta"] * np.linspace(0.0, 1.0, _AS_LUT_N))
                      / np.arcsinh(p_k["beta"])).astype(np.float32))
            hw = max(300, self._panels.winfo_width() - 10)
            hist = _draw_stretch_histogram(lum, curve, hw, 150,
                                           p_k["bg"] / p_k["white"],
                                           cfg["ghs"]["SP"] if ghs_on else None)
            self._hist_bgr = hist
            self._photo_hist = _bgr_to_photo(hist, hw, 150)
            self._lbl_hist.configure(image=self._photo_hist)
        self._set_diag(self._diagnostics(d, params, p_k, t_render))

    def _show(self):
        for key, lbl in self._lbls.items():
            bgr = self._imgs.get(key)
            if bgr is None:
                lbl.configure(image="")
                continue
            photo = _bgr_to_photo(bgr, self._W, self._H)
            setattr(self, f"_photo_{key}", photo)     # keep a reference (GC)
            lbl.configure(image=photo)
            win = self._zoom_wins.get(key)
            if win is not None and win.winfo_exists():
                win.update_image(bgr)

    def _diagnostics(self, d: dict, params: dict, p_k: "dict | None",
                     t_render: float) -> str:
        prep, rows = d["prep"], d["rows"]
        fw, fh = prep["full_w"], prep["full_h"]
        L = []
        L.append(f"Séquence : {d['n_full']} images à empiler "
                 f"({os.path.basename(d['seq_first'])} → {os.path.basename(d['seq_last'])}), "
                 f"{fw}×{fh} px. Aperçu : {d['k']} premières images réduites à "
                 f"{prep['w']}×{prep['h']} px.")
        L.append(f"Référence : {d['ref_stars']} étoiles détectées (k·σ = "
                 f"{d['cfg']['k_sigma']:.1f}).")
        bad = [r["name"] for r in rows if r["status"] == "not_aligned"]
        line = (f"Alignement : {d['aligned']} / {d['k']} alignées")
        if bad:
            line += f", {len(bad)} non alignée(s) ({', '.join(bad[:4])}{'…' if len(bad) > 4 else ''})"
        if d["skipped"]:
            line += f", {d['skipped']} ignorée(s) (illisible ou autre taille)"
        L.append(line + ".")
        ws = [r["weight"] for r in rows if r["status"] == "aligned"]
        if len(ws) > 1:
            L.append(f"Poids (bruit) : min {min(ws):.2f} · médiane {float(np.median(ws)):.2f}"
                     f" · max {max(ws):.2f}" + ("" if d["cfg"]["weighting"]
                                               else "  (pondération désactivée)"))
        rej = [(r["rejected"], r["name"]) for r in rows if r.get("rejected") is not None]
        if d["cfg"]["kappa"] <= 0:
            L.append("Rejet σ : désactivé.")
        elif rej:
            worst = max(rej)
            L.append(f"Rejet σ (κ = {d['cfg']['kappa']:.2f}) : {100 * np.mean([x for x, _ in rej]):.2f} % "
                     f"des pixels rejetés en moyenne, max {100 * worst[0]:.2f} % ({worst[1]}).")
        else:
            L.append(f"Rejet σ : actif à partir de la 11e image — augmenter K pour le voir agir.")
        sat = prep["sat_level"]
        is_fits = os.path.splitext(d["seq"][0])[1].lower() in (".fit", ".fits")
        if not sat:
            L.append("Saturation capteur : aucune détectée dans la 1re image.")
        else:
            L.append("Saturation capteur : " + ("détectée" if is_fits else "niveau 8 bits")
                     + f" ({sat:.0f} en unités du stack) — couleur neutralisée sur les "
                       "cœurs d'étoiles écrêtés. Un halo coloré autour des étoiles "
                       "brillantes vient de l'optique (aberration chromatique).")
        cfg_now = _stack_video_config(params)
        if p_k is not None and cfg_now["stretch"] == "ghs":
            g = cfg_now["ghs"]
            lut = _ghs_curve_lut(g)
            bg_x = p_k["bg"] / p_k["white"]
            bg_out = float(lut[int(round(np.clip(bg_x, 0, 1) * (_AS_LUT_N - 1)))])
            L.append(f"Étirement : GHS manuel (SP {g['SP']:.4f}, D {g['D']:.0f}, b {g['b']:.1f}, "
                     f"LP {g['LP']:.4f}, HP {g['HP']:.2f}) — fond à {100 * bg_x:.1f} % du blanc "
                     f"en entrée → {100 * bg_out:.0f} % en sortie (l'auto viserait "
                     f"{100 * p_k['target']:.0f} %, β = {p_k['beta']:.0f}).")
        elif p_k is not None:
            L.append(f"Étirement ({d['k']} images) : fond visé {100 * p_k['target']:.0f} %, "
                     f"β = {p_k['beta']:.0f}, gains R/V/B = "
                     + "/".join(f"{g:.2f}" for g in p_k["gains"])
                     + f", rampe du fond auto n_eff = {p_k['n_eff']:.1f} / {_AS_RAMP_IMAGES}"
                       " (baisse du bruit mesurée, gradient de ciel inclus).")

        # Video + cost estimate for the full run
        fps = int(params.get("fps", 30))
        n_video = max(3, int(round(float(params["sv_duration"]) * fps)))
        n_use = d["n_full"]
        mw = int(params.get("sv_max_width", 1920))
        vw, vh = ((mw - mw % 2, int(round(fh * mw / fw)) // 2 * 2)
                  if 0 < mw < fw else (fw, fh))
        hold = (n_video - 1) / max(1, n_use)
        L.append(f"Vidéo : {n_video} images à {fps} ips ({n_video / fps:.1f} s), {vw}×{vh} px — "
                 + (f"chaque étape tenue ~{hold:.1f} image(s)." if hold >= 1 else
                    f"~{n_use - (n_video - 1)} étapes intermédiaires sautées "
                    "(toutes les images restent empilées)."))
        k = max(1, d["k"])
        px_ratio = (fw * fh) / float(prep["w"] * prep["h"])
        load_per_img = max(0.0, d["t_total"] - d["t_stack"]) / k
        # Measured on M31 (4056×2160, 597 images): 26 min real vs ×1.3 → 47 min
        # estimated — the small preview frames carry a fixed per-image cost
        # that doesn't scale with the pixel count.
        stack_per_img = d["t_stack"] / k * px_ratio * 0.7
        render_per_img = t_render * (vw * vh) / float(prep["w"] * prep["h"]) * 1.5
        per_img = max(load_per_img, stack_per_img, render_per_img)
        est = _stack_video_estimate(n_use, fw, fh, n_video, vw, vh,
                                    per_img / (fw * fh / 1e6))
        avail = _mem_available_gb()
        mem_line = f"≈ {est['mem_gb']:.1f} Go de RAM"
        if avail is not None:
            mem_line += f" (disponible : {avail:.1f} Go)"
            if est["mem_gb"] > 0.9 * avail:
                mem_line += "  ⚠ risque de manque de mémoire"
        secs = est["seconds"]
        dur = f"{secs / 60:.0f} min" if secs >= 90 else f"{max(1, secs):.0f} s"
        L.append(f"Estimation du traitement complet : ≈ {dur} "
                 f"(~{per_img:.1f} s/image), {mem_line}.")
        for wmsg in d["warns"]:
            L.append("⚠ " + wmsg)
        return "\n".join(L)

    # ── Misc ──────────────────────────────────────────────────────────────────

    def _ghs_from_auto(self):
        p_k = getattr(self, "_p_k", None)
        if p_k is None:
            self._set_status("Attendre la fin du mini-stack avant « partir de l'auto ».")
            return
        self.master._sv_ghs_from_auto(p_k["beta"])
        self._set_status(f"GHS initialisé sur l'étirement auto : SP 0, D {p_k['beta']:.0f}, b 0.")

    def _set_status(self, text: str):
        self._lbl_status.configure(text=text)

    def _set_diag(self, text: str):
        self._diag.configure(state="normal")
        self._diag.delete("1.0", "end")
        self._diag.insert("1.0", text)
        self._diag.configure(state="disabled")

    def _open_zoom(self, key: str):
        bgr = self._imgs.get(key)
        if bgr is None:
            return
        win = self._zoom_wins.get(key)
        if win is not None and win.winfo_exists():
            win.lift()
            win.focus_force()
            return
        self._zoom_wins[key] = _ZoomWindow(self, self._titles[key].cget("text"), bgr)

    def _on_window_resize(self, event):
        if event.widget is not self:
            return
        if self._resize_job:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(150, self._apply_resize)

    def _apply_resize(self):
        self._resize_job = None
        self.update_idletasks()
        total_w = self._panels.winfo_width()
        total_h = self._panels.winfo_height()
        if total_w < 60 or total_h < 60:
            return
        new_w, new_h = max(120, total_w // 3 - 20), max(100, total_h - 50)
        if abs(new_w - self._W) < 4 and abs(new_h - self._H) < 4:
            return
        self._W, self._H = new_w, new_h
        for lbl in self._lbls.values():
            lbl.configure(width=self._W, height=self._H)
        self._show()

    def destroy(self):
        self._gen += 1            # stop a running mini-stack
        super().destroy()


# ─── GUI ──────────────────────────────────────────────────────────────────────

class AllSkyApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("AllSky Video Processor")
        self.geometry("920x780")
        self.minsize(820, 600)
        self.resizable(True, True)
        self._src_dir: str = ""
        self._processing: bool = False
        self._preview_win: PreviewWindow | None = None
        self._sv_preview_win: StackVideoPreview | None = None
        self._build_ui()
        self._load_settings()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        self._save_settings()
        self.destroy()

    # ── Settings persistence ──────────────────────────────────────────────────

    def _setting_vars(self) -> list:
        """Ordered list of (key, tkVar) for all persistent UI parameters."""
        return [
            ("fits_bayer",       self._fits_bayer),
            ("fits_pattern",     self._fits_pattern_var),
            ("fps",              self._fps_var),
            ("use_stack",        self._use_stack),
            ("stack_n",          self._stack_n),
            ("use_ema",          self._use_ema),
            ("alpha",            self._alpha_var),
            ("use_sat_boost",    self._use_sat_boost),
            ("sat_boost",        self._sat_boost_var),
            ("sat_thresh",       self._sat_thresh_var),
            ("contrast",         self._contrast_var),
            ("brightness",       self._brightness_var),
            ("saturation",       self._saturation_var),
            ("rgb_r",            self._rgb_r_var),
            ("rgb_g",            self._rgb_g_var),
            ("rgb_b",            self._rgb_b_var),
            ("use_ghs",          self._use_ghs),
            ("ghs_SP",           self._ghs_SP),
            ("ghs_D",            self._ghs_D),
            ("ghs_b",            self._ghs_b),
            ("ghs_LP",           self._ghs_LP),
            ("ghs_HP",           self._ghs_HP),
            ("ghs_lin",          self._ghs_lin),
            ("ghs_gamma",        self._ghs_gamma),
            ("use_star",         self._use_star),
            ("star_mode",        self._star_mode_var),
            ("star_decay",       self._star_decay),
            ("star_boost",       self._star_boost),
            ("star_thresh",      self._star_thresh),
            ("use_sat_enhanced", self._use_sat_enhanced),
            ("sat_win_det",      self._sat_win_det),
            ("sat_thresh2",      self._sat_thresh2),
            ("sat_hough_thresh", self._sat_hough_thresh),
            ("sat_hough_min_len", self._sat_hough_min_len),
            ("sat_hough_max_gap", self._sat_hough_max_gap),
            ("sat_hough_tunnel", self._sat_hough_tunnel),
            ("sat_hough_dg_win", self._sat_hough_dg_win),
            ("sat_temporal_win", self._sat_temporal_win),
            ("sat_decay",        self._sat_decay),
            ("sat_win_bg",       self._sat_win_bg),
            ("sat_bg_src",       self._sat_bg_src_var),
            ("sat_beta",         self._sat_beta),
            ("sat_color",        self._sat_color_var),
            ("sat_stretch_SP",   self._sat_stretch_SP),
            ("sat_stretch_D",    self._sat_stretch_D),
            ("sat_use_mask",     self._sat_use_mask),
            ("sat_mask_radius",  self._sat_mask_radius),
            ("use_align_stack",   self._use_align_stack),
            ("align_n",           self._align_n),
            ("align_mask_radius", self._align_mask_radius),
            ("align_ground_cutoff", self._align_ground_cutoff),
            ("align_k_sigma",     self._align_k_sigma),
            ("align_out",         self._align_out_var),
            ("sv_duration",       self._sv_duration),
            ("sv_max_width",      self._sv_max_width_var),
            ("sv_progression",    self._sv_progression_var),
            ("sv_target_auto",    self._sv_target_auto),
            ("sv_target",         self._sv_target),
            ("sv_lce",            self._sv_lce),
            ("sv_saturation",     self._sv_saturation),
            ("sv_kappa",          self._sv_kappa),
            ("sv_remove_green",   self._sv_remove_green),
            ("sv_linearize",      self._sv_linearize),
            ("sv_hot_pixels",     self._sv_hot_pixels),
            ("sv_bg_norm",        self._sv_bg_norm),
            ("sv_weighting",      self._sv_weighting),
            ("sv_counter",        self._sv_counter),
            ("sv_stretch",        self._sv_stretch_var),
            ("sv_ghs_SP",         self._sv_ghs_SP),
            ("sv_ghs_D",          self._sv_ghs_D),
            ("sv_ghs_b",          self._sv_ghs_b),
            ("sv_ghs_LP",         self._sv_ghs_LP),
            ("sv_ghs_HP",         self._sv_ghs_HP),
        ]

    def _save_settings(self):
        data: dict = {
            "src_dir":  self._src_dir,
            "out_name": self._out_entry.get(),
            "align_ref_path": self._align_ref_path,
        }
        for key, var in self._setting_vars():
            data[key] = var.get()
        try:
            os.makedirs(os.path.dirname(_SETTINGS_FILE), exist_ok=True)
            with open(_SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except OSError:
            pass

    def _load_settings(self):
        try:
            with open(_SETTINGS_FILE, encoding="utf-8") as f:
                data: dict = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return

        # Restore source directory (re-count images, update label)
        src = data.get("src_dir", "")
        if src and os.path.isdir(src):
            self._src_dir = src
            exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG",
                    "*.fit", "*.fits", "*.FIT", "*.FITS")
            paths: list = []
            for e in exts:
                paths.extend(glob.glob(os.path.join(src, e)))
            n = len(paths)
            short = os.path.basename(src) or src
            self._lbl_dir.configure(
                text=f"{short}  •  {n} image{'s' if n != 1 else ''} "
                     f"trouvée{'s' if n != 1 else ''}")

        # Restore output filename
        out = data.get("out_name", "")
        if out:
            self._out_entry.delete(0, "end")
            self._out_entry.insert(0, out)

        # Restore Stack & Align reference image
        ref = data.get("align_ref_path", "")
        if ref and os.path.isfile(ref):
            self._align_ref_path = ref
            self._lbl_align_ref.configure(text=os.path.basename(ref))

        # Restore all tkVar parameters
        for key, var in self._setting_vars():
            if key in data:
                try:
                    var.set(data[key])
                except Exception:
                    pass

        # Setting a checkbox variable doesn't fire its command: re-sync the
        # Stack & Align sub-frames (the video panel depends on two settings).
        self._toggle_align_stack()

    # ── Layout helpers ────────────────────────────────────────────────────────

    def _section(self, parent, title: str):
        ctk.CTkLabel(parent, text=title,
                     font=ctk.CTkFont(size=13, weight="bold")).pack(
            anchor="w", pady=(14, 2))
        ctk.CTkFrame(parent, height=1, fg_color="#555555").pack(fill="x", pady=(0, 6))

    def _slider_row(self, parent, label: str, variable,
                    from_: float, to: float, step: float,
                    integer: bool = False, decimals: int = 2) -> ctk.CTkSlider:
        """Creates a labelled slider row and returns the slider widget."""
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=2)

        ctk.CTkLabel(row, text=label, width=238, anchor="w").pack(side="left")

        fmt = (lambda v: str(int(round(v)))) if integer else (lambda v: f"{v:.{decimals}f}")
        val_lbl = ctk.CTkLabel(row, text=fmt(variable.get()), width=50, anchor="e")
        val_lbl.pack(side="right", padx=(0, 4))
        # Keep the value label in sync with programmatic changes too
        # (settings restore, « Partir de l'auto »), not only slider drags.
        variable.trace_add("write", lambda *_: val_lbl.configure(text=fmt(variable.get())))

        n_steps = max(1, round((to - from_) / step))

        def on_change(raw):
            if integer:
                v = int(round(float(raw)))
                variable.set(v)
            else:
                v = round(float(raw) / step) * step
                variable.set(round(v, 10))

        slider = ctk.CTkSlider(row, from_=from_, to=to,
                               number_of_steps=n_steps,
                               variable=variable,
                               command=on_change)
        slider.pack(side="left", fill="x", expand=True, padx=8)
        return slider

    def _set_frame_state(self, frame: ctk.CTkFrame, enabled: bool):
        """Recursively enable/disable all widgets inside a frame."""
        state = "normal" if enabled else "disabled"
        for widget in frame.winfo_children():
            try:
                widget.configure(state=state)
            except Exception:
                pass
            for child in widget.winfo_children():
                try:
                    child.configure(state=state)
                except Exception:
                    pass

    # ── Full UI ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Pinned header: source directory + video output (always visible) ──
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=14, pady=(12, 4))

        self._section(header, "Répertoire Source & Sortie")
        src_row = ctk.CTkFrame(header, fg_color="transparent")
        src_row.pack(fill="x")
        ctk.CTkButton(src_row, text="Parcourir…", width=120,
                      command=self._browse_src).pack(side="left")
        self._lbl_dir = ctk.CTkLabel(src_row, text="Aucun dossier sélectionné",
                                      anchor="w", wraplength=560)
        self._lbl_dir.pack(side="left", padx=10, fill="x", expand=True)
        ctk.CTkLabel(header,
                     text="Formats acceptés : .jpg / .jpeg / .png / .fit / .fits",
                     font=ctk.CTkFont(size=11), text_color="gray60").pack(
            anchor="w", padx=4, pady=(2, 4))

        out_row = ctk.CTkFrame(header, fg_color="transparent")
        out_row.pack(fill="x", pady=(2, 2))
        ctk.CTkLabel(out_row, text="Fichier de sortie :", width=150, anchor="w").pack(side="left")
        self._out_entry = ctk.CTkEntry(out_row, width=260,
                                        placeholder_text="allsky_timelapse.mp4")
        self._out_entry.insert(0, "allsky_timelapse.mp4")
        self._out_entry.pack(side="left", padx=8)
        ctk.CTkLabel(out_row, text="FPS :", width=40, anchor="e").pack(side="left", padx=(16, 4))
        self._fps_var = ctk.IntVar(value=30)
        ctk.CTkSlider(out_row, from_=10, to=60, number_of_steps=10,
                      variable=self._fps_var, width=140).pack(side="left", padx=4)
        self._fps_lbl = ctk.CTkLabel(out_row, text=str(self._fps_var.get()), width=30, anchor="w")
        self._fps_lbl.pack(side="left")
        self._fps_var.trace_add(
            "write",
            lambda *_: self._fps_lbl.configure(text=str(self._fps_var.get())))

        # FITS Bayer debayer option
        self._fits_bayer       = ctk.BooleanVar(value=False)
        self._fits_pattern_var = ctk.StringVar(value="RGGB")
        ctk.CTkCheckBox(header,
                         text="Débayeriser les FITS (capteur RAW Bayer)",
                         variable=self._fits_bayer,
                         command=self._toggle_fits_bayer).pack(anchor="w", pady=(4, 0))
        self._fits_bayer_sub = ctk.CTkFrame(header, fg_color="transparent")
        self._fits_bayer_sub.pack(fill="x", padx=28)
        bayer_row = ctk.CTkFrame(self._fits_bayer_sub, fg_color="transparent")
        bayer_row.pack(fill="x", pady=2)
        ctk.CTkLabel(bayer_row, text="Matrice Bayer :", width=150, anchor="w").pack(side="left")
        ctk.CTkOptionMenu(bayer_row,
                          variable=self._fits_pattern_var,
                          values=["RGGB", "BGGR", "GRBG", "GBRG"],
                          width=120).pack(side="left", padx=8)
        ctk.CTkLabel(self._fits_bayer_sub,
                     text="RGGB = ZWO/Sony standard  •  Ignoré si le FITS est déjà en couleur (3 canaux).",
                     font=ctk.CTkFont(size=11), text_color="gray60",
                     wraplength=640).pack(anchor="w", pady=(0, 2))
        self._set_frame_state(self._fits_bayer_sub, False)

        # ── Tabbed settings (each tab scrolls independently — no more one
        #    giant page-long scroll) ────────────────────────────────────────
        tabview = ctk.CTkTabview(self)
        tabview.pack(fill="both", expand=True, padx=14, pady=(4, 4))

        tab_names = ["Réduction du Bruit", "Image & Couleur",
                     "Star Trail", "Satellites / Avions", "Stack & Align"]
        for name in tab_names:
            tabview.add(name)

        def _tab_scroll(name: str) -> ctk.CTkScrollableFrame:
            f = ctk.CTkScrollableFrame(tabview.tab(name), fg_color="transparent")
            f.pack(fill="both", expand=True)
            return f

        tab_bruit = _tab_scroll("Réduction du Bruit")
        tab_image = _tab_scroll("Image & Couleur")
        tab_star  = _tab_scroll("Star Trail")
        tab_sat   = _tab_scroll("Satellites / Avions")
        tab_align = _tab_scroll("Stack & Align")

        # ── Noise reduction ───────────────────────────────────────────────
        # Stacking Glissant
        self._use_stack = ctk.BooleanVar(value=False)
        self._stack_n   = ctk.IntVar(value=3)
        ctk.CTkCheckBox(tab_bruit,
                         text="Activer le Stacking Glissant",
                         variable=self._use_stack,
                         command=self._toggle_stack).pack(anchor="w")
        self._stack_sub = ctk.CTkFrame(tab_bruit, fg_color="transparent")
        self._stack_sub.pack(fill="x", padx=28)
        self._slider_row(self._stack_sub, "Nombre d'images (N)",
                         self._stack_n, 2, 10, 1, integer=True)
        self._set_frame_state(self._stack_sub, False)

        # EMA
        self._use_ema   = ctk.BooleanVar(value=False)
        self._alpha_var = ctk.DoubleVar(value=0.35)
        ctk.CTkCheckBox(tab_bruit,
                         text="Activer le Filtre EMA (Moyenne Mobile Exponentielle)",
                         variable=self._use_ema,
                         command=self._toggle_ema).pack(anchor="w", pady=(8, 0))
        self._ema_sub = ctk.CTkFrame(tab_bruit, fg_color="transparent")
        self._ema_sub.pack(fill="x", padx=28)
        self._slider_row(self._ema_sub, "Facteur Alpha (α)",
                         self._alpha_var, 0.05, 1.00, 0.05)
        self._set_frame_state(self._ema_sub, False)

        # Boost satellites (timelapse)
        self._use_sat_boost  = ctk.BooleanVar(value=False)
        self._sat_boost_var  = ctk.DoubleVar(value=2.0)
        self._sat_thresh_var = ctk.IntVar(value=15)
        ctk.CTkCheckBox(tab_bruit,
                         text="Activer le Boost Satellites (Timelapse)",
                         variable=self._use_sat_boost,
                         command=self._toggle_sat_boost).pack(anchor="w", pady=(8, 0))
        self._sat_sub = ctk.CTkFrame(tab_bruit, fg_color="transparent")
        self._sat_sub.pack(fill="x", padx=28)
        self._slider_row(self._sat_sub, "Boost satellites",
                         self._sat_boost_var, 1.0, 8.0, 0.5)
        self._slider_row(self._sat_sub, "Seuil de détection (niveau)",
                         self._sat_thresh_var, 0, 60, 1, integer=True)
        ctk.CTkLabel(self._sat_sub,
                     text="Amplifie les transitoires (satellites) par rapport au fond stable (étoiles).\n"
                          "2–4 = satellites bien visibles  •  >5 = très agressif  •  seuil typique JPEG : 10–20.",
                     font=ctk.CTkFont(size=11), text_color="gray60",
                     wraplength=600).pack(anchor="w", pady=(0, 4))
        self._set_frame_state(self._sat_sub, False)

        ctk.CTkLabel(tab_bruit,
                     text="ℹ Pipeline bruit : Stacking → Boost Satellites → EMA.",
                     font=ctk.CTkFont(size=11), text_color="gray60").pack(
            anchor="w", padx=6, pady=(8, 4))

        # ── Linear correction + Color ─────────────────────────────────────
        self._section(tab_image, "Correction Linéaire et Couleur")

        self._contrast_var   = ctk.DoubleVar(value=1.0)
        self._brightness_var = ctk.IntVar(value=0)
        self._saturation_var = ctk.DoubleVar(value=1.0)
        self._rgb_r_var      = ctk.DoubleVar(value=1.0)
        self._rgb_g_var      = ctk.DoubleVar(value=1.0)
        self._rgb_b_var      = ctk.DoubleVar(value=1.0)

        self._slider_row(tab_image, "Contraste — centré sur 128 (α)",
                         self._contrast_var,   0.5, 3.0, 0.05)
        self._slider_row(tab_image, "Luminosité — décalage offset (β)",
                         self._brightness_var, -100, 100, 1, integer=True)
        self._slider_row(tab_image, "Saturation  (0 = niveaux de gris, 1 = neutre)",
                         self._saturation_var, 0.0, 3.0, 0.05)

        ctk.CTkLabel(tab_image, text="Balance RVB :",
                     font=ctk.CTkFont(size=12, weight="bold")).pack(
            anchor="w", padx=4, pady=(8, 2))
        self._slider_row(tab_image, "  Rouge (R)",
                         self._rgb_r_var, 0.5, 3.0, 0.05)
        self._slider_row(tab_image, "  Vert  (G)",
                         self._rgb_g_var, 0.5, 3.0, 0.05)
        self._slider_row(tab_image, "  Bleu  (B)",
                         self._rgb_b_var, 0.5, 3.0, 0.05)

        ctk.CTkLabel(tab_image,
                     text="ℹ Pipeline couleur : Contraste/Luminosité → Saturation → Balance RVB → GHS",
                     font=ctk.CTkFont(size=11), text_color="gray60",
                     wraplength=620).pack(anchor="w", padx=6, pady=(4, 0))

        # ── GHS ───────────────────────────────────────────────────────────
        self._section(tab_image, "Stretch GHS (Generalized Hyperbolic Stretch)")
        self._use_ghs = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(tab_image,
                         text="Activer le Stretch GHS",
                         variable=self._use_ghs,
                         command=self._toggle_ghs).pack(anchor="w")

        self._ghs_sub = ctk.CTkFrame(tab_image, fg_color="transparent")
        self._ghs_sub.pack(fill="x", padx=28)

        self._ghs_SP = ctk.DoubleVar(value=0.10)
        self._ghs_D  = ctk.DoubleVar(value=4.0)
        self._ghs_b  = ctk.DoubleVar(value=1.0)
        self._ghs_LP = ctk.DoubleVar(value=0.00)
        self._ghs_HP = ctk.DoubleVar(value=1.00)

        self._slider_row(self._ghs_sub, "SP — Symmetry Point (fond de ciel)",
                         self._ghs_SP, 0.005, 0.70, 0.005)
        self._slider_row(self._ghs_sub, "D  — Stretch Factor (intensité)",
                         self._ghs_D,  0.0,  25.0, 0.5)
        self._slider_row(self._ghs_sub, "b  — Highlight Compression (étoiles)",
                         self._ghs_b,  0.0,  30.0, 0.5)
        self._slider_row(self._ghs_sub, "LP — Low Point (protection fond de ciel)",
                         self._ghs_LP, 0.000, 0.20, 0.005)
        self._slider_row(self._ghs_sub, "HP — Highlight Point (protection hautes lumières)",
                         self._ghs_HP, 0.30, 1.00, 0.01)

        # Gamma linearisation (JPEG ISP)
        self._ghs_lin = ctk.BooleanVar(value=False)
        self._ghs_gamma = ctk.DoubleVar(value=2.2)
        ctk.CTkCheckBox(self._ghs_sub,
                         text="Linéariser avant GHS  (JPEG ISP / gamma encodé)",
                         variable=self._ghs_lin,
                         command=self._toggle_ghs_gamma).pack(anchor="w", pady=(8, 0))
        self._ghs_gamma_sub = ctk.CTkFrame(self._ghs_sub, fg_color="transparent")
        self._ghs_gamma_sub.pack(fill="x", padx=20)
        self._slider_row(self._ghs_gamma_sub, "Gamma caméra (γ)",
                         self._ghs_gamma, 1.0, 3.0, 0.1)
        ctk.CTkLabel(self._ghs_gamma_sub,
                     text="2.2 = sRGB standard  •  Après activation, abaissez SP à 0.005–0.05 "
                          "pour cibler le fond de ciel linéarisé.",
                     font=ctk.CTkFont(size=11), text_color="gray60",
                     wraplength=580).pack(anchor="w", pady=(0, 4))
        self._set_frame_state(self._ghs_gamma_sub, False)

        self._set_frame_state(self._ghs_sub, False)

        # ── Star Trail ───────────────────────────────────────────────────
        self._use_star = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(tab_star,
                         text="Activer le Star Trail",
                         variable=self._use_star,
                         command=self._toggle_star).pack(anchor="w")

        self._star_sub = ctk.CTkFrame(tab_star, fg_color="transparent")
        self._star_sub.pack(fill="x", padx=28)

        # Mode selector
        mode_row = ctk.CTkFrame(self._star_sub, fg_color="transparent")
        mode_row.pack(fill="x", pady=(4, 2))
        ctk.CTkLabel(mode_row, text="Mode de sortie :", width=150, anchor="w").pack(side="left")
        self._star_mode_var = ctk.StringVar(value="Progressif — vidéo animée (.mp4)")
        ctk.CTkOptionMenu(
            mode_row,
            variable=self._star_mode_var,
            values=["Progressif — vidéo animée (.mp4)",
                    "Image finale — cliché unique (.png)"],
            width=300,
        ).pack(side="left", padx=8)

        # Decay slider
        self._star_decay = ctk.DoubleVar(value=1.00)
        self._slider_row(self._star_sub,
                         "Persistance (decay)",
                         self._star_decay, 0.90, 1.00, 0.01)
        ctk.CTkLabel(self._star_sub,
                     text="1.00 = traînées permanentes  •  < 1.00 = effet comète (les vieilles traînées s'effacent)",
                     font=ctk.CTkFont(size=11), text_color="gray60",
                     wraplength=600).pack(anchor="w", pady=(0, 6))

        # Satellite boost slider
        self._star_boost = ctk.DoubleVar(value=1.0)
        self._slider_row(self._star_sub,
                         "Boost satellites",
                         self._star_boost, 1.0, 8.0, 0.5)
        ctk.CTkLabel(self._star_sub,
                     text="Amplifie les signaux transitoires (satellites) par rapport au fond stable (étoiles).\n"
                          "1.0 = désactivé  •  2–4 = satellites bien visibles  •  >5 = très agressif.",
                     font=ctk.CTkFont(size=11), text_color="gray60",
                     wraplength=600).pack(anchor="w", pady=(0, 6))

        self._star_thresh = ctk.IntVar(value=15)
        self._slider_row(self._star_sub,
                         "Seuil de détection (niveau)",
                         self._star_thresh, 0, 60, 1, integer=True)
        ctk.CTkLabel(self._star_sub,
                     text="Marge minimum au-dessus du fond pour qu'un signal soit considéré comme transitoire.\n"
                          "Montez ce seuil pour supprimer le bruit amplifié (typique JPEG : 10–20).\n"
                          "Un satellite dépasse généralement le fond de 30–100+ niveaux.",
                     font=ctk.CTkFont(size=11), text_color="gray60",
                     wraplength=600).pack(anchor="w", pady=(0, 4))

        ctk.CTkLabel(tab_star,
                     text="ℹ En mode Star Trail, les modules Stacking et EMA ne s'appliquent pas.",
                     font=ctk.CTkFont(size=11), text_color="gray60").pack(
            anchor="w", padx=6, pady=(6, 0))

        self._set_frame_state(self._star_sub, False)

        # ── Satellite / aircraft trail enhancement ────────────────────────
        self._use_sat_enhanced = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(tab_sat,
                         text="Activer le Renforcement des Traînées",
                         variable=self._use_sat_enhanced,
                         command=self._toggle_sat_enhanced).pack(anchor="w")

        self._sat_enh_sub = ctk.CTkFrame(tab_sat, fg_color="transparent")
        self._sat_enh_sub.pack(fill="x", padx=28)

        # Detection
        ctk.CTkLabel(self._sat_enh_sub,
                     text="Détection :",
                     font=ctk.CTkFont(size=12, weight="bold")).pack(
            anchor="w", pady=(6, 1))

        self._sat_win_det   = ctk.IntVar(value=7)
        self._sat_thresh2   = ctk.IntVar(value=15)
        self._slider_row(self._sat_enh_sub,
                         "Fenêtre médiane de détection (nb images, impair)",
                         self._sat_win_det, 3, 15, 2, integer=True)
        self._slider_row(self._sat_enh_sub,
                         "Seuil de détection (niveau above fond)",
                         self._sat_thresh2, 0, 60, 1, integer=True)
        ctk.CTkLabel(self._sat_enh_sub,
                     text="Fenêtre impaire ≥ 5 : les satellites disparaissent de la médiane.  "
                          "Seuil 3-8 pour FITS/PNG, 12-20 pour JPEG.",
                     font=ctk.CTkFont(size=11), text_color="gray60",
                     wraplength=580).pack(anchor="w", pady=(0, 4))

        # Hough line isolation
        ctk.CTkLabel(self._sat_enh_sub,
                     text="Isolation par Transformée de Hough :",
                     font=ctk.CTkFont(size=12, weight="bold")).pack(
            anchor="w", pady=(6, 1))

        self._sat_hough_thresh  = ctk.IntVar(value=15)
        self._sat_hough_min_len = ctk.IntVar(value=12)
        self._sat_hough_max_gap = ctk.IntVar(value=4)
        self._sat_hough_tunnel  = ctk.IntVar(value=5)
        self._sat_hough_dg_win  = ctk.IntVar(value=3)
        self._slider_row(self._sat_enh_sub,
                         "Seuil de vote (pixels alignés requis)",
                         self._sat_hough_thresh, 5, 40, 1, integer=True)
        self._slider_row(self._sat_enh_sub,
                         "Longueur minimale du segment (pixels)",
                         self._sat_hough_min_len, 5, 40, 1, integer=True)
        self._slider_row(self._sat_enh_sub,
                         "Saut maximal entre segments (pixels)",
                         self._sat_hough_max_gap, 0, 20, 1, integer=True)
        self._slider_row(self._sat_enh_sub,
                         "Épaisseur du tunnel de sélection (pixels)",
                         self._sat_hough_tunnel, 1, 15, 1, integer=True)
        self._slider_row(self._sat_enh_sub,
                         "Fenêtre Dg combinée pour détection Hough (frames)",
                         self._sat_hough_dg_win, 1, 7, 1, integer=True)
        ctk.CTkLabel(self._sat_enh_sub,
                     text="Permet de baisser le seuil ADU pour capter les satellites faibles : "
                          "Hough ne retient que les alignements de pixels (un satellite), pas le bruit "
                          "dispersé. Les pixels d'origine sont conservés dans le tunnel détecté "
                          "(rendu naturel, sans lignes redessinées). "
                          "Seuil de vote 10-20 et longueur min. 10-20 px pour poses de 3-6s  •  "
                          "Saut 4-8 px pour relier les pointillés d'un satellite ou avion.  •  "
                          "Fenêtre Dg 3-5 : cumule N cartes Dg avant Hough → détecte les traînées "
                          "trop faibles sur une seule frame (texture per-frame préservée).",
                     font=ctk.CTkFont(size=11), text_color="gray60",
                     wraplength=580).pack(anchor="w", pady=(0, 4))

        # Persistence accumulator
        ctk.CTkLabel(self._sat_enh_sub,
                     text="Persistance :",
                     font=ctk.CTkFont(size=12, weight="bold")).pack(
            anchor="w", pady=(6, 1))

        self._sat_temporal_win = ctk.IntVar(value=3)
        self._slider_row(self._sat_enh_sub,
                         "Fenêtre temporelle de lissage E (frames)",
                         self._sat_temporal_win, 1, 9, 1, integer=True)
        ctk.CTkLabel(self._sat_enh_sub,
                     text="Max glissant sur N frames consécutives : comble les trous où Hough rate la traînée.  "
                          "1 = désactivé  •  3 = trous de 1-2 frames comblés (recommandé)  •  7 = trous de 3-4 frames.",
                     font=ctk.CTkFont(size=11), text_color="gray60",
                     wraplength=580).pack(anchor="w", pady=(0, 4))
        self._sat_decay     = ctk.DoubleVar(value=0.97)
        self._slider_row(self._sat_enh_sub,
                         "Atténuation par frame (α — 1.0 = traînée permanente)",
                         self._sat_decay, 0.80, 1.00, 0.01)
        ctk.CTkLabel(self._sat_enh_sub,
                     text="0.95 = évanouissement rapide (étoile filante)  •  "
                          "0.97-0.99 = traînée longue  •  1.00 = toute la nuit.",
                     font=ctk.CTkFont(size=11), text_color="gray60",
                     wraplength=580).pack(anchor="w", pady=(0, 4))

        # Background
        ctk.CTkLabel(self._sat_enh_sub,
                     text="Fond de ciel :",
                     font=ctk.CTkFont(size=12, weight="bold")).pack(
            anchor="w", pady=(6, 1))

        self._sat_win_bg    = ctk.IntVar(value=31)
        self._sat_bg_src_var = ctk.StringVar(value="EMA lissé")
        self._slider_row(self._sat_enh_sub,
                         "Nb images fond de ciel (EMA equiv. / Moyenne glissante)",
                         self._sat_win_bg, 5, 101, 2, integer=True)
        bg_row = ctk.CTkFrame(self._sat_enh_sub, fg_color="transparent")
        bg_row.pack(fill="x", pady=2)
        ctk.CTkLabel(bg_row, text="Source du fond de ciel :",
                     width=150, anchor="w").pack(side="left")
        ctk.CTkOptionMenu(bg_row, variable=self._sat_bg_src_var,
                          values=["EMA lissé", "Moyenne glissante (N images)", "Image courante"],
                          width=240).pack(side="left", padx=8)
        ctk.CTkLabel(self._sat_enh_sub,
                     text="EMA lissé : fond lisse à faible coût mémoire.  "
                          "Moyenne glissante : moyenne exacte sur N images (plus précis, ~N×25 Mo RAM).  "
                          "Image courante : étoiles fluides mais fond bruité.",
                     font=ctk.CTkFont(size=11), text_color="gray60",
                     wraplength=580).pack(anchor="w", pady=(0, 4))

        # Fusion & stretch
        ctk.CTkLabel(self._sat_enh_sub,
                     text="Fusion et Stretch :",
                     font=ctk.CTkFont(size=12, weight="bold")).pack(
            anchor="w", pady=(6, 1))

        self._sat_beta      = ctk.DoubleVar(value=1.5)
        self._sat_color_var = ctk.StringVar(value="Blanc")
        self._sat_stretch_SP = ctk.DoubleVar(value=0.02)
        self._sat_stretch_D  = ctk.DoubleVar(value=12.0)
        self._slider_row(self._sat_enh_sub, "Gain des traînées (β)",
                         self._sat_beta, 0.5, 3.0, 0.1)
        color_row = ctk.CTkFrame(self._sat_enh_sub, fg_color="transparent")
        color_row.pack(fill="x", pady=2)
        ctk.CTkLabel(color_row, text="Couleur des traînées :",
                     width=150, anchor="w").pack(side="left")
        ctk.CTkOptionMenu(color_row, variable=self._sat_color_var,
                          values=["Blanc", "Teinté (chrominance originale)"],
                          width=240).pack(side="left", padx=8)
        self._slider_row(self._sat_enh_sub,
                         "Stretch traînée — Point de symétrie (SP)",
                         self._sat_stretch_SP, 0.005, 0.10, 0.005)
        self._slider_row(self._sat_enh_sub,
                         "Stretch traînée — Intensité (D)",
                         self._sat_stretch_D, 2.0, 20.0, 1.0)
        ctk.CTkLabel(self._sat_enh_sub,
                     text="SP bas + D élevé = satellites faibles très boostés sans saturer le fond.  "
                          "Ajuster β pour doser la visibilité globale des traînées.",
                     font=ctk.CTkFont(size=11), text_color="gray60",
                     wraplength=580).pack(anchor="w", pady=(0, 4))

        # Horizon mask
        ctk.CTkLabel(self._sat_enh_sub,
                     text="Masque d'horizon :",
                     font=ctk.CTkFont(size=12, weight="bold")).pack(
            anchor="w", pady=(6, 1))

        self._sat_use_mask   = ctk.BooleanVar(value=False)
        self._sat_mask_radius = ctk.DoubleVar(value=0.45)
        ctk.CTkCheckBox(self._sat_enh_sub,
                         text="Masque circulaire (exclut arbres / bâtiments)",
                         variable=self._sat_use_mask,
                         command=self._toggle_sat_mask).pack(anchor="w", pady=(2, 0))
        self._sat_mask_sub = ctk.CTkFrame(self._sat_enh_sub, fg_color="transparent")
        self._sat_mask_sub.pack(fill="x", padx=20)
        self._slider_row(self._sat_mask_sub,
                         "Rayon (fraction de min(H,W))",
                         self._sat_mask_radius, 0.20, 0.50, 0.01)
        ctk.CTkLabel(self._sat_mask_sub,
                     text="0.45 = typique Allsky 180°  •  Réduit les faux-positifs (branches, insectes).",
                     font=ctk.CTkFont(size=11), text_color="gray60",
                     wraplength=560).pack(anchor="w", pady=(0, 4))
        self._set_frame_state(self._sat_mask_sub, False)

        ctk.CTkLabel(tab_sat,
                     text="ℹ Renforcement Traînées : indépendant des modes Stacking / EMA / Star Trail.\n"
                          "  Le GHS principal s'applique au fond de ciel ; le stretch traînée est dédié aux satellites.",
                     font=ctk.CTkFont(size=11), text_color="gray60",
                     wraplength=600).pack(anchor="w", padx=6, pady=(8, 0))

        self._set_frame_state(self._sat_enh_sub, False)

        # ── Stack & Align — "freeze the ground, align the sky" ────────────
        self._use_align_stack = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(tab_align,
                         text="Activer l'Empilement Aligné (ciel aligné, sol figé)",
                         variable=self._use_align_stack,
                         command=self._toggle_align_stack).pack(anchor="w")

        self._align_sub = ctk.CTkFrame(tab_align, fg_color="transparent")
        self._align_sub.pack(fill="x", padx=28)

        self._align_ref_path: str = ""
        ref_row = ctk.CTkFrame(self._align_sub, fg_color="transparent")
        ref_row.pack(fill="x", pady=(4, 2))
        ctk.CTkButton(ref_row, text="Choisir l'image de référence…", width=220,
                      command=self._pick_align_ref).pack(side="left")
        self._lbl_align_ref = ctk.CTkLabel(
            ref_row, text="Aucune — utilise la première image du dossier",
            anchor="w")
        self._lbl_align_ref.pack(side="left", padx=10, fill="x", expand=True)

        out_mode_row = ctk.CTkFrame(self._align_sub, fg_color="transparent")
        out_mode_row.pack(fill="x", pady=(2, 4))
        ctk.CTkLabel(out_mode_row, text="Sortie :", width=150, anchor="w").pack(side="left")
        self._align_out_var = ctk.StringVar(value="Image finale (.png)")
        ctk.CTkOptionMenu(out_mode_row,
                          variable=self._align_out_var,
                          values=["Image finale (.png)",
                                  "Vidéo de progression du stack (.mp4)"],
                          command=lambda *_: self._toggle_align_stack(),
                          width=300).pack(side="left", padx=8)

        self._align_n = ctk.IntVar(value=10)
        self._slider_row(self._align_sub, "Nombre d'images à empiler (N)",
                         self._align_n, 3, 1500, 1, integer=True)
        ctk.CTkLabel(self._align_sub,
                     text="Les N images consécutives à partir de la référence (ou du début du "
                          "dossier si aucune n'est choisie) sont alignées sur le ciel puis "
                          "moyennées. N élevé = moins de bruit mais traitement plus long.",
                     font=ctk.CTkFont(size=11), text_color="gray60",
                     wraplength=600).pack(anchor="w", pady=(0, 6))

        self._align_mask_radius = ctk.DoubleVar(value=0.95)
        self._slider_row(self._align_sub, "Rayon du masque ciel/sol (fraction de la demi-diagonale)",
                         self._align_mask_radius, 0.20, 1.10, 0.01)
        ctk.CTkLabel(self._align_sub,
                     text="Le disque central = ciel (aligné), l'extérieur = sol/horizon (figé, "
                          "simplement moyenné). 1.0 = le disque atteint exactement les coins de "
                          "l'image (quel que soit le format du capteur). Contrairement au masque "
                          "satellites (qui exclut large pour réduire les faux-positifs), ici on "
                          "veut en général aligner un maximum de ciel : monter vers ~1.0 élimine "
                          "la couture visible entre étoiles nettes et traînées. Ne réduire que si "
                          "un vrai décor (arbres/bâtiments) encercle largement le ciel jusque "
                          "près du zénith.",
                     font=ctk.CTkFont(size=11), text_color="gray60",
                     wraplength=600).pack(anchor="w", pady=(0, 6))

        self._align_ground_cutoff = ctk.DoubleVar(value=0.0)
        self._slider_row(self._align_sub,
                         "Bas de l'image toujours figé (fraction de H, depuis le bas)",
                         self._align_ground_cutoff, 0.0, 0.5, 0.01)
        ctk.CTkLabel(self._align_sub,
                     text="0 = désactivé. Un disque centré ne peut pas exclure un décor concentré "
                          "en bas (toit, arbres) sans aussi couper le ciel des coins — ce réglage "
                          "force une bande horizontale en bas de l'image à rester non-alignée "
                          "(simple moyenne brute), quel que soit le rayon ci-dessus. À augmenter "
                          "si le décor du bas montre un léger dédoublement/fantôme après le "
                          "traitement ; viser juste au-dessus du point le plus haut du décor "
                          "(toit, cime d'arbre).",
                     font=ctk.CTkFont(size=11), text_color="gray60",
                     wraplength=600).pack(anchor="w", pady=(0, 6))

        self._align_k_sigma = ctk.DoubleVar(value=6.0)
        self._slider_row(self._align_sub,
                         "Sensibilité détection étoiles (k·σ au-dessus du fond)",
                         self._align_k_sigma, 3.0, 10.0, 0.5)
        ctk.CTkLabel(self._align_sub,
                     text="Plus bas = détecte plus d'étoiles faibles (mais plus de faux positifs). "
                          "Plus haut = ne garde que les étoiles brillantes (registration plus "
                          "fiable, mais risque d'échec si trop peu d'étoiles). 6.0 = point de "
                          "départ raisonnable pour un ciel JPEG typique.",
                     font=ctk.CTkFont(size=11), text_color="gray60",
                     wraplength=600).pack(anchor="w", pady=(0, 4))

        # ── Stack video (Multicam Live Stack pipeline) ─────────────────────
        # Sibling of _align_sub (not a child): _set_frame_state only reaches
        # two widget levels deep.
        self._section(tab_align, "Vidéo de progression du stack (pipeline Live Stack Multicam)")
        self._sv_sub = ctk.CTkFrame(tab_align, fg_color="transparent")
        self._sv_sub.pack(fill="x", padx=28)

        # Packed before _sv_sub (not inside it): the panel is disabled until
        # the video mode is on, and a disabled button looks almost the same.
        ctk.CTkButton(tab_align, text="Aperçu de la vidéo de stack…", height=32,
                      command=self._open_sv_preview).pack(anchor="w", padx=28, pady=(2, 2),
                                                          before=self._sv_sub)
        ctk.CTkLabel(self._sv_sub,
                     text="Mini-stack des K premières images (réduites) avec le même pipeline : "
                          "brute | 1 image | K images, diagnostics (alignement, poids, rejet σ, "
                          "saturation) et estimation du temps et de la mémoire. Les réglages de "
                          "rendu se mettent à jour instantanément, ceux d'empilement relancent "
                          "le mini-stack.",
                     font=ctk.CTkFont(size=11), text_color="gray60",
                     wraplength=600).pack(anchor="w", pady=(0, 6))

        self._sv_duration = ctk.IntVar(value=30)
        self._slider_row(self._sv_sub, "Durée de la vidéo (s)",
                         self._sv_duration, 5, 120, 1, integer=True)

        prog_row = ctk.CTkFrame(self._sv_sub, fg_color="transparent")
        prog_row.pack(fill="x", pady=2)
        ctk.CTkLabel(prog_row, text="Progression :", width=238, anchor="w").pack(side="left")
        self._sv_progression_var = ctk.StringVar(value="Linéaire")
        ctk.CTkOptionMenu(prog_row, variable=self._sv_progression_var,
                          values=["Linéaire", "Logarithmique"],
                          width=180).pack(side="left", padx=8)
        ctk.CTkLabel(self._sv_sub,
                     text="Image 1 = brute, image 2 = 1 image traitée, image 3 = 2 images "
                          "empilées + traitement… jusqu'à N. Le nombre d'images vidéo vaut "
                          "durée × FPS : si N est plus petit, chaque étape est tenue plusieurs "
                          "images ; s'il est plus grand, des étapes intermédiaires sont sautées "
                          "(mais toutes les images sont bien empilées). Logarithmique = plus de "
                          "temps sur les premières images, là où l'image évolue le plus.",
                     font=ctk.CTkFont(size=11), text_color="gray60",
                     wraplength=600).pack(anchor="w", pady=(0, 6))

        width_row = ctk.CTkFrame(self._sv_sub, fg_color="transparent")
        width_row.pack(fill="x", pady=2)
        ctk.CTkLabel(width_row, text="Largeur max de la vidéo (px) :", width=238,
                     anchor="w").pack(side="left")
        self._sv_max_width_var = ctk.StringVar(value="1920")
        ctk.CTkOptionMenu(width_row, variable=self._sv_max_width_var,
                          values=["1280", "1920", "2560", "3840", "Native"],
                          width=180).pack(side="left", padx=8)
        ctk.CTkLabel(self._sv_sub,
                     text="L'empilement et l'image finale PNG restent en pleine résolution ; "
                          "seule la vidéo est réduite (rendu plus rapide, fichier lisible partout).",
                     font=ctk.CTkFont(size=11), text_color="gray60",
                     wraplength=600).pack(anchor="w", pady=(0, 6))

        stretch_row = ctk.CTkFrame(self._sv_sub, fg_color="transparent")
        stretch_row.pack(fill="x", pady=2)
        ctk.CTkLabel(stretch_row, text="Étirement :", width=238, anchor="w").pack(side="left")
        self._sv_stretch_var = ctk.StringVar(value="Automatique (Multicam)")
        ctk.CTkOptionMenu(stretch_row, variable=self._sv_stretch_var,
                          values=["Automatique (Multicam)", "GHS manuel"],
                          command=lambda *_: self._toggle_align_stack(),
                          width=220).pack(side="left", padx=8)

        # Nested in _sv_sub: its rows are one level too deep for
        # _set_frame_state(_sv_sub), so _toggle_align_stack handles it apart.
        self._sv_ghs_sub = ctk.CTkFrame(self._sv_sub, fg_color="transparent")
        self._sv_ghs_sub.pack(fill="x", padx=20)
        self._sv_ghs_SP = ctk.DoubleVar(value=0.0)
        self._sv_ghs_D  = ctk.DoubleVar(value=100.0)
        self._sv_ghs_b  = ctk.DoubleVar(value=0.0)
        self._sv_ghs_LP = ctk.DoubleVar(value=0.0)
        self._sv_ghs_HP = ctk.DoubleVar(value=1.0)
        self._slider_row(self._sv_ghs_sub, "SP — Symmetry Point",
                         self._sv_ghs_SP, 0.0, 0.05, 0.0001, decimals=4)
        self._slider_row(self._sv_ghs_sub, "D  — Stretch Factor",
                         self._sv_ghs_D, 0.0, 500.0, 1.0, integer=True)
        self._slider_row(self._sv_ghs_sub, "b  — Highlight Compression",
                         self._sv_ghs_b, 0.0, 10.0, 0.1)
        self._slider_row(self._sv_ghs_sub, "LP — Low Point",
                         self._sv_ghs_LP, 0.0, 0.05, 0.0001, decimals=4)
        self._slider_row(self._sv_ghs_sub, "HP — Highlight Point",
                         self._sv_ghs_HP, 0.50, 1.00, 0.01)
        ctk.CTkLabel(self._sv_ghs_sub,
                     text="Courbe GHS appliquée entre les points noir et blanc mesurés sur le "
                          "stack (balance des blancs automatique, couleurs préservées). Échelle "
                          "linéaire : le fond n'est qu'à 0,1–3 % du blanc (0,3 % sur un stack "
                          "M31 en FITS 12 bits), d'où SP au 1/10000 et D jusqu'à 500 — placer "
                          "SP sous le marqueur « fond » de l'histogramme de l'aperçu. SP = 0, b = 0 et D = β reproduisent exactement "
                          "l'étirement automatique (bouton « Partir de l'auto » de l'aperçu). "
                          "Courbe fixe : le fond garde la même luminosité pendant toute la "
                          "vidéo, seul le bruit diminue. Indépendant des réglages GHS de "
                          "l'onglet « Image & Couleur ».",
                     font=ctk.CTkFont(size=11), text_color="gray60",
                     wraplength=560).pack(anchor="w", pady=(0, 6))

        self._sv_target_auto = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(self._sv_sub, text="Fond de ciel automatique (10 % → 18 % quand le bruit baisse)",
                        variable=self._sv_target_auto).pack(anchor="w", pady=2)
        self._sv_target = ctk.DoubleVar(value=0.15)
        self._slider_row(self._sv_sub, "Niveau du fond (si manuel)",
                         self._sv_target, 0.03, 0.40, 0.01)

        self._sv_lce = ctk.DoubleVar(value=0.5)
        self._slider_row(self._sv_sub, "Contraste local (0 = désactivé)",
                         self._sv_lce, 0.0, 1.5, 0.05)
        self._sv_saturation = ctk.DoubleVar(value=1.3)
        self._slider_row(self._sv_sub, "Saturation",
                         self._sv_saturation, 0.0, 3.0, 0.05)
        self._sv_kappa = ctk.DoubleVar(value=3.0)
        self._slider_row(self._sv_sub, "Rejet σ (κ, 0 = désactivé)",
                         self._sv_kappa, 0.0, 6.0, 0.25)
        ctk.CTkLabel(self._sv_sub,
                     text="Rejet σ au fil de l'eau (actif après 10 images) : une valeur à plus "
                          "de κ·σ au-dessus de la moyenne (satellite, avion, rayon cosmique) "
                          "n'est pas empilée. Contraste local et saturation sont ceux de "
                          "l'aperçu Live Stack de Multicam ; les réglages de l'onglet "
                          "« Image & Couleur » ne s'appliquent pas à cette vidéo.",
                     font=ctk.CTkFont(size=11), text_color="gray60",
                     wraplength=600).pack(anchor="w", pady=(0, 6))

        self._sv_remove_green = ctk.BooleanVar(value=False)
        self._sv_linearize    = ctk.BooleanVar(value=True)
        self._sv_hot_pixels   = ctk.BooleanVar(value=True)
        self._sv_bg_norm      = ctk.BooleanVar(value=True)
        self._sv_weighting    = ctk.BooleanVar(value=True)
        self._sv_counter      = ctk.BooleanVar(value=True)
        for text, var in (
                ("Retrait du vert (SCNR neutre moyen lissé)", self._sv_remove_green),
                ("Linéariser les JPEG/PNG (sRGB → linéaire) avant empilement", self._sv_linearize),
                ("Corriger les pixels chauds", self._sv_hot_pixels),
                ("Normaliser le fond de ciel sur l'image de référence", self._sv_bg_norm),
                ("Pondérer les images par leur bruit (σ_réf / σ)²", self._sv_weighting),
                ("Afficher le compteur d'images empilées", self._sv_counter)):
            ctk.CTkCheckBox(self._sv_sub, text=text, variable=var).pack(anchor="w", pady=2)

        ctk.CTkLabel(tab_align,
                     text="ℹ Empilement Aligné : détecte les étoiles, aligne le ciel par "
                          "rotation/translation (RANSAC) sur l'image de référence, moyenne le "
                          "ciel aligné et le sol figé (moyenne brute), puis fusionne avec un "
                          "dégradé (feather) à la frontière du masque. Produit une image PNG "
                          "unique, ou une vidéo <nom>_stackvideo.mp4 (+ image finale PNG) "
                          "montrant le stack se construire — indépendant des autres onglets.",
                     font=ctk.CTkFont(size=11), text_color="gray60",
                     wraplength=860).pack(anchor="w", padx=6, pady=(8, 0))

        self._set_frame_state(self._align_sub, False)
        self._set_frame_state(self._sv_sub, False)
        self._set_frame_state(self._sv_ghs_sub, False)

        # ── Pinned footer: live preview + progress + run (always visible) ──
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.pack(fill="x", padx=14, pady=(4, 12))

        ctk.CTkButton(footer,
                      text="Aperçu en direct  (Original | Fond de ciel | Filtre Satellites)…",
                      height=34,
                      command=self._open_preview).pack(fill="x", pady=(0, 2))
        ctk.CTkLabel(footer,
                     text="Reflète en temps réel les réglages de l'onglet actif : correction/GHS, "
                          "Stacking/EMA/Renforcement des Traînées pour le fond de ciel, et le filtre "
                          "Hough pour l'isolation des satellites.",
                     font=ctk.CTkFont(size=11), text_color="gray60",
                     wraplength=860).pack(anchor="w", padx=4, pady=(0, 8))

        self._progress_bar = ctk.CTkProgressBar(footer)
        self._progress_bar.set(0.0)
        self._progress_bar.pack(fill="x", pady=(4, 2))
        self._lbl_progress = ctk.CTkLabel(footer, text="En attente…", anchor="w")
        self._lbl_progress.pack(anchor="w", pady=(0, 8))

        self._btn_run = ctk.CTkButton(
            footer,
            text="Générer la Vidéo",
            height=50,
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self._start_processing,
        )
        self._btn_run.pack(fill="x")

        # Trace all params that affect the preview (background + correction + GHS)
        for var in (self._contrast_var, self._brightness_var,
                    self._saturation_var,
                    self._rgb_r_var, self._rgb_g_var, self._rgb_b_var,
                    self._fits_bayer, self._fits_pattern_var,
                    self._use_ghs,
                    self._ghs_SP, self._ghs_D, self._ghs_b,
                    self._ghs_LP, self._ghs_HP,
                    self._ghs_lin, self._ghs_gamma,
                    self._use_stack, self._stack_n,
                    self._use_ema, self._alpha_var,
                    self._use_sat_boost, self._sat_boost_var, self._sat_thresh_var,
                    self._use_sat_enhanced,
                    self._sat_win_det, self._sat_thresh2,
                    self._sat_hough_thresh, self._sat_hough_min_len,
                    self._sat_hough_max_gap, self._sat_hough_tunnel,
                    self._sat_win_bg, self._sat_bg_src_var,
                    self._sat_beta, self._sat_color_var,
                    self._sat_stretch_SP, self._sat_stretch_D):
            var.trace_add("write", self._notify_preview)

        # Stack-video preview: rendering settings re-render the cached
        # mini-stack, stacking settings recompute it.
        for var in (self._sv_target_auto, self._sv_target, self._sv_lce,
                    self._sv_saturation, self._sv_remove_green, self._sv_counter,
                    self._sv_duration, self._sv_progression_var,
                    self._sv_max_width_var, self._fps_var, self._sv_stretch_var,
                    self._sv_ghs_SP, self._sv_ghs_D, self._sv_ghs_b,
                    self._sv_ghs_LP, self._sv_ghs_HP):
            var.trace_add("write", lambda *_: self._notify_sv_preview(stack=False))
        for var in (self._align_n, self._align_mask_radius, self._align_ground_cutoff,
                    self._align_k_sigma, self._sv_kappa, self._sv_linearize,
                    self._sv_hot_pixels, self._sv_bg_norm, self._sv_weighting,
                    self._fits_bayer, self._fits_pattern_var):
            var.trace_add("write", lambda *_: self._notify_sv_preview(stack=True))

    # ── Toggle callbacks ──────────────────────────────────────────────────────

    def _toggle_stack(self):
        self._set_frame_state(self._stack_sub, self._use_stack.get())

    def _toggle_ema(self):
        self._set_frame_state(self._ema_sub, self._use_ema.get())

    def _toggle_sat_boost(self):
        self._set_frame_state(self._sat_sub, self._use_sat_boost.get())

    def _toggle_ghs(self):
        self._set_frame_state(self._ghs_sub, self._use_ghs.get())

    def _toggle_ghs_gamma(self):
        self._set_frame_state(self._ghs_gamma_sub, self._ghs_lin.get())

    def _toggle_star(self):
        self._set_frame_state(self._star_sub, self._use_star.get())

    def _toggle_fits_bayer(self):
        self._set_frame_state(self._fits_bayer_sub, self._fits_bayer.get())

    def _toggle_sat_enhanced(self):
        self._set_frame_state(self._sat_enh_sub, self._use_sat_enhanced.get())

    def _toggle_sat_mask(self):
        self._set_frame_state(self._sat_mask_sub, self._sat_use_mask.get())

    def _toggle_align_stack(self):
        on = self._use_align_stack.get()
        self._set_frame_state(self._align_sub, on)
        self._set_frame_state(self._sv_sub, on and self._align_is_video())
        self._set_frame_state(self._sv_ghs_sub, on and self._align_is_video()
                              and self._sv_stretch_var.get() == "GHS manuel")

    def _align_is_video(self) -> bool:
        return "mp4" in self._align_out_var.get().lower()

    def _pick_align_ref(self):
        path = filedialog.askopenfilename(
            title="Image de référence pour l'empilement aligné",
            initialdir=self._src_dir or None,
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.fit *.fits *.JPG *.JPEG *.PNG *.FIT *.FITS"),
                ("JPEG",   "*.jpg *.jpeg"),
                ("PNG",    "*.png"),
                ("FITS",   "*.fit *.fits"),
            ])
        if not path:
            return
        self._align_ref_path = path
        self._lbl_align_ref.configure(text=os.path.basename(path))
        self._notify_sv_preview(stack=True)

    # ── Preview ───────────────────────────────────────────────────────────────

    def _open_preview(self):
        if self._preview_win is not None and self._preview_win.winfo_exists():
            self._preview_win.lift()
            self._preview_win.focus_force()
            return
        self._preview_win = PreviewWindow(self, self._get_preview_params)

    def _get_preview_params(self) -> dict:
        params = self._collect_params()
        params["src_dir"] = self._src_dir
        params["fps"] = self._fps_var.get()
        return params

    def _ghs_dict(self) -> dict:
        return dict(
            SP    = self._ghs_SP.get(),
            D     = self._ghs_D.get(),
            b     = self._ghs_b.get(),
            LP    = self._ghs_LP.get(),
            HP    = self._ghs_HP.get(),
            gamma = self._ghs_gamma.get() if self._ghs_lin.get() else 1.0,
        )

    def _open_sv_preview(self):
        win = self._sv_preview_win
        if win is not None and win.winfo_exists():
            win.lift()
            win.focus_force()
            return
        if not self._src_dir:
            messagebox.showwarning("Dossier manquant",
                                   "Veuillez d'abord sélectionner un répertoire source.")
            return
        # The preview is about the stack video: switch that mode on, so the
        # settings panel is editable and « Générer » produces what is previewed.
        if not (self._use_align_stack.get() and self._align_is_video()):
            self._use_align_stack.set(True)
            self._align_out_var.set("Vidéo de progression du stack (.mp4)")
            self._toggle_align_stack()
        try:
            self._sv_preview_win = StackVideoPreview(self, self._get_preview_params)
        except Exception as exc:
            messagebox.showerror("Aperçu impossible",
                                 f"{exc}\n\n{traceback.format_exc()}")

    def _sv_ghs_from_auto(self, beta: float):
        """GHS curve identical to the current auto stretch: SP = 0, b = 0, D = β."""
        self._sv_ghs_SP.set(0.0)
        self._sv_ghs_D.set(int(round(min(500.0, max(0.0, beta)))))
        self._sv_ghs_b.set(0.0)
        self._sv_ghs_LP.set(0.0)
        self._sv_ghs_HP.set(1.0)
        self._sv_stretch_var.set("GHS manuel")
        self._toggle_align_stack()

    def _notify_sv_preview(self, stack: bool):
        win = self._sv_preview_win
        if win is not None and win.winfo_exists():
            (win.refresh_stack if stack else win.refresh_render)()

    def _notify_preview(self, *_):
        if self._preview_win is not None and self._preview_win.winfo_exists():
            self._preview_win.refresh()

    # ── Source browser ────────────────────────────────────────────────────────

    def _browse_src(self):
        d = filedialog.askdirectory(title="Sélectionner le répertoire d'images AllSky")
        if not d:
            return
        self._src_dir = d
        exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG",
                "*.fit", "*.fits", "*.FIT", "*.FITS")
        paths: list[str] = []
        for e in exts:
            paths.extend(glob.glob(os.path.join(d, e)))
        n = len(paths)
        short = os.path.basename(d) or d
        self._lbl_dir.configure(
            text=f"{short}  •  {n} image{'s' if n != 1 else ''} trouvée{'s' if n != 1 else ''}")
        self._notify_sv_preview(stack=True)

    # ── Start processing ──────────────────────────────────────────────────────

    def _collect_params(self) -> dict:
        """
        All UI parameters except src_dir/out_path/fps — shared by the actual
        processing pipelines (_start_processing) and the live preview windows,
        so the preview always reflects exactly what a real run would produce.
        """
        use_sat_enhanced = self._use_sat_enhanced.get()
        star_mode = ("final"
                     if "png" in self._star_mode_var.get().lower()
                     else "progressive")

        bayer_code = None
        if self._fits_bayer.get():
            bayer_code = _BAYER_MAP.get(self._fits_pattern_var.get())

        return dict(
            contrast   = self._contrast_var.get(),
            brightness = self._brightness_var.get(),
            saturation = self._saturation_var.get(),
            rgb_r      = self._rgb_r_var.get(),
            rgb_g      = self._rgb_g_var.get(),
            rgb_b      = self._rgb_b_var.get(),
            use_ghs    = self._use_ghs.get(),
            ghs        = self._ghs_dict(),
            bayer_code = bayer_code,
            use_star   = self._use_star.get(),
            use_sat_enhanced = use_sat_enhanced,
            # Timelapse-only
            use_stack  = self._use_stack.get(),
            stack_n    = self._stack_n.get(),
            use_ema    = self._use_ema.get(),
            alpha_ema  = self._alpha_var.get(),
            sat_boost        = self._sat_boost_var.get() if self._use_sat_boost.get() else 1.0,
            sat_thresh       = float(self._sat_thresh_var.get()),
            # Star trail-only
            star_mode  = star_mode,
            star_decay = self._star_decay.get(),
            star_boost  = self._star_boost.get(),
            star_thresh = float(self._star_thresh.get()),
            # Satellite enhancement
            sat_win_det    = self._sat_win_det.get(),
            sat_thresh2    = float(self._sat_thresh2.get()),
            sat_hough_thresh  = self._sat_hough_thresh.get(),
            sat_hough_min_len = self._sat_hough_min_len.get(),
            sat_hough_max_gap = self._sat_hough_max_gap.get(),
            sat_hough_tunnel  = self._sat_hough_tunnel.get(),
            sat_decay      = self._sat_decay.get(),
            sat_win_bg     = self._sat_win_bg.get(),
            sat_bg_src     = ("mean"    if "EMA"      in self._sat_bg_src_var.get() else
                              "rolling" if "Moyenne"  in self._sat_bg_src_var.get() else
                              "current"),
            sat_beta       = self._sat_beta.get(),
            sat_color      = ("white" if self._sat_color_var.get() == "Blanc"
                              else "tinted"),
            sat_stretch_SP = self._sat_stretch_SP.get(),
            sat_stretch_D  = self._sat_stretch_D.get(),
            sat_use_mask   = self._sat_use_mask.get(),
            sat_mask_radius = self._sat_mask_radius.get(),
            # Stack & Align
            use_align_stack   = self._use_align_stack.get(),
            align_n           = self._align_n.get(),
            align_mask_radius = self._align_mask_radius.get(),
            align_ground_cutoff = self._align_ground_cutoff.get(),
            align_k_sigma     = self._align_k_sigma.get(),
            align_ref_path    = self._align_ref_path,
            align_video       = self._align_is_video(),
            # Stack video (Multicam Live Stack pipeline)
            sv_duration       = self._sv_duration.get(),
            sv_progression    = ("log" if self._sv_progression_var.get() == "Logarithmique"
                                 else "linear"),
            sv_target         = None if self._sv_target_auto.get() else self._sv_target.get(),
            sv_local_contrast = self._sv_lce.get(),
            sv_saturation     = self._sv_saturation.get(),
            sv_kappa          = self._sv_kappa.get(),
            sv_remove_green   = self._sv_remove_green.get(),
            sv_linearize      = self._sv_linearize.get(),
            sv_hot_pixels     = self._sv_hot_pixels.get(),
            sv_bg_norm        = self._sv_bg_norm.get(),
            sv_weighting      = self._sv_weighting.get(),
            sv_counter        = self._sv_counter.get(),
            sv_max_width      = (0 if self._sv_max_width_var.get() == "Native"
                                 else int(self._sv_max_width_var.get())),
            sv_stretch        = ("ghs" if self._sv_stretch_var.get() == "GHS manuel"
                                 else "auto"),
            sv_ghs            = dict(SP=self._sv_ghs_SP.get(), D=self._sv_ghs_D.get(),
                                     b=self._sv_ghs_b.get(), LP=self._sv_ghs_LP.get(),
                                     HP=self._sv_ghs_HP.get()),
        )

    def _start_processing(self):
        if self._processing:
            return

        if not self._src_dir:
            messagebox.showwarning("Dossier manquant",
                                   "Veuillez d'abord sélectionner un répertoire source.")
            return

        out_name = self._out_entry.get().strip() or "allsky_timelapse.mp4"
        if not out_name.lower().endswith(".mp4"):
            out_name += ".mp4"
        out_path = os.path.join(self._src_dir, out_name)

        params = self._collect_params()
        use_star         = params["use_star"]
        use_sat_enhanced = params["use_sat_enhanced"]
        use_align_stack  = params["use_align_stack"]
        star_mode        = params["star_mode"]
        params["src_dir"]  = self._src_dir
        params["out_path"] = out_path
        params["fps"]       = self._fps_var.get()

        self._processing = True
        if use_star:
            lbl = ("Calcul Star Trail (image finale)…"
                   if star_mode == "final"
                   else "Génération Star Trail (vidéo)…")
        elif use_sat_enhanced:
            lbl = "Renforcement traînées satellites…"
        elif use_align_stack and params["align_video"]:
            lbl = "Vidéo de progression du stack…"
        elif use_align_stack:
            lbl = "Empilement aligné (détection étoiles + RANSAC)…"
        else:
            lbl = "Traitement en cours…"
        self._btn_run.configure(state="disabled", text=lbl)
        self._progress_bar.set(0.0)
        self._lbl_progress.configure(text="Démarrage…")

        if use_star:
            target_fn = process_star_trail
        elif use_sat_enhanced:
            target_fn = process_satellites
        elif use_align_stack and params["align_video"]:
            target_fn = process_stack_video
        elif use_align_stack:
            target_fn = process_stack_align
        else:
            target_fn = process_video
        threading.Thread(
            target=target_fn,
            args=(params,
                  self._cb_progress,
                  self._cb_done,
                  self._cb_error),
            daemon=True,
        ).start()

    # ── Thread callbacks (always dispatched via .after() to the GUI thread) ───

    def _cb_progress(self, done: int, total: int):
        frac = done / total if total else 0.0
        pct  = int(frac * 100)
        self.after(0, lambda: self._progress_bar.set(frac))
        self.after(0, lambda: self._lbl_progress.configure(
            text=f"{pct}%  —  {done} / {total} frames traitées"))

    def _cb_done(self, out_path: str, note: str = ""):
        self._processing = False
        self.after(0, lambda: self._progress_bar.set(1.0))
        self.after(0, lambda: self._lbl_progress.configure(text="Terminé avec succès."))
        self.after(0, lambda: self._btn_run.configure(
            state="normal", text="Générer la Vidéo"))
        msg = f"Traitement terminé avec succès :\n\n{out_path}"
        if note:
            msg += f"\n\n{note}"
        self.after(0, lambda: messagebox.showinfo("Traitement terminé", msg))

    def _cb_error(self, msg: str):
        self._processing = False
        self.after(0, lambda: self._btn_run.configure(
            state="normal", text="Générer la Vidéo"))
        self.after(0, lambda: messagebox.showerror("Erreur de traitement", msg))


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Enable GPU acceleration via OpenCL (AMD RX 570 / any OpenCL device).
    # cv2.UMat operations in _detect_E_with_load will use the GPU automatically.
    if cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(True)
        dev = cv2.ocl.Device.getDefault()
        print(f"[GPU] OpenCL activé : {dev.name()} "
              f"({dev.maxComputeUnits()} unités, "
              f"{dev.globalMemSize() // 1024 // 1024} Mo)")
    else:
        print("[GPU] OpenCL non disponible — traitement CPU uniquement")

    app = AllSkyApp()
    app.mainloop()
