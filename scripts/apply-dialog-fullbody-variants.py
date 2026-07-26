#!/usr/bin/env python3

import argparse
import math
import re
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
_bundled_python = _repo_root / "build/python-packages"
if _bundled_python.is_dir():
    sys.path.insert(0, str(_bundled_python))

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


FACE_RE = re.compile(r"^(?P<char>.+)_face_(?P<face>.+)\.webp$")

# Keep fanservice changes to clearly mature/adult-coded characters. Childlike or
# ambiguous characters get expressive/action outfit variants instead.
MATURE_GLAMOUR_CHARS = {"reimu", "marisa", "youmu", "hina", "iku", "elly", "yumemi"}
PRIMARY_BASES = {"reimu", "marisa", "youmu", "hina", "iku", "elly", "yumemi", "cirno", "kurumi", "wriggle", "scuttle"}

FACE_ANCHORS = {
    "reimu": (0.50, 0.112, 0.185),
    "marisa": (0.50, 0.118, 0.185),
    "youmu": (0.50, 0.125, 0.180),
    "hina": (0.50, 0.128, 0.180),
    "iku": (0.49, 0.145, 0.175),
    "elly": (0.51, 0.125, 0.180),
    "yumemi": (0.50, 0.130, 0.170),
    "cirno": (0.50, 0.128, 0.170),
    "kurumi": (0.50, 0.130, 0.170),
    "wriggle": (0.50, 0.120, 0.170),
}


def expression_family(face):
    if face in {"happy", "smile", "relaxed", "calm"}:
        return "happy"
    if face in {"smug", "proud", "tsun", "tsun_blush", "chuuni"}:
        return "smug"
    if face in {"angry", "irritated", "outraged", "assertive", "serious", "shouting"}:
        return "angry"
    if face in {"surprised", "puzzled", "eeeeh", "unsettled", "inquisitive", "concerned"}:
        return "surprised"
    if face in {"eyes_closed", "sigh"}:
        return "closed"
    if face in {"embarrassed", "blush"} or "blush" in face:
        return "blush"
    if face in {"defeated", "sad", "beaten"}:
        return "sad"
    return "normal"


def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def alpha_bbox(alpha):
    ys, xs = np.where(alpha > 8)
    if len(xs) == 0:
        return 0, 0, alpha.shape[1], alpha.shape[0]
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def bbox_uv(alpha):
    h, w = alpha.shape
    x0, y0, x1, y1 = alpha_bbox(alpha)
    xs = np.arange(w, dtype=np.float32)[None, :]
    ys = np.arange(h, dtype=np.float32)[:, None]
    u = (xs - x0) / max(x1 - x0 - 1, 1)
    v = (ys - y0) / max(y1 - y0 - 1, 1)
    return np.clip(u, 0.0, 1.0), np.clip(v, 0.0, 1.0), (x0, y0, x1, y1)


def normalize_canvas(img, pad_x_ratio=0.20, pad_y_ratio=0.12, min_x=96, min_y=96):
    img = img.convert("RGBA")
    alpha = np.asarray(img.getchannel("A"))
    x0, y0, x1, y1 = alpha_bbox(alpha)
    content = img.crop((x0, y0, x1, y1))
    cw = max(x1 - x0, 1)
    ch = max(y1 - y0, 1)
    px = max(min_x, int(cw * pad_x_ratio))
    py = max(min_y, int(ch * pad_y_ratio))
    out = Image.new("RGBA", (cw + 2 * px, ch + 2 * py), (0, 0, 0, 0))
    out.alpha_composite(content, (px, py))
    return out


def find_alphamap_path(path):
    candidate = path.with_suffix(f".alphamap{path.suffix}")
    return candidate if candidate.exists() else None


def write_matching_alphamap(path, size):
    amap = find_alphamap_path(path)
    if amap is not None:
        Image.new("RGB", size, (255, 255, 255)).save(amap, "WEBP", lossless=True, quality=100, method=4)


def masked_blend(rgb, color, amount, mask):
    color = np.array(color, dtype=np.float32)
    return rgb * (1.0 - amount * mask) + color * (amount * mask)


def body_alpha_array(img):
    return np.asarray(img.getchannel("A")).astype(np.float32) / 255.0


def alpha_composite_on_body(base, layer):
    body = body_alpha_array(base)
    arr = np.asarray(layer.convert("RGBA")).astype(np.float32)
    arr[..., 3] *= body
    layer = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")
    out = base.copy()
    out.alpha_composite(layer)
    return out


def estimate_skin_color(img):
    arr = np.asarray(img.convert("RGBA")).astype(np.float32)
    rgb = arr[..., :3]
    alpha = arr[..., 3:4] / 255.0
    lum = rgb[..., 0:1] * 0.2126 + rgb[..., 1:2] * 0.7152 + rgb[..., 2:3] * 0.0722
    maxc = rgb.max(axis=2, keepdims=True)
    minc = rgb.min(axis=2, keepdims=True)
    chroma = maxc - minc
    mask = (
        (alpha > 0.05) &
        (lum > 95) & (lum < 245) &
        (rgb[..., 0:1] > rgb[..., 1:2] * 0.86) &
        (rgb[..., 1:2] > rgb[..., 2:3] * 0.58) &
        (chroma < 125)
    )[..., 0]
    if mask.sum() < 64:
        return (246, 202, 184)
    sample = rgb[mask]
    color = np.median(sample, axis=0)
    return tuple(int(np.clip(c, 120, 255)) for c in color)


def base_for_face(dialog_dir, charname, face):
    return dialog_dir / f"{charname}.webp"


def pose_params(family, face):
    params = {
        "happy": (-0.082, 0.108, -0.098, -0.050, -4.6),
        "smug": (0.096, -0.118, 0.116, 0.052, 5.0),
        "angry": (-0.092, 0.120, -0.116, -0.044, -4.2),
        "surprised": (0.102, -0.084, 0.122, 0.072, 5.6),
        "closed": (-0.050, -0.048, 0.062, -0.058, -3.0),
        "blush": (0.088, -0.106, 0.092, 0.058, 4.4),
        "sad": (-0.104, 0.064, -0.090, -0.058, -5.2),
    }.get(family, (0.0, 0.0, 0.0, 0.0, 0.0))
    if "tsun" in face:
        params = (params[0] * 1.15, params[1] * 1.2, params[2] * 1.1, params[3], params[4] + 1.0)
    return params


def pose_warp(img, family, face):
    img = normalize_canvas(img, min_x=140, min_y=150)
    lean, hip, shoulder, head, angle = pose_params(family, face)
    resampling = getattr(Image, "Resampling", Image).BICUBIC

    iw, ih = img.size
    arr = np.asarray(img).astype(np.float32)
    y = np.linspace(0.0, 1.0, ih, dtype=np.float32)[:, None]
    x = np.arange(iw, dtype=np.float32)[None, :]

    def band(center, width):
        return np.exp(-np.square((y - center) / width))

    offset = iw * (
        lean * (y - 0.5) * 0.92 +
        head * band(0.145, 0.090) +
        shoulder * band(0.340, 0.155) +
        hip * band(0.610, 0.185)
    )

    src_x = x - offset
    x0 = np.floor(src_x).astype(np.int32)
    x1 = x0 + 1
    mix = (src_x - x0)[..., None]
    valid = (x0 >= 0) & (x1 < iw)
    yy = np.arange(ih)[:, None]
    x0 = np.clip(x0, 0, iw - 1)
    x1 = np.clip(x1, 0, iw - 1)
    warped = arr[yy, x0] * (1.0 - mix) + arr[yy, x1] * mix
    warped[~valid] = 0
    transformed = Image.fromarray(np.clip(warped, 0, 255).astype(np.uint8), "RGBA")
    transformed = transformed.rotate(angle, resample=resampling, expand=False, fillcolor=(0, 0, 0, 0))
    return normalize_canvas(transformed, min_x=140, min_y=150)


def glamour_grade(img, charname, face):
    family = expression_family(face)
    rgba = img.convert("RGBA")

    color = 1.10
    contrast = 1.07
    brightness = 1.02
    if family == "sad":
        color = 0.88
        brightness = 0.94
    elif family in {"happy", "blush", "smug"}:
        color = 1.18
        brightness = 1.05
    elif family in {"angry", "surprised"}:
        contrast = 1.14

    rgba = ImageEnhance.Color(rgba).enhance(color)
    rgba = ImageEnhance.Contrast(rgba).enhance(contrast)
    rgba = ImageEnhance.Brightness(rgba).enhance(brightness)

    arr = np.asarray(rgba).astype(np.float32) / 255.0
    rgb = arr[..., :3]
    alpha = arr[..., 3:4]
    visible = alpha > 0.02
    lum = rgb[..., 0:1] * 0.2126 + rgb[..., 1:2] * 0.7152 + rgb[..., 2:3] * 0.0722
    maxc = rgb.max(axis=2, keepdims=True)
    minc = rgb.min(axis=2, keepdims=True)
    chroma = maxc - minc
    u, v, _ = bbox_uv(np.asarray(rgba.getchannel("A")))
    u = u[..., None]
    v = v[..., None]
    center = 1.0 - np.clip(np.abs(u - 0.5) * 2.0, 0.0, 1.0)

    skin = (
        visible &
        (lum > 0.34) & (lum < 0.96) &
        (rgb[..., 0:1] > rgb[..., 1:2] * 0.88) &
        (rgb[..., 1:2] > rgb[..., 2:3] * 0.62) &
        (chroma < 0.52)
    )
    skin_amt = skin.astype(np.float32)
    rgb = masked_blend(rgb, (1.0, 0.74, 0.69), 0.09, skin_amt)

    if charname in MATURE_GLAMOUR_CHARS:
        torso_v = smoothstep(0.20, 0.34, v) * (1.0 - smoothstep(0.56, 0.72, v))
        hip_v = smoothstep(0.48, 0.60, v) * (1.0 - smoothstep(0.84, 0.98, v))
        leg_v = smoothstep(0.62, 0.76, v)
        cloth = (visible & ~skin).astype(np.float32)
        blouse = cloth * torso_v * (0.74 + 0.26 * smoothstep(0.28, 0.88, lum))
        skirt = cloth * hip_v * (0.82 + 0.18 * (1.0 - smoothstep(0.34, 0.78, lum)))
        stocking = visible.astype(np.float32) * leg_v * (1.0 - smoothstep(0.60, 0.92, lum))
        sheer = blouse * smoothstep(0.42, 0.92, lum) * center

        rgb = masked_blend(rgb, (0.99, 0.98, 0.95), 0.58, blouse)
        rgb = masked_blend(rgb, (1.0, 0.72, 0.82), 0.24, sheer)
        rgb = masked_blend(rgb, (0.030, 0.027, 0.045), 0.48, skirt)
        rgb = masked_blend(rgb, (0.035, 0.030, 0.055), 0.34, stocking)
    else:
        cloth = visible.astype(np.float32) * (1.0 - smoothstep(0.70, 0.96, lum))
        rgb = masked_blend(rgb, (0.20, 0.28, 0.46), 0.18, cloth * smoothstep(0.22, 0.72, v))

    mood_color = {
        "happy": (1.0, 0.82, 0.48),
        "smug": (0.98, 0.70, 1.0),
        "angry": (1.0, 0.30, 0.30),
        "surprised": (0.60, 0.92, 1.0),
        "closed": (0.76, 0.82, 1.0),
        "blush": (1.0, 0.48, 0.66),
        "sad": (0.42, 0.58, 1.0),
    }.get(family, (0.90, 0.94, 1.0))
    aura = visible.astype(np.float32) * smoothstep(0.12, 0.48, v) * (1.0 - smoothstep(0.70, 0.96, v))
    rgb = masked_blend(rgb, mood_color, 0.055 if family == "normal" else 0.13, aura)

    arr[..., :3] = np.where(visible, np.clip(rgb, 0.0, 1.0), 0.0)
    return Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8), "RGBA")


def draw_body_line(draw, pts, fill, width):
    draw.line(pts, fill=fill, width=max(1, int(width)), joint="curve")


def draw_adult_glamour_outfit(img, charname, face):
    family = expression_family(face)
    alpha = np.asarray(img.getchannel("A"))
    x0, y0, x1, y1 = alpha_bbox(alpha)
    w = max(x1 - x0, 1)
    h = max(y1 - y0, 1)
    skin = estimate_skin_color(img)

    def X(u):
        return x0 + w * u

    def Y(v):
        return y0 + h * v

    accent = {
        "happy": (255, 206, 92),
        "smug": (222, 120, 255),
        "angry": (255, 72, 86),
        "surprised": (108, 228, 255),
        "closed": (180, 190, 255),
        "blush": (255, 118, 162),
        "sad": (92, 130, 255),
    }.get(family, (230, 230, 255))

    silhouette = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(silhouette, "RGBA")
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    lw = max(2, int(w * 0.012))

    # Emotion-specific outer layers deliberately change the read of the full
    # portrait, so variants do not collapse into a face swap at gameplay size.
    curious_surprise = face in {"puzzled", "inquisitive", "concerned", "unsettled"}
    open_surprise = family == "surprised" and not curious_surprise
    happy_raised = family == "happy" and face not in {"relaxed", "calm"}

    if family == "happy":
        sdraw.polygon([(X(0.16), Y(0.26)), (X(0.43), Y(0.18)), (X(0.36), Y(0.62)), (X(0.08), Y(0.74))], fill=(255, 230, 145, 110))
        sdraw.polygon([(X(0.58), Y(0.18)), (X(0.88), Y(0.24)), (X(0.84), Y(0.78)), (X(0.60), Y(0.60))], fill=(255, 230, 145, 104))
        if happy_raised:
            sdraw.line([(X(0.30), Y(0.28)), (X(0.14), Y(0.13)), (X(0.10), Y(0.06))], fill=(*skin, 128), width=max(lw * 2, 5))
            sdraw.ellipse((X(0.080), Y(0.035), X(0.135), Y(0.090)), fill=(*skin, 128))
    elif family == "smug":
        sdraw.polygon([(X(0.12), Y(0.25)), (X(0.39), Y(0.20)), (X(0.34), Y(0.98)), (X(0.05), Y(0.82))], fill=(28, 18, 48, 140))
        sdraw.polygon([(X(0.62), Y(0.20)), (X(0.92), Y(0.27)), (X(0.92), Y(0.83)), (X(0.66), Y(0.98))], fill=(28, 18, 48, 132))
        sdraw.line([(X(0.71), Y(0.34)), (X(0.87), Y(0.50)), (X(0.78), Y(0.58))], fill=(*skin, 178), width=max(lw * 3, 6))
    elif family == "angry":
        sdraw.polygon([(X(0.10), Y(0.21)), (X(0.90), Y(0.21)), (X(0.78), Y(0.56)), (X(0.22), Y(0.56))], fill=(150, 18, 36, 128))
        sdraw.polygon([(X(0.20), Y(0.28)), (X(0.00), Y(0.42)), (X(0.08), Y(0.66)), (X(0.28), Y(0.50))], fill=(150, 18, 36, 92))
    elif family == "surprised":
        if curious_surprise:
            sdraw.polygon([(X(0.10), Y(0.26)), (X(0.44), Y(0.18)), (X(0.42), Y(0.72)), (X(0.04), Y(0.62))], fill=(210, 202, 138, 88))
            sdraw.polygon([(X(0.56), Y(0.24)), (X(0.86), Y(0.33)), (X(0.72), Y(0.82)), (X(0.54), Y(0.58))], fill=(92, 92, 118, 72))
        else:
            sdraw.polygon([(X(0.02), Y(0.24)), (X(0.40), Y(0.16)), (X(0.34), Y(0.88)), (X(-0.04), Y(0.78))], fill=(140, 236, 255, 90))
            sdraw.polygon([(X(0.60), Y(0.16)), (X(0.98), Y(0.24)), (X(1.04), Y(0.78)), (X(0.66), Y(0.88))], fill=(140, 236, 255, 88))
            sdraw.line([(X(0.26), Y(0.32)), (X(0.08), Y(0.36)), (X(0.00), Y(0.48))], fill=(*skin, 120), width=max(lw * 2, 5))
            sdraw.line([(X(0.74), Y(0.32)), (X(0.94), Y(0.36)), (X(1.00), Y(0.48))], fill=(*skin, 120), width=max(lw * 2, 5))
    elif family == "blush":
        sdraw.polygon([(X(0.08), Y(0.25)), (X(0.92), Y(0.25)), (X(0.82), Y(0.90)), (X(0.20), Y(0.90))], fill=(255, 142, 186, 98))
        sdraw.line([(X(0.29), Y(0.30)), (X(0.14), Y(0.43)), (X(0.22), Y(0.55))], fill=(*skin, 160), width=max(lw * 2, 5))
    elif family in {"sad", "closed"}:
        sdraw.polygon([(X(0.08), Y(0.20)), (X(0.92), Y(0.20)), (X(0.74), Y(0.80)), (X(0.26), Y(0.80))], fill=(118, 142, 255, 92))
        sdraw.polygon([(X(0.20), Y(0.28)), (X(0.80), Y(0.28)), (X(0.66), Y(0.58)), (X(0.34), Y(0.58))], fill=(230, 238, 255, 72))

    # Clear, visible outfit differences: off-shoulder blouse, open collar,
    # high-slit mini skirt, stockings, straps. Non-explicit, but deliberately
    # fanservice-coded and more revealing than the base portraits.
    draw.polygon([(X(0.23), Y(0.238)), (X(0.77), Y(0.228)), (X(0.67), Y(0.54)), (X(0.33), Y(0.56))], fill=(255, 255, 255, 98))
    draw.polygon([(X(0.405), Y(0.232)), (X(0.595), Y(0.232)), (X(0.50), Y(0.445))], fill=(*skin, 214))
    draw.arc((X(0.34), Y(0.215), X(0.66), Y(0.365)), 14, 166, fill=(255, 255, 255, 205), width=lw)
    draw.polygon([(X(0.28), Y(0.500)), (X(0.73), Y(0.486)), (X(0.66), Y(0.675)), (X(0.36), Y(0.704))], fill=(16, 14, 25, 176))
    slit_side = 1 if family in {"smug", "blush", "surprised"} else -1
    if slit_side > 0:
        draw.polygon([(X(0.545), Y(0.515)), (X(0.660), Y(0.545)), (X(0.600), Y(0.825)), (X(0.505), Y(0.720))], fill=(*skin, 200))
        draw_body_line(draw, [(X(0.55), Y(0.52)), (X(0.60), Y(0.80))], (*accent, 205), lw)
    else:
        draw.polygon([(X(0.455), Y(0.515)), (X(0.340), Y(0.545)), (X(0.400), Y(0.818)), (X(0.495), Y(0.720))], fill=(*skin, 196))
        draw_body_line(draw, [(X(0.45), Y(0.52)), (X(0.40), Y(0.79))], (*accent, 205), lw)

    draw.polygon([(X(0.31), Y(0.66)), (X(0.46), Y(0.66)), (X(0.43), Y(0.97)), (X(0.25), Y(0.97))], fill=(18, 15, 26, 146))
    draw.polygon([(X(0.54), Y(0.66)), (X(0.70), Y(0.66)), (X(0.77), Y(0.97)), (X(0.58), Y(0.97))], fill=(18, 15, 26, 146))
    draw_body_line(draw, [(X(0.31), Y(0.66)), (X(0.46), Y(0.66))], (255, 225, 238, 170), lw)
    draw_body_line(draw, [(X(0.54), Y(0.66)), (X(0.72), Y(0.66))], (255, 225, 238, 170), lw)
    draw_body_line(draw, [(X(0.42), Y(0.57)), (X(0.35), Y(0.68))], (24, 18, 32, 220), lw)
    draw_body_line(draw, [(X(0.58), Y(0.57)), (X(0.66), Y(0.68))], (24, 18, 32, 220), lw)
    draw_body_line(draw, [(X(0.43), Y(0.22)), (X(0.57), Y(0.22))], (24, 18, 34, 215), lw + 1)

    if family == "smug":
        draw.polygon([(X(0.37), Y(0.33)), (X(0.63), Y(0.33)), (X(0.60), Y(0.54)), (X(0.40), Y(0.54))], fill=(28, 18, 44, 108))
        for i in range(4):
            draw_body_line(draw, [(X(0.43), Y(0.36 + i * 0.04)), (X(0.57), Y(0.38 + i * 0.04))], (*accent, 180), lw)
        if face == "chuuni":
            draw.polygon([(X(0.32), Y(0.25)), (X(0.68), Y(0.25)), (X(0.63), Y(0.62)), (X(0.37), Y(0.62))], fill=(18, 14, 36, 128))
            draw_body_line(draw, [(X(0.35), Y(0.31)), (X(0.65), Y(0.58))], (178, 132, 255, 210), lw + 1)
            draw_body_line(draw, [(X(0.65), Y(0.31)), (X(0.35), Y(0.58))], (178, 132, 255, 210), lw + 1)
    elif family == "angry":
        draw_body_line(draw, [(X(0.23), Y(0.49)), (X(0.76), Y(0.50))], (*accent, 230), lw + 3)
        draw.rectangle((X(0.46), Y(0.49), X(0.55), Y(0.535)), fill=(255, 220, 120, 210))
        draw.polygon([(X(0.24), Y(0.245)), (X(0.48), Y(0.230)), (X(0.42), Y(0.42)), (X(0.20), Y(0.44))], fill=(190, 28, 44, 126))
        draw.polygon([(X(0.52), Y(0.230)), (X(0.78), Y(0.245)), (X(0.80), Y(0.44)), (X(0.58), Y(0.42))], fill=(190, 28, 44, 126))
    elif family == "blush":
        draw.polygon([(X(0.30), Y(0.25)), (X(0.70), Y(0.25)), (X(0.63), Y(0.50)), (X(0.37), Y(0.51))], fill=(255, 150, 190, 42))
        draw_body_line(draw, [(X(0.30), Y(0.245)), (X(0.70), Y(0.245))], (255, 218, 232, 210), lw + 2)
    elif family == "sad":
        for i in range(3):
            draw_body_line(draw, [(X(0.30 + i * 0.13), Y(0.27)), (X(0.40 + i * 0.10), Y(0.58))], (160, 180, 255, 80), lw)
    elif family == "surprised":
        if curious_surprise:
            draw.polygon([(X(0.28), Y(0.25)), (X(0.58), Y(0.24)), (X(0.52), Y(0.57)), (X(0.26), Y(0.58))], fill=(255, 250, 230, 66))
            draw.polygon([(X(0.55), Y(0.50)), (X(0.76), Y(0.53)), (X(0.70), Y(0.69)), (X(0.54), Y(0.65))], fill=(34, 28, 50, 112))
        else:
            draw.polygon([(X(0.27), Y(0.24)), (X(0.73), Y(0.24)), (X(0.68), Y(0.58)), (X(0.32), Y(0.58))], fill=(255, 255, 255, 34))

    # Gloss strokes make the sheer outfit read clearly in the small in-game view.
    for i in range(5):
        x = X(0.30 + i * 0.09)
        draw_body_line(draw, [(x, Y(0.28)), (x + w * 0.075, Y(0.55))], (255, 255, 255, 44), max(1, w * 0.007))

    out = silhouette.copy()
    out.alpha_composite(img)
    return alpha_composite_on_body(out, layer)


def draw_action_outfit(img, charname, face):
    family = expression_family(face)
    alpha = np.asarray(img.getchannel("A"))
    x0, y0, x1, y1 = alpha_bbox(alpha)
    w = max(x1 - x0, 1)
    h = max(y1 - y0, 1)

    def X(u):
        return x0 + w * u

    def Y(v):
        return y0 + h * v

    accent = {
        "happy": (255, 214, 82),
        "smug": (205, 132, 255),
        "angry": (255, 82, 92),
        "surprised": (98, 228, 255),
        "closed": (180, 195, 255),
        "sad": (90, 130, 255),
    }.get(family, (210, 230, 255))

    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    lw = max(2, int(w * 0.014))
    draw.polygon([(X(0.18), Y(0.28)), (X(0.82), Y(0.28)), (X(0.72), Y(0.62)), (X(0.28), Y(0.62))], fill=(*accent, 66))
    draw.arc((X(0.30), Y(0.20), X(0.70), Y(0.43)), 10, 170, fill=(*accent, 180), width=lw)
    draw_body_line(draw, [(X(0.32), Y(0.42)), (X(0.68), Y(0.42))], (255, 255, 255, 160), lw)
    for i in range(4):
        px = X(0.34 + i * 0.10)
        py = Y(0.55 + 0.03 * (i % 2))
        draw.line((px - lw, py, px + lw, py), fill=(*accent, 200), width=lw)
        draw.line((px, py - lw, px, py + lw), fill=(*accent, 200), width=lw)
    return alpha_composite_on_body(img, layer)


def draw_expression(img, charname, face, face_path=None):
    family = expression_family(face)
    out = img.convert("RGBA")
    alpha = np.asarray(out.getchannel("A"))
    x0, y0, x1, y1 = alpha_bbox(alpha)
    w = max(x1 - x0, 1)
    h = max(y1 - y0, 1)
    ax, ay, scale = FACE_ANCHORS.get(charname, (0.50, 0.125, 0.175))
    cx = x0 + w * ax
    cy = y0 + h * ay
    fw = w * scale
    fh = h * scale * 0.62
    skin = estimate_skin_color(out)

    layer = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    lw = max(2, int(w * 0.012))
    eye_y = cy - fh * 0.06
    mouth_y = cy + fh * 0.28
    eye_dx = fw * 0.25
    eye_w = fw * 0.22

    # Repaint the face area first. The base portraits already include a neutral
    # face, so every emotion variant must replace that region instead of drawing
    # loose marks on top of it.
    draw.ellipse((cx - fw * 0.62, cy - fh * 0.50, cx + fw * 0.62, cy + fh * 0.58), fill=(*skin, 118))
    draw.ellipse((cx - fw * 0.48, cy - fh * 0.34, cx + fw * 0.48, cy + fh * 0.42), fill=(255, 244, 235, 24))
    ink = (44, 30, 40, 230)
    out.alpha_composite(layer)
    layer = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")

    placed_reference = False
    if face_path is not None and Path(face_path).exists():
        with Image.open(face_path) as face_img:
            ref = face_img.convert("RGBA")
        ref_alpha = np.asarray(ref.getchannel("A"))
        rx0, ry0, rx1, ry1 = alpha_bbox(ref_alpha)
        if rx1 > rx0 and ry1 > ry0:
            ref = ref.crop((rx0, ry0, rx1, ry1))
            resampling = getattr(Image, "Resampling", Image).LANCZOS
            ref_aspect = ref.width / max(ref.height, 1)
            target_w = int(round(fw * {
                "angry": 2.45,
                "surprised": 2.65,
                "smug": 2.18,
                "happy": 2.05,
                "blush": 2.12,
                "closed": 1.92,
                "sad": 2.08,
            }.get(family, 2.05)))
            target_h = int(round(target_w / max(ref_aspect, 0.1)))
            max_h = int(round(fh * (2.10 if family in {"angry", "surprised"} else 1.62)))
            if target_h > max_h:
                target_h = max_h
                target_w = int(round(target_h * ref_aspect))
            ref = ref.resize((max(1, target_w), max(1, target_h)), resampling)
            px = int(round(cx - ref.width * 0.50))
            py = int(round(cy - ref.height * {
                "angry": 0.58,
                "surprised": 0.60,
                "happy": 0.54,
                "closed": 0.52,
                "sad": 0.52,
            }.get(family, 0.55)))
            out.alpha_composite(ref, (px, py))
            placed_reference = True

    if placed_reference:
        if family == "blush":
            draw.ellipse((cx - fw * 0.60, cy + fh * 0.02, cx - fw * 0.24, cy + fh * 0.26), fill=(255, 82, 132, 118))
            draw.ellipse((cx + fw * 0.24, cy + fh * 0.02, cx + fw * 0.60, cy + fh * 0.26), fill=(255, 82, 132, 118))
        elif family == "sad":
            draw.ellipse((cx + eye_dx + eye_w * 0.28, eye_y + fh * 0.10, cx + eye_dx + eye_w * 0.64, eye_y + fh * 0.38), fill=(106, 176, 255, 150))
        out.alpha_composite(layer)
        return out

    if family == "happy":
        draw.arc((cx - eye_dx - eye_w, eye_y - fh * 0.10, cx - eye_dx + eye_w, eye_y + fh * 0.18), 10, 170, fill=ink, width=lw)
        draw.arc((cx + eye_dx - eye_w, eye_y - fh * 0.10, cx + eye_dx + eye_w, eye_y + fh * 0.18), 10, 170, fill=ink, width=lw)
        draw.arc((cx - fw * 0.18, mouth_y - fh * 0.08, cx + fw * 0.18, mouth_y + fh * 0.18), 8, 172, fill=(130, 36, 50, 220), width=lw)
    elif family == "smug":
        draw.line((cx - eye_dx - eye_w, eye_y - fh * 0.05, cx - eye_dx + eye_w, eye_y + fh * 0.02), fill=ink, width=lw)
        draw.arc((cx + eye_dx - eye_w, eye_y - fh * 0.08, cx + eye_dx + eye_w, eye_y + fh * 0.15), 190, 350, fill=ink, width=lw)
        draw.arc((cx - fw * 0.06, mouth_y - fh * 0.06, cx + fw * 0.24, mouth_y + fh * 0.12), 190, 345, fill=(120, 32, 48, 230), width=lw)
    elif family == "angry":
        draw.line((cx - eye_dx - eye_w, eye_y - fh * 0.18, cx - eye_dx + eye_w, eye_y + fh * 0.02), fill=(80, 22, 28, 240), width=lw + 1)
        draw.line((cx + eye_dx - eye_w, eye_y + fh * 0.02, cx + eye_dx + eye_w, eye_y - fh * 0.18), fill=(80, 22, 28, 240), width=lw + 1)
        draw.line((cx - fw * 0.14, mouth_y + fh * 0.08, cx + fw * 0.15, mouth_y - fh * 0.04), fill=(120, 24, 34, 235), width=lw)
    elif family == "surprised":
        draw.ellipse((cx - eye_dx - eye_w * 0.42, eye_y - fh * 0.11, cx - eye_dx + eye_w * 0.42, eye_y + fh * 0.11), fill=(42, 32, 48, 190))
        draw.ellipse((cx + eye_dx - eye_w * 0.42, eye_y - fh * 0.11, cx + eye_dx + eye_w * 0.42, eye_y + fh * 0.11), fill=(42, 32, 48, 190))
        draw.ellipse((cx - eye_dx - eye_w * 0.14, eye_y - fh * 0.08, cx - eye_dx + eye_w * 0.06, eye_y + fh * 0.01), fill=(255, 255, 255, 145))
        draw.ellipse((cx + eye_dx - eye_w * 0.14, eye_y - fh * 0.08, cx + eye_dx + eye_w * 0.06, eye_y + fh * 0.01), fill=(255, 255, 255, 145))
        draw.ellipse((cx - fw * 0.060, mouth_y - fh * 0.030, cx + fw * 0.060, mouth_y + fh * 0.115), fill=(94, 28, 42, 205))
    elif family == "closed":
        draw.arc((cx - eye_dx - eye_w, eye_y - fh * 0.08, cx - eye_dx + eye_w, eye_y + fh * 0.13), 190, 350, fill=ink, width=lw)
        draw.arc((cx + eye_dx - eye_w, eye_y - fh * 0.08, cx + eye_dx + eye_w, eye_y + fh * 0.13), 190, 350, fill=ink, width=lw)
        draw.line((cx - fw * 0.10, mouth_y, cx + fw * 0.10, mouth_y), fill=(116, 38, 48, 220), width=lw)
    elif family == "blush":
        draw.arc((cx - eye_dx - eye_w, eye_y - fh * 0.06, cx - eye_dx + eye_w, eye_y + fh * 0.14), 190, 350, fill=ink, width=lw)
        draw.arc((cx + eye_dx - eye_w, eye_y - fh * 0.06, cx + eye_dx + eye_w, eye_y + fh * 0.14), 190, 350, fill=ink, width=lw)
        draw.ellipse((cx - fw * 0.52, cy + fh * 0.04, cx - fw * 0.22, cy + fh * 0.24), fill=(255, 82, 132, 110))
        draw.ellipse((cx + fw * 0.22, cy + fh * 0.04, cx + fw * 0.52, cy + fh * 0.24), fill=(255, 82, 132, 110))
        draw.ellipse((cx - fw * 0.045, mouth_y - fh * 0.02, cx + fw * 0.045, mouth_y + fh * 0.08), fill=(118, 36, 50, 220))
    elif family == "sad":
        draw.line((cx - eye_dx - eye_w, eye_y - fh * 0.02, cx - eye_dx + eye_w, eye_y + fh * 0.08), fill=ink, width=lw)
        draw.line((cx + eye_dx - eye_w, eye_y + fh * 0.08, cx + eye_dx + eye_w, eye_y - fh * 0.02), fill=ink, width=lw)
        draw.arc((cx - fw * 0.16, mouth_y, cx + fw * 0.16, mouth_y + fh * 0.20), 200, 340, fill=(92, 36, 52, 220), width=lw)
        draw.ellipse((cx + eye_dx + eye_w * 0.30, eye_y + fh * 0.08, cx + eye_dx + eye_w * 0.65, eye_y + fh * 0.35), fill=(106, 176, 255, 170))

    out.alpha_composite(layer)
    return out


def draw_mood_effects(img, charname, face):
    family = expression_family(face)
    out = img.convert("RGBA")
    alpha = np.asarray(out.getchannel("A"))
    x0, y0, x1, y1 = alpha_bbox(alpha)
    w = max(x1 - x0, 1)
    h = max(y1 - y0, 1)
    draw = ImageDraw.Draw(out, "RGBA")
    head_cx = x0 + w * 0.50
    head_cy = y0 + h * 0.18
    face_w = w * 0.34
    face_h = h * 0.16

    if family == "angry":
        for i in range(5):
            y = y0 + h * (0.31 + i * 0.040)
            draw.line((x0 + w * 0.74, y, x1 - w * 0.03, y - h * 0.030), fill=(255, 54, 72, 58), width=max(2, int(w * 0.012)))
            draw.line((x0 + w * 0.04, y + h * 0.035, x0 + w * 0.22, y), fill=(255, 54, 72, 34), width=max(2, int(w * 0.010)))
    elif family == "surprised":
        for i in range(8):
            a = i / 8.0 * math.tau
            sx = head_cx + math.cos(a) * face_w * 0.82
            sy = head_cy + math.sin(a) * face_h * 0.90
            r = max(3, int(w * 0.010))
            draw.line((sx - r, sy, sx + r, sy), fill=(160, 232, 255, 150), width=2)
            draw.line((sx, sy - r, sx, sy + r), fill=(160, 232, 255, 150), width=2)
    elif family == "smug":
        draw.arc((head_cx - face_w * 0.75, head_cy - face_h * 0.65, head_cx + face_w * 0.75, head_cy + face_h * 0.82), 205, 330, fill=(255, 214, 255, 96), width=max(2, int(w * 0.012)))
    elif family in {"sad", "closed"}:
        for i in range(4):
            x = x0 + w * (0.25 + i * 0.16)
            draw.line((x, y0 + h * 0.08, x - w * 0.04, y0 + h * 0.34), fill=(90, 128, 255, 42), width=max(2, int(w * 0.010)))

    return out.filter(ImageFilter.UnsharpMask(radius=0.85, percent=62, threshold=3))


def make_variant(dialog_dir, face_path):
    match = FACE_RE.match(face_path.name)
    if not match:
        return None, None
    charname = match.group("char")
    face = match.group("face")
    if face == "normal":
        return None, None

    base_path = base_for_face(dialog_dir, charname, face)
    if not base_path.exists():
        raise RuntimeError(f"missing base portrait for {face_path}: {base_path}")

    with Image.open(base_path) as base:
        out = normalize_canvas(base.convert("RGBA"), min_x=140, min_y=150)

    family = expression_family(face)
    out = pose_warp(out, family, face)
    out = glamour_grade(out, charname, face)
    if charname in MATURE_GLAMOUR_CHARS:
        out = draw_adult_glamour_outfit(out, charname, face)
    else:
        out = draw_action_outfit(out, charname, face)
    out = draw_expression(out, charname, face)
    out = draw_mood_effects(out, charname, face)
    out = normalize_canvas(out, min_x=140, min_y=150)
    return dialog_dir / f"{charname}_variant_{face}.webp", out


def normalize_base_portraits(dialog_dir, write):
    normalized = []
    for name in sorted(PRIMARY_BASES):
        path = dialog_dir / f"{name}.webp"
        if not path.exists():
            continue
        with Image.open(path) as img:
            out = normalize_canvas(img, min_x=140, min_y=150)
        normalized.append((path, out))
        if write:
            out.save(path, "WEBP", lossless=True, quality=100, method=4)
            write_matching_alphamap(path, out.size)
    return normalized


def normalize_leftover_variants(dialog_dir, generated_paths, write):
    generated_paths = {p.resolve() for p in generated_paths}
    normalized = []
    for path in sorted(dialog_dir.glob("*_variant_*.webp")):
        if path.resolve() in generated_paths or ".alphamap" in path.name:
            continue
        with Image.open(path) as img:
            out = normalize_canvas(img, min_x=140, min_y=150)
        normalized.append((path, out))
        if write:
            out.save(path, "WEBP", lossless=True, quality=100, method=4)
            write_matching_alphamap(path, out.size)
    return normalized


def main():
    parser = argparse.ArgumentParser(description="Generate full-body dialog emotion variants with pose/outfit/expression changes.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    dialog_dir = repo_root / "atlas/portraits/dialog"
    normalized = normalize_base_portraits(dialog_dir, args.write)
    for path, image in normalized:
        print(f"{path.relative_to(repo_root).as_posix()} normalized {image.size}")

    face_paths = sorted(p for p in dialog_dir.glob("*_face_*.webp") if FACE_RE.match(p.name))
    generated = []

    for face_path in face_paths:
        out_path, image = make_variant(dialog_dir, face_path)
        if out_path is None:
            continue
        generated.append(out_path)
        if args.write:
            image.save(out_path, "WEBP", lossless=True, quality=100, method=4)
            write_matching_alphamap(out_path, image.size)
        print(f"{out_path.relative_to(repo_root).as_posix()} {image.size}")

    leftover = normalize_leftover_variants(dialog_dir, generated, args.write)
    for path, image in leftover:
        print(f"{path.relative_to(repo_root).as_posix()} normalized {image.size}")

    print(f"normalized base portraits: {len(normalized)}")
    print(f"full-body emotion variants: {len(generated)}")
    print(f"normalized leftover variants: {len(leftover)}")


if __name__ == "__main__":
    main()
