#!/usr/bin/env python3

import argparse
import io
import math
import re
import subprocess
import sys
from pathlib import Path
from statistics import median

_repo_root = Path(__file__).resolve().parents[1]
_bundled_python = _repo_root / "build/python-packages"
if _bundled_python.is_dir():
    sys.path.insert(0, str(_bundled_python))

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


FACE_RE = re.compile(r"^(?P<char>.+)_face_(?P<face>.+)\.webp$")
DEFAULT_SCALE = 0.5


IRIS_COLORS = {
    "reimu": (104, 56, 82, 255),
    "marisa": (120, 76, 54, 255),
    "youmu": (96, 132, 154, 255),
    "cirno": (78, 142, 232, 255),
    "hina": (78, 138, 88, 255),
    "wriggle": (82, 146, 78, 255),
    "kurumi": (142, 68, 96, 255),
    "iku": (140, 92, 178, 255),
    "elly": (150, 78, 132, 255),
    "yumemi": (106, 106, 188, 255),
    "scuttle": (100, 176, 192, 255),
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
    if face in {"embarrassed", "blush"}:
        return "blush"
    if face in {"defeated", "sad"}:
        return "sad"
    return "normal"


def visible_bbox(img):
    bbox = img.getbbox()
    if bbox is None:
        raise RuntimeError("empty portrait image")
    return bbox


def detect_face_box(img):
    arr = np.asarray(img.convert("RGBA"))
    alpha = arr[..., 3] > 32
    ys, xs = np.where(alpha)
    if len(xs) == 0:
        raise RuntimeError("empty portrait alpha")

    vx0, vy0, vx1, vy1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    vh = vy1 - vy0
    upper = np.zeros_like(alpha)
    upper[vy0:int(vy0 + vh * 0.34), :] = True

    rgb = arr[..., :3].astype(np.float32)
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    maxc = rgb.max(axis=2)
    minc = rgb.min(axis=2)
    skin = (
        alpha
        & upper
        & (r > 118)
        & (g > 76)
        & (b > 55)
        & (r > g * 0.88)
        & (g > b * 0.66)
        & ((maxc - minc) < 118)
    )

    sy, sx = np.where(skin)
    if len(sx) > 60:
        top_limit = np.percentile(sy, 42)
        top_skin = sy <= top_limit
        if top_skin.sum() > 30:
            sx = sx[top_skin]
            sy = sy[top_skin]

        x0 = int(np.percentile(sx, 8))
        x1 = int(np.percentile(sx, 92)) + 1
        y0 = int(np.percentile(sy, 4))
        y1 = int(np.percentile(sy, 96)) + 1
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        fw = max(x1 - x0, int((vx1 - vx0) * 0.10), 58)
        fh = max(y1 - y0, int(vh * 0.060), 50)
    else:
        cx = (vx0 + vx1) // 2
        cy = int(vy0 + vh * 0.18)
        fw = max(int((vx1 - vx0) * 0.18), 82)
        fh = max(int(vh * 0.11), 70)

    fw = min(fw, int((vx1 - vx0) * 0.26))
    fh = min(fh, int(vh * 0.16))
    return cx, cy, fw, fh


def median_skin_color(img, box):
    arr = np.asarray(img.convert("RGBA"))
    x0, y0, x1, y1 = box
    crop = arr[y0:y1, x0:x1, :]
    alpha = crop[..., 3] > 32
    rgb = crop[..., :3]
    if alpha.sum() < 10:
        return (238, 196, 184, 230)
    pixels = rgb[alpha]
    return tuple(int(median([int(v) for v in pixels[:, i]])) for i in range(3)) + (238,)


def draw_arc_line(draw, box, start, end, fill, width):
    draw.arc(box, start=start, end=end, fill=fill, width=width)


def draw_eye(draw, cx, cy, w, h, iris, mood, side):
    ink = (42, 32, 48, 235)
    white = (255, 248, 246, 232)
    if mood == "closed":
        draw_arc_line(draw, (cx - w, cy - h * 0.35, cx + w, cy + h * 0.70), 10, 170, ink, max(2, int(h * 0.17)))
        return

    if mood == "happy":
        draw_arc_line(draw, (cx - w, cy - h * 0.15, cx + w, cy + h * 0.95), 8, 172, ink, max(2, int(h * 0.16)))
        return

    if mood == "smug":
        box = (cx - w, cy - h * 0.35, cx + w, cy + h * 0.55)
        draw.ellipse(box, fill=white, outline=ink, width=max(2, int(h * 0.12)))
        draw.rectangle((cx - w - 2, cy - h * 0.52, cx + w + 2, cy - h * 0.08), fill=(0, 0, 0, 0))
    else:
        scale = 1.15 if mood == "surprised" else 1.0
        box = (cx - w * scale, cy - h * scale, cx + w * scale, cy + h * scale)
        draw.ellipse(box, fill=white, outline=ink, width=max(2, int(h * 0.13)))

    iris_r = max(3, int(min(w, h) * (0.62 if mood == "surprised" else 0.48)))
    draw.ellipse((cx - iris_r, cy - iris_r, cx + iris_r, cy + iris_r), fill=iris, outline=ink, width=max(1, iris_r // 4))
    pupil_r = max(2, iris_r // 2)
    draw.ellipse((cx - pupil_r, cy - pupil_r, cx + pupil_r, cy + pupil_r), fill=(28, 24, 36, 240))
    draw.ellipse((cx - iris_r * 0.28, cy - iris_r * 0.45, cx + iris_r * 0.12, cy - iris_r * 0.05), fill=(255, 255, 255, 230))

    lash_y = cy - h * (1.05 if mood == "angry" else 1.0)
    tilt = -1 if side < 0 else 1
    if mood == "angry":
        draw.line((cx - w, lash_y - tilt * h * 0.30, cx + w, lash_y + tilt * h * 0.16), fill=ink, width=max(2, int(h * 0.18)))
    else:
        draw.line((cx - w, lash_y, cx + w, lash_y - h * 0.10), fill=ink, width=max(2, int(h * 0.14)))


def draw_expression(patch, charname, face, face_box_on_patch):
    family = expression_family(face)
    img = patch.convert("RGBA")

    if family == "sad":
        img = ImageEnhance.Color(img).enhance(0.76)
        img = ImageEnhance.Brightness(img).enhance(0.88)
    elif family == "angry":
        img = ImageEnhance.Contrast(img).enhance(1.08)
    elif family in {"happy", "blush"}:
        img = ImageEnhance.Brightness(img).enhance(1.04)

    draw = ImageDraw.Draw(img, "RGBA")
    x0, y0, x1, y1 = face_box_on_patch
    fw = max(x1 - x0, 1)
    fh = max(y1 - y0, 1)
    skin = median_skin_color(img, (max(0, x0), max(0, y0), min(img.width, x1), min(img.height, y1)))

    iris = IRIS_COLORS.get(charname, (112, 92, 174, 255))
    eye_w = max(6, int(fw * 0.080))
    eye_h = max(5, int(fh * 0.066))
    eye_y = y0 + fh * (0.47 if family != "surprised" else 0.45)
    left_x = x0 + fw * 0.36
    right_x = x0 + fw * 0.64

    # Small translucent patches make the expression legible without covering
    # the portrait with a flat mask.
    mute = (skin[0], skin[1], skin[2], 92)
    for ex in (left_x, right_x):
        draw.ellipse((ex - eye_w * 1.35, eye_y - eye_h * 1.25, ex + eye_w * 1.35, eye_y + eye_h * 1.25), fill=mute)
    mouth_x = x0 + fw * 0.50
    mouth_y = y0 + fh * 0.72
    mouth_w = max(8, int(fw * 0.075))
    mouth_h = max(5, int(fh * 0.052))
    draw.ellipse((mouth_x - mouth_w * 1.25, mouth_y - mouth_h * 1.1, mouth_x + mouth_w * 1.25, mouth_y + mouth_h * 1.1), fill=mute)

    eye_mood = family
    if family in {"normal", "blush", "sad", "angry", "surprised", "smug", "closed", "happy"}:
        pass
    else:
        eye_mood = "normal"

    if family == "sad":
        eye_mood = "closed" if face == "defeated" else "normal"
    if family == "blush":
        eye_mood = "normal"

    draw_eye(draw, left_x, eye_y, eye_w, eye_h, iris, eye_mood, -1)
    draw_eye(draw, right_x, eye_y, eye_w, eye_h, iris, eye_mood, 1)

    ink = (42, 32, 48, 235)
    brow_w = max(2, int(fh * 0.025))
    brow_y = y0 + fh * 0.32
    if family == "angry":
        draw.line((left_x - eye_w, brow_y + 7, left_x + eye_w, brow_y - 7), fill=ink, width=brow_w)
        draw.line((right_x - eye_w, brow_y - 7, right_x + eye_w, brow_y + 7), fill=ink, width=brow_w)
    elif family == "surprised":
        draw.arc((left_x - eye_w, brow_y - 8, left_x + eye_w, brow_y + 8), 205, 335, fill=ink, width=brow_w)
        draw.arc((right_x - eye_w, brow_y - 8, right_x + eye_w, brow_y + 8), 205, 335, fill=ink, width=brow_w)
    elif family == "sad":
        draw.line((left_x - eye_w, brow_y - 4, left_x + eye_w, brow_y + 4), fill=ink, width=brow_w)
        draw.line((right_x - eye_w, brow_y + 4, right_x + eye_w, brow_y - 4), fill=ink, width=brow_w)

    if family in {"happy", "smug", "blush"}:
        draw_arc_line(draw, (mouth_x - mouth_w, mouth_y - mouth_h, mouth_x + mouth_w, mouth_y + mouth_h), 15, 165, ink, max(2, mouth_h // 3))
    elif family == "angry" and face == "shouting":
        draw.ellipse((mouth_x - mouth_w, mouth_y - mouth_h, mouth_x + mouth_w, mouth_y + mouth_h * 1.4), fill=(92, 38, 54, 230), outline=ink, width=max(2, mouth_h // 3))
    elif family == "surprised":
        draw.ellipse((mouth_x - mouth_w * 0.55, mouth_y - mouth_h, mouth_x + mouth_w * 0.55, mouth_y + mouth_h), fill=(82, 42, 58, 225), outline=ink, width=max(2, mouth_h // 3))
    elif family in {"sad", "closed"}:
        draw_arc_line(draw, (mouth_x - mouth_w, mouth_y, mouth_x + mouth_w, mouth_y + mouth_h * 1.8), 200, 340, ink, max(2, mouth_h // 3))
    else:
        draw.line((mouth_x - mouth_w * 0.55, mouth_y, mouth_x + mouth_w * 0.55, mouth_y + 1), fill=ink, width=max(2, mouth_h // 3))

    if family in {"blush", "happy", "embarrassed"} or "blush" in face:
        blush = (255, 92, 126, 72)
        by = y0 + fh * 0.62
        draw.ellipse((x0 + fw * 0.15, by - fh * 0.05, x0 + fw * 0.35, by + fh * 0.05), fill=blush)
        draw.ellipse((x0 + fw * 0.65, by - fh * 0.05, x0 + fw * 0.85, by + fh * 0.05), fill=blush)

    if family == "sad":
        draw.rectangle((x0, y0, x1, y0 + fh * 0.22), fill=(74, 92, 150, 38))

    return img.filter(ImageFilter.UnsharpMask(radius=0.8, percent=72, threshold=2))


def base_for_face(dialog_dir, charname, face):
    if face == "defeated":
        candidate = dialog_dir / f"{charname}_variant_defeated.webp"
        if candidate.exists():
            return candidate
    if charname == "elly" and face in {"defeated", "beaten"}:
        candidate = dialog_dir / "elly_variant_beaten.webp"
        if candidate.exists():
            return candidate
    return dialog_dir / f"{charname}.webp"


def git_head_bytes(repo_root, relpath):
    try:
        return subprocess.check_output(
            ["git", "show", f"HEAD:{relpath.as_posix()}"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def polish_original_overlay(img, face):
    family = expression_family(face)
    alpha = img.getchannel("A")
    out = img.convert("RGBA")

    saturation = 1.10
    contrast = 1.06
    brightness = 1.02
    if family == "sad":
        saturation = 0.92
        brightness = 0.97
    elif family in {"happy", "blush"}:
        saturation = 1.14
        brightness = 1.04
    elif family in {"angry", "smug", "surprised"}:
        contrast = 1.09

    out = ImageEnhance.Color(out).enhance(saturation)
    out = ImageEnhance.Contrast(out).enhance(contrast)
    out = ImageEnhance.Brightness(out).enhance(brightness)
    out.putalpha(alpha)
    return out.filter(ImageFilter.UnsharpMask(radius=0.75, percent=55, threshold=3))


def make_face_patch(repo_root, dialog_dir, face_path):
    match = FACE_RE.match(face_path.name)
    if not match:
        return None

    charname = match.group("char")
    face = match.group("face")
    image_rel = face_path.relative_to(repo_root)
    config_rel = Path("atlas/config/dialog") / f"{face_path.stem}.spr"

    head_image = git_head_bytes(repo_root, image_rel)
    head_config = git_head_bytes(repo_root, config_rel)
    if head_image and head_config:
        original = Image.open(io.BytesIO(head_image)).convert("RGBA")
        return polish_original_overlay(original, face), head_config.decode("utf-8")

    base_path = base_for_face(dialog_dir, charname, face)
    if not base_path.exists():
        raise RuntimeError(f"missing base portrait for {face_path}: {base_path}")

    base = Image.open(base_path).convert("RGBA")
    old_w, old_h = Image.open(face_path).size
    cx, cy, fw, fh = detect_face_box(base)

    patch_w = int(max(old_w, fw * 2.45, base.width * 0.22))
    patch_h = int(max(old_h, fh * 2.15, base.height * 0.14))
    patch_w = min(patch_w, int(base.width * 0.62))
    patch_h = min(patch_h, int(base.height * 0.34))

    family = expression_family(face)
    x_shift = {
        "angry": -0.04,
        "smug": 0.04,
        "surprised": 0.02,
        "sad": -0.02,
    }.get(family, 0.0)
    y_shift = {
        "happy": -0.02,
        "sad": 0.03,
        "closed": 0.02,
    }.get(family, 0.0)

    left = int(round(cx - patch_w * (0.50 + x_shift)))
    top = int(round(cy - patch_h * (0.42 + y_shift)))
    left = max(0, min(base.width - patch_w, left))
    top = max(0, min(base.height - patch_h, top))
    right = left + patch_w
    bottom = top + patch_h

    patch = base.crop((left, top, right, bottom))
    face_box = (
        int(cx - fw * 0.58 - left),
        int(cy - fh * 0.62 - top),
        int(cx + fw * 0.58 - left),
        int(cy + fh * 0.62 - top),
    )
    patch = draw_expression(patch, charname, face, face_box)

    config = {
        "padding_top": top * DEFAULT_SCALE,
        "padding_bottom": (base.height - bottom) * DEFAULT_SCALE,
        "padding_left": left * DEFAULT_SCALE,
        "padding_right": (base.width - right) * DEFAULT_SCALE,
    }
    return patch, config


def fmt(value):
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.6g}"


def write_config(path, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(config, str):
        path.write_text(config, encoding="utf-8")
    else:
        lines = [f"{key} = {fmt(value)}\n" for key, value in config.items() if abs(value) > 1e-6]
        path.write_text("".join(lines), encoding="utf-8")


def save_image(path, img):
    img.save(path, "WEBP", lossless=True, quality=100, method=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    dialog_dir = repo_root / "atlas/portraits/dialog"
    config_dir = repo_root / "atlas/config/dialog"
    face_paths = sorted(p for p in dialog_dir.glob("*_face_*.webp") if FACE_RE.match(p.name))

    print(f"dialog face variants: {len(face_paths)}")
    for idx, face_path in enumerate(face_paths, 1):
        patch, config = make_face_patch(repo_root, dialog_dir, face_path)
        if args.write:
            save_image(face_path, patch)
            write_config(config_dir / f"{face_path.stem}.spr", config)
        print(f"[{idx:03d}/{len(face_paths):03d}] {face_path.relative_to(repo_root).as_posix()} {patch.size}")

    print("dialog expression polish OK")


if __name__ == "__main__":
    main()
