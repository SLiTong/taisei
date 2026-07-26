#!/usr/bin/env python3

import argparse
import math
import re
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
_bundled_python = _repo_root / "build" / "python-packages"
if _bundled_python.is_dir():
    sys.path.insert(0, str(_bundled_python))

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


FRAME_RE = re.compile(r"\.frame\d{4}\.webp$")
FRAME_COUNT = 8
MAX_FRAME_HEIGHT = 1536
SEQUENCE = "0 1 2 3 4 5 6 7 6 5 4 3 2 1"

MOOD_COLORS = {
    "angry": (255, 74, 92),
    "assertive": (255, 94, 92),
    "irritated": (255, 74, 92),
    "outraged": (255, 64, 76),
    "shouting": (255, 70, 70),
    "serious": (120, 190, 255),
    "happy": (255, 210, 96),
    "smile": (255, 210, 96),
    "smug": (220, 132, 255),
    "proud": (220, 132, 255),
    "tsun": (235, 132, 255),
    "tsun_blush": (255, 150, 190),
    "blush": (255, 126, 170),
    "surprised": (130, 226, 255),
    "puzzled": (130, 226, 255),
    "concerned": (130, 226, 255),
    "inquisitive": (130, 226, 255),
    "unsettled": (130, 226, 255),
    "eyes_closed": (180, 196, 255),
    "sigh": (160, 180, 255),
    "sad": (90, 132, 255),
    "defeated": (92, 126, 255),
    "beaten": (92, 126, 255),
}


def is_fullbody_portrait(path):
    name = path.name
    return (
        path.suffix.lower() == ".webp"
        and ".alphamap." not in name
        and "_face_" not in name
        and "_misc_" not in name
        and not FRAME_RE.search(name)
    )


def alpha_bbox(img):
    alpha = np.asarray(img.getchannel("A"))
    ys, xs = np.where(alpha > 8)
    if len(xs) == 0:
        return 0, 0, img.width, img.height
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def portrait_mood(stem):
    if "_variant_" not in stem:
        return "normal"
    return stem.rsplit("_variant_", 1)[1]


def mood_color(stem):
    mood = portrait_mood(stem)
    return MOOD_COLORS.get(mood, (190, 220, 255))


def make_affine_frame(img, phase, bbox):
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.54
    w = max(x1 - x0, 1)
    h = max(y1 - y0, 1)

    breath = math.sin(phase)
    sway = math.sin(phase + 0.55)
    scale_x = 1.0 + 0.0065 * breath
    scale_y = 1.0 + 0.0110 * breath
    shift_x = 0.0080 * w * sway
    shift_y = -0.0050 * h * breath

    # PIL affine coefficients map output coordinates to source coordinates.
    a = 1.0 / scale_x
    e = 1.0 / scale_y
    c = cx - (cx + shift_x) / scale_x
    f = cy - (cy + shift_y) / scale_y
    return img.transform(
        img.size,
        Image.Transform.AFFINE,
        (a, 0, c, 0, e, f),
        resample=Image.Resampling.BICUBIC,
        fillcolor=(0, 0, 0, 0),
    )


def row_sway(img, phase, bbox):
    arr = np.asarray(img.convert("RGBA")).copy()
    h, w = arr.shape[:2]
    x0, y0, x1, y1 = bbox
    content_h = max(y1 - y0, 1)
    out = np.zeros_like(arr)

    for y in range(h):
        v = np.clip((y - y0) / content_h, 0.0, 1.0)
        hair_band = math.exp(-((v - 0.18) / 0.18) ** 2)
        hem_band = math.exp(-((v - 0.78) / 0.24) ** 2)
        body_band = 0.25 + 0.75 * v
        dx = (
            math.sin(phase + v * 2.4) * 2.8 * body_band
            + math.sin(phase * 1.7 + v * 8.0) * 3.8 * hair_band
            + math.sin(phase * 1.2 + v * 5.2) * 3.0 * hem_band
        )
        shift = int(round(dx))
        if shift > 0:
            out[y, shift:] = arr[y, : w - shift]
        elif shift < 0:
            out[y, : w + shift] = arr[y, -shift:]
        else:
            out[y] = arr[y]

    return Image.fromarray(out, "RGBA")


def add_video_effects(img, stem, phase, frame_index, bbox):
    x0, y0, x1, y1 = bbox
    w = max(x1 - x0, 1)
    h = max(y1 - y0, 1)
    color = mood_color(stem)

    alpha = np.asarray(img.getchannel("A")).astype(np.float32) / 255.0
    glow = img.filter(ImageFilter.GaussianBlur(radius=max(4, int(w * 0.018))))
    glow_arr = np.asarray(glow).astype(np.float32)
    glow_arr[..., 0] = color[0]
    glow_arr[..., 1] = color[1]
    glow_arr[..., 2] = color[2]
    glow_arr[..., 3] *= 0.12 + 0.05 * (0.5 + 0.5 * math.sin(phase))
    glow = Image.fromarray(np.clip(glow_arr, 0, 255).astype(np.uint8), "RGBA")

    out = Image.new("RGBA", img.size, (0, 0, 0, 0))
    out.alpha_composite(glow)
    out.alpha_composite(img)

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    rng_seed = sum(ord(c) for c in stem)

    for i in range(18):
        t = (frame_index / FRAME_COUNT + i * 0.173 + rng_seed * 0.0017) % 1.0
        px = x0 + w * ((i * 0.371 + rng_seed * 0.013) % 1.0)
        py = y0 + h * ((0.10 + t * 0.85) % 1.0)
        drift = math.sin(phase + i * 1.71) * w * 0.025
        r = max(1, int(w * (0.0025 + 0.0025 * ((i % 3) + 1))))
        a = int(42 + 42 * (1.0 - abs(t - 0.5) * 2.0))
        draw.line((px + drift - r * 2, py, px + drift + r * 2, py), fill=(*color, a), width=max(1, r))
        draw.line((px + drift, py - r * 2, px + drift, py + r * 2), fill=(255, 255, 255, max(14, a // 2)), width=1)

    arr = np.asarray(overlay).astype(np.float32)
    arr[..., 3] *= np.clip(alpha * 2.5, 0.0, 1.0)
    overlay = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")
    out.alpha_composite(overlay)

    shimmer = 1.0 + 0.020 * math.sin(phase + 0.4)
    out = ImageEnhance.Brightness(out).enhance(shimmer)
    out = ImageEnhance.Contrast(out).enhance(1.010)
    return out


def make_frame(source, stem, frame_index):
    phase = math.tau * frame_index / FRAME_COUNT
    bbox = alpha_bbox(source)
    frame = make_affine_frame(source, phase, bbox)
    frame = row_sway(frame, phase, bbox)
    frame = add_video_effects(frame, stem, phase, frame_index, bbox)
    return frame


def normalize_source_resolution(img):
    if img.height <= MAX_FRAME_HEIGHT:
        return img

    scale = MAX_FRAME_HEIGHT / img.height
    size = (max(1, round(img.width * scale)), MAX_FRAME_HEIGHT)
    return img.resize(size, Image.Resampling.LANCZOS)


def write_framegroup_config(config_dir, stem):
    path = config_dir / f"{stem}.framegroup.spr"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("autotrim = no\n", encoding="utf-8", newline="\n")


def write_animation(resources_dialog_dir, stem):
    path = resources_dialog_dir / f"{stem}.ani"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"@sprite_count = {FRAME_COUNT}\n\nmain = {SEQUENCE}\n", encoding="utf-8", newline="\n")


def generate_for_path(path, dialog_dir, config_dir, resources_dialog_dir, write):
    stem = path.stem
    with Image.open(path) as img:
        source = normalize_source_resolution(img.convert("RGBA"))

    outputs = []
    for i in range(FRAME_COUNT):
        frame = make_frame(source, stem, i)
        out = dialog_dir / f"{stem}.frame{i:04d}.webp"
        outputs.append(out)
        if write:
            frame.save(out, "WEBP", quality=84, method=4)

    if write:
        write_framegroup_config(config_dir, stem)
        write_animation(resources_dialog_dir, stem)

    return outputs


def main():
    parser = argparse.ArgumentParser(description="Generate engine-native animated dialog portrait clips.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    dialog_dir = repo / "atlas" / "portraits" / "dialog"
    config_dir = repo / "atlas" / "config" / "dialog"
    resources_dialog_dir = repo / "resources" / "00-taisei.pkgdir" / "gfx" / "dialog"

    sources = sorted(p for p in dialog_dir.glob("*.webp") if is_fullbody_portrait(p))
    if args.limit:
        sources = sources[: args.limit]

    total_frames = 0
    for source in sources:
        outputs = generate_for_path(source, dialog_dir, config_dir, resources_dialog_dir, args.write)
        total_frames += len(outputs)
        mode = "wrote" if args.write else "would write"
        print(f"{mode} {source.stem}: {len(outputs)} frames")

    print(f"portrait clips: {len(sources)}")
    print(f"frames: {total_frames}")
    print(f"frame_count_per_clip: {FRAME_COUNT}")


if __name__ == "__main__":
    main()
