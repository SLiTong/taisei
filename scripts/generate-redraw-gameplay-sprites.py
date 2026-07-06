#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter


FRAME_RE = re.compile(r"^(?P<name>.+)\.frame(?P<frame>\d+)\.png$")


@dataclass(frozen=True)
class SourceSpec:
    source: str
    crop_rel: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    height_scale: float = 1.02
    width_scale: float = 0.94
    y_shift: int = 0
    sharpen: float = 1.0
    alpha: float = 1.0
    tint: tuple[int, int, int] | None = None
    guard_start: float | None = None
    guard_alpha: float = 0.95
    leg_style: str | None = None


SOURCES: dict[str, SourceSpec] = {
    "reimu": SourceSpec(
        "atlas/portraits/dialog/reimu_redraw_normal.webp",
        height_scale=1.03,
        leg_style="reimu_player",
    ),
    "marisa": SourceSpec(
        "atlas/portraits/dialog/marisa_redraw_normal.webp",
        height_scale=1.03,
        leg_style="marisa_player",
    ),
    "youmu": SourceSpec(
        "atlas/portraits/dialog/youmu_redraw_normal.webp",
        height_scale=0.98,
        width_scale=0.86,
        y_shift=-4,
        guard_start=0.34,
        leg_style="youmu_player",
    ),
    "cirno": SourceSpec("atlas/portraits/dialog/cirno_redraw_normal.webp", height_scale=1.03),
    "hina": SourceSpec(
        "atlas/portraits/dialog/hina_redraw_normal.webp",
        crop_rel=(0.24, 0.0, 0.78, 1.0),
        height_scale=1.04,
        width_scale=0.98,
        guard_start=0.44,
        leg_style="hina_boss",
    ),
    "wriggle": SourceSpec("atlas/portraits/dialog/wriggle_redraw_normal.webp", height_scale=1.03),
    "wriggleex": SourceSpec(
        "atlas/portraits/dialog/wriggle_redraw_outraged.webp",
        height_scale=1.04,
        tint=(255, 210, 250),
    ),
    "kurumi": SourceSpec(
        "atlas/portraits/dialog/kurumi_redraw_normal.webp",
        height_scale=1.03,
        y_shift=-4,
        guard_start=0.40,
        leg_style="kurumi_boss",
    ),
    "iku": SourceSpec(
        "atlas/portraits/dialog/iku_redraw_normal.webp",
        crop_rel=(0.16, 0.0, 0.84, 1.0),
        height_scale=1.04,
    ),
    "iku_mid": SourceSpec(
        "atlas/portraits/dialog/iku_redraw_normal.webp",
        crop_rel=(0.12, 0.0, 0.88, 1.0),
        height_scale=1.05,
    ),
    "elly": SourceSpec(
        "atlas/portraits/dialog/elly_redraw_normal.webp",
        crop_rel=(0.24, 0.0, 0.82, 1.0),
        height_scale=1.04,
        width_scale=0.98,
        guard_start=0.36,
        leg_style="elly_boss",
    ),
    "scuttle": SourceSpec("atlas/portraits/dialog/scuttle_redraw_normal.webp", height_scale=1.04),
}


LEG_STYLES = {
    "reimu_player": {
        "upper": (244, 220, 203),
        "lower": (58, 36, 34),
        "shoe": (84, 34, 34),
        "hip": 0.62,
        "knee": 0.78,
        "ankle": 0.96,
        "spread": 0.090,
        "width": 0.060,
    },
    "marisa_player": {
        "upper": (245, 224, 208),
        "lower": (55, 45, 43),
        "shoe": (92, 62, 42),
        "hip": 0.62,
        "knee": 0.78,
        "ankle": 0.96,
        "spread": 0.090,
        "width": 0.060,
    },
    "youmu_player": {
        "upper": (236, 224, 207),
        "lower": (35, 82, 74),
        "shoe": (35, 175, 151),
        "hip": 0.58,
        "knee": 0.76,
        "ankle": 0.97,
        "spread": 0.095,
        "width": 0.060,
    },
    "hina_boss": {
        "upper": (245, 214, 204),
        "lower": (36, 19, 31),
        "shoe": (96, 20, 50),
        "hip": 0.66,
        "knee": 0.82,
        "ankle": 0.98,
        "spread": 0.078,
        "width": 0.050,
    },
    "kurumi_boss": {
        "upper": (245, 220, 205),
        "lower": (44, 33, 46),
        "shoe": (232, 218, 196),
        "hip": 0.61,
        "knee": 0.78,
        "ankle": 0.97,
        "spread": 0.080,
        "width": 0.050,
    },
    "elly_boss": {
        "upper": (245, 220, 204),
        "lower": (78, 28, 36),
        "shoe": (48, 18, 24),
        "hip": 0.69,
        "knee": 0.83,
        "ankle": 0.97,
        "spread": 0.075,
        "width": 0.048,
    },
}


def alpha_bbox(image: Image.Image) -> tuple[int, int, int, int] | None:
    return image.getchannel("A").getbbox()


def crop_source(path: Path, spec: SourceSpec) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    bbox = alpha_bbox(image)
    if bbox is None:
        raise ValueError(f"{path} has no visible pixels")

    content = image.crop(bbox)
    left, top, right, bottom = spec.crop_rel
    w, h = content.size
    crop = (
        max(0, min(w - 1, round(w * left))),
        max(0, min(h - 1, round(h * top))),
        max(1, min(w, round(w * right))),
        max(1, min(h, round(h * bottom))),
    )
    content = content.crop(crop)

    bbox = alpha_bbox(content)
    if bbox is None:
        raise ValueError(f"{path} crop has no visible pixels")

    return content.crop(bbox)


def add_outline(sprite: Image.Image, color: tuple[int, int, int, int] = (20, 22, 28, 180)) -> Image.Image:
    alpha = sprite.getchannel("A")
    outline_alpha = alpha.filter(ImageFilter.MaxFilter(3))
    outline_alpha = ImageChops.subtract(outline_alpha, alpha)
    outline = Image.new("RGBA", sprite.size, color)
    outline.putalpha(outline_alpha)

    out = Image.new("RGBA", sprite.size, (0, 0, 0, 0))
    out.alpha_composite(outline)
    out.alpha_composite(sprite)
    return out


def tint_sprite(sprite: Image.Image, tint: tuple[int, int, int] | None) -> Image.Image:
    if tint is None:
        return sprite

    overlay = Image.new("RGBA", sprite.size, (*tint, 42))
    out = Image.alpha_composite(sprite, overlay)
    out.putalpha(sprite.getchannel("A"))
    return out


def apply_fullbody_guard(generated: Image.Image, original: Image.Image, spec: SourceSpec) -> Image.Image:
    if spec.guard_start is None:
        return generated

    bbox = alpha_bbox(original)
    if bbox is None:
        return generated

    left, top, right, bottom = bbox
    guard_y = round(top + (bottom - top) * spec.guard_start)

    guard = original.copy()
    guard = ImageEnhance.Color(guard).enhance(1.16)
    guard = ImageEnhance.Contrast(guard).enhance(1.12)
    guard = ImageEnhance.Sharpness(guard).enhance(1.45)

    alpha = guard.getchannel("A")
    mask = Image.new("L", guard.size, 0)
    mask.paste(255, (0, guard_y, guard.width, guard.height))
    alpha = ImageChops.multiply(alpha, mask)
    alpha = alpha.point(lambda a: round(a * spec.guard_alpha))
    guard.putalpha(alpha)
    guard = add_outline(guard, (18, 18, 22, 150))

    if spec.y_shift:
        shifted = Image.new("RGBA", guard.size, (0, 0, 0, 0))
        shifted.alpha_composite(guard, (0, spec.y_shift))
        guard = shifted

    out = generated.copy()
    out.alpha_composite(guard)
    return out


def mute_youmu_ghost(sprite: Image.Image) -> Image.Image:
    bbox = alpha_bbox(sprite)
    if bbox is None:
        return sprite

    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0
    center_x = (x0 + x1) / 2
    start_y = y0 + h * 0.28
    side_band = max(10, w * 0.18)

    out = sprite.copy()
    pixels = out.load()
    for y in range(max(0, round(start_y)), y1):
        for x in range(x0, x1):
            r, g, b, a = pixels[x, y]
            if a == 0:
                continue

            is_pale = min(r, g, b) > 175 and max(r, g, b) - min(r, g, b) < 55
            is_side = abs(x - center_x) > side_band
            if is_pale and is_side:
                pixels[x, y] = (r, g, b, round(a * 0.45))

    return out


def draw_leg_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    color: tuple[int, int, int],
    width: int,
    alpha: int,
) -> None:
    outline_width = width + 6
    draw.line((start, end), fill=(12, 10, 15, 245), width=outline_width)
    draw.line((start, end), fill=(*color, min(255, alpha + 12)), width=width)


def apply_readable_legs(sprite: Image.Image, spec: SourceSpec, frame_index: int) -> Image.Image:
    if spec.leg_style is None:
        return sprite

    style = LEG_STYLES[spec.leg_style]
    if spec.leg_style == "youmu_player":
        sprite = mute_youmu_ghost(sprite)

    bbox = alpha_bbox(sprite)
    if bbox is None:
        return sprite

    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0
    center_x = (x0 + x1) / 2

    sway = math.sin(frame_index * 0.9 + 0.4)
    step = math.sin(frame_index * 0.83)
    hip_y = y0 + h * style["hip"]
    knee_y = y0 + h * style["knee"]
    ankle_y = min(sprite.height - 3, y0 + h * style["ankle"])
    spread = max(4.0, w * style["spread"])
    leg_width = max(3, round(w * style["width"]))
    foot_w = max(8, round(leg_width * 2.8))
    foot_h = max(4, round(leg_width * 1.3))

    layer = Image.new("RGBA", sprite.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")

    for side in (-1, 1):
        side_f = float(side)
        hip = (
            center_x + side_f * spread * 0.55 + sway * w * 0.010,
            hip_y,
        )
        knee = (
            center_x + side_f * spread * (0.75 + 0.08 * step),
            knee_y,
        )
        ankle = (
            center_x + side_f * spread * (0.95 - 0.06 * step),
            ankle_y,
        )

        draw_leg_line(draw, hip, knee, style["upper"], leg_width, 230)
        draw_leg_line(draw, knee, ankle, style["lower"], leg_width, 240)

        fx = ankle[0] + side_f * foot_w * 0.18
        fy = ankle[1] + foot_h * 0.20
        draw.ellipse(
            (
                fx - foot_w / 2,
                fy - foot_h / 2,
                fx + foot_w / 2,
                fy + foot_h / 2,
            ),
            fill=(12, 10, 14, 220),
        )
        draw.ellipse(
            (
                fx - foot_w / 2 + 1,
                fy - foot_h / 2 + 1,
                fx + foot_w / 2 - 1,
                fy + foot_h / 2 - 1,
            ),
            fill=(*style["shoe"], 245),
        )

    out = sprite.copy()
    out.alpha_composite(layer)
    return add_outline(out, (12, 12, 16, 120))


def transform_sprite(source: Image.Image, spec: SourceSpec, canvas_size: tuple[int, int],
                     bbox: tuple[int, int, int, int], frame_index: int) -> Image.Image:
    canvas_w, canvas_h = canvas_size
    left, top, right, bottom = bbox
    bbox_w = right - left
    bbox_h = bottom - top

    sway = math.sin((frame_index / 2.0) + 0.35)
    breathe = 1.0 + 0.018 * math.sin(frame_index * 1.11)
    max_w = canvas_w * spec.width_scale
    max_h = min(canvas_h - 4, bbox_h * spec.height_scale) * breathe

    scale = min(max_w / source.width, max_h / source.height)
    new_size = (
        max(1, round(source.width * scale)),
        max(1, round(source.height * scale)),
    )

    sprite = source.resize(new_size, Image.Resampling.LANCZOS)
    sprite = ImageEnhance.Sharpness(sprite).enhance(spec.sharpen)
    sprite = add_outline(tint_sprite(sprite, spec.tint))

    if spec.alpha != 1.0:
        alpha = sprite.getchannel("A").point(lambda a: round(a * spec.alpha))
        sprite.putalpha(alpha)

    target_center_x = (left + right) / 2 + sway * max(1.0, bbox_w * 0.035)
    target_bottom = bottom + spec.y_shift
    x = round(target_center_x - sprite.width / 2)
    y = round(target_bottom - sprite.height)

    x = max(-sprite.width + 1, min(canvas_w - 1, x))
    y = max(-sprite.height + 1, min(canvas_h - 1, y))

    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    canvas.alpha_composite(sprite, (x, y))
    return canvas


def frame_groups(kind_dir: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for path in sorted(kind_dir.glob("*.frame*.png")):
        match = FRAME_RE.match(path.name)
        if not match:
            continue
        groups.setdefault(match["name"], []).append(path)
    return groups


def copy_originals(groups: dict[str, list[Path]], backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    for paths in groups.values():
        for path in paths:
            target = backup_dir / path.name
            if not target.exists():
                shutil.copy2(path, target)


def render_strip(paths: list[Path], max_frame_w: int, label: str) -> Image.Image:
    frames = [Image.open(p).convert("RGBA") for p in paths]
    scale = min(1.0, max_frame_w / max(img.width for img in frames))
    tile_w = max(32, round(max(img.width for img in frames) * scale) + 10)
    tile_h = max(44, round(max(img.height for img in frames) * scale) + 28)
    sheet = Image.new("RGBA", (tile_w * len(frames), tile_h), (21, 24, 30, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 6), label, fill=(235, 238, 245, 255))
    for idx, img in enumerate(frames):
        if scale != 1.0:
            img = img.resize((round(img.width * scale), round(img.height * scale)), Image.Resampling.NEAREST)
        x = idx * tile_w + (tile_w - img.width) // 2
        y = tile_h - img.height - 6
        sheet.alpha_composite(img, (x, y))
    return sheet


def render_previews(out_dir: Path, backup_root: Path, atlas_root: Path, characters: list[tuple[str, str]]) -> None:
    for kind, name in characters:
        current = sorted((atlas_root / kind).glob(f"{name}.frame*.png"))
        before = sorted((backup_root / kind).glob(f"{name}.frame*.png"))
        if not current or not before:
            continue

        max_frame_w = 46 if kind == "player" else 64
        before_strip = render_strip(before, max_frame_w, f"{kind}/{name} before")
        after_strip = render_strip(current, max_frame_w, f"{kind}/{name} redraw")
        w = max(before_strip.width, after_strip.width)
        preview = Image.new("RGBA", (w, before_strip.height + after_strip.height), (14, 16, 20, 255))
        preview.alpha_composite(before_strip, (0, 0))
        preview.alpha_composite(after_strip, (0, before_strip.height))
        preview.save(out_dir / f"{kind}-{name}-before-after.png")

    player_rows = []
    boss_rows = []
    for kind, name in characters:
        rows = player_rows if kind == "player" else boss_rows
        current = sorted((atlas_root / kind).glob(f"{name}.frame*.png"))
        if current:
            rows.append(render_strip(current, 42 if kind == "player" else 54, f"{name}"))

    for filename, rows in [("player-redraw-contact.png", player_rows), ("boss-redraw-contact.png", boss_rows)]:
        if not rows:
            continue
        w = max(r.width for r in rows)
        h = sum(r.height for r in rows)
        sheet = Image.new("RGBA", (w, h), (14, 16, 20, 255))
        y = 0
        for row in rows:
            sheet.alpha_composite(row, (0, y))
            y += row.height
        sheet.save(out_dir / filename)


def generate(args: argparse.Namespace) -> None:
    atlas_root = args.atlas_root
    out_dir = args.out_dir
    backup_root = out_dir / "original-png"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = []
    processed: list[tuple[str, str]] = []

    for kind in ("player", "boss"):
        groups = frame_groups(atlas_root / kind)
        copy_originals(groups, backup_root / kind)

        for name, paths in sorted(groups.items()):
            if name not in SOURCES:
                continue

            spec = SOURCES[name]
            source = crop_source(args.repo_root / spec.source, spec)
            processed.append((kind, name))

            for path in paths:
                frame_index = int(FRAME_RE.match(path.name)["frame"])
                anchor_path = backup_root / kind / path.name
                original = Image.open(anchor_path if anchor_path.exists() else path).convert("RGBA")
                bbox = alpha_bbox(original)
                if bbox is None:
                    raise ValueError(f"{anchor_path if anchor_path.exists() else path} has no visible pixels")

                generated = transform_sprite(source, spec, original.size, bbox, frame_index)
                generated = apply_fullbody_guard(generated, original, spec)
                generated = apply_readable_legs(generated, spec, frame_index)
                generated.save(path)

                new_bbox = alpha_bbox(generated)
                metrics.append({
                    "kind": kind,
                    "name": name,
                    "frame": frame_index,
                    "path": str(path),
                    "canvas": list(original.size),
                    "old_bbox": list(bbox),
                    "new_bbox": list(new_bbox) if new_bbox else None,
                })

    (out_dir / "generated-frame-metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    render_previews(out_dir, backup_root, atlas_root, processed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--atlas-root", type=Path, default=Path("atlas/common"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/redraw/gameplay-sprite-replacement"))
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    args.atlas_root = (args.repo_root / args.atlas_root).resolve()
    args.out_dir = (args.repo_root / args.out_dir).resolve()
    generate(args)


if __name__ == "__main__":
    main()
