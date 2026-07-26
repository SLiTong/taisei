#!/usr/bin/env python3

import argparse
from collections import deque
from pathlib import Path
from statistics import median

from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps


SOURCE_DIR = Path("build/character-redraw-ai/sources")
TARGET_ROOTS = (
    Path("atlas/common/player"),
    Path("atlas/common/boss"),
    Path("atlas/common/enemy"),
)

SOURCE_BY_PREFIX = {
    "reimu": "reimu",
    "marisa": "marisa",
    "youmu": "youmu",
    "cirno": "cirno",
    "elly": "elly",
    "hina": "hina",
    "iku_mid": "iku",
    "iku": "iku",
    "kurumi": "kurumi",
    "scuttle": "scuttle",
    "wriggleex": "wriggle",
    "wriggle": "wriggle",
    "bigfairy": "cirno",
    "hugefairy": "cirno",
    "superfairy": "cirno",
    "fairy_blue": "cirno",
    "fairy_red": "elly",
}

TINTS = {
    "fairy_blue": (0.78, 0.90, 1.18),
    "fairy_red": (1.18, 0.82, 0.82),
    "bigfairy": (0.86, 0.96, 1.14),
    "hugefairy": (0.90, 0.98, 1.10),
    "superfairy": (1.08, 0.94, 1.08),
    "wriggleex": (0.86, 1.16, 0.88),
}


def is_target(path):
    if path.suffix.lower() not in {".png", ".webp"}:
        return False
    if path.name == "swirl.png":
        return False
    return any(path.is_relative_to(root) for root in TARGET_ROOTS)


def prefix_for(path):
    return path.stem.split(".frame", 1)[0]


def source_name_for(path):
    prefix = prefix_for(path)
    try:
        return SOURCE_BY_PREFIX[prefix]
    except KeyError as exc:
        raise RuntimeError(f"No redraw source mapped for {path}") from exc


def border_key(image):
    rgb = image.convert("RGB")
    width, height = rgb.size
    step = max(1, min(width, height) // 96)
    samples = []
    for x in range(0, width, step):
        samples.append(rgb.getpixel((x, 0)))
        samples.append(rgb.getpixel((x, height - 1)))
    for y in range(0, height, step):
        samples.append(rgb.getpixel((0, y)))
        samples.append(rgb.getpixel((width - 1, y)))
    return tuple(int(median(p[channel] for p in samples)) for channel in range(3))


def smoothstep(value):
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def largest_alpha_component(image):
    alpha = image.getchannel("A")
    pixels = alpha.load()
    width, height = image.size
    seen = bytearray(width * height)
    best = []

    for y in range(height):
        for x in range(width):
            idx = y * width + x
            if seen[idx] or pixels[x, y] <= 8:
                continue

            seen[idx] = 1
            queue = deque([(x, y)])
            component = []

            while queue:
                cx, cy = queue.popleft()
                component.append((cx, cy))
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < width and 0 <= ny < height:
                        nidx = ny * width + nx
                        if not seen[nidx] and pixels[nx, ny] > 8:
                            seen[nidx] = 1
                            queue.append((nx, ny))

            if len(component) > len(best):
                best = component

    if not best:
        raise RuntimeError("source cutout is empty")

    mask = Image.new("L", image.size, 0)
    mask_pixels = mask.load()
    for x, y in best:
        mask_pixels[x, y] = 255

    kept = Image.new("RGBA", image.size, (0, 0, 0, 0))
    kept.alpha_composite(image)
    kept.putalpha(ImageChops.multiply(image.getchannel("A"), mask))
    return kept.crop(kept.getbbox())


def remove_chroma_key(path):
    image = Image.open(path).convert("RGBA")
    key = border_key(image)
    pixels = image.load()
    width, height = image.size

    for y in range(height):
        for x in range(width):
            red, green, blue, alpha = pixels[x, y]
            distance = max(abs(red - key[0]), abs(green - key[1]), abs(blue - key[2]))
            green_dominance = green - max(red, blue)
            key_like = distance < 42 or (green > 80 and green_dominance > 18)

            if key_like:
                if distance <= 18 or green_dominance > 35:
                    out_alpha = 0
                elif distance >= 150:
                    out_alpha = alpha
                else:
                    out_alpha = int(alpha * smoothstep((distance - 18) / 132.0))
            else:
                out_alpha = alpha

            if out_alpha < 5:
                pixels[x, y] = (0, 0, 0, 0)
            else:
                if green > max(red, blue) + 8 and out_alpha < 245:
                    green = min(green, max(red, blue) + 2)
                pixels[x, y] = (red, green, blue, out_alpha)

    return largest_alpha_component(image)


def tint_texture(texture, prefix):
    if prefix not in TINTS:
        return texture

    factors = TINTS[prefix]
    red, green, blue, alpha = texture.split()
    red = red.point(lambda v: min(255, int(v * factors[0])))
    green = green.point(lambda v: min(255, int(v * factors[1])))
    blue = blue.point(lambda v: min(255, int(v * factors[2])))
    return Image.merge("RGBA", (red, green, blue, alpha))


def draw_frame_from_source(frame, source, prefix):
    frame_rgba = frame.convert("RGBA")
    bbox = frame_rgba.getbbox()
    if bbox is None:
        return frame_rgba

    left, top, right, bottom = bbox
    target_size = (right - left, bottom - top)
    alpha = frame_rgba.crop(bbox).getchannel("A")

    texture = source.resize(target_size, Image.Resampling.LANCZOS)
    texture = tint_texture(texture, prefix)

    # The original alpha is used only as animation pose/anchor data; all color is
    # regenerated from the new style source.
    texture.putalpha(alpha)

    edge = alpha.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(3))
    shaded = ImageEnhance.Color(texture).enhance(1.12)
    shaded = ImageEnhance.Contrast(shaded).enhance(1.08)

    pixels = shaded.load()
    edge_pixels = edge.load()
    width, height = shaded.size
    for y in range(height):
        v = y / max(height - 1, 1)
        for x in range(width):
            red, green, blue, out_alpha = pixels[x, y]
            if out_alpha == 0:
                continue

            # Compact sprites need bolder line work and top-left shine to read
            # after atlas scaling.
            light = 1.08 - 0.18 * v
            red = min(255, int(red * light))
            green = min(255, int(green * light))
            blue = min(255, int(blue * light))

            if edge_pixels[x, y] > 16:
                red = int(red * 0.28)
                green = int(green * 0.26)
                blue = int(blue * 0.36)

            pixels[x, y] = (red, green, blue, out_alpha)

    output = Image.new("RGBA", frame_rgba.size, (0, 0, 0, 0))
    output.alpha_composite(shaded, (left, top))
    return output


def iter_targets(repo_root):
    for root in TARGET_ROOTS:
        for path in sorted((repo_root / root).glob("*")):
            if path.is_file() and is_target(path.relative_to(repo_root)):
                yield path


def save_image(path, image):
    if path.suffix.lower() == ".webp":
        image.save(path, lossless=True, quality=100, method=6)
    else:
        image.save(path, optimize=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--sources", type=Path, default=SOURCE_DIR)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    sources_dir = (repo_root / args.sources).resolve()
    source_cache = {}

    def get_source(name):
        if name not in source_cache:
            source_cache[name] = remove_chroma_key(sources_dir / f"{name}.png")
        return source_cache[name]

    targets = list(iter_targets(repo_root))
    print(f"sprite frames: {len(targets)}")

    for idx, path in enumerate(targets, start=1):
        rel = path.relative_to(repo_root)
        prefix = prefix_for(path)
        source = get_source(source_name_for(path))
        with Image.open(path) as frame:
            redrawn = draw_frame_from_source(frame, source, prefix)
            if redrawn.size != frame.size:
                raise RuntimeError(f"size changed for {rel}: {frame.size} -> {redrawn.size}")
            if redrawn.getextrema()[3][1] == 0:
                raise RuntimeError(f"fully transparent output: {rel}")

        if args.write:
            save_image(path, redrawn)

        print(f"[{idx:03d}/{len(targets):03d}] {rel.as_posix()} <- {source_name_for(path)}")

    print("sprite redraw OK")


if __name__ == "__main__":
    main()
