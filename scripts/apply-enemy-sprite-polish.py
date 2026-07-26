#!/usr/bin/env python3

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


ENEMY_SPECS = {
    "fairy_blue": {
        "hair": (78, 166, 255, 255),
        "dress": (36, 96, 214, 255),
        "accent": (214, 244, 255, 255),
        "wing": (118, 222, 255, 92),
        "scale": 1.0,
    },
    "fairy_red": {
        "hair": (255, 104, 124, 255),
        "dress": (210, 48, 78, 255),
        "accent": (255, 224, 228, 255),
        "wing": (255, 164, 202, 92),
        "scale": 1.0,
    },
    "bigfairy": {
        "hair": (105, 208, 255, 255),
        "dress": (54, 122, 236, 255),
        "accent": (242, 252, 255, 255),
        "wing": (140, 234, 255, 105),
        "scale": 1.23,
    },
    "hugefairy": {
        "hair": (152, 126, 255, 255),
        "dress": (74, 72, 190, 255),
        "accent": (255, 238, 174, 255),
        "wing": (188, 174, 255, 112),
        "scale": 1.34,
    },
    "superfairy": {
        "hair": (255, 92, 190, 255),
        "dress": (88, 42, 122, 255),
        "accent": (255, 230, 132, 255),
        "wing": (255, 142, 218, 120),
        "scale": 1.48,
    },
}


def rgba_mul(color, amount):
    r, g, b, a = color
    return (min(255, int(r * amount)), min(255, int(g * amount)), min(255, int(b * amount)), a)


def draw_chibi_fairy(size, spec, frame):
    aa = 4
    w, h = size
    img = Image.new("RGBA", (w * aa, h * aa), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img, "RGBA")

    def p(x, y):
        return (int(round(x * aa)), int(round(y * aa)))

    def box(cx, cy, rx, ry):
        return (
            int(round((cx - rx) * aa)),
            int(round((cy - ry) * aa)),
            int(round((cx + rx) * aa)),
            int(round((cy + ry) * aa)),
        )

    def line(points, fill, width=1):
        draw.line([p(x, y) for x, y in points], fill=fill, width=max(1, int(width * aa)), joint="curve")

    def ellipse(cx, cy, rx, ry, fill, outline=None, width=1):
        draw.ellipse(box(cx, cy, rx, ry), fill=fill, outline=outline, width=max(1, int(width * aa)))

    def polygon(points, fill, outline=None):
        pts = [p(x, y) for x, y in points]
        draw.polygon(pts, fill=fill)
        if outline:
            draw.line(pts + [pts[0]], fill=outline, width=max(1, int(1.45 * aa)), joint="curve")

    cx = w / 2
    s = spec["scale"]
    bob = [0, -1.8, 1.2, -0.6][frame]
    wing = [1.0, 0.58, 1.18, 0.72][frame]
    tilt = [-2.5, 1.0, 2.5, -1.0][frame]

    outline = (28, 24, 45, 235)
    skin = (250, 205, 186, 255)
    blush = (255, 100, 126, 70)
    hair = spec["hair"]
    dress = spec["dress"]
    accent = spec["accent"]
    wing_color = spec["wing"]

    head_y = h * 0.32 + bob
    body_y = h * 0.58 + bob
    head_rx = w * 0.145 * s
    head_ry = h * 0.105 * s
    body_rx = w * 0.135 * s
    body_h = h * 0.235 * s

    # Wings behind the body.
    for side in (-1, 1):
        wx = cx + side * w * 0.18 * s
        wy = body_y - h * 0.09 * s
        wing_rx = w * 0.22 * s
        wing_ry = h * 0.105 * s * wing
        ellipse(wx, wy, wing_rx, wing_ry, wing_color, rgba_mul(wing_color, 0.55), 1.1)
        ellipse(wx + side * w * 0.05 * s, wy + h * 0.08 * s, wing_rx * 0.72, wing_ry * 0.70, wing_color, rgba_mul(wing_color, 0.55), 1.0)

    # Legs and arms.
    line([(cx - w * 0.06 * s, body_y + body_h * 0.42), (cx - w * 0.11 * s, body_y + body_h * 0.82)], outline, 2.0)
    line([(cx + w * 0.06 * s, body_y + body_h * 0.42), (cx + w * 0.11 * s, body_y + body_h * 0.82)], outline, 2.0)
    line([(cx - body_rx * 0.82, body_y - body_h * 0.12), (cx - body_rx * 1.46, body_y + body_h * 0.08)], outline, 2.2)
    line([(cx + body_rx * 0.82, body_y - body_h * 0.12), (cx + body_rx * 1.46, body_y + body_h * 0.08)], outline, 2.2)

    # Dress, blouse, and ribbon.
    polygon([
        (cx - body_rx, body_y - body_h * 0.42),
        (cx + body_rx, body_y - body_h * 0.42),
        (cx + body_rx * 1.28, body_y + body_h * 0.42),
        (cx, body_y + body_h * 0.72),
        (cx - body_rx * 1.28, body_y + body_h * 0.42),
    ], dress, outline)
    polygon([
        (cx - body_rx * 0.72, body_y - body_h * 0.42),
        (cx + body_rx * 0.72, body_y - body_h * 0.42),
        (cx + body_rx * 0.40, body_y - body_h * 0.05),
        (cx - body_rx * 0.40, body_y - body_h * 0.05),
    ], (248, 246, 238, 255), outline)
    polygon([
        (cx, body_y - body_h * 0.08),
        (cx - body_rx * 0.36, body_y + body_h * 0.12),
        (cx - body_rx * 0.04, body_y + body_h * 0.20),
    ], accent, outline)
    polygon([
        (cx, body_y - body_h * 0.08),
        (cx + body_rx * 0.36, body_y + body_h * 0.12),
        (cx + body_rx * 0.04, body_y + body_h * 0.20),
    ], accent, outline)

    # Head and hair. Big shapes first for readability.
    ellipse(cx, head_y, head_rx, head_ry, skin, outline, 1.7)
    polygon([
        (cx - head_rx * 1.05, head_y - head_ry * 0.35),
        (cx - head_rx * 0.55, head_y - head_ry * 1.16),
        (cx, head_y - head_ry * 0.92 + tilt * 0.05),
        (cx + head_rx * 0.62, head_y - head_ry * 1.10),
        (cx + head_rx * 1.12, head_y - head_ry * 0.22),
        (cx + head_rx * 0.62, head_y + head_ry * 0.18),
        (cx, head_y - head_ry * 0.05),
        (cx - head_rx * 0.62, head_y + head_ry * 0.20),
    ], hair, outline)
    for side in (-1, 1):
        ellipse(cx + side * head_rx * 1.0, head_y + head_ry * 0.05, head_rx * 0.30, head_ry * 0.62, hair, outline, 1.4)

    # Face details.
    eye_y = head_y + head_ry * 0.10
    for side in (-1, 1):
        ex = cx + side * head_rx * 0.38
        ellipse(ex, eye_y, head_rx * 0.15, head_ry * 0.20, (250, 250, 255, 245), outline, 1.0)
        ellipse(ex, eye_y, head_rx * 0.075, head_ry * 0.115, rgba_mul(hair, 0.72), None, 1.0)
        ellipse(ex - head_rx * 0.03, eye_y - head_ry * 0.05, head_rx * 0.035, head_ry * 0.045, (255, 255, 255, 230), None, 1.0)
    draw.arc(box(cx, head_y + head_ry * 0.33, head_rx * 0.22, head_ry * 0.18), 20, 160, fill=outline, width=max(1, int(1.4 * aa)))
    ellipse(cx - head_rx * 0.58, head_y + head_ry * 0.30, head_rx * 0.18, head_ry * 0.09, blush, None, 1)
    ellipse(cx + head_rx * 0.58, head_y + head_ry * 0.30, head_rx * 0.18, head_ry * 0.09, blush, None, 1)

    # Highlights and readable silhouette glints.
    line([(cx - head_rx * 0.48, head_y - head_ry * 0.74), (cx - head_rx * 0.05, head_y - head_ry * 0.88)], (255, 255, 255, 95), 1.1)
    line([(cx - body_rx * 0.30, body_y - body_h * 0.30), (cx + body_rx * 0.10, body_y + body_h * 0.18)], (255, 255, 255, 72), 1.2)

    img = img.filter(ImageFilter.UnsharpMask(radius=1.0 * aa, percent=95, threshold=2))
    return img.resize(size, Image.Resampling.LANCZOS)


def save_image(path, image):
    image.save(path, "WEBP", lossless=True, quality=100, method=6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    enemy_dir = repo_root / "atlas/common/enemy"
    paths = []
    for name in ENEMY_SPECS:
        paths.extend(sorted(enemy_dir.glob(f"{name}.frame*.webp")))

    print(f"enemy frames: {len(paths)}")
    for idx, path in enumerate(paths, 1):
        prefix = path.stem.split(".frame", 1)[0]
        frame = int(path.stem.rsplit("frame", 1)[1])
        with Image.open(path) as src:
            img = draw_chibi_fairy(src.size, ENEMY_SPECS[prefix], frame % 4)
        if args.write:
            save_image(path, img)
        print(f"[{idx:03d}/{len(paths):03d}] {path.relative_to(repo_root).as_posix()} {img.size}")

    print("enemy sprite polish OK")


if __name__ == "__main__":
    main()
