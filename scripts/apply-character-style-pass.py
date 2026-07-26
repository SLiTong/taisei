#!/usr/bin/env python3

import argparse
import csv
import io
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT_REL_TARGETS = [
    Path('atlas/common/player'),
    Path('atlas/common/boss'),
    Path('atlas/common/enemy'),
    Path('atlas/portraits/dialog'),
]

PREVIEW_SAMPLES = [
    Path('atlas/portraits/dialog/reimu.webp'),
    Path('atlas/portraits/dialog/marisa.webp'),
    Path('atlas/portraits/dialog/youmu.webp'),
    Path('atlas/portraits/dialog/cirno.webp'),
    Path('atlas/portraits/dialog/iku.webp'),
    Path('atlas/portraits/dialog/yumemi.webp'),
    Path('atlas/common/player/reimu.frame0004.png'),
    Path('atlas/common/player/marisa.frame0004.png'),
    Path('atlas/common/boss/cirno.frame0000.png'),
    Path('atlas/common/boss/iku.frame0000.png'),
    Path('atlas/common/enemy/fairy_blue.frame0000.webp'),
    Path('atlas/common/enemy/fairy_red.frame0000.webp'),
]


def is_character_image(path: Path) -> bool:
    if path.suffix.lower() not in {'.png', '.webp'}:
        return False

    if '.alphamap' in path.name:
        return False

    # This is an abstract enemy/effect glyph, not a character illustration.
    if path.as_posix().endswith('/atlas/common/enemy/swirl.png'):
        return False

    return True


def iter_targets(repo_root: Path):
    for rel_root in ROOT_REL_TARGETS:
        root = repo_root / rel_root
        if not root.is_dir():
            continue

        for path in sorted(root.rglob('*')):
            if path.is_file() and is_character_image(path):
                yield path


def target_group(path: Path) -> str:
    rel = path.as_posix()
    if '/atlas/common/player/' in rel:
        return 'player'
    if '/atlas/common/boss/' in rel:
        return 'boss'
    if '/atlas/common/enemy/' in rel:
        return 'enemy'
    if '/atlas/portraits/dialog/' in rel:
        if '_face_' in path.name or '_face.' in path.name:
            return 'portrait-face'
        if '_variant_' in path.name:
            return 'portrait-variant'
        if '_misc_' in path.name:
            return 'portrait-misc'
        return 'portrait-base'
    return 'unknown'


def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def image_bbox(alpha: np.ndarray):
    ys, xs = np.where(alpha[..., 0] > 0.02)
    if len(xs) == 0 or len(ys) == 0:
        return None

    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def bbox_uv(alpha: np.ndarray):
    h, w = alpha.shape[:2]
    bbox = image_bbox(alpha)
    x_grid = np.arange(w, dtype=np.float32)[None, :, None]
    y_grid = np.arange(h, dtype=np.float32)[:, None, None]

    if bbox is None:
        return x_grid / max(w - 1, 1), y_grid / max(h - 1, 1)

    x0, y0, x1, y1 = bbox
    bw = max(x1 - x0 - 1, 1)
    bh = max(y1 - y0 - 1, 1)
    return (x_grid - x0) / bw, (y_grid - y0) / bh


def recolor_rgba(img: Image.Image, strength: float) -> Image.Image:
    rgba = img.convert('RGBA')
    arr = np.asarray(rgba).astype(np.float32) / 255.0

    rgb = arr[..., :3]
    alpha = arr[..., 3:4]
    visible = alpha > 0.01

    lum = (
        rgb[..., 0:1] * 0.2126 +
        rgb[..., 1:2] * 0.7152 +
        rgb[..., 2:3] * 0.0722
    )

    # Soft modern-anime color grade: cleaner highlights, cooler shadows,
    # slightly richer midtones, without changing silhouette or dimensions.
    contrast = 1.0 + 0.14 * strength
    sat = 1.0 + 0.18 * strength
    graded = 0.5 + (rgb - 0.5) * contrast
    graded = lum + (graded - lum) * sat

    highlight = smoothstep(0.58, 0.92, lum) * strength
    shadow = (1.0 - smoothstep(0.16, 0.48, lum)) * strength
    mid = smoothstep(0.22, 0.62, lum) * (1.0 - smoothstep(0.70, 0.92, lum)) * strength

    warm_highlight = np.array([1.00, 0.965, 0.900], dtype=np.float32)
    cool_shadow = np.array([0.105, 0.120, 0.230], dtype=np.float32)
    rose_mid = np.array([1.00, 0.710, 0.740], dtype=np.float32)

    graded = graded * (1.0 - 0.08 * highlight) + warm_highlight * (0.08 * highlight)
    graded = graded * (1.0 - 0.09 * shadow) + cool_shadow * (0.09 * shadow)

    # Skin-like areas get a tiny rose lift. The condition is deliberately broad
    # and weak so it enriches existing art instead of repainting costumes.
    skin_like = (
        (rgb[..., 0:1] > rgb[..., 1:2] * 0.92) &
        (rgb[..., 1:2] > rgb[..., 2:3] * 0.72) &
        (lum > 0.38) &
        (lum < 0.92)
    )
    graded = np.where(
        skin_like,
        graded * (1.0 - 0.035 * mid) + rose_mid * (0.035 * mid),
        graded,
    )

    # Pearly shine pass, most visible on hair, white cloth, and glossy fabric.
    shine = smoothstep(0.66, 0.95, lum) * strength
    shine_color = np.array([0.90, 0.92, 1.00], dtype=np.float32)
    graded = graded * (1.0 - 0.045 * shine) + shine_color * (0.045 * shine)

    graded = np.clip(graded, 0.0, 1.0)
    arr[..., :3] = np.where(visible, graded, 0.0)

    out = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), 'RGBA')

    # Dark violet-blue line reinforcement, derived from luminance/alpha edges.
    gray = Image.fromarray(np.clip((lum[..., 0] * alpha[..., 0]) * 255.0, 0, 255).astype(np.uint8), 'L')
    alpha_img = Image.fromarray(np.clip(alpha[..., 0] * 255.0, 0, 255).astype(np.uint8), 'L')
    edge = Image.blend(
        gray.filter(ImageFilter.FIND_EDGES),
        alpha_img.filter(ImageFilter.FIND_EDGES),
        0.35,
    ).filter(ImageFilter.GaussianBlur(radius=0.45))

    edge_arr = np.asarray(edge).astype(np.float32) / 255.0
    edge_arr = smoothstep(0.04, 0.38, edge_arr)[..., None] * alpha
    out_arr = np.asarray(out).astype(np.float32) / 255.0
    line_color = np.array([0.085, 0.075, 0.145], dtype=np.float32)
    line_amt = 0.16 * strength * edge_arr
    out_arr[..., :3] = out_arr[..., :3] * (1.0 - line_amt) + line_color * line_amt

    # A subtle crispness pass helps old low-detail sprites read closer to the
    # high-polish reference style without changing their layout.
    sharpened = Image.fromarray(np.clip(out_arr * 255.0, 0, 255).astype(np.uint8), 'RGBA')
    sharpened = sharpened.filter(ImageFilter.UnsharpMask(radius=1.1, percent=int(75 * strength), threshold=3))
    return sharpened


def masked_blend(rgb, color, amount, mask):
    color = np.array(color, dtype=np.float32)
    return rgb * (1.0 - amount * mask) + color * (amount * mask)


def modern_repaint_rgba(img: Image.Image, group: str) -> Image.Image:
    """Layout-locked redraw pass toward a polished modern anime look.

    This intentionally does not alter image dimensions or alpha. Portraits keep
    their existing face/base overlay geometry; sprites keep frame anchors.
    """
    base_strength = {
        'player': 0.88,
        'boss': 0.92,
        'enemy': 0.76,
        'portrait-base': 1.22,
        'portrait-variant': 1.18,
        'portrait-face': 1.10,
        'portrait-misc': 1.00,
    }.get(group, 1.0)

    rgba = recolor_rgba(img, base_strength).convert('RGBA')
    arr = np.asarray(rgba).astype(np.float32) / 255.0
    rgb = arr[..., :3]
    alpha = arr[..., 3:4]
    visible = alpha > 0.01

    lum = (
        rgb[..., 0:1] * 0.2126 +
        rgb[..., 1:2] * 0.7152 +
        rgb[..., 2:3] * 0.0722
    )
    maxc = rgb.max(axis=2, keepdims=True)
    minc = rgb.min(axis=2, keepdims=True)
    chroma = maxc - minc
    red_bias = rgb[..., 0:1] - np.maximum(rgb[..., 1:2], rgb[..., 2:3])
    blue_bias = rgb[..., 2:3] - np.maximum(rgb[..., 0:1], rgb[..., 1:2])

    u, v = bbox_uv(alpha)
    center = 1.0 - np.clip(np.abs(u - 0.5) * 2.0, 0.0, 1.0)
    portrait = group.startswith('portrait')
    full_body = group in {'portrait-base', 'portrait-variant'}

    skin = (
        visible &
        (lum > 0.34) & (lum < 0.95) &
        (rgb[..., 0:1] > rgb[..., 1:2] * 0.90) &
        (rgb[..., 1:2] > rgb[..., 2:3] * 0.64) &
        (red_bias > -0.04) &
        (rgb[..., 0:1] < rgb[..., 1:2] * 1.65) &
        (chroma < 0.50)
    )
    hair_color = (
        visible &
        (chroma > 0.10) &
        ((v < 0.48) | portrait) &
        ~skin
    )
    line = visible & (lum < 0.22)

    # Skin gets soft rose subsurface color and cleaner highlights.
    skin_amount = skin.astype(np.float32)
    rgb = masked_blend(rgb, (1.0, 0.73, 0.68), 0.105, skin_amount)
    blush_zone = smoothstep(0.20, 0.42, v) * (1.0 - smoothstep(0.48, 0.66, v)) * center
    rgb = masked_blend(rgb, (1.0, 0.42, 0.55), 0.055, skin_amount * blush_zone)

    # Modern anime hair rendering: cooler shadows and pearl-blue highlight ramps.
    hair_amount = hair_color.astype(np.float32)
    cool_hair_shadow = (1.0 - smoothstep(0.18, 0.55, lum)) * hair_amount
    hair_highlight = smoothstep(0.58, 0.88, lum) * hair_amount
    rgb = masked_blend(rgb, (0.09, 0.08, 0.18), 0.10, cool_hair_shadow)
    rgb = masked_blend(rgb, (0.82, 0.88, 1.0), 0.09, hair_highlight)

    if full_body:
        # The reference direction is office-fashion anime: bright blouse-like
        # upper torso, dark skirt/stocking-like lower shapes, while preserving
        # each character's silhouette and original line art.
        torso_v = smoothstep(0.22, 0.34, v) * (1.0 - smoothstep(0.54, 0.68, v))
        lower_v = smoothstep(0.48, 0.60, v) * (1.0 - smoothstep(0.88, 0.98, v))
        torso = (visible & ~skin & (lum > 0.14)).astype(np.float32) * torso_v
        lower = (visible & ~skin).astype(np.float32) * lower_v
        accessory = visible.astype(np.float32) * (1.0 - skin_amount) * (1.0 - torso) * (1.0 - lower)

        blouse_mask = torso * (0.70 + 0.30 * smoothstep(0.18, 0.72, lum))
        skirt_mask = lower * (0.72 + 0.28 * (1.0 - smoothstep(0.30, 0.78, lum)))

        rgb = masked_blend(rgb, (0.965, 0.970, 0.950), 0.62, blouse_mask)
        rgb = masked_blend(rgb, (0.038, 0.038, 0.060), 0.52, skirt_mask)
        rgb = masked_blend(rgb, (0.16, 0.18, 0.30), 0.08, accessory)

        blouse_shine = blouse_mask * smoothstep(0.48, 0.90, lum)
        rgb = masked_blend(rgb, (1.0, 1.0, 0.98), 0.18, blouse_shine)

    if group in {'player', 'boss', 'enemy'}:
        # Small in-game sprites need stronger silhouette readability than large
        # portraits. Push outlines and local contrast, but avoid costume remaps.
        sprite_shadow = visible.astype(np.float32) * (1.0 - smoothstep(0.16, 0.46, lum))
        rgb = masked_blend(rgb, (0.055, 0.052, 0.110), 0.12, sprite_shadow)

    # Inked contour pass. The line mask uses luminance and alpha edges so it
    # follows the existing drawing instead of inventing new outlines.
    gray = Image.fromarray(np.clip(lum[..., 0] * alpha[..., 0] * 255, 0, 255).astype(np.uint8), 'L')
    alpha_img = Image.fromarray(np.clip(alpha[..., 0] * 255, 0, 255).astype(np.uint8), 'L')
    edge = Image.blend(
        gray.filter(ImageFilter.FIND_EDGES),
        alpha_img.filter(ImageFilter.FIND_EDGES),
        0.42,
    ).filter(ImageFilter.GaussianBlur(radius=0.35))
    edge_arr = smoothstep(0.035, 0.34, np.asarray(edge).astype(np.float32) / 255.0)[..., None] * alpha
    ink = np.maximum(line.astype(np.float32) * 0.20, edge_arr)
    rgb = masked_blend(rgb, (0.050, 0.043, 0.105), 0.19, ink)

    # Fine sparkle/highlight flecks, limited to visible bright areas; this helps
    # portraits read closer to the glossy sample style without adding objects.
    glossy = visible.astype(np.float32) * smoothstep(0.72, 0.96, lum)
    rgb = masked_blend(rgb, (1.0, 0.96, 0.90), 0.035, glossy)

    arr[..., :3] = np.where(visible, np.clip(rgb, 0.0, 1.0), 0.0)
    out = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), 'RGBA')
    out = out.filter(ImageFilter.UnsharpMask(radius=1.0, percent=52 if portrait else 68, threshold=2))
    return out


def style_image(path: Path, mode: str) -> Image.Image:
    rel = path.as_posix()
    strength = 1.0

    if '/atlas/common/player/' in rel:
        strength = 0.88
    elif '/atlas/common/boss/' in rel:
        strength = 0.90
    elif '/atlas/common/enemy/' in rel:
        strength = 0.78
    elif '/atlas/portraits/dialog/' in rel:
        strength = 1.06

    with Image.open(path) as img:
        if mode == 'style':
            return recolor_rgba(img, strength)
        if mode == 'redraw':
            return modern_repaint_rgba(img, target_group(path))

    raise ValueError(f'unknown mode: {mode}')


def save_image(path: Path, img: Image.Image):
    if path.suffix.lower() == '.webp':
        img.save(path, 'WEBP', lossless=True, method=6)
    else:
        img.save(path, optimize=True)


def checkerboard(size, cell=16):
    w, h = size
    bg = Image.new('RGBA', size, (220, 224, 232, 255))
    draw = ImageDraw.Draw(bg)
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            if ((x // cell) + (y // cell)) % 2:
                draw.rectangle((x, y, x + cell - 1, y + cell - 1), fill=(244, 246, 250, 255))
    return bg


def fit_on_checker(img: Image.Image, box_size):
    canvas = checkerboard(box_size)
    subject = img.convert('RGBA')
    subject.thumbnail((box_size[0] - 18, box_size[1] - 32), Image.Resampling.LANCZOS)
    x = (box_size[0] - subject.width) // 2
    y = (box_size[1] - subject.height) // 2
    canvas.alpha_composite(subject, (x, y))
    return canvas


def write_manifest(repo_root: Path, targets, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'path',
            'group',
            'width',
            'height',
            'alpha_bbox',
            'has_alphamap',
        ])
        writer.writeheader()
        for path in targets:
            with Image.open(path) as img:
                rgba = img.convert('RGBA')
                arr = np.asarray(rgba).astype(np.float32) / 255.0
                bbox = image_bbox(arr[..., 3:4])
            alphamap = path.with_suffix(f'.alphamap{path.suffix}')
            writer.writerow({
                'path': path.relative_to(repo_root).as_posix(),
                'group': target_group(path),
                'width': rgba.width,
                'height': rgba.height,
                'alpha_bbox': '' if bbox is None else ','.join(map(str, bbox)),
                'has_alphamap': 'yes' if alphamap.is_file() else 'no',
            })


def git_head_image(repo_root: Path, path: Path):
    rel = path.relative_to(repo_root).as_posix()
    try:
        data = subprocess.check_output(
            ['git', 'show', f'HEAD:{rel}'],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    return Image.open(io.BytesIO(data))


def has_dialog_face_sprite_config(repo_root: Path, path: Path) -> bool:
    rel = path.relative_to(repo_root)
    if rel.parts[:3] != ('atlas', 'portraits', 'dialog'):
        return False

    config = repo_root / 'atlas' / 'config' / 'dialog' / f'{path.stem}.spr'
    if not config.is_file():
        return False

    text = config.read_text(encoding='utf-8')
    return 'padding_left' in text or 'padding_right' in text or 'padding_bottom' in text


def verify_layout(repo_root: Path, targets):
    failures = []

    for path in targets:
        rel = path.relative_to(repo_root).as_posix()
        group = target_group(path)
        with Image.open(path) as current:
            current_rgba = current.convert('RGBA')
            original = git_head_image(repo_root, path)
            if original is not None:
                try:
                    if (
                        current.size != original.size
                        and not (
                            group == 'portrait-face'
                            and has_dialog_face_sprite_config(repo_root, path)
                        )
                        and group not in {'portrait-base', 'portrait-variant', 'portrait-misc'}
                    ):
                        failures.append(f'{rel}: size changed {original.size} -> {current.size}')
                finally:
                    original.close()

            if current_rgba.getextrema()[3][1] == 0:
                failures.append(f'{rel}: alpha channel is fully transparent')
            elif group in {'portrait-base', 'portrait-variant'} and '_misc_' not in path.name:
                alpha = np.asarray(current_rgba).astype(np.float32)[..., 3:4] / 255.0
                bbox = image_bbox(alpha)

                if bbox is not None:
                    x0, y0, x1, y1 = bbox
                    margins = (
                        x0,
                        y0,
                        current_rgba.width - x1,
                        current_rgba.height - y1,
                    )

                    if min(margins) < 8:
                        failures.append(f'{rel}: portrait alpha too close to edge {margins}')

        alphamap = path.with_suffix(f'.alphamap{path.suffix}')
        if alphamap.is_file():
            with Image.open(path) as base, Image.open(alphamap) as amap:
                if base.size != amap.size:
                    failures.append(
                        f'{rel}: alphamap size mismatch {base.size} != {amap.size}'
                    )

    if failures:
        print('layout verification failed:')
        for failure in failures:
            print(f'  - {failure}')
        raise SystemExit(1)

    print(f'layout verification OK: {len(targets)} images')


def write_preview(repo_root: Path, out_path: Path, mode: str):
    samples = [repo_root / p for p in PREVIEW_SAMPLES if (repo_root / p).is_file()]
    cell = (220, 260)
    label_h = 38
    cols = 4
    rows = len(samples)
    sheet = Image.new('RGBA', (cols * cell[0], rows * (cell[1] + label_h)), (26, 28, 36, 255))
    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype('arial.ttf', 14)
    except OSError:
        font = ImageFont.load_default()

    for row, path in enumerate(samples):
        y = row * (cell[1] + label_h)
        with Image.open(path) as before:
            before_rgba = before.convert('RGBA')
        after_rgba = style_image(path, mode)

        sheet.alpha_composite(fit_on_checker(before_rgba, cell), (0, y + label_h))
        sheet.alpha_composite(fit_on_checker(after_rgba, cell), (cell[0], y + label_h))

        draw.text((8, y + 6), path.relative_to(repo_root).as_posix(), fill=(238, 240, 246, 255), font=font)
        draw.text((8, y + 22), 'before', fill=(170, 178, 196, 255), font=font)
        draw.text((cell[0] + 8, y + 22), f'after {mode} pass', fill=(170, 220, 255, 255), font=font)

        # Duplicate the after cell at 2x and 1x comparison columns for small sprites.
        zoom = after_rgba.copy()
        zoom.thumbnail((cell[0] - 18, cell[1] - 32), Image.Resampling.NEAREST)
        zoom_cell = checkerboard(cell)
        zoom_cell.alpha_composite(zoom, ((cell[0] - zoom.width) // 2, (cell[1] - zoom.height) // 2))
        sheet.alpha_composite(zoom_cell, (cell[0] * 2, y + label_h))
        draw.text((cell[0] * 2 + 8, y + 22), 'after nearest view', fill=(170, 220, 255, 255), font=font)

        diff = Image.blend(before_rgba.resize(after_rgba.size), after_rgba, 0.5)
        sheet.alpha_composite(fit_on_checker(diff, cell), (cell[0] * 3, y + label_h))
        draw.text((cell[0] * 3 + 8, y + 22), 'blend check', fill=(170, 220, 255, 255), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.convert('RGB').save(out_path, optimize=True)


def main():
    parser = argparse.ArgumentParser(description='Apply a non-destructive-layout character art style pass.')
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--mode', choices=('style', 'redraw'), default='redraw')
    parser.add_argument('--preview', type=Path, default=None)
    parser.add_argument('--manifest', type=Path, default=None)
    parser.add_argument('--verify-layout', action='store_true')
    parser.add_argument('--write', action='store_true', help='Overwrite target atlas source images.')
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    targets = list(iter_targets(repo_root))

    print(f'character images: {len(targets)}')
    if args.manifest:
        write_manifest(repo_root, targets, args.manifest)
        print(f'manifest: {args.manifest}')

    if args.preview:
        write_preview(repo_root, args.preview, args.mode)
        print(f'preview: {args.preview}')

    if args.write:
        for idx, path in enumerate(targets, 1):
            rel = path.relative_to(repo_root).as_posix()
            img = style_image(path, args.mode)
            save_image(path, img)
            print(f'[{idx:03d}/{len(targets):03d}] {rel}')

    if args.verify_layout:
        verify_layout(repo_root, targets)


if __name__ == '__main__':
    main()
