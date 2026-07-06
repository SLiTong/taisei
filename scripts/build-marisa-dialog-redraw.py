#!/usr/bin/env python3
"""Build anatomy-stable Marisa dialog portraits from one body and face sheet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
EXPRESSIONS = (
    "normal",
    "happy",
    "inquisitive",
    "puzzled",
    "smug",
    "surprised",
    "sweat_smile",
    "unamused",
)

# Pillow affine transforms map output coordinates back into source coordinates.
# These coefficients align the expression sheet's eyes and mouth to the master.
SOURCE_FROM_MASTER = (
    0.88745,
    0.02112,
    -190.35,
    0.03674,
    1.11960,
    -187.88,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--master",
        type=Path,
        default=ROOT / "misc/redraw-source/marisa/dialog-master.png",
    )
    parser.add_argument(
        "--expression-sheet",
        type=Path,
        default=ROOT / "misc/redraw-source/marisa/dialog-expression-sheet-chromakey.png",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "atlas/portraits/dialog",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=ROOT / "artifacts/redraw/dialog-anatomy-fix/frames",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=ROOT / "artifacts/redraw/dialog-anatomy-fix/audit.json",
    )
    return parser.parse_args()


def split_sheet(sheet: Image.Image) -> list[Image.Image]:
    cells: list[Image.Image] = []
    for row in range(2):
        top = round(row * sheet.height / 2)
        bottom = round((row + 1) * sheet.height / 2)
        for column in range(4):
            left = round(column * sheet.width / 4)
            right = round((column + 1) * sheet.width / 4)
            cells.append(sheet.crop((left, top, right, bottom)).convert("RGBA"))
    return cells


def source_feature_mask(size: tuple[int, int], include_sweat: bool) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((116, 210, 296, 362), fill=255)
    if include_sweat:
        draw.ellipse((258, 208, 318, 304), fill=255)
    return mask.filter(ImageFilter.GaussianBlur(9))


def warp_face(cell: Image.Image, master_size: tuple[int, int], include_sweat: bool) -> tuple[Image.Image, Image.Image]:
    warped = cell.transform(
        master_size,
        Image.Transform.AFFINE,
        SOURCE_FROM_MASTER,
        resample=Image.Resampling.BICUBIC,
    )
    mask = source_feature_mask(cell.size, include_sweat).transform(
        master_size,
        Image.Transform.AFFINE,
        SOURCE_FROM_MASTER,
        resample=Image.Resampling.BICUBIC,
    )
    return warped, mask


def normalize(image: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
    frame_size = (512, 995)
    horizontal_padding = 16
    top_padding = 16
    bottom_padding = 24
    crop = image.crop(bbox)
    scale = min(
        (frame_size[0] - 2 * horizontal_padding) / crop.width,
        (frame_size[1] - top_padding - bottom_padding) / crop.height,
    )
    size = (round(crop.width * scale), round(crop.height * scale))
    resized = crop.resize(size, Image.Resampling.LANCZOS)
    frame = Image.new("RGBA", frame_size, (0, 0, 0, 0))
    left = (frame.width - resized.width) // 2
    top = frame.height - bottom_padding - resized.height
    frame.alpha_composite(resized, (left, top))
    return frame


def changed_pixel_count(a: Image.Image, b: Image.Image) -> int:
    diff = ImageChops.difference(a.convert("RGB"), b.convert("RGB"))
    return sum(1 for value in diff.convert("L").get_flattened_data() if value > 3)


def changed_bbox(a: Image.Image, b: Image.Image) -> tuple[int, int, int, int] | None:
    return ImageChops.difference(a.convert("RGB"), b.convert("RGB")).getbbox()


def main() -> None:
    args = parse_args()
    master = Image.open(args.master).convert("RGBA")
    sheet = Image.open(args.expression_sheet).convert("RGB")
    cells = split_sheet(sheet)
    if len(cells) != len(EXPRESSIONS):
        raise SystemExit("Expression sheet must contain exactly eight cells.")

    alpha = master.getchannel("A").point(lambda value: 255 if value > 8 else 0)
    content_bbox = alpha.getbbox()
    if content_bbox is None:
        raise SystemExit("No visible pixels found in the dialog master.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.preview_dir.mkdir(parents=True, exist_ok=True)
    frames: list[Image.Image] = []

    for index, (expression, cell) in enumerate(zip(EXPRESSIONS, cells, strict=True)):
        if index == 0:
            variant = master.copy()
        else:
            face, mask = warp_face(cell, master.size, expression == "sweat_smile")
            variant = Image.composite(face, master, mask)
            variant.putalpha(master.getchannel("A"))
        frame = normalize(variant, content_bbox)
        frames.append(frame)
        frame.save(
            args.output_dir / f"marisa_redraw_{expression}.webp",
            "WEBP",
            lossless=True,
            method=6,
            exact=True,
        )
        frame.save(args.preview_dir / f"{index + 1:02d}-{expression}.png")

    normal = frames[0]
    audit = {
        "frame_size": list(normal.size),
        "source_content_bbox": list(content_bbox),
        "expressions": [],
    }
    for expression, frame in zip(EXPRESSIONS, frames, strict=True):
        alpha_bbox = frame.getchannel("A").point(lambda value: 255 if value > 8 else 0).getbbox()
        change_bbox = changed_bbox(normal, frame)
        audit["expressions"].append(
            {
                "name": expression,
                "alpha_bbox": list(alpha_bbox) if alpha_bbox else None,
                "changed_pixels_from_normal": changed_pixel_count(normal, frame),
                "change_bbox_from_normal": list(change_bbox) if change_bbox else None,
                "bottom_padding": frame.height - alpha_bbox[3] if alpha_bbox else None,
            }
        )

    expected_bbox = audit["expressions"][0]["alpha_bbox"]
    for item in audit["expressions"]:
        if item["alpha_bbox"] != expected_bbox:
            raise SystemExit(f"Anchor drift detected in {item['name']}: {item['alpha_bbox']}")
        if item["bottom_padding"] is None or item["bottom_padding"] < 16:
            raise SystemExit(f"Incomplete bottom padding in {item['name']}")
        if item["name"] != "normal" and item["changed_pixels_from_normal"] < 1000:
            raise SystemExit(f"Expression motion is too subtle in {item['name']}")
        if item["change_bbox_from_normal"] and item["change_bbox_from_normal"][3] > 300:
            raise SystemExit(f"Body pixels changed outside the face in {item['name']}")

    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
