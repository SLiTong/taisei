#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from PIL import Image

import argparse
import json
import sys


REIMU_MAP = {
    "reimu.webp": "reimu_glamour_confident_clean.png",
    "reimu_variant_happy.webp": "reimu_glamour_happy.png",
    "reimu_variant_smug.webp": "reimu_glamour_happy.png",
    "reimu_variant_surprised.webp": "reimu_glamour_surprised.png",
    "reimu_variant_puzzled.webp": "reimu_glamour_surprised.png",
    "reimu_variant_unsettled.webp": "reimu_glamour_battleworn.png",
    "reimu_variant_sigh.webp": "reimu_glamour_battleworn.png",
    "reimu_variant_unamused.webp": "reimu_glamour_battleworn.png",
    "reimu_variant_annoyed.webp": "reimu_glamour_angry.png",
    "reimu_variant_irritated.webp": "reimu_glamour_angry.png",
    "reimu_variant_outraged.webp": "reimu_glamour_angry.png",
    "reimu_variant_assertive.webp": "reimu_glamour_power.png",
}

MARISA_MAP = {
    "marisa.webp": "marisa_glamour_confident.png",
    "marisa_variant_happy.webp": "marisa_glamour_happy_b.png",
    "marisa_variant_smug.webp": "marisa_glamour_happy_a.png",
    "marisa_variant_surprised.webp": "marisa_glamour_surprised.png",
    "marisa_variant_puzzled.webp": "marisa_glamour_surprised.png",
    "marisa_variant_inquisitive.webp": "marisa_glamour_surprised.png",
    "marisa_variant_sweat_smile.webp": "marisa_glamour_battleworn.png",
    "marisa_variant_unamused.webp": "marisa_glamour_angry.png",
}

YOUMU_MAP = {
    "youmu.webp": "youmu_glamour_confident.png",
    "youmu_variant_happy.webp": "youmu_glamour_happy.png",
    "youmu_variant_relaxed.webp": "youmu_glamour_happy.png",
    "youmu_variant_surprised.webp": "youmu_glamour_surprised.png",
    "youmu_variant_puzzled.webp": "youmu_glamour_surprised.png",
    "youmu_variant_eeeeh.webp": "youmu_glamour_surprised.png",
    "youmu_variant_embarrassed.webp": "youmu_glamour_surprised.png",
    "youmu_variant_chuuni.webp": "youmu_glamour_power.png",
    "youmu_variant_smug.webp": "youmu_glamour_power.png",
    "youmu_variant_sigh.webp": "youmu_glamour_battleworn.png",
    "youmu_variant_eyes_closed.webp": "youmu_glamour_battleworn.png",
    "youmu_variant_unamused.webp": "youmu_glamour_angry.png",
}

ELLY_MAP = {
    "elly.webp": "elly_glamour_normal.png",
    "elly_variant_smug.webp": "elly_glamour_smug.png",
    "elly_variant_angry.webp": "elly_glamour_angry.png",
    "elly_variant_shouting.webp": "elly_glamour_shouting.png",
    "elly_variant_blush.webp": "elly_glamour_blush.png",
    "elly_variant_eyes_closed.webp": "elly_glamour_eyes_closed.png",
    "elly_variant_beaten.webp": "elly_glamour_defeated.png",
}

HINA_MAP = {
    "hina.webp": "hina_glamour_normal.png",
    "hina_variant_serious.webp": "hina_glamour_serious_ritual.png",
    "hina_variant_concerned.webp": "hina_glamour_concerned.png",
    "hina_variant_defeated.webp": "hina_glamour_defeated.png",
}

IKU_MAP = {
    "iku.webp": "iku_glamour_normal.png",
    "iku_variant_smile.webp": "iku_glamour_smile.png",
    "iku_variant_serious.webp": "iku_glamour_serious_lightning.png",
    "iku_variant_eyes_closed.webp": "iku_glamour_eyes_closed.png",
    "iku_variant_defeated.webp": "iku_glamour_defeated.png",
}

YUMEMI_MAP = {
    "yumemi.webp": "yumemi_glamour_normal.png",
    "yumemi_variant_smug.webp": "yumemi_glamour_smug.png",
    "yumemi_variant_surprised.webp": "yumemi_glamour_surprised.png",
    "yumemi_variant_sad.webp": "yumemi_glamour_sad.png",
    "yumemi_variant_sigh.webp": "yumemi_glamour_sigh.png",
    "yumemi_variant_eyes_closed.webp": "yumemi_glamour_eyes_closed.png",
    "yumemi_variant_defeated.webp": "yumemi_glamour_defeated.png",
}


def qa_image(path: Path) -> dict:
    im = Image.open(path).convert("RGBA")
    alpha = im.getchannel("A")
    bbox = alpha.getbbox()

    if bbox is None:
        raise ValueError(f"{path} has no visible pixels")

    w, h = im.size
    left, top, right, bottom = bbox
    margins = {
        "left": left,
        "top": top,
        "right": w - right,
        "bottom": h - bottom,
    }

    visible = 0
    green_spill = 0

    for r, g, b, a in im.getdata():
        if a > 32:
            visible += 1
            # Hina-style emerald hair is an intended character color, not chroma-key spill.
            preserved_emerald = g > 80 and b > 45 and r < 90 and (g - b) < 90

            if g > max(r, b) + 25 and g > 100 and not preserved_emerald:
                green_spill += 1

    spill_ratio = green_spill / max(1, visible)

    return {
        "size": [w, h],
        "bbox": list(bbox),
        "margins": margins,
        "green_spill_ratio": round(spill_ratio, 6),
        "passes": {
            "canvas_size": (w, h) == (1024, 2048),
            "padding": min(margins.values()) >= 40,
            "green_spill": spill_ratio < 0.001,
        },
    }


def save_webp(src: Path, dst: Path) -> dict:
    qa = qa_image(src)

    if not all(qa["passes"].values()):
        raise ValueError(f"{src} failed QA: {qa}")

    im = Image.open(src).convert("RGBA")
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, "WEBP", lossless=True, quality=100, method=6)

    alphamap = dst.with_suffix(f".alphamap{dst.suffix}")
    if alphamap.exists():
        white = Image.new("RGB", im.size, (255, 255, 255))
        white.save(alphamap, "WEBP", lossless=True, quality=100, method=6)
        qa["alphamap"] = str(alphamap)

    qa["output"] = str(dst)
    return qa


def apply_reimu(repo: Path) -> dict:
    return apply_character(repo, REIMU_MAP)


def apply_character(repo: Path, asset_map: dict[str, str]) -> dict:
    alpha_dir = repo / "build" / "ai-portrait-pipeline" / "alpha"
    dialog_dir = repo / "atlas" / "portraits" / "dialog"

    report = {}

    for target_name, source_name in asset_map.items():
        src = alpha_dir / source_name
        dst = dialog_dir / target_name

        if not src.exists():
            raise FileNotFoundError(src)

        report[target_name] = save_webp(src, dst)

    return report


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Apply AI-generated full-body portrait assets")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--character", choices=("reimu", "marisa", "youmu", "elly", "hina", "iku", "yumemi"), default="reimu")
    args = parser.parse_args(argv[1:])

    repo = args.repo.resolve()

    if args.character == "reimu":
        report = apply_reimu(repo)
    elif args.character == "marisa":
        report = apply_character(repo, MARISA_MAP)
    elif args.character == "youmu":
        report = apply_character(repo, YOUMU_MAP)
    elif args.character == "elly":
        report = apply_character(repo, ELLY_MAP)
    elif args.character == "hina":
        report = apply_character(repo, HINA_MAP)
    elif args.character == "iku":
        report = apply_character(repo, IKU_MAP)
    elif args.character == "yumemi":
        report = apply_character(repo, YUMEMI_MAP)
    else:
        raise AssertionError(args.character)

    report_path = repo / "build" / "ai-portrait-pipeline" / f"{args.character}_apply_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
