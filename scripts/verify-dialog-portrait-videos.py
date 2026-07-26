#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
_bundled_python = _repo_root / "build" / "python-packages"
if _bundled_python.is_dir():
    sys.path.insert(0, str(_bundled_python))

from PIL import Image, ImageChops, ImageStat


FRAME_COUNT = 8
MIN_FRAME_DIFF = 0.010


def is_fullbody_portrait(path):
    name = path.name
    return (
        path.suffix.lower() == ".webp"
        and ".alphamap." not in name
        and "_face_" not in name
        and "_misc_" not in name
        and ".frame" not in name
    )


def mean_abs_diff(a, b):
    diff = ImageChops.difference(a.convert("RGBA"), b.convert("RGBA"))
    stat = ImageStat.Stat(diff)
    return sum(stat.mean) / (4 * 255)


def verify_clip(repo, source):
    stem = source.stem
    dialog_dir = repo / "atlas" / "portraits" / "dialog"
    resource_dir = repo / "resources" / "00-taisei.pkgdir" / "gfx" / "dialog"
    frame_paths = [dialog_dir / f"{stem}.frame{i:04d}.webp" for i in range(FRAME_COUNT)]
    spr_paths = [resource_dir / f"{stem}.frame{i:04d}.spr" for i in range(FRAME_COUNT)]
    ani_path = resource_dir / f"{stem}.ani"

    missing = [str(p.relative_to(repo)) for p in [ani_path, *frame_paths, *spr_paths] if not p.exists()]
    frame_diff = None
    size = None

    if not missing:
        with Image.open(frame_paths[0]) as a, Image.open(frame_paths[FRAME_COUNT // 2]) as b:
            size = a.size
            frame_diff = mean_abs_diff(a, b)

    return {
        "clip": stem,
        "size": size,
        "frame0_mid_diff": frame_diff,
        "missing": missing,
        "passes": not missing and frame_diff is not None and frame_diff >= MIN_FRAME_DIFF,
    }


def main():
    parser = argparse.ArgumentParser(description="Verify generated dialog portrait video clips.")
    parser.add_argument("--repo-root", type=Path, default=_repo_root)
    parser.add_argument("--out", type=Path, default=Path("build/playtest/animated-dialog-video-verification.json"))
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    sources = sorted(p for p in (repo / "atlas" / "portraits" / "dialog").glob("*.webp") if is_fullbody_portrait(p))
    clips = [verify_clip(repo, p) for p in sources]
    failures = [c for c in clips if not c["passes"]]

    summary = {
        "source_count": len(sources),
        "clip_count": len(clips),
        "frame_count_per_clip": FRAME_COUNT,
        "min_required_frame_diff": MIN_FRAME_DIFF,
        "min_observed_frame_diff": min(c["frame0_mid_diff"] for c in clips if c["frame0_mid_diff"] is not None),
        "failure_count": len(failures),
        "failures": failures,
        "clips": clips,
    }

    out = args.out
    if not out.is_absolute():
        out = repo / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({k: summary[k] for k in summary if k != "clips"}, indent=2))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
