# Animated Portrait Videoization Design

## Goal

Replace static dialog portraits with engine-native looping animated portrait clips while keeping static sprites as fallbacks.

## Acceptance Target

- Dialog portraits visibly change texture frames over time, not only position or scale.
- Emotion variants remain distinct full-body images, so face, pose, costume, and mood effects can differ by dialogue state.
- All animated portraits keep transparent backgrounds, stable dimensions, and complete full-body framing.
- Existing sprite resources keep working in menus, cut-ins, and preload code.
- Verification includes a successful Windows build plus runtime frame captures with measurable pixel differences in portrait regions.

## Architecture

Taisei already supports `.ani` resources whose frames map to `dialog/<name>.frame0000.spr` sprite resources. The portrait module will prefer an optional animation resource with the same base name as the static sprite, sample the `main` sequence by frame number, and fall back to the static sprite when no animation exists.

Dialog actors currently cache a rendered composite sprite until the character face or variant changes. The cache key will gain an animation frame slot, so animated portraits are recomposited only when the sampled frame changes.

To avoid destroying and recreating the composite texture every few frames, each dialog actor owns a small composite frame cache. Face or variant changes invalidate the cache; repeated animation loops reuse the already-composited frame textures.

## Asset Pipeline

`scripts/generate-dialog-portrait-videos.py` will generate short transparent WebP frame loops for every full-body dialog portrait and full-body emotion variant under `atlas/portraits/dialog`. The script will also write matching `.ani` files to `resources/00-taisei.pkgdir/gfx/dialog` and matching atlas config `.spr` files so `gen-atlas-portraits-fast` can produce frame sprites.

The generated motion is intentionally non-explicit: slow breathing, body sway, small hair/cloth deformation, shimmer highlights, and mood particles. Characters that are childlike or ambiguous remain non-sexualized.

## Test Plan

- Run `python scripts/generate-dialog-portrait-videos.py --write`.
- Run the portraits atlas target to regenerate `portraits*.tsatlas.zst` and dialog frame `.spr` files.
- Run `meson compile -C build/windows-mingw`.
- Run stage dialog framedumps with `TAISEI_SKIP_TO_DIALOG=1`.
- Compare portrait-region screenshots from two different frames and require a non-trivial pixel difference.
- For strict regression checks, set `TAISEI_DIALOG_PORTRAIT_STATIC_POSE=1` to disable draw-layer breathing and bobbing. The portrait region must still show pixel differences from animated source frames.
- Run `python scripts/verify-dialog-portrait-videos.py`; it must report zero missing clip resources and zero clips below the minimum frame-difference threshold.
