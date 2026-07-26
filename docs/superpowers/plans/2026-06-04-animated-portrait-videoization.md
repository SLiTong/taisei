# Animated Portrait Videoization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build engine-native animated dialog portraits with static fallback.

**Architecture:** Reuse Taisei `.ani` resources next to existing dialog sprite resources. Add frame-aware portrait rendering and a generator for transparent portrait frame loops.

**Tech Stack:** C, Taisei resource system, Python, Pillow, NumPy, Meson atlas targets.

---

### Task 1: Frame-Aware Portrait Rendering

**Files:**
- Modify: `src/portrait.c`
- Modify: `src/portrait.h`
- Modify: `src/dialog.h`
- Modify: `src/dialog.c`

- [ ] Add optional animation lookup using the existing static portrait resource name.
- [ ] Add `portrait_render_byname_frame(charname, variant, face, frame, out)` and keep `portrait_render_byname` as frame-zero compatibility wrapper.
- [ ] Add `composite_anim_frame` to `DialogActor`.
- [ ] In `dialog_draw`, compute a 12 FPS animation frame from `global.frames`, mark the composite dirty when it changes, and render with `portrait_render_byname_frame`.
- [ ] Build with `meson compile -C build/windows-mingw`.

### Task 2: Dialog Portrait Animation Assets

**Files:**
- Create: `scripts/generate-dialog-portrait-videos.py`
- Create/update: `atlas/portraits/dialog/*.frame0000.webp` through `*.frame0007.webp`
- Create/update: `atlas/config/dialog/*.framegroup.spr`
- Create/update: `resources/00-taisei.pkgdir/gfx/dialog/*.ani`

- [ ] Generate 8 transparent frames for each full-body portrait and variant.
- [ ] Use consistent canvas size and copied padding config per source portrait.
- [ ] Write each `.ani` as `@sprite_count = 8` and `main = 0 1 2 3 4 5 6 7 6 5 4 3 2 1`.
- [ ] Run `python scripts/generate-dialog-portrait-videos.py --write`.
- [ ] Run `meson compile -C build/windows-mingw gen-atlas-portraits-fast`.

### Task 3: Runtime Verification

**Files:**
- Create/update: `build/playtest/animated-dialog-*`
- Create: `scripts/verify-dialog-portrait-videos.py`

- [ ] Run the game with `TAISEI_SKIP_TO_DIALOG=1` and framedump enabled.
- [ ] Capture at least two frames from a live dialog sequence.
- [ ] Compare portrait-region pixels and record the difference ratio.
- [ ] Run `python scripts/verify-dialog-portrait-videos.py` and require zero failures.
- [ ] Re-run `meson compile -C build/windows-mingw`.
