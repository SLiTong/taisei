# Character Redraw Pipeline

This branch replaces Taisei character-facing atlas assets with a layout-locked
modern anime repaint pass. The goal is to move the existing characters toward a
cleaner office-fashion / polished anime look while preserving all runtime
contracts: filenames, dimensions, alpha channels, sprite anchors, expression
overlay alignment, and atlas resource names.

## Scope

The current character asset scope is:

- `atlas/common/player`: player animation frames
- `atlas/common/boss`: boss animation frames
- `atlas/common/enemy`: humanoid/fairy enemy frames
- `atlas/portraits/dialog`: dialog base portraits, variants, expression layers,
  and portrait misc overlays

The following are intentionally excluded:

- `.alphamap` files, because they are companion masks used by the atlas
  generator and must stay dimensionally paired with their base images
- `atlas/common/enemy/swirl.png`, because it is an abstract effect glyph rather
  than a character illustration
- bullets, items, UI, spell circles, stage effects, and other non-character art

## Commands

Generate a manifest and preview without modifying assets:

```powershell
$env:PYTHONPATH = (Resolve-Path 'build\python-packages').Path
python scripts\apply-character-style-pass.py `
  --mode redraw `
  --manifest build\character-redraw-preview\character-assets-manifest.csv `
  --preview build\character-redraw-preview\character-redraw-preview.png
```

Apply the repaint pass to all in-scope source images:

```powershell
$env:PYTHONPATH = (Resolve-Path 'build\python-packages').Path
python scripts\apply-character-style-pass.py --mode redraw --write
```

Regenerate the affected atlases:

```powershell
$env:PYTHONPATH = (Resolve-Path 'build\python-packages').Path
python scripts\gen-atlas.py atlas\config atlas\portraits resources\00-taisei.pkgdir\gfx --border=2 --tsatlas-sort=perimeter --format tsatlas --width=512 --height=512 --samples=256 --single --tsatlas-zstd-level=22
python scripts\gen-atlas.py atlas\config atlas\common resources\00-taisei.pkgdir\gfx --border=2 --format tsatlas --width=512 --height=512 --samples=256 --single --tsatlas-zstd-level=22
```

## Quality Gates

- Every target image keeps its original width and height.
- Every target image keeps an alpha channel.
- No target filename, path, or sprite config name changes.
- Dialog face overlays remain visually aligned with their base portraits.
- `common.tsatlas.zst` and `portraits.tsatlas.zst` regenerate successfully.
- `meson compile -C build\windows-mingw` succeeds.
- The installed Windows build launches from `build-test-windows`.
- A real playtest shows player, enemy, boss, and dialog/portrait assets without
  missing sprites, broken alpha, or atlas coordinate errors.
