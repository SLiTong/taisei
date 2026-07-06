#!/usr/bin/env python3
"""
Audit dialogue redraw portrait coverage.

This checks the dialogue scripts for actor/face/variant combinations and verifies
that matching *_redraw_*.webp sources plus generated .spr resources exist.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


TASK_RE = re.compile(
    r"DIALOG_TASK\((\w+),\s*(\w+)\)\s*\{"
    r"(?P<body>.*?)"
    r"(?=\n\}\s*\n\n/\*|\n\}\s*\n\s*/\*\n \* Register|\Z)",
    re.S,
)


def record_combo(combos, faces_by_char, variants_by_char, task_combos, task, actor_name, actors):
    if actor_name not in actors:
        return

    state = actors[actor_name]
    combo = (state["char"], state.get("variant"), state.get("face") or "normal")
    combos.add(combo)
    faces_by_char.setdefault(combo[0], set()).add(combo[2])

    if combo[1]:
        variants_by_char.setdefault(combo[0], set()).add(combo[1])

    task_combos.setdefault(task, set()).add(combo)


def scan_dialog_scripts(root: Path):
    combos = set()
    faces_by_char = {}
    variants_by_char = {}
    task_combos = {}

    for path in sorted((root / "src" / "dialog").glob("*.c")):
        text = path.read_text(encoding="utf-8", errors="replace")

        for match in TASK_RE.finditer(text):
            protagonist, task_name = match.group(1), match.group(2)
            task_id = f"{protagonist}_{task_name}"
            actors = {}

            for line in match.group("body").splitlines():
                code = line.split("//", 1)[0]

                for actor_match in re.finditer(r"ACTOR_(?:LEFT|RIGHT)\((\w+)\)", code):
                    name = actor_match.group(1)
                    actors[name] = {
                        "char": name,
                        "face": "normal",
                        "variant": None,
                    }
                    record_combo(combos, faces_by_char, variants_by_char, task_combos, task_id, name, actors)

                for variant_match in re.finditer(r"VARIANT\((\w+),\s*(\w+)\)", code):
                    name, variant = variant_match.groups()
                    if name in actors:
                        actors[name]["variant"] = variant
                        record_combo(combos, faces_by_char, variants_by_char, task_combos, task_id, name, actors)

                for face_match in re.finditer(r"FACE\((\w+),\s*(\w+)\)", code):
                    name, face = face_match.groups()
                    if name in actors:
                        actors[name]["face"] = face
                        record_combo(combos, faces_by_char, variants_by_char, task_combos, task_id, name, actors)

                for message_match in re.finditer(r"MSG(?:_UNSKIPPABLE)?\((\w+)\s*,", code):
                    record_combo(combos, faces_by_char, variants_by_char, task_combos, task_id, message_match.group(1), actors)

    return combos, faces_by_char, variants_by_char, task_combos


def audit(root: Path):
    combos, faces_by_char, variants_by_char, task_combos = scan_dialog_scripts(root)

    webps = {p.stem for p in (root / "atlas" / "portraits" / "dialog").glob("*_redraw*.webp")}
    sprs = {p.stem for p in (root / "resources" / "00-taisei.pkgdir" / "gfx" / "dialog").glob("*_redraw*.spr")}
    installed_sprs = {
        p.stem
        for p in (root / "dist" / "windows-clean" / "data" / "00-taisei.pkgdir" / "gfx" / "dialog").glob("*_redraw*.spr")
    }

    missing_base_webp = []
    missing_exact_webp = []
    missing_required_spr = []
    missing_installed_spr = []
    variant_combos = set()

    for char, variant, face in sorted(combos, key=lambda item: (item[0], item[1] or "", item[2])):
        base = f"{char}_redraw_{face}"

        if base not in webps:
            missing_base_webp.append(base)
        if base not in sprs:
            missing_required_spr.append(base)
        if installed_sprs and base not in installed_sprs:
            missing_installed_spr.append(base)

        if variant:
            variant_combos.add((char, variant, face))
            exact = f"{char}_redraw_{variant}_{face}"

            if exact not in webps:
                missing_exact_webp.append(exact)
            if exact not in sprs:
                missing_required_spr.append(exact)
            if installed_sprs and exact not in installed_sprs:
                missing_installed_spr.append(exact)

    return {
        "dialog_combo_count": len(combos),
        "variant_combo_count": len(variant_combos),
        "redraw_webp_count": len(webps),
        "redraw_spr_count": len(sprs),
        "installed_redraw_spr_count": len(installed_sprs),
        "missing_base_webp": sorted(set(missing_base_webp)),
        "missing_exact_webp": sorted(set(missing_exact_webp)),
        "missing_required_spr": sorted(set(missing_required_spr)),
        "missing_installed_spr": sorted(set(missing_installed_spr)),
        "faces_by_char": {key: sorted(value) for key, value in sorted(faces_by_char.items())},
        "variants_by_char": {key: sorted(value) for key, value in sorted(variants_by_char.items())},
        "tasks": {
            key: [
                {"char": char, "variant": variant, "face": face}
                for char, variant, face in sorted(value, key=lambda item: (item[0], item[1] or "", item[2]))
            ]
            for key, value in sorted(task_combos.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--no-tasks", action="store_true", help="Omit per-task details from JSON output")
    args = parser.parse_args()

    result = audit(args.root.resolve())

    if args.no_tasks:
        result = dict(result)
        result.pop("tasks", None)

    print(json.dumps(result, indent=2, ensure_ascii=False))

    failed = any(
        result[key]
        for key in (
            "missing_base_webp",
            "missing_exact_webp",
            "missing_required_spr",
            "missing_installed_spr",
        )
    )

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
