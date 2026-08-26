#!/usr/bin/env python3
"""Generate SHA-256 manifest for the FAIR-X IJIS revision artifact.

The manifest intentionally excludes itself and raw datasets. It covers the
paper-facing canonical tables, their upstream revision audits, executable
revision scripts/workflows, the pinned environment, and retained external or
secondary result directories.
"""
from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ARTIFACT_MANIFEST.md"

EXACT = [
    ROOT / "README.md",
    ROOT / "requirements.txt",
]
TREES = [
    ROOT / ".github" / "workflows",
    ROOT / "scripts",
    ROOT / "results" / "canonical",
    ROOT / "results" / "revision_audits",
    ROOT / "results" / "external_malbehavd",
    ROOT / "results" / "external_malbehavd_temporal",
    ROOT / "results" / "beth_limit_lifting",
    ROOT / "results" / "robust_threshold_validation",
]
SKIP_NAMES = {"ARTIFACT_MANIFEST.md", ".DS_Store"}
SKIP_SUFFIXES = {".pyc", ".pkl", ".pt", ".pth"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def selected_files() -> list[Path]:
    files: set[Path] = set()
    for path in EXACT:
        if path.is_file():
            files.add(path)
    for tree in TREES:
        if not tree.exists():
            continue
        for path in tree.rglob("*"):
            if not path.is_file() or path.name in SKIP_NAMES:
                continue
            if path.suffix.lower() in SKIP_SUFFIXES:
                continue
            files.add(path)
    # The generator is part of the manifest, but the manifest itself is not.
    files.add(Path(__file__).resolve())
    return sorted(files, key=lambda p: p.relative_to(ROOT).as_posix())


def main() -> None:
    files = selected_files()
    lines = [
        "# FAIR-X / FAIR-BETH Artifact Manifest",
        "",
        f"Generated: {date.today().isoformat()}",
        "",
        "This manifest records SHA-256 digests for revision-critical executable and machine-readable artifacts. The scientific manuscript is intentionally kept outside this code repository. Raw BETH and MalBehavD datasets are not redistributed and are therefore not hashed here; the BETH raw-file identity audit is recorded separately in `results/canonical/dataset_inventory.csv`.",
        "",
        "The manifest excludes itself to avoid a self-referential digest. Binary model checkpoints and transient pickle artifacts are also excluded; canonical claims are derived from committed text/CSV/JSON outputs.",
        "",
        "| File | SHA-256 |",
        "|---|---|",
    ]
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        lines.append(f"| `{rel}` | `{digest(path)}` |")
    lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT} with {len(files)} entries")


if __name__ == "__main__":
    main()
