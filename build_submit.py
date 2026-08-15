#!/usr/bin/env python3
"""Build a minimal submit.zip with only the allowed top-level entries."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def collect_files(root: Path, model_path: Path | None = None) -> list[tuple[Path, Path]]:
    required = [root / "script.py", root / "requirements.txt"]
    missing = [str(path) for path in required if not path.is_file()]
    model_dir = root / "model"
    # Keep the ZIP minimal and mirror script.py's model preference.
    preferred_model = model_path if model_path is not None else model_dir / "final_model.pkl"
    legacy_model = model_dir / "rf.pkl"
    support_files = sorted(
        path for path in model_dir.rglob("*")
        if path.is_file() and path.suffix != ".pkl" and "__pycache__" not in path.parts
    )
    if preferred_model.is_file():
        model_files = [preferred_model, *support_files]
    elif legacy_model.is_file():
        model_files = [legacy_model, *support_files]
    else:
        model_files = []
    if not model_files:
        missing.append(str(model_dir / "<model file>"))
    if missing:
        raise FileNotFoundError(f"missing submission inputs: {missing}")
    paths = required + model_files
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"symlinks are not allowed in submission: {path}")
    collected = []
    for path in paths:
        if path == preferred_model:
            collected.append((path, Path("model/final_model.pkl")))
        else:
            collected.append((path, path.relative_to(root)))
    return collected


def build_zip(root: Path, output: Path, model_path: Path | None = None) -> None:
    files = collect_files(root, model_path)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for source, relative in files:
            archive.write(source, relative.as_posix())
    with zipfile.ZipFile(output) as archive:
        roots = {Path(name).parts[0] for name in archive.namelist()}
        if roots != {"model", "script.py", "requirements.txt"}:
            raise ValueError(f"unexpected ZIP roots: {sorted(roots)}")
        bad = archive.testzip()
        if bad:
            raise ValueError(f"corrupt ZIP member: {bad}")
    print(f"built {output} ({output.stat().st_size / 1024 / 1024:.2f} MiB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("submit.zip"))
    parser.add_argument("--model-path", type=Path)
    args = parser.parse_args()
    model_path = args.model_path.resolve() if args.model_path is not None else None
    build_zip(args.root.resolve(), args.output.resolve(), model_path)


if __name__ == "__main__":
    main()
