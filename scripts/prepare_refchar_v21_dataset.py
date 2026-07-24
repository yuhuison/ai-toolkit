#!/usr/bin/env python3
"""Materialize RefChar WebDataset shards into AI Toolkit folder layout."""

from __future__ import annotations

import argparse
import os
import tarfile
from pathlib import Path


MEMBER_TARGETS = {
    ".target.jpg": ("target", ".jpg"),
    ".source.jpg": ("ref0", ".jpg"),
    ".i2i.txt": ("target", ".txt"),
}


def destination_for(member_name: str, output: Path) -> Path | None:
    for suffix, (folder, extension) in MEMBER_TARGETS.items():
        if member_name.endswith(suffix):
            key = member_name[: -len(suffix)]
            return output / folder / f"{key}{extension}"
    return None


def write_member(source: tarfile.TarFile, member: tarfile.TarInfo, dest: Path) -> None:
    payload = source.extractfile(member)
    if payload is None:
        raise RuntimeError(f"Unable to read {member.name}")
    data = payload.read()
    temporary = dest.with_suffix(dest.suffix + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, dest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shards",
        type=Path,
        default=Path("/workspace/fusal-refchar-v20/shards"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/workspace/refchar_v21/dataset"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite existing files instead of validating and skipping them.",
    )
    args = parser.parse_args()

    shard_paths = sorted(args.shards.glob("*.tar"))
    if not shard_paths:
        raise SystemExit(f"No tar shards found in {args.shards}")

    (args.output / "target").mkdir(parents=True, exist_ok=True)
    (args.output / "ref0").mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    keys_by_kind: dict[str, set[str]] = {
        ".target.jpg": set(),
        ".source.jpg": set(),
        ".i2i.txt": set(),
    }

    for shard_path in shard_paths:
        with tarfile.open(shard_path, "r") as shard:
            for member in shard:
                dest = destination_for(member.name, args.output)
                if dest is None:
                    continue
                suffix = next(
                    item for item in MEMBER_TARGETS if member.name.endswith(item)
                )
                key = member.name[: -len(suffix)]
                if key in keys_by_kind[suffix]:
                    raise RuntimeError(f"Duplicate {suffix} key: {key}")
                keys_by_kind[suffix].add(key)

                if dest.exists() and not args.overwrite:
                    if dest.stat().st_size != member.size:
                        raise RuntimeError(
                            f"Existing file size mismatch: {dest} "
                            f"({dest.stat().st_size} != {member.size})"
                        )
                    skipped += 1
                    continue

                write_member(shard, member, dest)
                written += 1

        print(f"Processed {shard_path.name}", flush=True)

    target_keys = keys_by_kind[".target.jpg"]
    source_keys = keys_by_kind[".source.jpg"]
    caption_keys = keys_by_kind[".i2i.txt"]
    if not (target_keys == source_keys == caption_keys):
        raise SystemExit(
            "Dataset key mismatch: "
            f"target={len(target_keys)} source={len(source_keys)} "
            f"caption={len(caption_keys)}"
        )
    empty_captions = [
        key
        for key in caption_keys
        if not (args.output / "target" / f"{key}.txt")
        .read_text(encoding="utf-8")
        .strip()
    ]
    if empty_captions:
        raise SystemExit(f"Empty captions found: {len(empty_captions)}")

    print(
        f"Ready: {len(target_keys)} pairs; written={written}; skipped={skipped}; "
        f"output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
