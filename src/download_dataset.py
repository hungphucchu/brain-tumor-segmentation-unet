from __future__ import annotations
import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download mateuszbuda/lgg-mri-segmentation with kagglehub."
    )
    parser.add_argument(
        "--dest-name",
        default="lgg_mri_kagglehub",
        help="Name of symlink (or junction) under data/raw/ pointing at the download.",
    )
    parser.add_argument(
        "--no-link",
        action="store_true",
        help="Only download and print path; do not create data/raw/<dest-name>.",
    )
    args = parser.parse_args()

    try:
        import kagglehub
    except ImportError as e:
        print("Missing package: pip install kagglehub", file=sys.stderr)
        raise SystemExit(1) from e

    repo_root = Path.cwd()
    print("Downloading mateuszbuda/lgg-mri-segmentation (this may take a while)...")
    path = Path(kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation"))
    path = path.resolve()
    print("Path to dataset files:", path)
    link_target = path / "kaggle_3m"
    if link_target.is_dir():
        print("Linking via kaggle_3m/ (slice folders live here).")
    else:
        link_target = path

    if args.no_link:
        print("\nDefault config uses data/raw. Either:")
        print(f"  python -m src.eda --config configs/default.yaml --data-root {link_target}")
        print("or set data_root in configs/default.yaml to the path above.")
        return

    raw_dir = repo_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    link_path = raw_dir / args.dest_name
    target = link_target.resolve()

    if link_path.is_symlink():
        if link_path.resolve() == target:
            print(f"Symlink already points here: {link_path} -> {target}")
        else:
            link_path.unlink()
            link_path.symlink_to(target, target_is_directory=True)
            print(f"Updated symlink {link_path} -> {target}")
    elif link_path.exists():
        print(
            f"Refusing to overwrite existing path (not our symlink): {link_path}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    else:
        try:
            link_path.symlink_to(target, target_is_directory=True)
            print(f"Created symlink {link_path} -> {target}")
        except OSError as err:
            print(f"Could not create symlink ({err}).", file=sys.stderr)
            print("On Windows, enable Developer Mode or run as admin for symlinks.", file=sys.stderr)
            print(f"Use this path with --data-root instead:\n  {link_target}")
            raise SystemExit(1) from err

    from .data_files import discover_pairs

    n = len(discover_pairs(raw_dir))
    print(f"Found {n} image/mask pairs under {raw_dir} (recursive).")
    if n == 0:
        kg = path / "kaggle_3m"
        hint = kg if kg.is_dir() else path
        print(
            "No pairs matched *_mask.tif layout under data/raw. Try:\n"
            f"  python -m src.eda --config configs/default.yaml --data-root {hint}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
