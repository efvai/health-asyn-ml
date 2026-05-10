"""
float_conv.py — Check float64→float32 quantization error for a dataset and
optionally reconstruct it as float32.

Usage:
    python utils/float_conv.py data_set_4
    python utils/float_conv.py          # will prompt for path
"""

import os
import sys
import shutil
import numpy as np
from pathlib import Path


def compute_max_error(dataset_path: Path) -> tuple[float, int]:
    """Return (global max abs error, number of .dat files scanned)."""
    dat_files = list(dataset_path.rglob("*.dat"))
    if not dat_files:
        return 0.0, 0

    global_max = 0.0
    for f in dat_files:
        data = np.fromfile(f, dtype=np.float64)
        if data.size == 0:
            continue
        err = np.abs(data - data.astype(np.float32).astype(np.float64)).max()
        if err > global_max:
            global_max = err

    return global_max, len(dat_files)


def reconstruct_float32(src: Path, dst: Path) -> int:
    """Mirror src tree into dst, converting .dat files to float32.

    Returns the number of .dat files written.
    """
    converted = 0
    for root, dirs, files in os.walk(src):
        root_path = Path(root)
        rel = root_path.relative_to(src)
        out_dir = dst / rel
        out_dir.mkdir(parents=True, exist_ok=True)

        for fname in files:
            src_file = root_path / fname
            dst_file = out_dir / fname

            if fname.lower().endswith(".dat"):
                data = np.fromfile(src_file, dtype=np.float64)
                data.astype(np.float32).tofile(dst_file)
                converted += 1
            else:
                shutil.copy2(src_file, dst_file)

    return converted


def main() -> None:
    # --- resolve input path ---
    if len(sys.argv) > 1:
        raw_path = sys.argv[1]
    else:
        raw_path = input("Dataset path: ").strip()

    dataset_path = Path(raw_path).resolve()
    if not dataset_path.is_dir():
        print(f"Error: '{dataset_path}' is not a directory.")
        sys.exit(1)

    # --- scan and report error ---
    print(f"Scanning '{dataset_path}' ...")
    max_err, n_files = compute_max_error(dataset_path)

    if n_files == 0:
        print("No .dat files found.")
        sys.exit(0)

    print(f"  .dat files scanned : {n_files}")
    print(f"  Max |double - float32| error : {max_err:.6e}")

    # --- prompt user ---
    answer = input("Reconstruct as float32? [Y/N]: ").strip().upper()
    if answer != "Y":
        print("Aborted — no files written.")
        return

    # --- derive output path ---
    # Works for both absolute and relative inputs; keep the original parent.
    src_abs = Path(raw_path).resolve()
    out_path = src_abs.parent / (src_abs.name + "_float32")

    if out_path.exists():
        print(f"Output directory already exists: '{out_path}'")
        overwrite = input("Overwrite? [Y/N]: ").strip().upper()
        if overwrite != "Y":
            print("Aborted.")
            return
        shutil.rmtree(out_path)

    print(f"Writing float32 dataset to '{out_path}' ...")
    converted = reconstruct_float32(src_abs, out_path)
    print(f"Done. {converted} .dat file(s) converted.")


if __name__ == "__main__":
    main()
