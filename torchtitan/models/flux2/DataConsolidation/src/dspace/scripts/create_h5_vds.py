import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import h5py


def _read_csv_paths(csv_path: Path) -> List[str]:
    with open(csv_path, "r") as fh:
        reader = csv.reader(fh)
        items = [row[0].strip() for row in reader if row and row[0].strip()]

    return items


def _probe_and_validate(
    files: List[str], keys: Tuple[str, ...]
) -> Tuple[Dict[str, dict], List[Tuple[str, int]]]:
    """
    Returns:
        key_meta: { key: {"dtype": np.dtype, "trailing": tuple[int, ...]} }
        per_file_lengths: list of (file_path, length) where length is along axis 0 for the first key
    """
    if not files:
        raise ValueError("No h5 shard files provided.")

    key_meta: Dict[str, dict] = {}
    with h5py.File(files[0], "r") as f0:
        for key in keys:
            ds = f0[key]
            print(ds.dtype)
            key_meta[key] = {"dtype": ds.dtype, "trailing": ds.shape[1:]}

    per_file_lengths: List[Tuple[str, int]] = []
    for fp in files:
        with h5py.File(fp, "r") as f:
            lens = []
            for key in keys:
                ds = f[key]
                if ds.shape[1:] != key_meta[key]["trailing"]:
                    raise ValueError(
                        f"Trailing shape mismatch for '{key}' in file '{fp}':"
                        f"expected {key_meta[key]['trailing']}, got {ds.shape[1:]}."
                    )
                if ds.dtype != key_meta[key]["dtype"]:
                    raise ValueError(
                        f"dtype mismatch for '{key}' in file '{fp}':"
                        f"expected {key_meta[key]['dtype']}, got {ds.dtype}."
                    )
                lens.append(ds.shape[0])
            # all ds len should be equal in a shard
            if len(set(lens)) != 1:
                raise ValueError(
                    f"Axis-0 lengths differ across keys in {fp}: {dict(zip(keys, lens))}"
                )

            per_file_lengths.append((fp, lens[0]))

    return key_meta, per_file_lengths


def _build_vds(
    out_path: Path,
    keys: Tuple[str, ...],
    key_meta: Dict[str, dict],
    per_file_lengths: List[Tuple[str, int]],
) -> Dict[str, dict]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, dict | list] = {"keys": {}, "files": []}
    offsets = {}
    totals = {}

    # Precompute global offsets per file (based on first key)
    off = 0
    for fp, n in per_file_lengths:
        manifest["files"].append({"path": fp, "length": n, "offset": off})
        offsets[fp] = off
        off += n
    total_len = off

    with h5py.File(out_path, "w", libver="latest") as fout:
        for key in keys:
            trailing = key_meta[key]["trailing"]
            dtype = key_meta[key]["dtype"]
            layout = h5py.VirtualLayout(shape=(total_len,) + trailing, dtype=dtype)

            # Map each file's slice
            for fp, n in per_file_lengths:
                vs = h5py.VirtualSource(fp, f"/{key}", shape=(n,) + trailing)
                a = offsets[fp]
                layout[a : a + n] = vs

            fout.create_virtual_dataset(f"/{key}", layout)
            totals[key] = total_len
            manifest["keys"][key] = {
                "dtype": str(dtype),
                "trailing_shape": list(trailing),
                "total_length": total_len,
            }

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Create a VDS that concatenates identical datasets across many HDF5 shards (axis 0)"
    )
    parser.add_argument(
        "--csv",
        required=True,
        type=Path,
        help="csv with one HDF5 file path per line (absolute paths)",
    )
    parser.add_argument(
        "--keys",
        nargs="+",
        default=["image_bytes", "captions", "filenames"],
        help="Dataset keys to concatenate: (default: image_bytes, captions, filenames)",
    )
    parser.add_argument(
        "--out", required=True, type=Path, help="Output VDS path, e.g. merged_vds.h5"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional path to write JSON manifest with file lengths and offsets",
    )

    args = parser.parse_args()

    files = _read_csv_paths(args.csv)
    # Validate against the dataset layout
    key_meta, per_file_lengths = _probe_and_validate(files, args.keys)
    manifest = _build_vds(args.out, tuple(args.keys), key_meta, per_file_lengths)

    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with open(args.manifest, "w") as fh:
            json.dump(manifest, fh, indent=2)

    print(f"[OK] Wrote VDS to {args.out}")
    print("Summary:")
    for key in args.keys:
        info = manifest["keys"][key]
        print(
            f"  /{key}: total: {info["total_length"]} trailing_shape={tuple(info["trailing_shape"])}"
        )


if __name__ == "__main__":
    main()
