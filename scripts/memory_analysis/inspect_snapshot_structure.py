#!/usr/bin/env python3
"""Quick script to inspect the structure of memory snapshot data."""

import pickle
import sys
from pathlib import Path


def inspect_snapshot(path: Path):
    """Inspect snapshot structure."""
    print(f"Loading: {path}")

    with open(path, "rb") as f:
        snapshot = pickle.load(f)

    print(f"\nTop-level keys: {list(snapshot.keys())}")

    if "segments" in snapshot:
        print(f"\nNumber of segments: {len(snapshot['segments'])}")
        if snapshot["segments"]:
            seg = snapshot["segments"][0]
            print(f"First segment keys: {list(seg.keys())}")

            if "blocks" in seg and seg["blocks"]:
                block = seg["blocks"][0]
                print(f"\nFirst block keys: {list(block.keys())}")
                print(f"Block state: {block.get('state', 'N/A')}")
                print(f"Block size: {block.get('size', 0) / 1024**3:.2f} GB")

                frames = block.get("frames", [])
                print(f"\nNumber of frames: {len(frames)}")
                if frames:
                    print(f"Frame structure: {list(frames[0].keys())}")
                    print(f"\nFirst 10 frames:")
                    for i, frame in enumerate(frames[:10]):
                        print(
                            f"  {i}: {frame['filename']}:{frame['line']} {frame['name']}"
                        )

    # Check for other metadata
    if "device_traces" in snapshot:
        print(f"\nHas device_traces: {len(snapshot['device_traces'])} entries")

    if "device_allocator_info" in snapshot:
        print(f"Has device_allocator_info")

    # Look for categorized data
    for key in snapshot.keys():
        if "category" in key.lower() or "trace" in key.lower():
            print(f"\nFound key: {key}")
            print(f"  Type: {type(snapshot[key])}")
            if isinstance(snapshot[key], (list, dict)):
                print(f"  Length/Size: {len(snapshot[key])}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_snapshot_structure.py <snapshot.pickle>")
        sys.exit(1)

    inspect_snapshot(Path(sys.argv[1]))
