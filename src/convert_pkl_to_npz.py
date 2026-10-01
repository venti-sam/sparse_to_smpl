#!/usr/bin/env python3
"""
One-time script to convert SMPLH .pkl body models (which contain
chumpy objects) into clean .npz files that can be loaded by smplx
without chumpy installed.

Usage:
    python3 convert_pkl_to_npz.py

Requires: chumpy, numpy (run with Python ≤3.9 where chumpy works)
"""
import pickle
import numpy as np
import os
import glob


def convert_pkl(pkl_path):
    """Convert a single .pkl file to .npz, stripping chumpy objects."""
    print(f"Converting: {pkl_path}")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    clean = {}
    for key, val in data.items():
        if hasattr(val, "r"):
            # chumpy object → extract numpy array
            clean[key] = np.array(val.r)
        elif hasattr(val, "toarray"):
            # scipy sparse → dense
            clean[key] = val.toarray()
        elif isinstance(val, np.ndarray):
            clean[key] = val
        else:
            clean[key] = val

    npz_path = pkl_path.replace(".pkl", ".npz")
    np.savez(npz_path, **clean)
    print(f"  → Saved: {npz_path}")
    return npz_path


if __name__ == "__main__":
    body_model_dir = os.path.join(
        os.path.dirname(__file__), "..", "support_data", "body_models"
    )
    pkl_files = glob.glob(os.path.join(body_model_dir, "**/*.pkl"), recursive=True)

    # Filter out symlinks
    pkl_files = [f for f in pkl_files if not os.path.islink(f)]

    if not pkl_files:
        print("No .pkl files found!")
    else:
        print(f"Found {len(pkl_files)} .pkl files to convert.\n")
        for pkl_path in pkl_files:
            convert_pkl(pkl_path)
        print("\nDone! You can now use ext='npz' with smplx.create().")
