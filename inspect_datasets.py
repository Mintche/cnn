"""Inspect dataset adapters before implementing or training a CNN."""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__:
    from .data import load_dataset
else:
    from data import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["digits", "hdf5", "npz", "mnist", "fashion_mnist", "cifar10"],
        default="digits",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="HDF5 directory or NPZ file, depending on --dataset.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, args.data_dir, seed=args.seed)
    print(f"Input shape: {dataset.input_shape}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Train:      {dataset.train.size:>6} samples")
    print(f"Validation: {dataset.validation.size:>6} samples")
    print(f"Test:       {dataset.test.size:>6} samples")


if __name__ == "__main__":
    main()
