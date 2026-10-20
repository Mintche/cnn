"""Dataset adapters and batch iteration for the JAX CNN experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import h5py
import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class DatasetSplit:
    """One dataset split using channels-last image arrays."""

    images: np.ndarray
    labels: np.ndarray

    @property
    def size(self) -> int:
        return int(self.images.shape[0])


@dataclass(frozen=True)
class DatasetBundle:
    """Train, validation, and test data plus their shape metadata."""

    train: DatasetSplit
    validation: DatasetSplit
    test: DatasetSplit
    num_classes: int
    input_shape: tuple[int, int, int]
    class_names: tuple[str, ...] = ()


def _prepare_images(images: np.ndarray) -> np.ndarray:
    """Convert [0, 1] or byte-range images to float32 channels-last arrays."""

    images = np.asarray(images)
    if images.ndim == 3:
        images = images[..., np.newaxis]
    if images.ndim != 4:
        raise ValueError("Images must have shape (N, H, W) or (N, H, W, C)")

    images = images.astype(np.float32)
    if images.size:
        minimum = float(images.min())
        maximum = float(images.max())
        if not np.isfinite(minimum) or not np.isfinite(maximum):
            raise ValueError("Images must contain only finite values")
        if minimum < 0 or maximum > 255:
            raise ValueError("Images must use the [0, 1] or [0, 255] value range")
        if maximum > 1:
            images /= 255.0
    return images


def _prepare_labels(labels: np.ndarray, n_samples: int) -> np.ndarray:
    """Convert class indices or one-hot labels to a one-dimensional array."""

    labels = np.asarray(labels)
    if labels.ndim == 1:
        result = labels
    elif labels.ndim == 2 and 1 in labels.shape:
        result = labels.reshape(-1)
    elif labels.ndim == 2:
        axis = 1 if labels.shape[0] == n_samples else 0
        result = np.argmax(labels, axis=axis)
    else:
        raise ValueError("Labels must be a one-dimensional or two-dimensional array")

    result = np.asarray(result, dtype=np.int32).reshape(-1)
    if result.shape[0] != n_samples:
        raise ValueError("The number of labels must match the number of images")
    return result


def _make_split(images: np.ndarray, labels: np.ndarray) -> DatasetSplit:
    images = _prepare_images(images)
    labels = _prepare_labels(labels, images.shape[0])
    return DatasetSplit(images=images, labels=labels)

def _train_test_split(
    images: np.ndarray,
    labels: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    np.random.seed(seed)
    n = images.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)

    shuffled_images = images[indices]
    shuffled_labels = labels[indices]
    
    split_idx = int(n * (1.0 - validation_fraction))
    
    train_images = shuffled_images[:split_idx]
    validation_images = shuffled_images[split_idx:]
    train_labels = shuffled_labels[:split_idx]
    validation_labels = shuffled_labels[split_idx:]
    
    return train_images, validation_images, train_labels, validation_labels


def _split_train_validation(
    images: np.ndarray,
    labels: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> tuple[DatasetSplit, DatasetSplit]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    train_images, validation_images, train_labels, validation_labels = _train_test_split(
        images,
        labels,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    return _make_split(train_images, train_labels), _make_split(
        validation_images, validation_labels
    )


def _bundle(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    validation_fraction: float,
    seed: int,
    class_names: tuple[str, ...] = (),
) -> DatasetBundle:
    train, validation = _split_train_validation(
        train_images,
        train_labels,
        validation_fraction,
        seed,
    )
    test = _make_split(test_images, test_labels)
    input_shape = tuple(int(dim) for dim in train.images.shape[1:])
    num_classes = int(max(train.labels.max(), test.labels.max()) + 1)
    return DatasetBundle(train, validation, test, num_classes, input_shape, class_names)


def load_hdf5_dataset(
    data_dir: str | Path,
    validation_fraction: float = 0.1,
    seed: int = 0,
) -> DatasetBundle:
    """Load ``trainset.hdf5`` and ``testset.hdf5`` from a directory."""

    data_dir = Path(data_dir)
    with h5py.File(data_dir / "trainset.hdf5", "r") as train_file:
        train_images = np.asarray(train_file["X_train"][:])
        train_labels = np.asarray(train_file["Y_train"][:])
    with h5py.File(data_dir / "testset.hdf5", "r") as test_file:
        test_images = np.asarray(test_file["X_test"][:])
        test_labels = np.asarray(test_file["Y_test"][:])

    return _bundle(
        train_images,
        train_labels,
        test_images,
        test_labels,
        validation_fraction,
        seed,
    )

def iter_batches(
    split: DatasetSplit,
    batch_size: int,
    key: jax.Array | None = None,
    shuffle: bool = True,
) -> Iterator[dict[str, jax.Array]]:
    """Yield JAX batches with deterministic, explicit-key shuffling."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    indices = np.arange(split.size)
    if shuffle:
        if key is None:
            raise ValueError("A JAX PRNG key is required when shuffle=True")
        indices = np.asarray(jax.random.permutation(key, split.size))

    for start in range(0, split.size, batch_size):
        batch_indices = indices[start : start + batch_size]
        yield {
            "images": jnp.asarray(split.images[batch_indices]),
            "labels": jnp.asarray(split.labels[batch_indices]),
        }