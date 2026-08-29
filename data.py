"""Dataset adapters and batch iteration for the JAX CNN experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

if __package__:
    from .config import DatasetName
else:
    from config import DatasetName


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
    """Convert images to float32 channels-last arrays in the [0, 1] range."""

    images = np.asarray(images)
    if images.ndim == 3:
        images = images[..., np.newaxis]
    if images.ndim != 4:
        raise ValueError("Images must have shape (N, H, W) or (N, H, W, C)")

    images = images.astype(np.float32)
    if images.size and images.max() > 1:
        images /= 255.0 if images.max() <= 255 else images.max()
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


def _split_train_validation(
    images: np.ndarray,
    labels: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> tuple[DatasetSplit, DatasetSplit]:
    train_images, validation_images, train_labels, validation_labels = train_test_split(
        images,
        labels,
        test_size=validation_fraction,
        random_state=seed,
        stratify=labels,
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


def load_digits_dataset(
    validation_fraction: float = 0.1,
    seed: int = 0,
) -> DatasetBundle:
    """Load scikit-learn's small 8x8 handwritten-digits dataset."""

    digits = load_digits()
    train_images, test_images, train_labels, test_labels = train_test_split(
        digits.images.astype(np.float32) / 16.0,
        digits.target,
        test_size=0.2,
        random_state=seed,
        stratify=digits.target,
    )
    class_names = tuple(str(index) for index in range(10))
    return _bundle(
        train_images,
        train_labels,
        test_images,
        test_labels,
        validation_fraction,
        seed,
        class_names,
    )


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


def load_npz_dataset(
    path: str | Path,
    validation_fraction: float = 0.1,
    seed: int = 0,
) -> DatasetBundle:
    """Load a portable NPZ dataset with four standard array keys.

    Required keys are ``X_train``, ``y_train``, ``X_test``, and ``y_test``.
    """

    with np.load(path) as archive:
        return _bundle(
            archive["X_train"],
            archive["y_train"],
            archive["X_test"],
            archive["y_test"],
            validation_fraction,
            seed,
        )


def load_dataset(
    name: DatasetName,
    data_dir: str | Path | None = None,
    validation_fraction: float = 0.1,
    seed: int = 0,
) -> DatasetBundle:
    """Load a named dataset through a single stable entry point.

    ``digits``, ``hdf5``, and ``npz`` are provided as working adapters. The
    other names are reserved extension points for MNIST, Fashion-MNIST, and
    CIFAR-10 adapters.
    """

    if name == "digits":
        return load_digits_dataset(validation_fraction, seed)
    if name == "hdf5":
        if data_dir is None:
            raise ValueError("data_dir is required for the hdf5 dataset")
        return load_hdf5_dataset(data_dir, validation_fraction, seed)
    if name == "npz":
        if data_dir is None:
            raise ValueError("data_dir must point to an NPZ file")
        return load_npz_dataset(data_dir, validation_fraction, seed)
    if name in {"mnist", "fashion_mnist", "cifar10"}:
        raise NotImplementedError(
            f"No adapter is implemented yet for {name!r}. Add it here without changing the training API."
        )
    raise ValueError(f"Unknown dataset: {name}")


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
