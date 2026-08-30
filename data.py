"""Dataset adapters and batch iteration for the JAX CNN experiments."""

from __future__ import annotations

import hashlib
import pickle
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from sklearn.datasets import fetch_openml, load_digits
from sklearn.model_selection import train_test_split

if __package__:
    from .config import DatasetName
else:
    from config import DatasetName

OPENML_DATASETS = {
    "mnist": {
        "data_id": 554,
        "class_names": tuple(str(index) for index in range(10)),
    },
    "fashion_mnist": {
        "data_id": 40996,
        "class_names": (
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ),
    },
}

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_FILENAME = "cifar-10-python.tar.gz"
CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"
CIFAR10_ROOT = "cifar-10-batches-py"
CIFAR10_TRAIN_BATCHES = tuple(f"data_batch_{index}" for index in range(1, 6))
CIFAR10_TEST_BATCH = "test_batch"


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


def _split_train_validation(
    images: np.ndarray,
    labels: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> tuple[DatasetSplit, DatasetSplit]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
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


def _file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_cifar10_archive(data_dir: Path, download: bool) -> Path:
    """Return a verified CIFAR-10 archive, downloading it when permitted."""

    archive_path = data_dir / CIFAR10_FILENAME
    if archive_path.is_file() and _file_md5(archive_path) == CIFAR10_MD5:
        return archive_path
    if not download:
        raise FileNotFoundError(
            f"A valid CIFAR-10 archive was not found at {archive_path}. "
            "Enable downloading or place the official archive there."
        )

    data_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = archive_path.with_suffix(archive_path.suffix + ".part")
    print(f"Downloading CIFAR-10 to {archive_path} ...")
    try:
        urllib.request.urlretrieve(CIFAR10_URL, temporary_path)
        if _file_md5(temporary_path) != CIFAR10_MD5:
            raise RuntimeError("The downloaded CIFAR-10 archive has an invalid MD5 checksum")
        temporary_path.replace(archive_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return archive_path


def _read_cifar10_batch(
    archive: tarfile.TarFile,
    filename: str,
) -> tuple[np.ndarray, np.ndarray]:
    member_name = f"{CIFAR10_ROOT}/{filename}"
    member = archive.extractfile(member_name)
    if member is None:
        raise ValueError(f"CIFAR-10 archive is missing {member_name}")
    payload = pickle.load(member, encoding="bytes")
    flat_images = np.asarray(payload[b"data"], dtype=np.uint8)
    labels = np.asarray(payload[b"labels"], dtype=np.int32)
    if flat_images.ndim != 2 or flat_images.shape[1] != 3 * 32 * 32:
        raise ValueError(f"Invalid CIFAR-10 image array in {filename}")
    images = flat_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return images, labels


def load_cifar10_dataset(
    data_dir: str | Path,
    validation_fraction: float = 0.1,
    seed: int = 0,
    download: bool = True,
) -> DatasetBundle:
    """Load CIFAR-10 from the official archive, downloading it once if needed."""

    archive_path = _ensure_cifar10_archive(Path(data_dir), download)
    with tarfile.open(archive_path, "r:gz") as archive:
        train_parts = [
            _read_cifar10_batch(archive, filename)
            for filename in CIFAR10_TRAIN_BATCHES
        ]
        test_images, test_labels = _read_cifar10_batch(
            archive,
            CIFAR10_TEST_BATCH,
        )
        metadata_file = archive.extractfile(f"{CIFAR10_ROOT}/batches.meta")
        if metadata_file is None:
            raise ValueError("CIFAR-10 archive is missing its class metadata")
        metadata = pickle.load(metadata_file, encoding="bytes")

    train_images = np.concatenate([part[0] for part in train_parts])
    train_labels = np.concatenate([part[1] for part in train_parts])
    class_names = tuple(
        name.decode("utf-8") if isinstance(name, bytes) else str(name)
        for name in metadata[b"label_names"]
    )
    return _bundle(
        train_images,
        train_labels,
        test_images,
        test_labels,
        validation_fraction,
        seed,
        class_names,
    )


def load_openml_image_dataset(
    name: str,
    data_dir: str | Path,
    validation_fraction: float = 0.1,
    seed: int = 0,
) -> DatasetBundle:
    """Load MNIST or Fashion-MNIST from OpenML's local download cache."""

    specification = OPENML_DATASETS[name]

    images, labels = fetch_openml(
        data_id=specification["data_id"],
        data_home=Path(data_dir) / "openml",
        return_X_y=True,
        as_frame=False,
        parser="auto",
    )

    images = np.asarray(images, dtype=np.uint8).reshape(-1, 28, 28, 1)
    labels = np.asarray(labels, dtype=np.int32)
    if images.shape[0] != 70_000 or labels.shape != (70_000,):
        raise ValueError(f"Unexpected sample count for {name}: {images.shape[0]}")

    return _bundle(
        images[:60_000],
        labels[:60_000],
        images[60_000:],
        labels[60_000:],
        validation_fraction,
        seed,
        specification["class_names"],
    )


def load_dataset(
    name: DatasetName,
    data_dir: str | Path | None = None,
    validation_fraction: float = 0.1,
    seed: int = 0,
    download: bool = True,
) -> DatasetBundle:
    """Load a named dataset through a single stable entry point.

    All configured names are working adapters. CIFAR-10, MNIST, and
    Fashion-MNIST are downloaded and cached on first use.
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
    if name == "cifar10":
        if data_dir is None:
            data_dir = Path(__file__).resolve().parent / "datasets"
        return load_cifar10_dataset(
            data_dir,
            validation_fraction,
            seed,
            download,
        )
    if name in {"mnist", "fashion_mnist"}:
        if data_dir is None:
            data_dir = Path(__file__).resolve().parent / "datasets"

        return load_openml_image_dataset(
            name,
            data_dir,
            validation_fraction,
            seed,
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
