"""Tests for dataset validation and the dependency-free CIFAR-10 adapter."""

from __future__ import annotations

import hashlib
import io
import pickle
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import data


def _add_pickle(archive: tarfile.TarFile, name: str, payload: dict) -> None:
    content = pickle.dumps(payload)
    member = tarfile.TarInfo(name)
    member.size = len(content)
    archive.addfile(member, io.BytesIO(content))


class Cifar10DatasetTests(unittest.TestCase):
    def test_official_archive_format_is_loaded_without_network_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / data.CIFAR10_FILENAME
            train_labels = np.repeat(np.arange(10, dtype=np.int32), 2)
            train_images = np.repeat(
                train_labels[:, np.newaxis],
                3 * 32 * 32,
                axis=1,
            ).astype(np.uint8)
            test_labels = np.arange(10, dtype=np.int32)
            test_images = np.repeat(
                test_labels[:, np.newaxis],
                3 * 32 * 32,
                axis=1,
            ).astype(np.uint8)

            with tarfile.open(archive_path, "w:gz") as archive:
                for index, batch_name in enumerate(data.CIFAR10_TRAIN_BATCHES):
                    start = index * 4
                    stop = start + 4
                    _add_pickle(
                        archive,
                        f"{data.CIFAR10_ROOT}/{batch_name}",
                        {
                            b"data": train_images[start:stop],
                            b"labels": train_labels[start:stop].tolist(),
                        },
                    )
                _add_pickle(
                    archive,
                    f"{data.CIFAR10_ROOT}/{data.CIFAR10_TEST_BATCH}",
                    {b"data": test_images, b"labels": test_labels.tolist()},
                )
                _add_pickle(
                    archive,
                    f"{data.CIFAR10_ROOT}/batches.meta",
                    {b"label_names": [f"class-{index}".encode() for index in range(10)]},
                )

            checksum = hashlib.md5(archive_path.read_bytes()).hexdigest()
            with patch.object(data, "CIFAR10_MD5", checksum):
                dataset = data.load_cifar10_dataset(
                    temporary_directory,
                    validation_fraction=0.5,
                    download=False,
                )

            self.assertEqual(dataset.input_shape, (32, 32, 3))
            self.assertEqual(dataset.num_classes, 10)
            self.assertEqual(dataset.train.size, 10)
            self.assertEqual(dataset.validation.size, 10)
            self.assertEqual(dataset.test.size, 10)
            self.assertEqual(dataset.class_names[0], "class-0")
            self.assertEqual(dataset.train.images.dtype, np.float32)
            self.assertLessEqual(float(dataset.train.images.max()), 1.0)

    def test_download_can_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(FileNotFoundError, "Enable downloading"):
                data.load_cifar10_dataset(temporary_directory, download=False)


class ImagePreparationTests(unittest.TestCase):
    def test_values_outside_supported_ranges_are_rejected(self) -> None:
        images = np.array([[[[256.0]]]], dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "value range"):
            data._prepare_images(images)


if __name__ == "__main__":
    unittest.main()
