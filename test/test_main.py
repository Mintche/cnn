"""Tests for command-line experiment configuration."""

from __future__ import annotations

import unittest

import numpy as np

from data import DatasetBundle, DatasetSplit
from main import model_config_for_dataset


class DatasetModelConfigTests(unittest.TestCase):
    def test_mnist_datasets_use_three_stages_for_28_pixel_images(self) -> None:
        empty_split = DatasetSplit(
            images=np.empty((0, 28, 28, 1), dtype=np.float32),
            labels=np.empty((0,), dtype=np.int32),
        )
        dataset = DatasetBundle(
            train=empty_split,
            validation=empty_split,
            test=empty_split,
            num_classes=10,
            input_shape=(28, 28, 1),
        )

        for dataset_name in ("mnist", "fashion_mnist"):
            with self.subTest(dataset=dataset_name):
                config = model_config_for_dataset(dataset_name, dataset, 0.1)
                self.assertEqual(config.depths, (2, 2, 4))
                self.assertEqual(config.dims, (32, 64, 128))


if __name__ == "__main__":
    unittest.main()
