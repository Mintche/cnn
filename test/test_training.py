"""Small end-to-end training test."""

from __future__ import annotations

import unittest

import jax
import numpy as np
import optax

from config import ModelConfig
from data import DatasetSplit
from model import build_forward_fn, init_cnn
from train import create_state, evaluate, fit


class TrainingIntegrationTests(unittest.TestCase):
    def test_one_epoch_connects_data_model_optimizer_and_evaluation(self) -> None:
        generator = np.random.default_rng(0)
        images = generator.random((12, 8, 8, 1), dtype=np.float32)
        labels = np.arange(12, dtype=np.int32) % 2
        train_split = DatasetSplit(images[:8], labels[:8])
        validation_split = DatasetSplit(images[8:], labels[8:])
        config = ModelConfig(
            depths=(1,),
            dims=(4,),
            num_classes=2,
            drop_path_rate=0.1,
        )
        init_key, fit_key = jax.random.split(jax.random.PRNGKey(0))
        params = init_cnn(init_key, (8, 8, 1), config)
        optimizer = optax.adamw(learning_rate=1e-3)
        state = create_state(params, optimizer)
        forward_fn = build_forward_fn(config)

        state, history = fit(
            state,
            train_split,
            validation_split,
            forward_fn,
            optimizer,
            epochs=1,
            batch_size=4,
            key=fit_key,
            show_progress=False,
        )
        metrics = evaluate(
            state.params,
            validation_split,
            forward_fn,
            batch_size=4,
        )

        self.assertEqual(state.step, 2)
        self.assertEqual(len(history), 1)
        self.assertTrue(np.isfinite(history[0]["train_loss"]))
        self.assertTrue(np.isfinite(metrics["loss"]))
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
