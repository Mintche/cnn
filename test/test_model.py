"""Tests for CNN parameter initialization and the forward pass."""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from config import ModelConfig
from model import apply_cnn, build_forward_fn, init_cnn


class InitCnnTests(unittest.TestCase):
    def test_parameter_tree_matches_configuration(self) -> None:
        config = ModelConfig(depths=(1, 2), dims=(4, 8), num_classes=3)

        params = init_cnn(jax.random.PRNGKey(0), (16, 16, 3), config)

        self.assertEqual(set(params), {"stem", "stages", "downsamples", "head"})
        self.assertEqual(params["stem"]["conv"]["w"].shape, (4, 4, 3, 4))
        self.assertEqual(params["stem"]["conv"]["b"].shape, (4,))

        self.assertEqual(len(params["stages"]), 2)
        self.assertEqual([len(stage) for stage in params["stages"]], [1, 2])
        first_block = params["stages"][0][0]
        self.assertEqual(first_block["dwconv"]["w"].shape, (7, 7, 1, 4))
        self.assertEqual(first_block["pwconv1"]["w"].shape, (4, 16))
        self.assertEqual(first_block["grn"]["gamma"].shape, (1, 1, 16))
        self.assertEqual(first_block["pwconv2"]["w"].shape, (16, 4))

        self.assertEqual(len(params["downsamples"]), 1)
        self.assertEqual(params["downsamples"][0]["conv"]["w"].shape, (2, 2, 4, 8))
        self.assertEqual(params["head"]["linear"]["w"].shape, (8, 3))
        self.assertEqual(params["head"]["linear"]["b"].shape, (3,))

    def test_initialization_is_reproducible_for_the_same_key(self) -> None:
        config = ModelConfig(depths=(1,), dims=(4,), num_classes=2)
        key = jax.random.PRNGKey(7)

        first = init_cnn(key, (8, 8, 1), config)
        second = init_cnn(key, (8, 8, 1), config)

        first_leaves = jax.tree_util.tree_leaves(first)
        second_leaves = jax.tree_util.tree_leaves(second)
        self.assertEqual(len(first_leaves), len(second_leaves))
        for first_leaf, second_leaf in zip(first_leaves, second_leaves):
            np.testing.assert_array_equal(np.asarray(first_leaf), np.asarray(second_leaf))

    def test_depths_and_dims_must_have_the_same_length(self) -> None:
        with self.assertRaisesRegex(ValueError, "depths and dims"):
            ModelConfig(depths=(1, 1), dims=(4,), num_classes=2)

    def test_spatial_shape_must_support_every_stage(self) -> None:
        config = ModelConfig(depths=(1, 1, 1), dims=(4, 8, 16), num_classes=2)

        with self.assertRaisesRegex(ValueError, "at least 16 pixels"):
            init_cnn(jax.random.PRNGKey(0), (8, 8, 1), config)


class ApplyCnnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = ModelConfig(depths=(1, 1), dims=(4, 8), num_classes=3)
        init_key, image_key = jax.random.split(jax.random.PRNGKey(11))
        cls.params = init_cnn(init_key, (16, 16, 3), cls.config)
        cls.images = jax.random.normal(image_key, (2, 16, 16, 3))

    def test_evaluation_returns_finite_logits_with_expected_shape(self) -> None:
        logits = apply_cnn(self.params, self.images)

        self.assertEqual(logits.shape, (2, self.config.num_classes))
        self.assertTrue(bool(jnp.all(jnp.isfinite(logits))))

    def test_evaluation_does_not_depend_on_a_random_key(self) -> None:
        without_key = apply_cnn(self.params, self.images, training=False, key=None)
        with_key = apply_cnn(
            self.params,
            self.images,
            training=False,
            key=jax.random.PRNGKey(1),
        )

        np.testing.assert_array_equal(np.asarray(without_key), np.asarray(with_key))

    def test_training_is_reproducible_for_the_same_random_key(self) -> None:
        key = jax.random.PRNGKey(5)

        first = apply_cnn(self.params, self.images, training=True, key=key)
        second = apply_cnn(self.params, self.images, training=True, key=key)

        self.assertEqual(first.shape, (2, self.config.num_classes))
        self.assertTrue(bool(jnp.all(jnp.isfinite(first))))
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))

    def test_training_requires_a_random_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "PRNG key"):
            apply_cnn(self.params, self.images, training=True, key=None)

    def test_configured_drop_path_uses_the_random_key(self) -> None:
        config = ModelConfig(
            depths=(1, 1),
            dims=(4, 8),
            num_classes=3,
            drop_path_rate=0.5,
        )
        forward = build_forward_fn(config)
        images = jnp.tile(self.images, (16, 1, 1, 1))

        first = forward(self.params, images, True, jax.random.PRNGKey(1))
        second = forward(self.params, images, True, jax.random.PRNGKey(2))

        self.assertFalse(np.array_equal(np.asarray(first), np.asarray(second)))

    def test_forward_pass_is_jittable_and_differentiable(self) -> None:
        @jax.jit
        def loss_fn(params):
            logits = apply_cnn(params, self.images, training=False)
            return jnp.mean(jnp.square(logits))

        loss, gradients = jax.value_and_grad(loss_fn)(self.params)

        self.assertTrue(bool(jnp.isfinite(loss)))
        gradient_leaves = jax.tree_util.tree_leaves(gradients)
        self.assertGreater(len(gradient_leaves), 0)
        self.assertTrue(
            all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in gradient_leaves)
        )


if __name__ == "__main__":
    unittest.main()
