"""JAX training utilities independent of the CNN architecture."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax

if __package__:
    from .data import DatasetSplit, iter_batches
    from .model import ForwardFn, cross_entropy_logits
else:
    from data import DatasetSplit, iter_batches
    from model import ForwardFn, cross_entropy_logits


class TrainState(NamedTuple):
    """Minimal immutable state carried through a JAX update step."""

    step: int
    params: Any
    optimizer_state: optax.OptState


def create_state(params: Any, optimizer: optax.GradientTransformation) -> TrainState:
    """Initialize optimizer state for a parameter pytree."""

    return TrainState(step=0, params=params, optimizer_state=optimizer.init(params))


def make_train_step(
    forward_fn: ForwardFn,
    optimizer: optax.GradientTransformation,
):
    """Build a JIT-compiled update function for the supplied model."""

    @jax.jit
    def train_step(
        state: TrainState,
        batch: dict[str, jax.Array],
        key: jax.Array,
    ) -> tuple[TrainState, dict[str, jax.Array]]:
        def loss_fn(params: Any) -> jax.Array:
            logits = forward_fn(params, batch["images"], True, key)
            return cross_entropy_logits(logits, batch["labels"])

        loss, gradients = jax.value_and_grad(loss_fn)(state.params)
        updates, optimizer_state = optimizer.update(
            gradients,
            state.optimizer_state,
            state.params,
        )
        params = optax.apply_updates(state.params, updates)
        metrics = {"loss": loss}
        return state._replace(
            step=state.step + 1,
            params=params,
            optimizer_state=optimizer_state,
        ), metrics

    return train_step


@jax.jit
def batch_metrics(logits: jax.Array, labels: jax.Array) -> dict[str, jax.Array]:
    """Compute loss and accuracy for one logits batch."""

    loss = cross_entropy_logits(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return {"loss": loss, "accuracy": accuracy}


def evaluate(
    params: Any,
    split: DatasetSplit,
    forward_fn: ForwardFn,
    batch_size: int,
) -> dict[str, float]:
    """Evaluate a parameter tree on a complete dataset split."""

    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0
    for batch in iter_batches(split, batch_size, shuffle=False):
        logits = forward_fn(params, batch["images"], False, None)
        metrics = batch_metrics(logits, batch["labels"])
        batch_size_actual = int(batch["labels"].shape[0])
        total_loss += float(metrics["loss"]) * batch_size_actual
        total_correct += float(metrics["accuracy"]) * batch_size_actual
        total_samples += batch_size_actual

    if total_samples == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def fit(
    state: TrainState,
    train_split: DatasetSplit,
    validation_split: DatasetSplit,
    forward_fn: ForwardFn,
    optimizer: optax.GradientTransformation,
    epochs: int,
    batch_size: int,
    key: jax.Array,
) -> tuple[TrainState, list[dict[str, float]]]:
    """Run the generic training loop and return state plus epoch history."""

    if epochs <= 0:
        raise ValueError("epochs must be positive")

    train_step = make_train_step(forward_fn, optimizer)
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        key, epoch_key = jax.random.split(key)
        epoch_loss = 0.0
        seen_samples = 0
        for batch in iter_batches(train_split, batch_size, epoch_key, shuffle=True):
            key, step_key = jax.random.split(key)
            state, metrics = train_step(state, batch, step_key)
            current_batch_size = int(batch["labels"].shape[0])
            epoch_loss += float(metrics["loss"]) * current_batch_size
            seen_samples += current_batch_size

        validation_metrics = evaluate(
            state.params,
            validation_split,
            forward_fn,
            batch_size,
        )
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": epoch_loss / seen_samples,
                "validation_loss": validation_metrics["loss"],
                "validation_accuracy": validation_metrics["accuracy"],
            }
        )

    return state, history
