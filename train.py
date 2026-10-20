"""JAX training utilities independent of the CNN architecture."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import pickle
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

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


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a trusted local checkpoint containing state, key, and epoch metadata."""

    with Path(path).open("rb") as checkpoint_file:
        return pickle.load(checkpoint_file)


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


@lru_cache(maxsize=16)
def make_eval_step(forward_fn: ForwardFn):
    """Build one compiled inference-and-metrics step."""

    # Parameters remain an explicit argument so callers can evaluate any state.
    @jax.jit
    def eval_step_with_params(
        params: Any,
        batch: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        logits = forward_fn(params, batch["images"], False, None)
        return batch_metrics(logits, batch["labels"])

    return eval_step_with_params


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
    eval_step = make_eval_step(forward_fn)
    for batch in iter_batches(split, batch_size, shuffle=False):
        metrics = eval_step(params, batch)
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

def plot_metrics(
    history: list[dict[str, float]],
    output_path: str | Path | None = None,
    show: bool = True,
) -> Path | None:
    """Plot training and validation loss and accuracy from an epoch history.

    When ``output_path`` is provided, its parent directory is created if
    needed. Set ``show=False`` for non-interactive experiment campaigns.
    """

    if not history:
        raise ValueError("Cannot plot an empty training history")

    epochs = [int(metrics["epoch"]) for metrics in history]
    train_loss = [metrics["train_loss"] for metrics in history]
    validation_loss = [metrics["validation_loss"] for metrics in history]
    validation_accuracy = [metrics["validation_accuracy"] for metrics in history]

    figure, (loss_axis, accuracy_axis) = plt.subplots(
        ncols=2,
        figsize=(11, 4),
        constrained_layout=True,
    )

    loss_axis.plot(
        epochs,
        train_loss,
        color="tab:blue",
        marker="o",
        label="Training",
    )
    loss_axis.plot(
        epochs,
        validation_loss,
        color="tab:orange",
        marker="o",
        label="Validation",
    )
    loss_axis.set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    loss_axis.legend()
    loss_axis.grid(alpha=0.25)
    loss_axis.xaxis.set_major_locator(MaxNLocator(integer=True))

    if "train_accuracy" in history[0]:
        accuracy_axis.plot(
            epochs,
            [metrics["train_accuracy"] for metrics in history],
            color="tab:blue",
            marker="o",
            label="Training",
        )
    accuracy_axis.plot(
        epochs,
        validation_accuracy,
        color="tab:green",
        marker="o",
        label="Validation",
    )
    accuracy_axis.set(
        title="Accuracy",
        xlabel="Epoch",
        ylabel="Accuracy",
        ylim=(0.0, 1.0),
    )
    accuracy_axis.legend()
    accuracy_axis.grid(alpha=0.25)
    accuracy_axis.xaxis.set_major_locator(MaxNLocator(integer=True))

    saved_path = None
    if output_path is not None:
        saved_path = Path(output_path)
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(saved_path, dpi=160, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(figure)
    return saved_path


def fit(
    state: TrainState,
    train_split: DatasetSplit,
    validation_split: DatasetSplit,
    forward_fn: ForwardFn,
    optimizer: optax.GradientTransformation,
    epochs: int,
    batch_size: int,
    key: jax.Array,
    show_progress: bool = True,
    output_dir: str | Path | None = None,
) -> tuple[TrainState, list[dict[str, float]]]:
    """Return the final state and epoch history, with train metrics in eval mode.

    If output_dir is set, save history.json each epoch and best_checkpoint.pkl
    whenever validation loss improves. The checkpoint includes optimizer state,
    the next PRNG key, and the completed epoch number.
    """

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if train_split.size == 0:
        raise ValueError("Cannot train on an empty dataset")

    train_step = make_train_step(forward_fn, optimizer)
    history: list[dict[str, float]] = []
    best_validation_loss = float("inf")
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in tqdm(range(epochs), disable=not show_progress):
        key, epoch_key = jax.random.split(key)
        for batch in iter_batches(train_split, batch_size, epoch_key, shuffle=True):
            key, step_key = jax.random.split(key)
            state, _ = train_step(state, batch, step_key)

        train_metrics = evaluate(state.params, train_split, forward_fn, batch_size)
        validation_metrics = evaluate(
            state.params,
            validation_split,
            forward_fn,
            batch_size,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "validation_loss": validation_metrics["loss"],
                "validation_accuracy": validation_metrics["accuracy"],
            }
        )

        if output_dir is not None:
            (output_dir / "history.json").write_text(
                json.dumps(history, indent=2) + "\n", encoding="utf-8"
            )
            if validation_metrics["loss"] < best_validation_loss:
                checkpoint = {
                    "state": jax.device_get(state),
                    "key": jax.device_get(key),
                    "epoch": epoch + 1,
                    "validation_loss": validation_metrics["loss"],
                }
                with (output_dir / "best_checkpoint.pkl").open("wb") as checkpoint_file:
                    pickle.dump(checkpoint, checkpoint_file)
                best_validation_loss = validation_metrics["loss"]

    return state, history
