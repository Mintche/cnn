"""Run a small, reproducible ConvNeXt V2 classification experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import jax
import optax

if __package__:
    from . import config, data, model, train
else:
    import config
    import data
    import model
    import train


WORKING_DATASETS = (
    "digits",
    "hdf5",
    "npz",
    "mnist",
    "fashion_mnist",
    "cifar10",
)


def model_config_for_dataset(
    dataset_name: str,
    dataset: data.DatasetBundle,
    drop_path_rate: float,
) -> config.ModelConfig:
    """Choose a practical model size while matching the loaded class count."""

    if dataset_name == "digits":
        depths, dims = (2, 2), (32, 64)
    elif dataset_name == "cifar10":
        depths, dims = (2, 2, 6, 2), (40, 80, 160, 320)
    elif dataset_name == "hdf5":
        depths, dims = (2, 2, 4, 2), (32, 64, 128, 256)
    elif dataset_name in {"mnist", "fashion_mnist"}:
        depths, dims = (2, 2, 4), (32, 64, 128)
    else:
        layouts = (
            ((2,), (32,)),
            ((2, 2), (32, 64)),
            ((2, 2, 4), (32, 64, 128)),
            ((2, 2, 4, 2), (32, 64, 128, 256)),
        )
        minimum_side = min(dataset.input_shape[:2])
        if minimum_side < 4:
            raise ValueError("The ConvNeXt stem requires image sides of at least 4 pixels")
        stage_count = max(
            index + 1
            for index in range(len(layouts))
            if minimum_side >= 4 * 2**index
        )
        depths, dims = layouts[stage_count - 1]
    return config.ModelConfig(
        depths=depths,
        dims=dims,
        num_classes=dataset.num_classes,
        drop_path_rate=drop_path_rate,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=WORKING_DATASETS, default="digits")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=config.DataConfig().data_dir,
        help="HDF5 directory, NPZ file, or downloaded-dataset cache directory.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--drop-path-rate", type=float, default=0.1)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Require CIFAR-10 to be present in the local cache.",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--plot", type=Path, default=None)
    parser.add_argument("--show-plot", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    data_config = config.DataConfig(
        name=args.dataset,
        data_dir=args.data_dir,
        validation_fraction=args.validation_fraction,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    training_config = config.TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    dataset = data.load_dataset(
        name=data_config.name,
        data_dir=data_config.data_dir,
        validation_fraction=data_config.validation_fraction,
        seed=data_config.seed,
        download=not args.no_download,
    )
    model_config = model_config_for_dataset(
        data_config.name,
        dataset,
        args.drop_path_rate,
    )
    experiment = config.ExperimentConfig(data_config, model_config, training_config)

    key = jax.random.key(experiment.training.seed)
    init_key, fit_key = jax.random.split(key)
    params = model.init_cnn(init_key, dataset.input_shape, experiment.model)
    optimizer = optax.adamw(
        learning_rate=experiment.training.learning_rate,
        weight_decay=experiment.training.weight_decay,
    )
    state = train.create_state(params, optimizer)
    forward_fn = model.build_forward_fn(experiment.model)

    parameter_count = sum(leaf.size for leaf in jax.tree_util.tree_leaves(params))
    print(
        f"Dataset: {experiment.data.name} | input: {dataset.input_shape} | "
        f"classes: {dataset.num_classes} | parameters: {parameter_count:,}"
    )
    state, history = train.fit(
        state=state,
        train_split=dataset.train,
        validation_split=dataset.validation,
        forward_fn=forward_fn,
        optimizer=optimizer,
        epochs=experiment.training.epochs,
        batch_size=experiment.data.batch_size,
        key=fit_key,
        show_progress=not args.quiet,
    )
    test_metrics = train.evaluate(
        params=state.params,
        split=dataset.test,
        forward_fn=forward_fn,
        batch_size=experiment.data.batch_size,
    )
    print(
        f"Test loss: {test_metrics['loss']:.4f} | "
        f"test accuracy: {test_metrics['accuracy']:.2%}"
    )
    if args.plot is not None or args.show_plot:
        train.plot_metrics(history, output_path=args.plot, show=args.show_plot)


if __name__ == "__main__":
    main()
