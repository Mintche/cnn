"""Configuration objects shared by datasets, models, and training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


DatasetName = Literal["digits", "hdf5", "npz", "mnist", "fashion_mnist", "cifar10"]


@dataclass(frozen=True)
class DataConfig:
    """Configuration for one dataset experiment."""
    name: DatasetName = "hf5"
    data_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent / "datasets"
    )
    validation_fraction: float = 0.1
    batch_size: int = 64
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


@dataclass(frozen=True)
class ModelConfig:
    depths: tuple[int, ...] = (2, 2)
    dims: tuple[int, ...] = (32, 64)
    num_classes: int = 10
    drop_path_rate: float = 0.0

    def __post_init__(self) -> None:
        if not self.depths or len(self.depths) != len(self.dims):
            raise ValueError("depths and dims must be non-empty and have the same size")
        if any(depth <= 0 for depth in self.depths):
            raise ValueError("all stage depths must be positive")
        if any(dim <= 0 for dim in self.dims):
            raise ValueError("all stage dimensions must be positive")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if not 0.0 <= self.drop_path_rate < 1.0:
            raise ValueError("drop_path_rate must be in [0, 1)")


@dataclass(frozen=True)
class TrainingConfig:
    """Training hyperparameters."""

    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete configuration for a CNN experiment."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
