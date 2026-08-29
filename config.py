"""Configuration objects shared by datasets, models, and training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


DatasetName = Literal["digits", "hdf5", "npz", "mnist", "fashion_mnist", "cifar10"]


@dataclass(frozen=True)
class DataConfig:
    """Configuration for one dataset experiment."""
    name: DatasetName = "digits"
    data_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent / "data"
    )
    validation_fraction: float = 0.1
    batch_size: int = 64
    seed: int = 0


@dataclass(frozen=True)
class ModelConfig:
    depths: tuple[int, ...] = (2, 2)
    dims: tuple[int, ...] = (32, 64)
    num_classes: int = 10


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
