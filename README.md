# ConvNeXt V2 in JAX

This repository contains a from-scratch ConvNeXt V2 implementation in JAX. Data loading, model definition, and training are kept independent so that the architecture can evolve without coupling it to a particular dataset or training loop.

## Current status

The parameter initialization is implemented in [`model.py`](model.py). `init_cnn()` now builds the parameter pytree for:

- the 4 x 4 patchifying stem and its LayerNorm;
- a configurable sequence of ConvNeXt V2 stages;
- the LayerNorm and 2 x 2 convolution between consecutive stages;
- each block's 7 x 7 depthwise convolution, LayerNorm, 4x channel-expansion MLP, Global Response Normalization (GRN), and projection;
- the final LayerNorm and linear classification head.

The default configuration follows the ConvNeXt V2-Tiny stage layout: depths `(3, 3, 9, 3)` and dimensions `(96, 192, 384, 768)`. The number of output classes is configurable.

The forward pass in `apply_cnn()` is still intentionally unfinished, so the model cannot yet train or produce logits. Its next milestone is to apply the initialized operations in ConvNeXt V2 order, including GELU, GRN, residual connections, spatial downsampling, global average pooling, and the classification head.

The training module already defines the functional JAX interface expected by the model:

```text
params -> forward(params, images, training, key) -> logits
```

The training step uses `jax.value_and_grad`, `jax.jit`, immutable parameter state, and an Optax optimizer. It expects channels-last images with shape `(batch, height, width, channels)` and returns logits with shape `(batch, num_classes)`.

## Installation

Use Python 3.10 or newer. Create and activate a virtual environment:

```bash
python -m venv .venv
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

## Structure

```text
.
|-- .gitignore              # Local and generated files excluded from Git
|-- config.py               # Experiment, data, model, and training settings
|-- data.py                 # Dataset adapters and JAX batch iterator
|-- model.py                # ConvNeXt V2 initialization and forward pass
|-- train.py                # Generic JIT/autodiff training utilities
|-- inspect_datasets.py     # Check a dataset adapter from the command line
|-- requirements.txt        # Python dependencies
|-- datasets/               # Local HDF5 datasets, ignored by Git
`-- data/                   # Local/generated data, ignored by Git
```

## Dataset adapters

The following adapters are available now:

- `digits`: scikit-learn's small handwritten-digits dataset; useful for a fast smoke test.
- `hdf5`: local `trainset.hdf5` and `testset.hdf5` files containing `X_train`, `Y_train`, `X_test`, and `Y_test`.
- `npz`: a local archive containing `X_train`, `y_train`, `X_test`, and `y_test`.

MNIST, Fashion-MNIST, and CIFAR-10 are reserved extension points in `load_dataset()`. Add an adapter there while preserving the `DatasetBundle` interface; the training code should not need to know where the data came from.

Inspect the built-in dataset before writing the model:

```bash
python inspect_datasets.py --dataset digits
```

Inspect local HDF5 data with:

```bash
python inspect_datasets.py --dataset hdf5 --data-dir datasets
```

For a custom NPZ file:

```bash
python inspect_datasets.py --dataset npz --data-dir path/to/dataset.npz
```

Downloaded or generated data should go into `data/`. Local contents of `data/` and `datasets/` are ignored by Git; only their `.gitkeep` placeholders are versioned.

## Recommended model configurations

The default `ModelConfig` reproduces the ConvNeXt V2-Tiny stage sizes and is mainly appropriate for ImageNet-sized inputs. It is unnecessarily large for the datasets used in this project. The following smaller configurations are practical starting points, not tuned hyperparameters:

| Dataset | Input shape | Classes | `depths` | `dims` | Parameters | Final spatial size |
|---|---:|---:|---:|---:|---:|---:|
| `digits` | 8 x 8 x 1 | 10 | `(2, 2)` | `(32, 64)` | 104,106 | 1 x 1 |
| Current `hdf5` | 64 x 64 x 1 | 2 | `(2, 2, 4, 2)` | `(32, 64, 128, 256)` | 1,908,386 | 2 x 2 |
| `mnist` | 28 x 28 x 1 | 10 | `(2, 2, 4)` | `(32, 64, 128)` | 695,466 | 1 x 1 |
| `fashion_mnist` | 28 x 28 x 1 | 10 | `(2, 2, 4)` | `(32, 64, 128)` | 695,466 | 1 x 1 |
| `cifar10` | 32 x 32 x 3 | 10 | `(2, 2, 6, 2)` | `(40, 80, 160, 320)` | 3,390,610 | 1 x 1 |

They can be copied directly into an experiment:

```python
from config import ModelConfig

model_configs = {
    "digits": ModelConfig(
        depths=(2, 2),
        dims=(32, 64),
        num_classes=10,
    ),
    "hdf5": ModelConfig(
        depths=(2, 2, 4, 2),
        dims=(32, 64, 128, 256),
        num_classes=2,
    ),
    "mnist": ModelConfig(
        depths=(2, 2, 4),
        dims=(32, 64, 128),
        num_classes=10,
    ),
    "fashion_mnist": ModelConfig(
        depths=(2, 2, 4),
        dims=(32, 64, 128),
        num_classes=10,
    ),
    "cifar10": ModelConfig(
        depths=(2, 2, 6, 2),
        dims=(40, 80, 160, 320),
        num_classes=10,
    ),
}
```

The MNIST, Fashion-MNIST, and CIFAR-10 adapters are not implemented yet; these entries document the intended configurations for when they are added.

An `npz` archive has no fixed shape or number of classes. Set `num_classes` from the loaded `DatasetBundle`, and limit the number of stages according to the shortest image side. With the planned 4 x 4 stem and 2 x 2 downsampling layers, one, two, three, and four stages require minimum input sides of 4, 8, 16, and 32 pixels respectively. Suitable width progressions are `(32,)`, `(32, 64)`, `(32, 64, 128)`, and `(32, 64, 128, 256)`.

## Development roadmap

1. Implement the initial ConvNeXt V2 parameter tree. **Done.**
2. Validate every initialized parameter shape and the total parameter count. **Done.**
3. Implement the ConvNeXt V2 block, downsampling, global average pooling, and classification head in `apply_cnn()`. **Done.**
4. Add shape-focused tests for individual blocks and the complete forward pass.
5. Connect `init_cnn()`, `build_forward_fn()`, `create_state()`, and `fit()` in a small experiment script.
6. Add one dataset adapter at a time and compare the same metrics across datasets.

When creating an experiment, keep `ModelConfig.num_classes` synchronized with the loaded dataset. The recommended configurations above respect the spatial constraints of each known input resolution.

## References

- [ConvNeXt V2 paper](https://arxiv.org/abs/2301.00808)
- [Official ConvNeXt V2 implementation](https://github.com/facebookresearch/ConvNeXt-V2)
