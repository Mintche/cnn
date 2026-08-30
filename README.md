# ConvNeXt V2 in JAX

This repository contains a from-scratch ConvNeXt V2 implementation in JAX. Data loading, model definition, and training are kept independent so that the architecture can evolve without coupling it to a particular dataset or training loop.

## Current status

The model is trainable end to end. [`model.py`](model.py) implements the parameter pytree and forward pass for:

- the 4 x 4 patchifying stem and its LayerNorm;
- a configurable sequence of ConvNeXt V2 stages;
- the LayerNorm and 2 x 2 convolution between consecutive stages;
- each block's 7 x 7 depthwise convolution, LayerNorm, 4x channel-expansion MLP, Global Response Normalization (GRN), and projection;
- GELU, GRN, residual connections, progressive stochastic depth, global average pooling, and the classification head.

The default configuration is deliberately small: depths `(2, 2)` and dimensions `(32, 64)`, suitable for the built-in 8 x 8 digits dataset. The command-line experiment selects a practical configuration for each dataset and synchronizes the output classes automatically.

The training module already defines the functional JAX interface expected by the model:

```text
params -> forward(params, images, training, key) -> logits
```

The training step uses `jax.value_and_grad`, `jax.jit`, immutable parameter state, and an Optax optimizer. It expects channels-last images with shape `(batch, height, width, channels)` and returns logits with shape `(batch, num_classes)`.

### Deliberate implementation choice

Weights use Kaiming-normal initialization. This is an intentional educational choice rather than an exact reproduction of the official ConvNeXt V2 code, which uses a truncated normal distribution with standard deviation 0.02. The architecture and parameter shapes are unchanged.

Run a short experiment after installation:

```bash
python main.py --dataset digits --epochs 3
```

Save the learning curves without opening a window:

```bash
python main.py --dataset digits --epochs 3 --plot data/digits.png
```

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
|-- main.py                 # Runnable training experiment
|-- model.py                # ConvNeXt V2 initialization and forward pass
|-- train.py                # Generic JIT/autodiff training utilities
|-- inspect_datasets.py     # Check a dataset adapter from the command line
|-- requirements.txt        # Python dependencies
|-- test/                   # Unit and end-to-end tests
|-- datasets/               # Local datasets and CIFAR-10 cache, ignored by Git
`-- data/                   # Local/generated data, ignored by Git
```

## Dataset adapters

The following adapters are available now:

- `digits`: scikit-learn's small handwritten-digits dataset; useful for a fast smoke test.
- `hdf5`: local `trainset.hdf5` and `testset.hdf5` files containing `X_train`, `Y_train`, `X_test`, and `Y_test`.
- `npz`: a local archive containing `X_train`, `y_train`, `X_test`, and `y_test`.
- `cifar10`: the official Python archive, downloaded and MD5-verified on first use, then read from the local cache.

MNIST and Fashion-MNIST remain reserved extension points in `load_dataset()`.

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

Inspect or download CIFAR-10 (about 163 MB for the official compressed archive):

```bash
python inspect_datasets.py --dataset cifar10 --data-dir datasets
```

Use `--no-download` to require an already cached archive. Downloaded datasets go into `datasets/`; generated outputs can go into `data/`. Both directories are ignored by Git except for their `.gitkeep` placeholders.

## Recommended model configurations

The default `ModelConfig` is the small `digits` configuration. The following configurations are practical starting points, not tuned hyperparameters:

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

The MNIST and Fashion-MNIST adapters are not implemented yet; their entries document the intended configurations.

An `npz` archive has no fixed shape or number of classes. Set `num_classes` from the loaded `DatasetBundle`, and limit the number of stages according to the shortest image side. With the planned 4 x 4 stem and 2 x 2 downsampling layers, one, two, three, and four stages require minimum input sides of 4, 8, 16, and 32 pixels respectively. Suitable width progressions are `(32,)`, `(32, 64)`, `(32, 64, 128)`, and `(32, 64, 128, 256)`.

Run the complete test suite with:

```bash
python -m unittest discover -s test -v
```

## References

- [ConvNeXt V2 paper](https://arxiv.org/abs/2301.00808)
- [Official ConvNeXt V2 implementation](https://github.com/facebookresearch/ConvNeXt-V2)
