# ConvNeXt V2 in JAX

A from-scratch implementation of the ConvNeXt V2 architecture to understand
and experiment with the ideas in the [paper](https://arxiv.org/abs/2301.00808).
The goal is educational: build and train the network's components rather than
reproduce the paper's performance.

## Implementation

The model is written directly with JAX operations: a 4 × 4 stem,
7 × 7 depthwise convolutions, LayerNorm, MLPs, GELU, Global Response Normalization
(GRN), residual connections, and configurable stochastic depth.
Training uses JAX autodiff and JIT compilation, with AdamW through Optax.
Weights use Kaiming normal initialization.

The current experiment classifies cats and dogs from 128 × 128 RGB images.
The notebook downloads the data with KaggleHub, skips unreadable images,
and creates training, validation, and test splits.
The network has three stages with depths `(2, 2)` and dimensions
`(64, 128)`, totalingaround 390k parameters.

## Usage

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Open `main.ipynb` in an editor that supports notebooks, select the `.venv`
environment as the Python kernel, and run the cells in order.
The first run requires an internet connection to download the images.
Hyperparameters are set in the model preparation cell.

## Training outputs

At the end of each epoch, loss and accuracy are measured on both the training
and validation sets in evaluation mode, using the same weights. This adds a
pass over the training set and makes the two sets directly comparable.

The notebook saves the following files in `results/cats_dogs/`:

- `history.json`: metrics updated after each epoch.
- `best_checkpoint.pkl`: the best state by validation loss, including
  optimizer state, the PRNG key, and the epoch number.
- `metrics.png`: training and validation loss and accuracy curves.

`fit()` returns the final state. The notebook reloads the best checkpoint
with `train.load_checkpoint()` for the final test evaluation.
Use validation results to guide experiments and reserve the test set for the
final assessment. Reusing the same output directory overwrites previous results.

## Files

| File | Purpose |
|---|---|
| `main.ipynb` | Image preparation and the complete experiment |
| `model.py` | Network initialization and forward pass |
| `train.py` | Training, evaluation, checkpoints, and plots |
| `data.py` | Array preparation, splitting, batching, and HDF5 loading |
| `config.py` | Data, model, and training configurations |

Additional reference: [official implementation](https://github.com/facebookresearch/ConvNeXt-V2).
