"""Functional CNN contract.

This module intentionally does not contain a finished architecture. Implement
the parameter tree and the forward pass here, then keep the training code
agnostic to the model details.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

if __package__:
    from .config import ModelConfig
else:
    from config import ModelConfig


Params = Any
ForwardFn = Callable[[Params, jax.Array, bool, jax.Array | None], jax.Array]

#Based on the 2015 paper
def kaiming_init(
    key: jax.Array,
    shape: tuple[int, ...],
    is_depthwise: bool = False,
    mode: str = "fan_in",
) -> jax.Array:
    if len(shape) == 2:
        fan_in, fan_out = shape
    elif len(shape) == 4:
        kh, kw, c_in, c_out = shape
        if is_depthwise:
            fan_in = kh * kw
            fan_out = kh * kw
        else:
            fan_in = kh * kw * c_in
            fan_out = kh * kw * c_out
    else:
        raise ValueError(f"Shape must be 2D or 4D, got length {len(shape)}")
    fan = fan_in if mode == "fan_in" else fan_out
    return jax.random.normal(key, shape) * jnp.sqrt(2.0 / fan)

def init_convnext_block(
        key: jax.Array, 
        dim: int,
) -> dict:
    k1, k2, k3 = jax.random.split(key, 3)
    hidden_dim = 4 * dim
    return {
        # Depthwise 7x7 (H, W, in_channels, out_channels/group)
        "dwconv": {
            "w": kaiming_init(k1, (7, 7, 1, dim), is_depthwise=True),
            "b": jnp.zeros((dim,)),
        },
        # LayerNorm
        "norm": {
            "gamma": jnp.ones((dim,)),
            "beta": jnp.zeros((dim,)),
        },
        # Pointwise 1 (C -> 4C)
        "pwconv1": {
            "w": kaiming_init(k2, (dim, hidden_dim)),
            "b": jnp.zeros((hidden_dim,)),
        },
        # GRN
        "grn": {
            "gamma": jnp.zeros((1, 1, hidden_dim)),
            "beta": jnp.zeros((1, 1, hidden_dim)),
        },
        # Pointwise 2 (4C -> C)
        "pwconv2": {
            "w": kaiming_init(k3, (hidden_dim, dim)),
            "b": jnp.zeros((dim,)),
        },
    }

def init_downsample(
        key: jax.Array, 
        in_dim: int, 
        out_dim: int,
) -> dict:
    return {
        # LayerNorm
        "norm": {
            "gamma": jnp.ones((in_dim,)),
            "beta": jnp.zeros((in_dim,)),
        },
        "conv": {
            "w": kaiming_init(key, (2, 2, in_dim, out_dim)),
            "b": jnp.zeros((out_dim,)),
        },
    }

def init_cnn(
    key: jax.Array,
    input_shape: tuple[int, int, int],  # (H, W, C_in)
    config: ModelConfig,
) -> dict:
    if len(config.depths) != len(config.dims):
        raise ValueError(f"depths and dims must have the same size")
    in_channels = input_shape[-1]
    c0 = config.dims[0]
    params = {}

    # Embeding of 4x4 blocks
    key_stem, key = jax.random.split(key)
    params["stem"] = {
        "norm": {
                    "gamma": jnp.ones((c0,)),
                    "beta": jnp.zeros((c0,)),
                },
        "conv": {
                    "w": kaiming_init(key_stem, (4, 4, in_channels, c0)),
                    "b": jnp.zeros((c0,)),
                },
    }

    params["stages"] = []
    params["downsamples"] = []

    for stage_idx, (depth, dim) in enumerate(zip(config.depths, config.dims)):
        if stage_idx>0:
            # Downsample
            key_down, key = jax.random.split(key)
            params["downsamples"].append(
                init_downsample(key_down,config.dims[stage_idx-1],dim)
                )
        stage_blocks = []
        for _ in range(depth):
            key_block, key = jax.random.split(key)
            stage_blocks.append(init_convnext_block(key_block, dim))

        params["stages"].append(stage_blocks)

    last_dim = config.dims[-1]
    k_head, _ = jax.random.split(key)
    params["head"] = {
        "norm": {
            "gamma": jnp.ones((last_dim,)),
            "beta": jnp.zeros((last_dim,)),
        },
        "linear": {
            "w": kaiming_init(k_head, (last_dim, config.num_classes)),
            "b": jnp.zeros((config.num_classes,)),
        },
    }
    
    return params

def layer_norm(
        x: jax.Array,
        norm: Params,
) -> jax.Array:
    xnorm = (x - jnp.mean(x,axis=-1, keepdims=True))/jnp.sqrt(jnp.var(x,axis=-1,keepdims=True) + 1e-8)
    return xnorm * norm["gamma"] + norm["beta"]

def drop_path(
    key: jax.Array,
    x: jax.Array,
    drop_prob: float = 0.0,
) -> jax.Array:
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = jax.random.bernoulli(key, p=1.0 - drop_prob, shape=shape)
    return (x / (1.0 - drop_prob)) * mask

def apply_cnn(
    params: Params,
    images: jax.Array,
    training: bool = False,
    key: jax.Array | None = None,
) -> jax.Array:
    # stem
    x = jax.lax.conv_general_dilated(images,params["stem"]["conv"]["w"],
                                    dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
                                    window_strides=(4, 4),padding='VALID') + params["stem"]["conv"]["b"]
    x = layer_norm(x,params["stem"]["norm"])

    stages = len(params["stages"])

    for i in range(stages):
        for block in params["stages"][i]:
            res = x
            #Depthwise + LayerNorm
            x = jax.lax.conv_general_dilated(x,block["dwconv"]["w"],padding=((3, 3), (3, 3))
                                             ,dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
                                             feature_group_count=x.shape[-1],window_strides=(1, 1)) + block["dwconv"]["b"]
            x = layer_norm(x,block["norm"])
            #Pointwise1
            x = x@block["pwconv1"]["w"] + block["pwconv1"]["b"]
            x = jax.nn.gelu(x)
            # GRN (Global Response Normalization)
            g = jnp.linalg.norm(x, axis=(1, 2), keepdims=True)
            n = g / (jnp.mean(g, axis=-1, keepdims=True) + 1e-8)
            x = block["grn"]["gamma"] * (x * n) + block["grn"]["beta"] + x
            #Pointwise2
            x = x@block["pwconv2"]["w"] + block["pwconv2"]["b"]
            # Drop Path + Skip Connection
            if training:
                key, subkey = jax.random.split(key)
                x = res + drop_path(subkey,x,drop_prob=0.1)
            else:
                x = res + x
        if i != stages - 1:
            #Downsample
            x = layer_norm(x,params["downsamples"][i]["norm"])
            x = jax.lax.conv_general_dilated(x,params["downsamples"][i]["conv"]["w"],
                                             window_strides=(2, 2),padding='VALID',
                                            dimension_numbers=('NHWC', 'HWIO', 'NHWC')) + params["downsamples"][i]["conv"]["b"]
    # Head
    x = jnp.mean(x, axis=(1, 2))
    x = layer_norm(x,params["head"]["norm"])

    return x @ params["head"]["linear"]["w"] + params["head"]["linear"]["b"]


def build_forward_fn(config: ModelConfig) -> ForwardFn:
    """Bind model configuration to the training loop's forward-function API."""

    del config

    def forward(
        params: Params,
        images: jax.Array,
        training: bool = False,
        key: jax.Array | None = None,
    ) -> jax.Array:
        return apply_cnn(params, images, training, key)

    return forward


def check_input_shape(images: jax.Array, expected_shape: tuple[int, int, int]) -> None:
    """Fail early when a dataset does not match the model's image contract."""

    if images.ndim != 4 or tuple(images.shape[1:]) != expected_shape:
        raise ValueError(
            "Expected images with shape "
            f"(batch, {expected_shape[0]}, {expected_shape[1]}, {expected_shape[2]}), "
            f"received {tuple(images.shape)}"
        )


def cross_entropy_logits(logits: jax.Array, labels: jax.Array) -> jax.Array:
    """Compute mean multiclass cross-entropy from unnormalized logits."""

    log_probabilities = jax.nn.log_softmax(logits, axis=-1)
    sample_indices = jnp.arange(labels.shape[0])
    return -jnp.mean(log_probabilities[sample_indices, labels])
