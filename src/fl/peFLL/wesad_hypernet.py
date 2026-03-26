# Hypernetwork that generates personalised LSTM head parameters from
# a client embedding vector.
#
# Adapts from:
#   - PeFLL models.py CNNHyper
#
# DESIGN
#
# The hypernetwork takes an embedding vector d ∈ R^embedding_dim and
# produces all parameters of the client's model.head as an OrderedDict
# that can be loaded via torch.nn.utils.stateless.functional_call().
#
# Architecture:
#   embedding → shared MLP trunk → per-parameter output heads
#
# Each output head is a single nn.Linear that maps from the trunk's
# hidden dimension to the flattened parameter tensor.  The output is
# reshaped to the parameter's original shape.
#
# For the wrist model (input_size=115, hidden=32, 2-layer LSTM, 2 classes):
#   Total head parameters: 27,650
#   Breakdown:
#     lstm.weight_ih_l0: (128, 115) = 14,720
#     lstm.weight_hh_l0: (128, 32)  =  4,096
#     lstm.bias_ih_l0:   (128,)     =    128
#     lstm.bias_hh_l0:   (128,)     =    128
#     lstm.weight_ih_l1: (128, 32)  =  4,096
#     lstm.weight_hh_l1: (128, 32)  =  4,096
#     lstm.bias_ih_l1:   (128,)     =    128
#     lstm.bias_hh_l1:   (128,)     =    128
#     lstm_bn.weight:    (32,)      =     32
#     lstm_bn.bias:      (32,)      =     32
#     classifier.1.weight: (2, 32)  =     64
#     classifier.1.bias:   (2,)     =      2
#
from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn


class WESADHyperNet(nn.Module):
    """Hypernetwork that generates LSTM head parameters from an embedding.

    Follows PeFLL's CNNHyper pattern: a shared MLP trunk followed by
    per-parameter output heads.

    Parameters:
    embedding_dim : int
        Dimensionality of the input client embedding.
    hidden_dim : int
        Hidden dimension of the shared MLP trunk.
    head_param_shapes : OrderedDict[str, tuple]
        Parameter names and shapes for the target head module.
        Obtained from model.head.state_dict() — but only the learnable
        parameters (excludes BatchNorm running_mean/running_var/num_batches).
    n_hidden_layers : int
        Number of hidden layers in the shared trunk.  Default 3
        (matching PeFLL's CNNHyper which has 3 layers).
    """

    def __init__(
        self,
        embedding_dim: int = 64,  # Scott et al. 2024
        hidden_dim: int = 512,  # Scott et al. 2024
        head_param_shapes: OrderedDict[str, tuple] | None = None,
        n_hidden_layers: int = 3,  # directly from PeFLL's CNNHyper implementation
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Default: wrist model head parameter shapes
        # (input_size=115, lstm_hidden=32, lstm_layers=2, n_classes=2)
        if head_param_shapes is None:
            head_param_shapes = get_default_wrist_head_shapes()

        self.param_names = list(head_param_shapes.keys())
        self.param_shapes = head_param_shapes

        # Shared MLP trunk (matches PeFLL CNNHyper's self.mlp)
        trunk_layers = []
        trunk_layers.append(nn.Linear(embedding_dim, hidden_dim))
        trunk_layers.append(nn.ReLU())
        for _ in range(n_hidden_layers - 1):
            trunk_layers.append(nn.Linear(hidden_dim, hidden_dim))
            trunk_layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*trunk_layers)

        # Per-parameter output heads
        # Each head maps hidden_dim → flattened parameter size
        # Mirrors PeFLL CNNHyper's self.c1_weights, self.c2_weights, etc.
        self.output_heads = nn.ModuleDict()
        for name, shape in head_param_shapes.items():
            numel = 1
            for s in shape:
                numel *= s
            # Sanitise name for ModuleDict (replace dots with underscores)
            safe_name = name.replace(".", "_")
            self.output_heads[safe_name] = nn.Linear(hidden_dim, numel)

        # Store the mapping from safe names back to original names
        self._name_map = {name.replace(".", "_"): name for name in head_param_shapes}

        # total_generated = sum(
        #     1 for s in head_param_shapes.values() for _ in [s] for d in [1]
        # )
        total_params_generated = sum(_numel(s) for s in head_param_shapes.values())
        print(
            f"  WESADHyperNet: {len(self.param_names)} output heads, "
            f"{total_params_generated:,} total generated parameters"
        )

    def forward(self, embedding: torch.Tensor) -> OrderedDict:
        """Generate head parameters from a client embedding.

        Parameters:
        embedding : torch.Tensor
            (embedding_dim,)

        Returns:
        OrderedDict[str, torch.Tensor]
            Parameter name → generated tensor with correct shape.
            Keys match model.head.state_dict() (learnable params only).
            Can be used with torch.nn.utils.stateless.functional_call().
        """
        # Shared trunk
        features = self.trunk(embedding)  # (hidden_dim,)

        # Per-parameter generation
        params = OrderedDict()
        for safe_name, head in self.output_heads.items():
            original_name = self._name_map[safe_name]
            shape = self.param_shapes[original_name]
            flat = head(features)  # (numel,)
            params[original_name] = flat.view(*shape)

        return params


def get_default_wrist_head_shapes(
    input_size: int = 115,
    lstm_hidden: int = 32,
    lstm_layers: int = 2,
    n_classes: int = 2,
) -> OrderedDict:
    """Return the parameter shapes for the wrist model's head.

    These are the learnable parameters , so we exclude BatchNorm
    running_mean, running_var, num_batches_tracked.

    Parameters match WESADHead containing:
      - nn.LSTM(input_size, lstm_hidden, num_layers=lstm_layers)
      - nn.BatchNorm1d(lstm_hidden)
      - nn.Sequential(nn.Dropout(...), nn.Linear(lstm_hidden, n_classes))
    """
    h = lstm_hidden
    shapes: OrderedDict[str, tuple[int, ...]] = OrderedDict()

    # LSTM layer 0: input_size → hidden
    shapes["lstm.weight_ih_l0"] = (4 * h, input_size)
    shapes["lstm.weight_hh_l0"] = (4 * h, h)
    shapes["lstm.bias_ih_l0"] = (4 * h,)
    shapes["lstm.bias_hh_l0"] = (4 * h,)

    # LSTM layers 1..N-1: hidden → hidden
    for layer in range(1, lstm_layers):
        shapes[f"lstm.weight_ih_l{layer}"] = (4 * h, h)
        shapes[f"lstm.weight_hh_l{layer}"] = (4 * h, h)
        shapes[f"lstm.bias_ih_l{layer}"] = (4 * h,)
        shapes[f"lstm.bias_hh_l{layer}"] = (4 * h,)

    # BatchNorm
    shapes["lstm_bn.weight"] = (h,)
    shapes["lstm_bn.bias"] = (h,)

    # Classifier (index 1 because index 0 is Dropout which has no params)
    shapes["classifier.1.weight"] = (n_classes, h)
    shapes["classifier.1.bias"] = (n_classes,)

    return shapes


def get_head_shapes_from_model(head_module: nn.Module) -> OrderedDict:
    """Extract learnable parameter shapes from an actual head module.

    Excludes BatchNorm running statistics (running_mean, running_var,
    num_batches_tracked) which are not learnable parameters.

    Parameters:
    head_module : nn.Module
        The head module (e.g. fl_model.head).

    Returns:
    OrderedDict[str, tuple]
    """
    shapes = OrderedDict()
    for name, param in head_module.named_parameters():
        shapes[name] = tuple(param.shape)
    return shapes


def _numel(shape: tuple) -> int:
    """Compute number of elements from a shape tuple."""
    n = 1
    for s in shape:
        n *= s
    return n
