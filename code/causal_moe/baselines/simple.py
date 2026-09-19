"""Ablation/baseline models for SS6's evaluation.

SS6 lists as baselines: GeoMoE, GC-MoE, DyMoE (regional-scaled), plus
**plain-GNN and no-drift-handling ablations**. The published-method
baselines (GC-MoE, GraphMoRE-as-GeoMoE-substitute, DyMoE) are separate
external codebases (SS6/SS7) and are NOT implemented here.

This module implements the ablations that isolate THIS project's own
contributions -- the ones that answer "does the causal splitter actually
earn its place?", which is the question a reviewer asks first:

  1. `PersistenceBaseline`   -- tomorrow's OLR = today's OLR. The trivial
     forecast SS8 explicitly warns about ("avoiding a trivial persistence
     baseline" is why OLR was chosen as the target at all).
  2. `PlainGNNBaseline`      -- same encoder + expert capacity, but NO
     causal splitting: every candidate edge is used, at weight 1. This is
     the direct ablation of the splitter -- if it matches the full model,
     the rationale generator is not contributing.
  3. `RandomSubsetBaseline`  -- keeps a random top-r-sized edge subset,
     fixed across training. Controls for "does ANY sparsification help,
     regardless of whether it's the causal one?" -- without this, a win over
     PlainGNN could just be a regularization effect.

All three share the expert's architecture so the comparison isolates the
edge-selection mechanism rather than model capacity.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from causal_moe.experts.expert import PlaceExpert


class PersistenceBaseline:
    """No parameters: predict the target's own current OLR."""

    def __init__(self, olr_channel: int = 5):
        self.olr_channel = olr_channel

    def predict(self, x_all: np.ndarray, target: int) -> np.ndarray:
        """x_all: (n_samples, n_places, n_features) -> (n_samples,)"""
        return x_all[:, target, self.olr_channel]


class _FixedEdgeExpert(nn.Module):
    """Shared scaffold: a PlaceExpert fed a FIXED edge set/weighting instead
    of the splitter's learned causal subgraph."""

    def __init__(self, in_channels: int, edge_index: torch.Tensor, hidden_channels: int = 16):
        super().__init__()
        self.expert = PlaceExpert(in_channels=in_channels, hidden_channels=hidden_channels)
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_weight", torch.ones(edge_index.shape[1]))

    def forward(self, x_all: torch.Tensor, target: int) -> torch.Tensor:
        return self.expert(x_all, target, self.edge_index, self.edge_weight)


class PlainGNNBaseline(_FixedEdgeExpert):
    """Ablation: ALL candidate edges, uniform weight -- no causal selection."""


class RandomSubsetBaseline(_FixedEdgeExpert):
    """Ablation: a random r-fraction of candidate edges, fixed at init.

    Controls for generic sparsification benefit, so a win by the real
    splitter can't be explained by "fewer edges is just easier to fit".
    """

    def __init__(
        self,
        in_channels: int,
        edge_index: torch.Tensor,
        r: float,
        hidden_channels: int = 16,
        seed: int = 0,
    ):
        n_edges = edge_index.shape[1]
        k = max(1, int(round(r * n_edges)))
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(n_edges, size=k, replace=False))
        super().__init__(in_channels, edge_index[:, torch.from_numpy(keep)], hidden_channels)
