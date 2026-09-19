"""Per-place expert (§4.3), first version -- SS9 step 4 / SS5.3 stage 2.

§4.3's diagram: "Each active expert predicts using only its place's causal
subgraph (c~)." This is a SEPARATE model from DIRGNNSplitter's own
causal_head -- that head exists only to pressure the rationale generator
during the invariance-training mechanism (mean risk + variance-across-swaps,
§4.2) and is never meant to be the final forecast. The expert here is what
the eventual drift/lifecycle machinery (§4.4/§4.5, steps 5-6) will freeze,
retrain, spawn, or hibernate -- keeping it decoupled from the splitter means
those later stages only ever touch the expert, never the causal-discovery
mechanism itself.

Kept deliberately simple (a small MLP over aggregated causal-neighbor
features), matching the same "test the mechanism, not model capacity"
philosophy already used for splitter/dirgnn.py's SharedEncoder (§5.3's
build-order rationale: isolate one new failure source per stage).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


class PlaceExpert(nn.Module):
    """Forecasts ONE target place's future OLR from its own causal subgraph
    c~ (candidate source clusters the splitter selected as top-r causal
    drivers, §4.2), independent of the splitter's causal_head.

    Aggregation: weighted sum of each selected source's own lag-flattened
    features (edge weight = splitter's causal edge weight, so a more
    confidently-causal source contributes more), concatenated with the
    target's own features (self-persistence is always available even if a
    self-loop wasn't selected), then an MLP regressor.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(
        self,
        x_all: torch.Tensor,
        target: int,
        causal_edge_index: torch.Tensor,
        causal_edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        """x_all: (n_candidate_sources_total, in_channels) -- this
            timestep's own features for every cluster that could appear as
            a source (indexed by the SAME cluster ids causal_edge_index
            uses, i.e. the full 50-cluster feature row, not just the
            candidate subset -- simplest indexing, avoids a remapping step).
        target: the place this expert forecasts (int, unused directly here
            since target's own features come from x_all[target] -- kept as
            an argument for call-site clarity/symmetry with the splitter).
        causal_edge_index: (2, k) int64 -- rows are (source, target) pairs
            selected by the splitter's top-r cutoff for THIS place, target
            column is constant (== target) by construction (star graph,
            causal_moe.data.candidate_edges).
        causal_edge_weight: (k,) float -- splitter's edge scores for the
            selected edges (attached to the generator's graph -- gradient
            flows back into the generator too, same as DIRGNNSplitter's own
            causal_head, so the expert's own accuracy pressure also shapes
            which edges get selected).
        Returns: scalar tensor, this place's forecast.
        """
        sources = causal_edge_index[0]
        weights = causal_edge_weight
        weighted_sum = (x_all[sources] * weights.unsqueeze(-1)).sum(dim=0)
        weight_total = weights.sum().clamp_min(1e-6)
        aggregated = weighted_sum / weight_total  # weighted mean, scale-invariant to k

        target_features = x_all[target]
        combined = torch.cat([target_features, aggregated], dim=-1)
        return self.mlp(combined).squeeze(-1)


@dataclass
class ExpertLossOutput:
    total_loss: torch.Tensor
    mse: torch.Tensor
    prediction: torch.Tensor


def compute_expert_loss(
    expert: PlaceExpert,
    x_all: torch.Tensor,
    target: int,
    y_target: torch.Tensor,
    causal_edge_index: torch.Tensor,
    causal_edge_weight: torch.Tensor,
) -> ExpertLossOutput:
    """y_target: scalar tensor, this place's true OLR N days ahead."""
    prediction = expert(x_all, target, causal_edge_index, causal_edge_weight)
    mse = (prediction - y_target) ** 2
    return ExpertLossOutput(total_loss=mse, mse=mse, prediction=prediction)
