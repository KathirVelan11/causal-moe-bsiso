"""DIR-GNN causal rationale generator (§4.2), adapted from graph
classification to spatiotemporal regression.

Instance unit (confirmed with user, 2026-09-18): one place-graph at one
timestep. The candidate edge set is FIXED across all timesteps (the graph's
node set and every candidate node-pair, §4.2's rationale generator scores
edges of this fixed set); only node features vary per timestep. For the
CausalDynamics validation step, the candidate set is the fully-connected
graph including self-loops (n^2 pairs) -- no physical pre-filter exists
there, unlike the real BSISO mesh's lattice adjacency (confirmed with user,
2026-09-18).

Mechanism (§4.2, all pieces):
  1. Rationale generator: scores every candidate edge from node features at
     a given timestep (differentiable).
  2. Hard top-r% selection: verified against Wuyxin/DIR-GNN's actual code
     (utils/get_subgraph.py, split_graph) -- NOT straight-through or
     Gumbel-softmax. Rank the scores, keep the top-r% by count using
     numpy-style argpartition on a DETACHED copy (no gradient through which
     edges got picked), but keep the KEPT edges' weights attached (gradient
     flows through how strongly each kept edge is weighted).
  3. Intervener: for each anchor instance (one place, one timestep), swap
     its non-causal part s~ for another timestep's s~ from the same place's
     own history (both time directions, §4.2/§8), drawn from a memory bank.
  4. Shared encoder processes (causal edges + swapped non-causal edges).
  5. Causal classifier (here: regressor) -- gradients flow, produces the
     real prediction used for the main loss.
  6. Spurious classifier (regressor) -- gradient-blocked from the shared
     encoder/generator, trained only as a leakage gauge, never affects the
     main model.

Loss: minimize E_swaps[risk(causal_head(c~, swapped_s~))]
              + lambda * Var_swaps[risk(...)]
      plus a separate detached-gradient loss training the spurious head.

lambda warm-up: starts at 0, linear ramp over the first ~30% of training
(§4.2 decision -- simple default, revisit only if training looks unstable).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def split_graph_top_r(
    edge_scores: torch.Tensor,
    edge_index: torch.Tensor,
    r: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hard top-r% edge selection, DIR-GNN's actual mechanism (verified
    against Wuyxin/DIR-GNN utils/get_subgraph.py::split_graph).

    edge_scores: (n_edges,) float tensor, differentiable soft importance
        score per candidate edge (e.g. sigmoid output of the rationale
        generator).
    edge_index: (2, n_edges) int64 tensor, candidate edges (fixed across
        timesteps).
    r: fraction in (0, 1], top-r% of edges (by count) are kept as causal.

    Returns (causal_edge_index, causal_edge_weight, spurious_edge_index,
    spurious_edge_weight):
      - causal_edge_index / spurious_edge_index: (2, k) int64, no gradient
        (selection is a detached, discrete decision -- matches DIR-GNN: the
        argpartition/rank itself never gets a gradient).
      - causal_edge_weight / spurious_edge_weight: (k,) float, ATTACHED to
        edge_scores -- gradient flows through how strongly each selected
        edge is weighted, just not through which edges got picked.
    """
    n_edges = edge_scores.shape[0]
    if not (0.0 < r <= 1.0):
        raise ValueError(f"r must be in (0, 1], got {r}")

    n_causal = max(1, int(round(r * n_edges)))
    n_causal = min(n_causal, n_edges)

    # Detached copy for the selection decision itself -- no gradient through
    # "which edges got picked", matching DIR-GNN's use of numpy argpartition
    # on detached scores.
    scores_detached = edge_scores.detach().cpu().numpy()
    # argpartition: indices of the n_causal largest scores, unordered within
    # that partition (matches DIR-GNN's use of argpartition, not full sort).
    if n_causal < n_edges:
        causal_idx_np = np.argpartition(-scores_detached, n_causal - 1)[:n_causal]
    else:
        causal_idx_np = np.arange(n_edges)
    causal_mask_np = np.zeros(n_edges, dtype=bool)
    causal_mask_np[causal_idx_np] = True

    causal_idx = torch.from_numpy(np.nonzero(causal_mask_np)[0]).to(edge_index.device)
    spurious_idx = torch.from_numpy(np.nonzero(~causal_mask_np)[0]).to(edge_index.device)

    causal_edge_index = edge_index[:, causal_idx]
    spurious_edge_index = edge_index[:, spurious_idx]

    # Weights stay attached to the original (non-detached) edge_scores
    # tensor -- gradient flows through edge STRENGTH, not through selection.
    causal_edge_weight = edge_scores[causal_idx]
    spurious_edge_weight = edge_scores[spurious_idx]

    return causal_edge_index, causal_edge_weight, spurious_edge_index, spurious_edge_weight


@dataclass
class LambdaWarmupSchedule:
    """Linear warm-up: lambda=0 for the first `warmup_fraction` of training,
    then linearly ramps to `lambda_max` over the remainder (§4.2 default:
    revisit only if training looks unstable)."""

    lambda_max: float
    total_steps: int
    warmup_fraction: float = 0.3

    def value(self, step: int) -> float:
        warmup_steps = int(self.warmup_fraction * self.total_steps)
        if step < warmup_steps:
            return 0.0
        if self.total_steps <= warmup_steps:
            return self.lambda_max
        progress = (step - warmup_steps) / (self.total_steps - warmup_steps)
        return float(min(1.0, progress)) * self.lambda_max


class RationaleGenerator(nn.Module):
    """Scores every candidate edge from a trailing WINDOW of each endpoint
    node's history, not a single timestep (fixed 2026-09-18 -- see below).

    Original single-timestep design (in_channels-only, one instant) was
    diagnosed as a structural blind spot: whether an edge (src, dst) is
    truly causal is a property of the RELATIONSHIP between src's past and
    dst's values over time (e.g. lagged cross-correlation) -- info that
    provably does not exist in concat(x_src[t], x_dst[t]) alone, no matter
    how well the MLP is trained. Confirmed empirically on CausalDynamics AO
    before this fix: raw lagged |cross-correlation| between src/dst gives
    AUC ~0.62-0.74 separating true vs false edges, while the old
    single-timestep generator scored AUC~0.50 (chance) after training --
    the signal exists in the data but was architecturally unreachable.

    Fix: feed the generator a short rolling window (length `window_len`)
    of each endpoint's own scalar/channel history ending at the anchor
    timestep, flattened and concatenated, so the MLP itself can learn a
    correlation-like function instead of being fed only one instant.
    Matches DIR-GNN's own edge-scoring approach in spirit (an MLP over
    endpoint features) -- only the per-node feature widened to a window.
    """

    def __init__(self, in_channels: int, window_len: int, hidden_channels: int = 32):
        super().__init__()
        self.window_len = window_len
        window_channels = in_channels * window_len
        self.mlp = nn.Sequential(
            nn.Linear(2 * window_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x_window: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """x_window: (n_nodes, window_len, in_channels) each node's own
            trailing history, ending at the anchor timestep (lag 0 last).
        edge_index: (2, n_edges) candidate edges.
        Returns: (n_edges,) raw scores (pre-sigmoid)."""
        n_nodes = x_window.shape[0]
        x_flat = x_window.reshape(n_nodes, -1)  # (n_nodes, window_len * in_channels)
        src, dst = edge_index[0], edge_index[1]
        edge_feat = torch.cat([x_flat[src], x_flat[dst]], dim=-1)
        return self.mlp(edge_feat).squeeze(-1)


class SharedEncoder(nn.Module):
    """A minimal weighted-message-passing encoder: one round of
    weighted-sum aggregation from edge-weighted neighbors, then an MLP.
    Kept deliberately simple (this validation step tests the SPLITTER
    mechanism, not encoder capacity -- §5.3's build-order rationale).

    forward() takes possibly-MULTIPLE (edge_index, edge_weight,
    source_features) groups instead of one, so a caller can give causal
    edges the anchor instance's features and spurious edges a swapped
    instance's features in the SAME message-passing pass -- this is what
    makes the intervention (§4.2) real: c~'s messages stay anchored to
    today while only s~'s messages come from the swapped-in day, and both
    get aggregated together into each destination node before prediction.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 32):
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels + hidden_channels, hidden_channels),
            nn.ReLU(),
        )
        self.msg_lin = nn.Linear(in_channels, hidden_channels)

    def forward(
        self,
        x_self: torch.Tensor,
        edge_groups: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        n_nodes: int,
    ) -> torch.Tensor:
        """x_self: (n_nodes, in_channels) -- this instance's own features,
            used for the node's residual/self term (the "+x" in the final
            MLP), independent of which day each incoming message came from.
        edge_groups: list of (edge_index, edge_weight, source_features)
            triples. Each group's messages are built as
            msg_lin(source_features[src]) * edge_weight, then ALL groups'
            messages are summed into their destination nodes together.
        Returns: (n_nodes, hidden_channels) node embeddings.
        """
        hidden = self.msg_lin.out_features
        agg = torch.zeros(n_nodes, hidden, dtype=x_self.dtype, device=x_self.device)
        for edge_index, edge_weight, source_features in edge_groups:
            if edge_index.shape[1] == 0:
                continue
            src, dst = edge_index[0], edge_index[1]
            messages = self.msg_lin(source_features[src]) * edge_weight.unsqueeze(-1)
            agg.index_add_(0, dst, messages)
        return self.node_mlp(torch.cat([x_self, agg], dim=-1))


class RegressionHead(nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int = 1):
        super().__init__()
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.lin(h).squeeze(-1)


@dataclass
class SplitterOutput:
    causal_edge_index: torch.Tensor
    causal_edge_weight: torch.Tensor
    spurious_edge_index: torch.Tensor
    spurious_edge_weight: torch.Tensor
    edge_scores: torch.Tensor  # (n_edges,) sigmoid scores, pre-selection


@dataclass
class SplitterLossOutput:
    total_loss: torch.Tensor
    mean_risk: torch.Tensor
    variance_risk: torch.Tensor
    spurious_loss: torch.Tensor
    lambda_value: float
    predictions_per_swap: torch.Tensor  # (n_swaps, n_nodes)
    entropy_penalty: torch.Tensor | None = None  # mean binary entropy of edge scores, if entropy_weight > 0


class DIRGNNSplitter(nn.Module):
    """Full splitter (§4.2): rationale generator + shared encoder + causal
    head (gradients flow) + spurious head (gradient-blocked, leakage gauge
    only)."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        r: float = 0.5,
        generator_window_len: int = 1,
        generator_in_channels: int | None = None,
    ):
        """in_channels: per-timestep channel count fed to the SHARED
            ENCODER/prediction path (e.g. 3 for CausalDynamics' 3-lag
            flattened features).
        generator_in_channels: per-timestep RAW channel count fed to the
            rationale generator's history window (e.g. 1 for CausalDynamics'
            scalar nodes) -- may differ from in_channels since the generator
            now sees a window of raw values, not the lag-flattened vector
            (fix, 2026-09-18). Defaults to in_channels for backward
            compatibility (generator_window_len=1 callers that still pass a
            single lag-flattened snapshot)."""
        super().__init__()
        self.r = r
        gen_channels = generator_in_channels if generator_in_channels is not None else in_channels
        self.generator = RationaleGenerator(gen_channels, generator_window_len, hidden_channels)
        self.encoder = SharedEncoder(in_channels, hidden_channels)
        self.causal_head = RegressionHead(hidden_channels)
        self.spurious_head = RegressionHead(hidden_channels)

    def split(self, x_window: torch.Tensor, edge_index: torch.Tensor) -> SplitterOutput:
        """x_window: (n_nodes, generator_window_len, in_channels) -- the
        generator's history window (may be generator_window_len=1, i.e.
        shape (n_nodes, 1, in_channels), for single-timestep scoring)."""
        raw_scores = self.generator(x_window, edge_index)
        edge_scores = torch.sigmoid(raw_scores)
        c_idx, c_w, s_idx, s_w = split_graph_top_r(edge_scores, edge_index, self.r)
        return SplitterOutput(
            causal_edge_index=c_idx,
            causal_edge_weight=c_w,
            spurious_edge_index=s_idx,
            spurious_edge_weight=s_w,
            edge_scores=edge_scores,
        )

    def predict_causal(
        self, x_anchor: torch.Tensor, split: SplitterOutput, x_env: torch.Tensor,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        """The intervention (§4.2): causal edges' messages are built from
        the ANCHOR instance's own features (c~ stays anchored to today),
        spurious edges' messages are built from the SWAPPED environment
        instance's features (only s~'s context changes) -- both groups
        aggregate together into each destination node in one pass, so the
        prediction reflects "today's causal structure, but a different
        day's non-causal background." Gradients flow through this whole
        path (causal head).

        return_hidden: also return the encoder's hidden embedding h, before
            the causal_head projection -- needed by the CIA alignment term
            (causal_moe.splitter.cia), which operates on representations,
            not final predictions."""
        n_nodes = x_anchor.shape[0]
        edge_groups = [
            (split.causal_edge_index, split.causal_edge_weight, x_anchor),
            (split.spurious_edge_index, split.spurious_edge_weight, x_env),
        ]
        h = self.encoder(x_anchor, edge_groups, n_nodes)
        pred = self.causal_head(h)
        if return_hidden:
            return pred, h
        return pred

    def predict_spurious_gauge(self, x_anchor: torch.Tensor, split: SplitterOutput) -> torch.Tensor:
        """Spurious classifier: gradient-blocked from the shared
        encoder/generator (detached inputs), leakage gauge only -- never
        affects the main model's weights (§4.2). Uses spurious edges with
        the ANCHOR's own features (no swap here -- this measures how much
        of TODAY's true label leaks into the non-causal edges alone, not an
        invariance test)."""
        n_nodes = x_anchor.shape[0]
        s_idx = split.spurious_edge_index.detach()
        s_w = split.spurious_edge_weight.detach()
        x_detached = x_anchor.detach()
        with torch.no_grad():
            h = self.encoder(x_detached, [(s_idx, s_w, x_detached)], n_nodes)
        h = h.detach().requires_grad_(True)
        return self.spurious_head(h)

    def compute_loss(
        self,
        x_anchor: torch.Tensor,
        y_anchor: torch.Tensor,
        env_candidates: torch.Tensor,
        edge_index: torch.Tensor,
        lambda_value: float,
        x_generator_window: torch.Tensor | None = None,
        entropy_weight: float = 0.0,
        prior_rate_weight: float = 0.0,
        target_node_mask: torch.Tensor | None = None,
    ) -> SplitterLossOutput:
        """x_anchor: (n_nodes, in_channels) anchor instance's features (fed
            to the shared encoder/prediction path -- single timestep, this
            part of the design is unchanged).
        y_anchor: (n_nodes,) anchor instance's true target.
        env_candidates: (n_swaps, n_nodes, in_channels) other timesteps'
            features from the same place's memory bank (both directions,
            §4.2/§8), used as swapped-in environments for s~.
        edge_index: (2, n_edges) fixed candidate edges.
        lambda_value: current warm-up lambda (0 at start, ramped per
            LambdaWarmupSchedule).
        x_generator_window: (n_nodes, generator_window_len, in_channels)
            optional trailing history window fed ONLY to the rationale
            generator's scoring step, not the prediction path (fix,
            2026-09-18: the generator needs cross-time info to detect
            causal structure -- see RationaleGenerator docstring). If None,
            falls back to x_anchor reshaped as a length-1 window (backward
            compatible with generator_window_len=1).
        entropy_weight: coefficient for an optional sparsity/polarization
            penalty (candidate 2, §10 step 2, 2026-09-18) added directly on
            the generator's sigmoid edge scores -- mean binary entropy
            across all candidate edges, pushing scores toward 0/1 instead
            of clustering near 0.5. Motivated by the AO diagnostic: true
            vs false edge scores were nearly indistinguishable (0.428 vs
            0.439) and close to uniform, consistent with too little
            pressure to actually polarize the selection at high edge
            density. Default 0.0 (off, backward compatible) -- 0 disables
            the term entirely regardless of scores.
        prior_rate_weight: coefficient for a sparsity-RATE prior (adapted
            from Kipf et al. 2018 NRI's KL-to-sparse-prior term, which
            biases inferred edges toward a known low base rate rather than
            just polarizing individual scores -- verified against the
            official implementation, github.com/ethanfetaya/NRI, 2026-09-18).
            Penalizes KL(mean_edge_score || Bernoulli(self.r)) -- i.e. the
            AVERAGE score across all candidate edges is pushed toward the
            target causal ratio r (same r used for top-r selection), not
            just toward 0/1 individually. Complements entropy_weight: that
            term makes each score confident, this term makes the confident
            scores land at roughly the right RATE instead of e.g. all
            edges confidently voting "causal". Default 0.0 (off).
        target_node_mask: (n_nodes,) bool, optional. When given, risk and
            the spurious leakage gauge are computed ONLY over these node(s)
            instead of averaged across every node in the graph (added for
            SS9 step 4: a "one real place" star graph has candidate SOURCE
            clusters with real features but no meaningful forecast target
            of their own under this place's expert -- only the target
            cluster's prediction should count toward loss). Default None
            preserves the original all-node-averaged behavior (steps 2/3).
        """
        if x_generator_window is None:
            x_generator_window = x_anchor.unsqueeze(1)  # (n_nodes, 1, in_channels)
        split = self.split(x_generator_window, edge_index)

        n_swaps = env_candidates.shape[0]
        preds = []
        for i in range(n_swaps):
            pred = self.predict_causal(x_anchor, split, env_candidates[i])
            preds.append(pred)
        predictions_per_swap = torch.stack(preds, dim=0)  # (n_swaps, n_nodes)

        if target_node_mask is not None:
            sq_err = (predictions_per_swap - y_anchor.unsqueeze(0)) ** 2  # (n_swaps, n_nodes)
            risks = sq_err[:, target_node_mask].mean(dim=1)  # (n_swaps,)
        else:
            risks = ((predictions_per_swap - y_anchor.unsqueeze(0)) ** 2).mean(dim=1)  # (n_swaps,)
        mean_risk = risks.mean()
        variance_risk = risks.var(unbiased=False) if n_swaps > 1 else torch.zeros_like(mean_risk)

        total_loss = mean_risk + lambda_value * variance_risk

        eps = 1e-7
        entropy_penalty = None
        if entropy_weight > 0.0:
            p = split.edge_scores.clamp(eps, 1.0 - eps)
            entropy_penalty = -(p * p.log() + (1 - p) * (1 - p).log()).mean()
            total_loss = total_loss + entropy_weight * entropy_penalty

        if prior_rate_weight > 0.0:
            mean_score = split.edge_scores.clamp(eps, 1.0 - eps).mean()
            r = torch.tensor(self.r, dtype=mean_score.dtype, device=mean_score.device).clamp(eps, 1.0 - eps)
            prior_kl = mean_score * (mean_score.log() - r.log()) + (1 - mean_score) * (
                (1 - mean_score).log() - (1 - r).log()
            )
            total_loss = total_loss + prior_rate_weight * prior_kl

        # Spurious leakage gauge: predict the same target from spurious
        # edges alone, gradient-blocked -- measurement only.
        spurious_pred = self.predict_spurious_gauge(x_anchor, split)
        if target_node_mask is not None:
            spurious_loss = F.mse_loss(spurious_pred[target_node_mask], y_anchor[target_node_mask])
        else:
            spurious_loss = F.mse_loss(spurious_pred, y_anchor)

        return SplitterLossOutput(
            total_loss=total_loss,
            mean_risk=mean_risk,
            variance_risk=variance_risk,
            spurious_loss=spurious_loss,
            entropy_penalty=entropy_penalty,
            lambda_value=lambda_value,
            predictions_per_swap=predictions_per_swap,
        )
