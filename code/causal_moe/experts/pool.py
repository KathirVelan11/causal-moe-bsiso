"""Expert pool + top-k router (§4.5, §9 Phase 5 -- B11 audit fix, 2026-09-23).

Before this fix there was no router, no top-k gating, no expert pool
anywhere in the codebase (grep for router/top_k/gating returned nothing) --
step 7 trains 50 independent SINGLE-expert models, one per place, with no
mixture at all. This module adds the actual MoE mechanism the project's
design doc (§4.3/§4.5) specifies, in the simplest form that is genuinely a
top-k sparse MoE rather than a single-expert model in disguise:

  - `ExpertPool(place)`: a list of (expert, regime_signature, span) triples,
    one per hibernated generation (reuses `causal_moe.drift.archive`'s
    already-existing storage format -- an ExpertPool IS an ExpertArchive's
    contents, viewed for INFERENCE-TIME blending instead of single-match
    reactivation).
  - Router: cosine similarity between TODAY's causal-edge-score signature
    and each pool member's stored signature (same technique
    ExpertArchive.best_match already uses for reactivation, kept
    consistent per SS4.5's "one consistent mechanism, not two formulas"
    principle), softmax over the top-k most similar members, blend their
    predictions by that softmax weight.
  - Variant A = k=1, pool_size=1 (one active expert per place): this is
    what the codebase already does today (ExpertArchive.try_reactivate,
    single best match only). Kept as the explicit ablation baseline.
  - Variant B = k=3 blend over the full pool: the actual MoE this project's
    design doc calls for.

Design choice (per the B11 research recommendation, 2026-09-23): FREEZE the
per-generation experts once trained (reuse the archive's stored
state_dicts as-is) and route/blend at inference time only, rather than
jointly training a differentiable gating network. This avoids load-
balancing-loss instability and is buildable directly on top of the
existing per-place training pipeline and archive storage -- no change to
how any individual expert is trained.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from causal_moe.drift.archive import ArchivedExpert
from causal_moe.drift.channels import cosine_similarity
from causal_moe.experts.expert import PlaceExpert


@dataclass
class RoutingDecision:
    member_ids: list[str]  # ids of the top-k pool members actually used, most-similar first
    weights: np.ndarray  # (k,) softmax weights over those members, sums to 1
    similarities: np.ndarray  # (k,) raw cosine similarities, same order as member_ids


@dataclass
class ExpertPool:
    """Per-place pool of frozen, trained experts (one per hibernated
    generation), routed by causal-signature similarity at inference time.

    Deliberately built ON TOP of `ArchivedExpert`/`ExpertArchive`'s own
    storage shape (state_dict + signature + span) rather than a new
    format, so `run_step6_lifecycle.py`'s already-working
    hibernate/train-per-regime pipeline (B10 fix) can populate this pool
    directly -- an ExpertPool is that same archive, read for k>1 blending
    instead of single-match reactivation.
    """

    place: int
    in_channels: int
    hidden_channels: int = 16
    members: list[ArchivedExpert] = field(default_factory=list)

    def add(self, archived: ArchivedExpert) -> None:
        self.members.append(archived)

    @property
    def size(self) -> int:
        return len(self.members)

    def route(self, signature: np.ndarray, k: int, temperature: float = 1.0) -> RoutingDecision:
        """Cosine-similarity top-k routing (same similarity technique as
        `ExpertArchive.best_match`/SS4.4 Channel 2, per SS4.5's "one
        consistent mechanism" principle).

        signature: (n_edges,) today's causal-signature vector (same
            representation the archive's own signatures use -- centered
            edge scores, see `causal_moe.drift.channels.center_signatures`).
        k: number of pool members to blend (k=1 reproduces the existing
            single-best-match reactivation behaviour exactly; k>1 is the
            actual MoE).
        temperature: softmax temperature over the similarities -- lower
            sharpens toward the single best match, higher flattens toward
            a uniform blend. Default 1.0 (no sharpening/flattening).
        """
        if self.size == 0:
            raise ValueError("cannot route: pool is empty")
        k = min(k, self.size)
        sims = np.array([
            cosine_similarity(np.asarray(signature, dtype=np.float64), m.signature)
            for m in self.members
        ])
        top_idx = np.argsort(-sims)[:k]
        top_sims = sims[top_idx]
        # Softmax over the TOP-K similarities only (sparse routing -- the
        # (n-k) excluded members get exactly zero weight, not just a small
        # one, which is what makes this "top-k" rather than a dense blend).
        scaled = top_sims / max(temperature, 1e-8)
        exp = np.exp(scaled - scaled.max())
        weights = exp / exp.sum()
        return RoutingDecision(
            member_ids=[self.members[i].expert_id for i in top_idx],
            weights=weights,
            similarities=top_sims,
        )

    def predict(
        self,
        x_all: torch.Tensor,
        target: int,
        causal_edge_index: torch.Tensor,
        causal_edge_weight: torch.Tensor,
        signature: np.ndarray,
        k: int,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, RoutingDecision]:
        """Blended forecast: route to the top-k most similar pool members,
        run each FROZEN expert's forward pass, combine by the router's
        softmax weights. Returns (prediction, routing_decision) so callers
        can log which regimes were blended and with what weight.

        Variant A (k=1) reduces this to exactly one expert's own
        prediction, weight 1.0 -- the ablation baseline this project
        already runs today via single-match reactivation. Variant B (k=3,
        the default the plan calls for) is the actual mixture.
        """
        decision = self.route(signature, k=k, temperature=temperature)
        weighted_sum = torch.zeros(())
        member_by_id = {m.expert_id: m for m in self.members}
        with torch.no_grad():
            for member_id, weight in zip(decision.member_ids, decision.weights):
                expert = PlaceExpert(in_channels=self.in_channels, hidden_channels=self.hidden_channels)
                expert.load_state_dict(member_by_id[member_id].state_dict)
                expert.eval()
                pred = expert(x_all, target, causal_edge_index, causal_edge_weight)
                weighted_sum = weighted_sum + float(weight) * pred
        return weighted_sum, decision
