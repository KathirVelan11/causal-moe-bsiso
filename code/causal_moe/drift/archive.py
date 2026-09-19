"""Hibernate & reactivate (SS4.5) -- the expert archive and its similarity
search.

SS4.5's rationale: climate systems have genuinely recurring regimes
(seasons, El Nino/La Nina, decadal oscillations), so deleting an expert on
spawn (plain DyMoE) throws away a regime that will likely return, while
keeping every expert active forever doesn't scale across a multi-decade
deployment. Hibernate + reactivate bounds active compute without losing
information.

Matching logic, exactly as SS4.5 specifies:
  - compare the CURRENT causal signature against EVERY archived signature
    (not just the one being replaced),
  - using the SAME similarity technique as SS4.4's Channel 2 -- one
    consistent mechanism, not two formulas (cosine similarity over the
    importance-vector representation),
  - best match above threshold -> reactivate as a WARM START (continue
    training from those weights; the old regime is not assumed to be an
    exact repeat),
  - otherwise spawn a fresh expert.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch

from causal_moe.drift.channels import cosine_similarity


@dataclass
class ArchivedExpert:
    expert_id: str
    place: int
    signature: np.ndarray  # (n_edges,) mean causal signature while this expert was active
    state_dict: dict
    trained_from_index: int
    trained_to_index: int


@dataclass
class ReactivationDecision:
    reactivated: bool
    matched_expert_id: str | None
    similarity: float
    reason: str


@dataclass
class ExpertArchive:
    """Per-place archive of hibernated experts (SS4.5's "Archive A/B" in the
    SS4.3 diagram -- one archive per place lineage)."""

    place: int
    similarity_threshold: float = 0.9
    experts: list[ArchivedExpert] = field(default_factory=list)

    def hibernate(
        self,
        expert_id: str,
        expert: torch.nn.Module,
        signature: np.ndarray,
        trained_from_index: int,
        trained_to_index: int,
    ) -> ArchivedExpert:
        """Store a deep copy of the expert's weights plus the causal
        signature that characterized the regime it was trained for."""
        archived = ArchivedExpert(
            expert_id=expert_id,
            place=self.place,
            signature=np.asarray(signature, dtype=np.float64).copy(),
            state_dict=copy.deepcopy(expert.state_dict()),
            trained_from_index=trained_from_index,
            trained_to_index=trained_to_index,
        )
        self.experts.append(archived)
        return archived

    def best_match(self, signature: np.ndarray) -> tuple[ArchivedExpert | None, float]:
        """Similarity search across the WHOLE archive (SS4.5: "not just the
        one being replaced")."""
        if not self.experts:
            return None, 0.0
        sims = [cosine_similarity(np.asarray(signature, dtype=np.float64), a.signature) for a in self.experts]
        best_i = int(np.argmax(sims))
        return self.experts[best_i], float(sims[best_i])

    def try_reactivate(
        self,
        signature: np.ndarray,
        expert: torch.nn.Module,
    ) -> ReactivationDecision:
        """If the best archived match clears the threshold, load its weights
        INTO `expert` as a warm start (SS4.5: continue training from those
        weights, don't freeze them as permanently identical) and report the
        decision. Otherwise leave `expert` untouched -- the caller spawns
        fresh."""
        match, sim = self.best_match(signature)
        if match is None:
            return ReactivationDecision(False, None, 0.0, "archive empty -- spawn fresh")
        if sim < self.similarity_threshold:
            return ReactivationDecision(
                False, match.expert_id, sim,
                f"best match {match.expert_id} similarity {sim:.3f} < threshold "
                f"{self.similarity_threshold:.3f} -- spawn fresh",
            )
        expert.load_state_dict(copy.deepcopy(match.state_dict))
        return ReactivationDecision(
            True, match.expert_id, sim,
            f"reactivated {match.expert_id} as warm start (similarity {sim:.3f})",
        )
