"""CIA (Cross-environment Intra-class Alignment), adapted for regression --
attempted fix for the density/ranking weakness documented across steps 2,
3, 4 and 8 (§10).

Source: Wang et al., "Dissecting the Failure of Invariant Learning on
Graphs" (NeurIPS 2024, arXiv 2411.02847), already cited in §10 step 2 as
the theoretical explanation for why our VREx-style loss
(mean_risk + lambda*Var[risk]) has NON-UNIQUE optima on graphs -- many
edge selections reach equally low loss, only some of which are genuinely
causal, because nothing in the plain objective breaks that tie.

**What CIA adds, in the original (classification) paper:** an alignment
term that pulls together the hidden representations of two INSTANCES THAT
SHARE THE SAME CLASS LABEL, even when their non-causal environment/
background has been swapped. If the causal part is doing its job, two
same-outcome instances should look similar in representation space
REGARDLESS of which environment's spurious features got swapped in; if the
model is still leaning on the spurious part, swapping environments would
visibly perturb the representation even between same-class instances --
exactly the leak CIA is designed to catch and penalize.

**Why classification's mechanism can't be copied verbatim.** "Same class"
is a hard, binary yes/no in classification: label(a) == label(b). This
project's target (OLR N days ahead) is continuous, so no such binary
partition exists to define which pairs are the "same class" for alignment.

**Adaptation made here (this project's own design, not from the paper):**
replace the hard same-class test with a continuous CLOSENESS WEIGHT
between two instances' TRUE TARGETS -- a Gaussian kernel on
|y_a - y_b| -- so pairs with very similar true outcomes are pulled
together strongly, pairs with very different outcomes are pulled together
weakly or not at all, and the transition is smooth rather than a hard
cutoff (no arbitrary binning of a continuous target into artificial
"classes" is needed, which would have been the other option and would
have introduced an unmotivated hyperparameter -- the bin edges).

Concretely, for a BATCH of (anchor, environment) pairs, each anchor i
produces TWO hidden representations under two different sampled swapped
environments h_i^(1), h_i^(2) (both using the SAME causal edges/weights --
only the spurious half's context differs, exactly DIR-GNN's own
intervention design, §4.2). The CIA loss is a weighted sum over all pairs
(i, j) in the batch of:

    weight(i,j) * || h_i^(1) - h_j^(1) ||^2   [pulled together if y_i~y_j]

using representations from the SAME swap index so the comparison isolates
"do same-outcome anchors look alike" rather than mixing in swap-to-swap
noise -- that noise is already what the existing variance-across-swaps
term measures.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CIALossOutput:
    loss: torch.Tensor
    mean_pairwise_weight: torch.Tensor  # diagnostic: how much pull is actually being applied


def target_closeness_weights(y_batch: torch.Tensor, bandwidth: float) -> torch.Tensor:
    """y_batch: (batch,) true targets for a batch of anchors.
    bandwidth: kernel bandwidth (same units as y) -- controls how close two
        targets must be to count as "the same regime" for alignment
        purposes. Analogous to choosing bin width if targets had been
        discretized, but smooth instead of a hard cutoff.
    Returns: (batch, batch) symmetric weight matrix in (0, 1], diagonal = 1.
    """
    diff = y_batch.unsqueeze(0) - y_batch.unsqueeze(1)  # (batch, batch)
    return torch.exp(-(diff**2) / (2.0 * bandwidth**2))


def cia_alignment_loss(
    hidden_batch: torch.Tensor,
    y_batch: torch.Tensor,
    bandwidth: float,
    exclude_diagonal: bool = True,
) -> CIALossOutput:
    """hidden_batch: (batch, hidden_channels) -- one hidden representation
        per anchor in the batch, from predict_causal(..., return_hidden=True)
        under a FIXED swap index (see module docstring: comparisons must be
        same-swap-index to isolate the causal-vs-spurious question from
        swap-to-swap variance, which the existing VREx term already covers).
    y_batch: (batch,) true targets, matching hidden_batch row-for-row.
    bandwidth: passed to target_closeness_weights.
    exclude_diagonal: drop self-pairs (i==i) from the loss (they are
        trivially identical and would just dilute the average with zeros).

    Returns weighted mean squared distance between all pairs' hidden
    representations, weighted by how close their TRUE targets are -- a
    small value means same-outcome anchors already look alike (good,
    causal-only reasoning); a large value means they don't (spurious
    leakage into the "causal" representation).
    """
    batch = hidden_batch.shape[0]
    weights = target_closeness_weights(y_batch, bandwidth)  # (batch, batch)

    diffs = hidden_batch.unsqueeze(0) - hidden_batch.unsqueeze(1)  # (batch, batch, hidden)
    sq_dist = (diffs**2).sum(dim=-1)  # (batch, batch)

    if exclude_diagonal:
        mask = ~torch.eye(batch, dtype=torch.bool, device=hidden_batch.device)
        weights = weights * mask
        denom = weights.sum().clamp_min(1e-8)
    else:
        denom = weights.sum().clamp_min(1e-8)

    loss = (weights * sq_dist).sum() / denom
    mean_weight = weights.sum() / max(batch * batch - (batch if exclude_diagonal else 0), 1)
    return CIALossOutput(loss=loss, mean_pairwise_weight=mean_weight)
