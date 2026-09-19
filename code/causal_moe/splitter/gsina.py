"""GSINA (Graph Sinkhorn Attention), adapted for this project's regression
splitter -- second named fix for the ranking weakness documented across
steps 2, 3, 4 and 8 (§10), and now also confirmed NOT fixed by CIA
(`causal_moe/splitter/cia.py`, §10 "CIA-for-regression" entry, 2026-09-19).

Source: Liu et al., "GSINA: Improving Subgraph Extraction for Graph
Invariant Learning via Graph Sinkhorn Attention" (arXiv 2402.07191, 2024),
already cited in §10 step 2 as the second identified fix.

**What DIR-GNN's current selection does (`split_graph_top_r`,
`causal_moe/splitter/dirgnn.py`), and why it's suspected as part of the
problem:** compute a soft score per edge, then pick the top-r by count
using `numpy.argpartition` on a DETACHED copy -- the selection DECISION
itself gets zero gradient, only the selected edges' WEIGHTS do. This
matches DIR-GNN's own published code exactly (§4.2), but it means the
rationale generator only ever gets gradient signal of the form "make your
already-selected edges' weights more/less confident" -- never "you should
have selected a DIFFERENT edge instead". If the generator's initial scores
are close to uniform (exactly what steps 4/5/7/8 measured on real data:
std 0.0017-0.0051), there's very little pressure ever pushing a genuinely
better edge across the top-r cutoff, because crossing that cutoff isn't
something gradient descent can request -- it's a discrete accident of
which detached score happened to be marginally larger this step.

**What GSINA changes:** replaces the hard, detached top-r cutoff with a
SOFT, DIFFERENTIABLE, but still (approximately) SPARSE selection, using
Sinkhorn normalization -- so gradient can flow through "which edges get
emphasized", not only "how strongly is each selected edge weighted".

**This project's adaptation (not from the paper, same spirit as CIA's own
"regression-adapted" framing):** the original GSINA operates on a
bipartite-style attention/transport matrix within a full GNN readout. This
project's splitter already has a single per-edge scalar score (from
`RationaleGenerator`, §4.2) rather than a multi-row attention matrix, so
the adaptation made here treats "keep vs. drop" as a 2-column assignment
problem per edge -- column 0 = "kept" mass, column 1 = "dropped" mass --
and uses Sinkhorn iterations to jointly (a) push each edge toward a
near-one-hot kept/dropped decision (like the paper's sparsity property)
while (b) enforcing that the TOTAL kept mass across all edges sums to
~r*n_edges (matching the same target sparsity `r` the hard top-r selection
targets, so GSINA is a drop-in comparison, not a different hyperparameter
regime). This keeps the same "detach only the true discreteness, keep
everything else on the graph" spirit as DIR-GNN's own mechanism, but moves
the point of detachment from "which edges are top-r" (hard, in the
original) to "how many total iterations of Sinkhorn normalization run"
(a fixed hyperparameter, no detachment needed at all -- Sinkhorn iteration
is itself fully differentiable).

**Deliberately kept separate from CIA and from `dirgnn.py`'s existing
`split_graph_top_r`:** per the user's original request (§10), GSINA and CIA
must stay independently testable/comparable, never combined in the same
experiment. This module does not import `cia.py` or modify
`split_graph_top_r` -- it is an alternative SELECTION function with the
same signature shape, meant to be swapped in wherever
`DIRGNNSplitter.split()` currently calls `split_graph_top_r`, by a caller
that chooses one mechanism or the other, not both at once.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GSINASelection:
    causal_edge_index: torch.Tensor
    causal_edge_weight: torch.Tensor  # differentiable "kept" mass, one value per candidate edge (may include near-zero entries -- see note below)
    spurious_edge_index: torch.Tensor
    spurious_edge_weight: torch.Tensor
    keep_prob: torch.Tensor  # (n_edges,) differentiable soft keep-probability, pre-hard-split -- the quantity gradient actually flows through


def sinkhorn_keep_probability(
    edge_scores: torch.Tensor,
    r: float,
    n_iters: int = 20,
    temperature: float = 1.0,
    gumbel_noise: bool = False,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Differentiable soft keep-probability per edge, via Sinkhorn
    normalization over a 2-column (keep, drop) assignment matrix.

    edge_scores: (n_edges,) raw (pre-sigmoid) generator scores -- same
        input `split_graph_top_r` receives after a sigmoid, but this
        function does its own logit construction (see below), so pass the
        RAW scores here, not an already-sigmoided probability.
    r: target keep fraction in (0, 1] -- same meaning as DIR-GNN's top-r,
        so results are comparable at matched sparsity.
    n_iters: number of Sinkhorn row/column normalization rounds. More
        iterations -> closer to a hard, one-hot-per-row assignment
        (sparser, per the paper's own sparsity property); fewer -> softer,
        smoother gradients early in training. Fully differentiable at any
        iteration count -- no detaching, unlike DIR-GNN's argpartition.
    temperature: divides the logits before Sinkhorn -- lower temperature
        sharpens the eventual keep/drop decision (same role as a Gumbel-
        softmax temperature schedule, §4.2's related-work comparison).
    gumbel_noise: if True, add Gumbel(0,1) noise to the logits before
        normalizing (standard Sinkhorn-Gumbel practice, encourages
        exploration early in training, analogous to this project's
        already-used lambda warm-up idea for a different term). Off by
        default so the function is deterministic for tests; the training
        script controls this via its own args.
    generator: optional torch.Generator for reproducible Gumbel draws.

    Returns: (n_edges,) differentiable keep-probability in (0, 1), summing
    to approximately r * n_edges (the target sparsity budget) -- NOT
    exactly r*n_edges (Sinkhorn only converges to the target row/column
    sums in the limit of infinite iterations), which is expected and
    matches the paper's own "approximately sparse" framing.
    """
    if not (0.0 < r <= 1.0):
        raise ValueError(f"r must be in (0, 1], got {r}")
    n_edges = edge_scores.shape[0]

    # Build a (n_edges, 2) logit matrix: column 0 = "keep" logit (the raw
    # generator score itself -- higher score, more inclined to keep),
    # column 1 = "drop" logit (fixed at 0, i.e. keep/drop compete via their
    # DIFFERENCE, same as a 2-way softmax over [score, 0]).
    logits = torch.stack([edge_scores, torch.zeros_like(edge_scores)], dim=1)  # (n_edges, 2)
    logits = logits / temperature

    if gumbel_noise:
        u = torch.rand(logits.shape, generator=generator, dtype=logits.dtype, device=logits.device).clamp_min(1e-8)
        gumbel = -torch.log(-torch.log(u))
        logits = logits + gumbel

    # Sinkhorn in log-space for numerical stability. Target marginals:
    # each EDGE (row) should sum to 1 (it's either kept or dropped, softly)
    # -- that's the standard row-normalization every iteration enforces.
    # The COLUMN sum for "keep" is what encodes the sparsity budget: we
    # want the total keep-mass across all edges to be ~ r * n_edges, so
    # column 0 is target-normalized to that sum, column 1 to the remainder
    # (n_edges - r*n_edges), once per iteration -- this is what pulls the
    # solution toward "roughly r*n_edges edges fully kept, the rest fully
    # dropped" (the paper's sparsity property) rather than every edge
    # sitting at the same soft 0.5.
    target_keep_mass = r * n_edges
    target_drop_mass = n_edges - target_keep_mass
    log_p = torch.log_softmax(logits, dim=1)  # start: valid row distribution, (n_edges, 2)

    eps = 1e-8
    for _ in range(n_iters):
        # Column normalization: rescale each column (in log-space, i.e.
        # subtract logsumexp) so column sums match the target keep/drop
        # mass, then re-normalize rows back to sum-to-1 (alternating
        # projection, the standard Sinkhorn-Knopp scheme).
        col_logsumexp = torch.logsumexp(log_p, dim=0)  # (2,)
        target_log_mass = torch.log(
            torch.tensor([target_keep_mass, target_drop_mass], dtype=log_p.dtype, device=log_p.device).clamp_min(eps)
        )
        log_p = log_p + (target_log_mass - col_logsumexp).unsqueeze(0)
        # Row normalization: back to a valid per-edge distribution.
        log_p = log_p - torch.logsumexp(log_p, dim=1, keepdim=True)

    keep_prob = log_p[:, 0].exp()
    return keep_prob


def gsina_split(
    edge_scores: torch.Tensor,
    edge_index: torch.Tensor,
    r: float,
    n_iters: int = 20,
    temperature: float = 1.0,
    gumbel_noise: bool = False,
    hard_at_eval: bool = False,
    generator: torch.Generator | None = None,
) -> GSINASelection:
    """Drop-in alternative to `causal_moe.splitter.dirgnn.split_graph_top_r`
    -- same edge_index/r contract, but the causal/spurious split is a soft,
    fully differentiable partition (via `sinkhorn_keep_probability`) rather
    than a hard, gradient-detached top-r cutoff.

    edge_scores: (n_edges,) RAW generator scores (pre-sigmoid) -- matches
        `RationaleGenerator.forward`'s output directly; this function
        applies its own Sinkhorn-based transform instead of a sigmoid.
    edge_index: (2, n_edges) candidate edges, same convention as
        `split_graph_top_r`.
    r: target keep fraction, same meaning as DIR-GNN's top-r.
    hard_at_eval: if True AND the module is in eval mode is not checked
        here (this is a plain function, not an nn.Module) -- callers that
        want a hard decision at inference time should threshold
        `keep_prob` themselves (e.g. `keep_prob > 0.5`) using this flag as
        a reminder that GSINA is soft by default, unlike DIR-GNN's
        top-r (which is already hard everywhere). Kept as an explicit,
        named argument (rather than silently changing behavior) so a
        caller must opt in.

    Returns a GSINASelection where causal/spurious edge index/weight cover
    ALL candidate edges in both groups (unlike `split_graph_top_r`, which
    partitions edges into two disjoint sets) -- every edge appears in BOTH
    the causal and spurious groups, weighted by `keep_prob` and
    `1 - keep_prob` respectively. This is intentional, not a bug: a hard
    partition would reintroduce exactly the non-differentiable "which
    edges" decision GSINA exists to remove. The causal head therefore
    receives every edge's message, scaled by how strongly GSINA believes
    that edge is causal -- gradient can now shift MASS from one edge to
    another instead of only reweighting an already-fixed hard selection.
    """
    keep_prob = sinkhorn_keep_probability(
        edge_scores, r, n_iters=n_iters, temperature=temperature,
        gumbel_noise=gumbel_noise, generator=generator,
    )

    if hard_at_eval:
        # Straight-through style hard decision for inference/reporting
        # only: forward value is a hard 0/1 mask, but if this is ever
        # called with gradients enabled the caller has explicitly opted
        # into losing gradient through the mask (documented above) --
        # implemented via detach + add-back so autograd doesn't silently
        # break for a caller who calls this inside a training loop anyway.
        hard = (keep_prob > 0.5).to(keep_prob.dtype)
        keep_prob_used = hard.detach() + keep_prob - keep_prob.detach()
    else:
        keep_prob_used = keep_prob

    return GSINASelection(
        causal_edge_index=edge_index,
        causal_edge_weight=keep_prob_used,
        spurious_edge_index=edge_index,
        spurious_edge_weight=1.0 - keep_prob_used,
        keep_prob=keep_prob,
    )
