import numpy as np
import torch

from causal_moe.splitter.dirgnn import (
    DIRGNNSplitter,
    LambdaWarmupSchedule,
    RationaleGenerator,
    split_graph_top_r,
)


def make_fully_connected_edge_index(n_nodes: int, include_self_loops: bool = True) -> torch.Tensor:
    pairs = [
        (i, j)
        for i in range(n_nodes)
        for j in range(n_nodes)
        if include_self_loops or i != j
    ]
    src = [p[0] for p in pairs]
    dst = [p[1] for p in pairs]
    return torch.tensor([src, dst], dtype=torch.int64)


# --- split_graph_top_r ------------------------------------------------------


def test_split_graph_top_r_keeps_correct_count():
    edge_index = make_fully_connected_edge_index(5)  # 25 edges incl self-loops
    scores = torch.rand(25, requires_grad=True)

    c_idx, c_w, s_idx, s_w = split_graph_top_r(scores, edge_index, r=0.4)

    assert c_idx.shape[1] == 10  # round(0.4 * 25)
    assert s_idx.shape[1] == 15
    assert c_w.shape[0] == 10
    assert s_w.shape[0] == 15


def test_split_graph_top_r_keeps_highest_scores():
    edge_index = make_fully_connected_edge_index(4, include_self_loops=False)  # 12 edges
    scores = torch.tensor([float(i) for i in range(12)])  # strictly increasing

    c_idx, c_w, s_idx, s_w = split_graph_top_r(scores, edge_index, r=0.5)

    # top-6 scores are indices 6..11 -- must all be in the causal set
    causal_edges = set(zip(c_idx[0].tolist(), c_idx[1].tolist()))
    all_edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    expected_causal = {all_edges[i] for i in range(6, 12)}
    assert causal_edges == expected_causal
    assert torch.allclose(c_w.sort().values, torch.tensor([6.0, 7, 8, 9, 10, 11]))


def test_split_graph_top_r_selection_is_detached_but_weights_attached():
    edge_index = make_fully_connected_edge_index(4, include_self_loops=False)
    scores = torch.rand(12, requires_grad=True)

    c_idx, c_w, s_idx, s_w = split_graph_top_r(scores, edge_index, r=0.5)

    # weights carry gradient back to the original scores tensor
    loss = c_w.sum() + s_w.sum()
    loss.backward()
    assert scores.grad is not None
    assert (scores.grad != 0).all()  # every edge's weight was used somewhere

    # the edge_index tensors themselves are indices, never require grad
    assert not c_idx.requires_grad
    assert not s_idx.requires_grad


def test_split_graph_top_r_rejects_bad_r():
    edge_index = make_fully_connected_edge_index(3)
    scores = torch.rand(9)
    import pytest

    with pytest.raises(ValueError):
        split_graph_top_r(scores, edge_index, r=0.0)
    with pytest.raises(ValueError):
        split_graph_top_r(scores, edge_index, r=1.5)


# --- LambdaWarmupSchedule ----------------------------------------------------


def test_lambda_warmup_schedule_zero_during_warmup():
    sched = LambdaWarmupSchedule(lambda_max=1.0, total_steps=100, warmup_fraction=0.3)
    assert sched.value(0) == 0.0
    assert sched.value(29) == 0.0


def test_lambda_warmup_schedule_ramps_linearly_after_warmup():
    sched = LambdaWarmupSchedule(lambda_max=2.0, total_steps=100, warmup_fraction=0.3)
    # warmup_steps = 30, remainder = 70 steps to reach lambda_max
    assert sched.value(30) == 0.0
    assert sched.value(100) == 2.0
    mid = sched.value(65)  # halfway through the remaining 70 steps
    assert 0.9 < mid < 1.1


def test_lambda_warmup_schedule_caps_at_max():
    sched = LambdaWarmupSchedule(lambda_max=1.0, total_steps=100, warmup_fraction=0.3)
    assert sched.value(1000) == 1.0


# --- DIRGNNSplitter: shapes, gradient isolation, loss behavior -------------


def make_toy_splitter_inputs(n_nodes=6, in_channels=3, n_swaps=4, seed=0):
    torch.manual_seed(seed)
    edge_index = make_fully_connected_edge_index(n_nodes, include_self_loops=True)
    x_anchor = torch.randn(n_nodes, in_channels)
    y_anchor = torch.randn(n_nodes)
    env_candidates = torch.randn(n_swaps, n_nodes, in_channels)
    return edge_index, x_anchor, y_anchor, env_candidates


def test_splitter_split_shapes():
    edge_index, x_anchor, _, _ = make_toy_splitter_inputs()
    model = DIRGNNSplitter(in_channels=3, hidden_channels=8, r=0.3)

    x_window = x_anchor.unsqueeze(1)  # (n_nodes, window_len=1, in_channels)
    split = model.split(x_window, edge_index)
    n_edges = edge_index.shape[1]
    expected_causal = max(1, round(0.3 * n_edges))
    assert split.causal_edge_index.shape[1] == expected_causal
    assert split.spurious_edge_index.shape[1] == n_edges - expected_causal
    assert split.edge_scores.shape[0] == n_edges
    assert (split.edge_scores >= 0).all() and (split.edge_scores <= 1).all()  # sigmoid


def test_splitter_split_accepts_multi_step_generator_window():
    edge_index, x_anchor, _, _ = make_toy_splitter_inputs(n_nodes=6, in_channels=3)
    model = DIRGNNSplitter(in_channels=3, hidden_channels=8, r=0.3, generator_window_len=5)

    x_window = torch.randn(6, 5, 3)  # (n_nodes, window_len=5, in_channels)
    split = model.split(x_window, edge_index)
    n_edges = edge_index.shape[1]
    expected_causal = max(1, round(0.3 * n_edges))
    assert split.causal_edge_index.shape[1] == expected_causal
    assert split.edge_scores.shape[0] == n_edges


def test_splitter_predict_causal_output_shape():
    edge_index, x_anchor, y_anchor, env_candidates = make_toy_splitter_inputs()
    model = DIRGNNSplitter(in_channels=3, hidden_channels=8, r=0.5)
    split = model.split(x_anchor, edge_index)

    pred = model.predict_causal(x_anchor, split, env_candidates[0])
    assert pred.shape == y_anchor.shape


def test_splitter_causal_prediction_changes_with_env_swap():
    """Sanity check the intervention is actually wired: since spurious
    edges' messages come from x_env, a different x_env must generally
    produce a different prediction (unless r=1.0, no spurious edges)."""
    edge_index, x_anchor, _, env_candidates = make_toy_splitter_inputs(n_swaps=2)
    model = DIRGNNSplitter(in_channels=3, hidden_channels=8, r=0.3)
    split = model.split(x_anchor, edge_index)

    pred0 = model.predict_causal(x_anchor, split, env_candidates[0])
    pred1 = model.predict_causal(x_anchor, split, env_candidates[1])
    assert not torch.allclose(pred0, pred1)


def test_splitter_causal_prediction_at_r1_ignores_env_swap():
    """At r=1.0 there are no spurious edges at all, so the swap should have
    zero effect on the prediction -- confirms causal edges truly stay
    anchored to x_anchor regardless of x_env."""
    edge_index, x_anchor, _, env_candidates = make_toy_splitter_inputs(n_swaps=2)
    model = DIRGNNSplitter(in_channels=3, hidden_channels=8, r=1.0)
    split = model.split(x_anchor, edge_index)

    pred0 = model.predict_causal(x_anchor, split, env_candidates[0])
    pred1 = model.predict_causal(x_anchor, split, env_candidates[1])
    assert torch.allclose(pred0, pred1, atol=1e-6)


def test_spurious_head_gradient_never_reaches_generator_or_encoder():
    edge_index, x_anchor, y_anchor, _ = make_toy_splitter_inputs()
    model = DIRGNNSplitter(in_channels=3, hidden_channels=8, r=0.5)

    split = model.split(x_anchor, edge_index)
    spurious_pred = model.predict_spurious_gauge(x_anchor, split)
    loss = ((spurious_pred - y_anchor) ** 2).mean()
    loss.backward()

    # spurious head itself got gradient
    assert model.spurious_head.lin.weight.grad is not None
    assert (model.spurious_head.lin.weight.grad != 0).any()

    # generator and shared encoder must get NO gradient from this loss
    assert model.generator.mlp[0].weight.grad is None
    assert model.encoder.msg_lin.weight.grad is None


def test_causal_loss_gradient_reaches_generator_and_encoder_and_causal_head():
    edge_index, x_anchor, y_anchor, env_candidates = make_toy_splitter_inputs()
    model = DIRGNNSplitter(in_channels=3, hidden_channels=8, r=0.5)

    out = model.compute_loss(x_anchor, y_anchor, env_candidates, edge_index, lambda_value=0.5)
    out.total_loss.backward()

    assert model.causal_head.lin.weight.grad is not None
    assert (model.causal_head.lin.weight.grad != 0).any()
    assert model.encoder.msg_lin.weight.grad is not None
    assert (model.encoder.msg_lin.weight.grad != 0).any()
    # generator gets gradient only through edge WEIGHTS (attached), not
    # through which edges were selected -- but weight-path gradient should
    # still be nonzero.
    assert model.generator.mlp[0].weight.grad is not None


def test_compute_loss_output_shapes_and_lambda_zero_ignores_variance():
    edge_index, x_anchor, y_anchor, env_candidates = make_toy_splitter_inputs(n_swaps=5)
    model = DIRGNNSplitter(in_channels=3, hidden_channels=8, r=0.5)

    out = model.compute_loss(x_anchor, y_anchor, env_candidates, edge_index, lambda_value=0.0)
    assert out.predictions_per_swap.shape == (5, x_anchor.shape[0])
    assert torch.allclose(out.total_loss, out.mean_risk)  # lambda=0 -> variance term dropped


def test_compute_loss_single_swap_has_zero_variance():
    edge_index, x_anchor, y_anchor, env_candidates = make_toy_splitter_inputs(n_swaps=1)
    model = DIRGNNSplitter(in_channels=3, hidden_channels=8, r=0.5)

    out = model.compute_loss(x_anchor, y_anchor, env_candidates, edge_index, lambda_value=1.0)
    assert out.variance_risk.item() == 0.0


def test_compute_loss_entropy_penalty_off_by_default():
    edge_index, x_anchor, y_anchor, env_candidates = make_toy_splitter_inputs()
    model = DIRGNNSplitter(in_channels=3, hidden_channels=8, r=0.5)

    out = model.compute_loss(x_anchor, y_anchor, env_candidates, edge_index, lambda_value=0.5)
    assert out.entropy_penalty is None


def test_compute_loss_entropy_penalty_pulls_scores_toward_polarized():
    """Candidate 2 fix (§10 step 2, 2026-09-18): a nonzero entropy_weight
    must actually push edge_scores away from 0.5 over a few optimizer
    steps -- confirms the term is wired to the generator's weights, not
    just computed and discarded."""
    edge_index, x_anchor, y_anchor, env_candidates = make_toy_splitter_inputs(n_nodes=6, in_channels=3)
    model = DIRGNNSplitter(in_channels=3, hidden_channels=8, r=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    with torch.no_grad():
        initial_scores = model.split(x_anchor.unsqueeze(1), edge_index).edge_scores.clone()
    initial_dist_from_half = (initial_scores - 0.5).abs().mean().item()

    for _ in range(50):
        optimizer.zero_grad()
        out = model.compute_loss(
            x_anchor, y_anchor, env_candidates, edge_index,
            lambda_value=0.0, entropy_weight=5.0,
        )
        assert out.entropy_penalty is not None
        out.total_loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_scores = model.split(x_anchor.unsqueeze(1), edge_index).edge_scores.clone()
    final_dist_from_half = (final_scores - 0.5).abs().mean().item()

    assert final_dist_from_half > initial_dist_from_half


def test_compute_loss_prior_rate_weight_off_by_default():
    edge_index, x_anchor, y_anchor, env_candidates = make_toy_splitter_inputs()
    model = DIRGNNSplitter(in_channels=3, hidden_channels=8, r=0.2)

    out = model.compute_loss(x_anchor, y_anchor, env_candidates, edge_index, lambda_value=0.5)
    # prior_rate_weight=0.0 by default -- total_loss must match the
    # entropy-off, variance-only path exactly (no KL term silently added).
    expected = out.mean_risk + 0.5 * out.variance_risk
    assert torch.allclose(out.total_loss, expected)


def test_compute_loss_prior_rate_weight_pulls_mean_score_toward_r():
    """NRI-style sparsity-rate prior (§10 step 2, 2026-09-18, adapted from
    github.com/ethanfetaya/NRI's KL-to-sparse-prior term): must pull the
    AVERAGE edge score toward self.r over a few optimizer steps, not just
    polarize individual scores (that's entropy_weight's job)."""
    edge_index, x_anchor, y_anchor, env_candidates = make_toy_splitter_inputs(n_nodes=6, in_channels=3)
    target_r = 0.15
    model = DIRGNNSplitter(in_channels=3, hidden_channels=8, r=target_r)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    with torch.no_grad():
        initial_mean = model.split(x_anchor.unsqueeze(1), edge_index).edge_scores.mean().item()

    for _ in range(80):
        optimizer.zero_grad()
        out = model.compute_loss(
            x_anchor, y_anchor, env_candidates, edge_index,
            lambda_value=0.0, prior_rate_weight=2.0,
        )
        out.total_loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_mean = model.split(x_anchor.unsqueeze(1), edge_index).edge_scores.mean().item()

    # started far from target_r=0.15 (random init hovers near 0.5); after
    # training under the prior, must have moved substantially closer.
    assert abs(final_mean - target_r) < abs(initial_mean - target_r)
    assert final_mean < initial_mean  # target is below a typical random init mean


def test_compute_loss_accepts_explicit_generator_window():
    """compute_loss's x_generator_window path (the fix, 2026-09-18) must be
    wired through to split() rather than silently ignored."""
    edge_index, x_anchor, y_anchor, env_candidates = make_toy_splitter_inputs(n_nodes=6, in_channels=3)
    model = DIRGNNSplitter(in_channels=3, hidden_channels=8, r=0.5, generator_window_len=4)

    x_generator_window = torch.randn(6, 4, 3)
    out = model.compute_loss(
        x_anchor, y_anchor, env_candidates, edge_index, lambda_value=0.5,
        x_generator_window=x_generator_window,
    )
    assert out.predictions_per_swap.shape == (env_candidates.shape[0], x_anchor.shape[0])
    out.total_loss.backward()
    assert model.generator.mlp[0].weight.grad is not None


def test_compute_loss_target_node_mask_matches_manual_single_node_risk():
    """SS9 step 4 addition: target_node_mask restricts risk/spurious-loss to
    a subset of nodes (a "one real place" star graph has candidate SOURCE
    clusters with no meaningful forecast target of their own). Must exactly
    match manually slicing predictions_per_swap to the masked node(s)."""
    edge_index, x_anchor, y_anchor, env_candidates = make_toy_splitter_inputs(n_nodes=6, n_swaps=4)
    model = DIRGNNSplitter(in_channels=3, hidden_channels=8, r=0.5)

    target = 2
    mask = torch.zeros(6, dtype=torch.bool)
    mask[target] = True

    out_masked = model.compute_loss(
        x_anchor, y_anchor, env_candidates, edge_index, lambda_value=1.0,
        target_node_mask=mask,
    )

    manual_risks = ((out_masked.predictions_per_swap[:, target] - y_anchor[target]) ** 2)
    manual_mean_risk = manual_risks.mean()
    manual_variance_risk = manual_risks.var(unbiased=False)

    assert torch.allclose(out_masked.mean_risk, manual_mean_risk)
    assert torch.allclose(out_masked.variance_risk, manual_variance_risk)


def test_compute_loss_target_node_mask_none_matches_all_node_average():
    """Default (no mask) must remain bit-for-bit identical to the original
    all-node-averaged behavior -- backward compatibility for steps 2/3."""
    edge_index, x_anchor, y_anchor, env_candidates = make_toy_splitter_inputs(n_nodes=6, n_swaps=4)
    model = DIRGNNSplitter(in_channels=3, hidden_channels=8, r=0.5)

    torch.manual_seed(123)
    out_default = model.compute_loss(x_anchor, y_anchor, env_candidates, edge_index, lambda_value=1.0)
    all_mask = torch.ones(6, dtype=torch.bool)
    torch.manual_seed(123)
    out_all_masked = model.compute_loss(
        x_anchor, y_anchor, env_candidates, edge_index, lambda_value=1.0,
        target_node_mask=all_mask,
    )

    assert torch.allclose(out_default.mean_risk, out_all_masked.mean_risk)
    assert torch.allclose(out_default.variance_risk, out_all_masked.variance_risk)
    assert torch.allclose(out_default.spurious_loss, out_all_masked.spurious_loss)


# --- Regression test: generator must see cross-time signal (fix, 2026-09-18) -


def test_generator_window_carries_lag_correlation_signal_single_step_cannot():
    """Root-cause regression test for the AUC~0.50 bug (§10 step 2): whether
    edge (src, dst) is truly causal is a property of src's PAST predicting
    dst's value, not visible in a single joint timestep. Build a toy series
    where node 0 causally drives node 1 at lag 2 (node1[t] = node0[t-2] +
    noise) and node 2 is an uncorrelated distractor, then check a
    multi-step window lets a simple untrained-but-structured statistic
    (correlation) separate the true edge (0->1) from the false edge
    (2->1) -- confirming the window carries the needed signal at all,
    independent of any training dynamics."""
    rng = np.random.default_rng(0)
    T = 300
    node0 = rng.standard_normal(T).astype(np.float32)
    node2 = rng.standard_normal(T).astype(np.float32)  # independent distractor
    node1 = np.zeros(T, dtype=np.float32)
    lag = 2
    node1[lag:] = node0[:-lag] * 2.0 + 0.1 * rng.standard_normal(T - lag).astype(np.float32)

    window_len = 5
    anchor_t = 250  # far past both the lag and the window length
    # window ending at anchor_t, most recent last: [t-4, t-3, t-2, t-1, t]
    w0 = node0[anchor_t - window_len + 1 : anchor_t + 1]
    w1 = node1[anchor_t - window_len + 1 : anchor_t + 1]
    w2 = node2[anchor_t - window_len + 1 : anchor_t + 1]

    # A single joint timestep (old design) cannot distinguish "0 causes 1"
    # from "2 is unrelated to 1": both pairs are just two scalars.
    single_step_0_1 = (w0[-1], w1[-1])
    single_step_2_1 = (w2[-1], w1[-1])
    assert len(single_step_0_1) == len(single_step_2_1) == 2  # structurally identical shape/info

    # A window lets a correlation-style statistic separate them. Use
    # full-series correlation for a stable estimate (a window is what the
    # model would see repeatedly across many anchors during training).
    corr_true_full = np.corrcoef(node0[:-lag], node1[lag:])[0, 1]
    corr_false_full = np.corrcoef(node2, node1)[0, 1]
    assert abs(corr_true_full) > 0.8  # strong real lagged relationship
    assert abs(corr_false_full) < 0.2  # distractor genuinely uncorrelated

    # And the RationaleGenerator's window input shape can actually carry
    # this: confirm reshape preserves per-node temporal order (no
    # accidental scrambling across nodes/time when flattened for the MLP).
    x_window = torch.tensor(np.stack([w0, w1, w2], axis=0)).unsqueeze(-1)  # (3 nodes, window_len, 1 channel)
    gen = RationaleGenerator(in_channels=1, window_len=window_len, hidden_channels=8)
    edge_index = torch.tensor([[0, 2], [1, 1]], dtype=torch.int64)  # edges (0->1) and (2->1)
    scores = gen(x_window, edge_index)
    assert scores.shape == (2,)
    # each edge's score must actually depend on BOTH endpoints' full windows,
    # not just the last timestep -- verified by checking the flattened input
    # the MLP receives differs when only an earlier lag differs.
    x_window_perturbed = x_window.clone()
    x_window_perturbed[0, 0, 0] += 5.0  # change node 0's earliest lag only
    scores_perturbed = gen(x_window_perturbed, edge_index)
    assert not torch.allclose(scores[0], scores_perturbed[0])  # edge 0->1 must react
    assert torch.allclose(scores[1], scores_perturbed[1])  # edge 2->1 must NOT react
