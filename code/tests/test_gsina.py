import torch

from causal_moe.splitter.gsina import gsina_split, sinkhorn_keep_probability


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


# --- sinkhorn_keep_probability ----------------------------------------------


def test_sinkhorn_keep_probability_shape_and_range():
    scores = torch.randn(20)
    keep = sinkhorn_keep_probability(scores, r=0.3, n_iters=20)
    assert keep.shape == (20,)
    assert (keep >= 0).all() and (keep <= 1).all()


def test_sinkhorn_keep_probability_rejects_bad_r():
    scores = torch.randn(10)
    try:
        sinkhorn_keep_probability(scores, r=0.0)
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        sinkhorn_keep_probability(scores, r=1.5)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_sinkhorn_keep_probability_mass_approximately_matches_r():
    """Total keep-mass across all edges should converge toward r * n_edges
    -- this is the sparsity-budget property that makes GSINA comparable to
    DIR-GNN's exact-count top-r selection at the same r."""
    torch.manual_seed(0)
    n_edges = 50
    scores = torch.randn(n_edges)
    keep = sinkhorn_keep_probability(scores, r=0.3, n_iters=50)
    target_mass = 0.3 * n_edges
    assert abs(keep.sum().item() - target_mass) < 1.0  # within 1 edge's worth of mass


def test_sinkhorn_keep_probability_prefers_higher_scores():
    """Sinkhorn should still respect score ORDER: given a clear separation
    between high- and low-scoring edges, the high scorers should end up
    with higher keep-probability on average."""
    torch.manual_seed(0)
    scores = torch.cat([torch.full((5,), 5.0), torch.full((15,), -5.0)])  # 5 clearly-high, 15 clearly-low
    keep = sinkhorn_keep_probability(scores, r=0.25, n_iters=50)  # r*20 = 5, matches the high group exactly
    assert keep[:5].mean() > keep[5:].mean()
    # at this large a score gap and enough iterations, should be close to a
    # clean separation: high group near 1, low group near 0
    assert keep[:5].mean() > 0.8
    assert keep[5:].mean() < 0.2


def test_sinkhorn_keep_probability_differentiable():
    scores = torch.randn(12, requires_grad=True)
    keep = sinkhorn_keep_probability(scores, r=0.4, n_iters=20)
    loss = keep.sum()
    loss.backward()
    assert scores.grad is not None
    assert (scores.grad != 0).any()


def test_sinkhorn_keep_probability_gradient_can_move_mass_between_edges():
    """The key claimed advantage over split_graph_top_r: gradient should be
    able to push mass FROM one edge TO another, not just reweight an
    already-fixed hard selection. Verify indirectly: perturbing one edge's
    score changes ANOTHER edge's keep-probability (a coupling effect that
    split_graph_top_r's detached argpartition cannot produce for edges on
    the same side of the cutoff)."""
    torch.manual_seed(0)
    base_scores = torch.randn(10)
    keep_base = sinkhorn_keep_probability(base_scores, r=0.3, n_iters=30)

    perturbed_scores = base_scores.clone()
    perturbed_scores[0] += 10.0  # make edge 0 dramatically more attractive
    keep_perturbed = sinkhorn_keep_probability(perturbed_scores, r=0.3, n_iters=30)

    # edge 0's own keep-prob should rise
    assert keep_perturbed[0] > keep_base[0]
    # the total mass budget is conserved (~r*n_edges either way), so SOME
    # other edge's mass must have been redistributed away
    other_mass_base = keep_base[1:].sum()
    other_mass_perturbed = keep_perturbed[1:].sum()
    assert other_mass_perturbed < other_mass_base


def test_sinkhorn_temperature_sharpens_selection():
    """Lower temperature should push keep-probabilities closer to {0, 1}
    (sharper, more top-r-like) than a higher temperature, matching the
    docstring's claim (same role as Gumbel-softmax temperature)."""
    torch.manual_seed(0)
    scores = torch.randn(20)
    keep_sharp = sinkhorn_keep_probability(scores, r=0.3, n_iters=30, temperature=0.1)
    keep_soft = sinkhorn_keep_probability(scores, r=0.3, n_iters=30, temperature=5.0)

    def polarization(p):
        # distance from 0.5, averaged -- higher = more polarized/sparse
        return (p - 0.5).abs().mean()

    assert polarization(keep_sharp) > polarization(keep_soft)


# --- gsina_split -------------------------------------------------------------


def test_gsina_split_shapes_cover_all_edges_in_both_groups():
    edge_index = make_fully_connected_edge_index(4, include_self_loops=False)  # 12 edges
    scores = torch.randn(12)
    out = gsina_split(scores, edge_index, r=0.5, n_iters=20)

    # unlike split_graph_top_r, GSINA weights EVERY edge in both groups
    assert out.causal_edge_index.shape[1] == 12
    assert out.spurious_edge_index.shape[1] == 12
    assert out.causal_edge_weight.shape[0] == 12
    assert out.spurious_edge_weight.shape[0] == 12
    assert torch.equal(out.causal_edge_index, edge_index)
    assert torch.equal(out.spurious_edge_index, edge_index)


def test_gsina_split_causal_and_spurious_weights_sum_to_one_per_edge():
    edge_index = make_fully_connected_edge_index(4, include_self_loops=False)
    scores = torch.randn(12)
    out = gsina_split(scores, edge_index, r=0.5, n_iters=20)
    assert torch.allclose(out.causal_edge_weight + out.spurious_edge_weight, torch.ones(12), atol=1e-5)


def test_gsina_split_gradient_flows_through_both_weight_groups():
    edge_index = make_fully_connected_edge_index(4, include_self_loops=False)
    scores = torch.randn(12, requires_grad=True)
    out = gsina_split(scores, edge_index, r=0.5, n_iters=20)
    loss = out.causal_edge_weight.sum() + out.spurious_edge_weight.sum()
    loss.backward()
    assert scores.grad is not None


def test_gsina_split_hard_at_eval_produces_binary_weights():
    edge_index = make_fully_connected_edge_index(4, include_self_loops=False)
    scores = torch.tensor([10.0, 10.0, 10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0])
    out = gsina_split(scores, edge_index, r=0.25, n_iters=30, hard_at_eval=True)  # r*12 = 3, matches the high group
    unique_vals = torch.unique(torch.round(out.causal_edge_weight * 1000) / 1000)
    assert set(unique_vals.tolist()).issubset({0.0, 1.0})


def test_gsina_split_hard_at_eval_still_allows_gradient_to_flow():
    """Straight-through: forward is hard, but gradient should still reach
    the original scores (via the detach+add-back trick), matching how
    split_graph_top_r keeps weights attached even though selection is hard."""
    edge_index = make_fully_connected_edge_index(4, include_self_loops=False)
    scores = torch.randn(12, requires_grad=True)
    out = gsina_split(scores, edge_index, r=0.5, n_iters=20, hard_at_eval=True)
    loss = out.causal_edge_weight.sum()
    loss.backward()
    assert scores.grad is not None
    assert (scores.grad != 0).any()


def test_gsina_split_keep_prob_field_matches_soft_probability():
    edge_index = make_fully_connected_edge_index(4, include_self_loops=False)
    torch.manual_seed(0)
    scores = torch.randn(12)
    out = gsina_split(scores, edge_index, r=0.5, n_iters=20)
    assert torch.allclose(out.keep_prob, out.causal_edge_weight)  # hard_at_eval off by default -> identical


def test_gsina_split_rejects_bad_r():
    edge_index = make_fully_connected_edge_index(3)
    scores = torch.randn(9)
    try:
        gsina_split(scores, edge_index, r=0.0)
        assert False, "expected ValueError"
    except ValueError:
        pass
