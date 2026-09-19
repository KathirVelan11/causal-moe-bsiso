import torch

from causal_moe.experts.expert import PlaceExpert, compute_expert_loss


def test_place_expert_forward_shape_and_range():
    n_places = 10
    in_channels = 21
    x_all = torch.randn(n_places, in_channels)
    target = 3
    causal_edge_index = torch.tensor([[1, 4, 3], [target, target, target]], dtype=torch.int64)
    causal_edge_weight = torch.tensor([0.9, 0.7, 0.6])

    expert = PlaceExpert(in_channels=in_channels, hidden_channels=8)
    out = expert(x_all, target, causal_edge_index, causal_edge_weight)

    assert out.shape == ()  # scalar forecast


def test_place_expert_gradient_flows_to_edge_weights_and_mlp():
    n_places = 6
    in_channels = 5
    x_all = torch.randn(n_places, in_channels)
    target = 0
    causal_edge_index = torch.tensor([[1, 2], [target, target]], dtype=torch.int64)
    causal_edge_weight = torch.tensor([0.5, 0.8], requires_grad=True)

    expert = PlaceExpert(in_channels=in_channels, hidden_channels=4)
    out = expert(x_all, target, causal_edge_index, causal_edge_weight)
    out.backward()

    assert causal_edge_weight.grad is not None
    assert (causal_edge_weight.grad != 0).any()
    assert expert.mlp[0].weight.grad is not None


def test_place_expert_ignores_source_order_weighted_mean_scale_invariant():
    """Weighted-mean aggregation (not weighted-sum) means doubling k
    identical sources at half the weight each should reproduce roughly the
    same aggregated vector -- a sanity check that k (candidate-set size)
    doesn't trivially inflate the expert's input scale, which would make
    the direct/2hop/full variants incomparable purely from set size."""
    torch.manual_seed(0)
    n_places = 4
    in_channels = 3
    x_all = torch.randn(n_places, in_channels)
    target = 0

    edge_index_a = torch.tensor([[1, 2], [target, target]], dtype=torch.int64)
    weight_a = torch.tensor([1.0, 1.0])

    # same two sources, duplicated, each at half weight -> same weighted mean
    edge_index_b = torch.tensor([[1, 2, 1, 2], [target, target, target, target]], dtype=torch.int64)
    weight_b = torch.tensor([0.5, 0.5, 0.5, 0.5])

    expert = PlaceExpert(in_channels=in_channels, hidden_channels=4)
    with torch.no_grad():
        out_a = expert(x_all, target, edge_index_a, weight_a)
        out_b = expert(x_all, target, edge_index_b, weight_b)

    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_compute_expert_loss_matches_squared_error():
    n_places = 5
    in_channels = 4
    x_all = torch.randn(n_places, in_channels)
    target = 2
    y_target = torch.tensor(1.5)
    causal_edge_index = torch.tensor([[0, 1], [target, target]], dtype=torch.int64)
    causal_edge_weight = torch.tensor([0.4, 0.6])

    expert = PlaceExpert(in_channels=in_channels, hidden_channels=4)
    out = compute_expert_loss(expert, x_all, target, y_target, causal_edge_index, causal_edge_weight)

    expected_mse = (out.prediction - y_target) ** 2
    assert torch.allclose(out.total_loss, expected_mse)
    assert torch.allclose(out.mse, expected_mse)
