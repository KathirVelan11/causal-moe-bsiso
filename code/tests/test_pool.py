import numpy as np
import pytest
import torch

from causal_moe.drift.archive import ArchivedExpert
from causal_moe.experts.expert import PlaceExpert
from causal_moe.experts.pool import ExpertPool


def make_member(expert_id: str, signature: np.ndarray, in_channels: int, hidden_channels: int = 16) -> ArchivedExpert:
    expert = PlaceExpert(in_channels=in_channels, hidden_channels=hidden_channels)
    return ArchivedExpert(
        expert_id=expert_id,
        place=0,
        signature=signature,
        state_dict=expert.state_dict(),
        trained_from_index=0,
        trained_to_index=100,
    )


def test_route_empty_pool_raises():
    pool = ExpertPool(place=0, in_channels=6)
    with pytest.raises(ValueError):
        pool.route(np.array([1.0, 0.0, 0.0]), k=1)


def test_route_k1_picks_best_match_exactly():
    pool = ExpertPool(place=0, in_channels=6)
    pool.add(make_member("gen1", np.array([1.0, 0.0, 0.0]), in_channels=6))
    pool.add(make_member("gen2", np.array([0.0, 1.0, 0.0]), in_channels=6))
    decision = pool.route(np.array([0.9, 0.1, 0.0]), k=1)
    assert decision.member_ids == ["gen1"]
    assert decision.weights == pytest.approx(np.array([1.0]))


def test_route_k_clamped_to_pool_size():
    pool = ExpertPool(place=0, in_channels=6)
    pool.add(make_member("gen1", np.array([1.0, 0.0]), in_channels=6))
    decision = pool.route(np.array([1.0, 0.0]), k=5)
    assert len(decision.member_ids) == 1


def test_route_topk_weights_sum_to_one():
    pool = ExpertPool(place=0, in_channels=6)
    pool.add(make_member("gen1", np.array([1.0, 0.0, 0.0]), in_channels=6))
    pool.add(make_member("gen2", np.array([0.9, 0.1, 0.0]), in_channels=6))
    pool.add(make_member("gen3", np.array([0.0, 0.0, 1.0]), in_channels=6))
    decision = pool.route(np.array([1.0, 0.0, 0.0]), k=2)
    assert len(decision.member_ids) == 2
    assert decision.weights.sum() == pytest.approx(1.0)
    # Most similar should get the largest weight.
    assert decision.weights[0] >= decision.weights[1]
    # The orthogonal member (gen3) should never be selected in top-2.
    assert "gen3" not in decision.member_ids


def test_predict_k1_matches_single_expert_forward():
    torch.manual_seed(0)
    in_channels = 6
    n_clusters = 5
    target = 2
    pool = ExpertPool(place=target, in_channels=in_channels)
    member = make_member("gen1", np.array([1.0, 0.0]), in_channels=in_channels)
    pool.add(member)

    x_all = torch.randn(n_clusters, in_channels)
    edge_index = torch.tensor([[0, 1], [target, target]])
    edge_weight = torch.tensor([0.5, 0.5])

    pred, decision = pool.predict(x_all, target, edge_index, edge_weight, np.array([1.0, 0.0]), k=1)

    ref_expert = PlaceExpert(in_channels=in_channels, hidden_channels=16)
    ref_expert.load_state_dict(member.state_dict)
    ref_expert.eval()
    with torch.no_grad():
        ref_pred = ref_expert(x_all, target, edge_index, edge_weight)

    assert torch.allclose(pred, ref_pred, atol=1e-6)
    assert decision.member_ids == ["gen1"]


def test_predict_blends_multiple_experts():
    torch.manual_seed(1)
    in_channels = 6
    n_clusters = 5
    target = 2
    pool = ExpertPool(place=target, in_channels=in_channels)
    m1 = make_member("gen1", np.array([1.0, 0.0]), in_channels=in_channels)
    m2 = make_member("gen2", np.array([0.9, 0.1]), in_channels=in_channels)
    pool.add(m1)
    pool.add(m2)

    x_all = torch.randn(n_clusters, in_channels)
    edge_index = torch.tensor([[0, 1], [target, target]])
    edge_weight = torch.tensor([0.5, 0.5])

    pred, decision = pool.predict(x_all, target, edge_index, edge_weight, np.array([1.0, 0.0]), k=2)

    e1 = PlaceExpert(in_channels=in_channels, hidden_channels=16)
    e1.load_state_dict(m1.state_dict)
    e1.eval()
    e2 = PlaceExpert(in_channels=in_channels, hidden_channels=16)
    e2.load_state_dict(m2.state_dict)
    e2.eval()
    with torch.no_grad():
        p1 = e1(x_all, target, edge_index, edge_weight)
        p2 = e2(x_all, target, edge_index, edge_weight)

    w = dict(zip(decision.member_ids, decision.weights))
    expected = w["gen1"] * p1 + w["gen2"] * p2
    assert torch.allclose(pred, expected, atol=1e-6)
