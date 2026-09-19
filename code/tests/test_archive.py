import numpy as np
import torch

from causal_moe.drift.archive import ExpertArchive
from causal_moe.experts.expert import PlaceExpert


def make_expert(seed: int) -> PlaceExpert:
    torch.manual_seed(seed)
    return PlaceExpert(in_channels=21, hidden_channels=8)


def test_archive_starts_empty_and_spawns_fresh():
    archive = ExpertArchive(place=22)
    expert = make_expert(0)

    decision = archive.try_reactivate(np.array([0.5, 0.5, 0.5]), expert)

    assert decision.reactivated is False
    assert decision.matched_expert_id is None
    assert "empty" in decision.reason


def test_hibernate_then_reactivate_on_matching_signature():
    archive = ExpertArchive(place=22, similarity_threshold=0.9)
    old_expert = make_expert(1)
    signature = np.array([0.9, 0.1, 0.1, 0.8])
    archive.hibernate("A-1", old_expert, signature, trained_from_index=0, trained_to_index=500)

    # a fresh expert with different weights
    new_expert = make_expert(99)
    before = new_expert.mlp[0].weight.clone()

    decision = archive.try_reactivate(signature.copy(), new_expert)

    assert decision.reactivated is True
    assert decision.matched_expert_id == "A-1"
    assert decision.similarity > 0.99
    # weights were actually loaded (warm start), i.e. they changed
    assert not torch.allclose(before, new_expert.mlp[0].weight)
    assert torch.allclose(old_expert.mlp[0].weight, new_expert.mlp[0].weight)


def test_no_reactivation_when_signature_is_dissimilar():
    archive = ExpertArchive(place=22, similarity_threshold=0.9)
    archive.hibernate("A-1", make_expert(1), np.array([1.0, 0.0, 0.0]), 0, 100)

    new_expert = make_expert(2)
    before = new_expert.mlp[0].weight.clone()

    decision = archive.try_reactivate(np.array([0.0, 1.0, 0.0]), new_expert)

    assert decision.reactivated is False
    assert decision.matched_expert_id == "A-1"  # reports what it compared against
    assert decision.similarity < 0.9
    assert torch.allclose(before, new_expert.mlp[0].weight)  # untouched


def test_best_match_searches_whole_archive_not_just_last():
    """SS4.5 explicitly requires comparing against EVERY archived signature,
    not only the expert being replaced."""
    archive = ExpertArchive(place=22, similarity_threshold=0.9)
    archive.hibernate("A-1", make_expert(1), np.array([1.0, 0.0, 0.0]), 0, 100)
    archive.hibernate("A-2", make_expert(2), np.array([0.0, 1.0, 0.0]), 100, 200)
    archive.hibernate("A-3", make_expert(3), np.array([0.0, 0.0, 1.0]), 200, 300)

    # query matches the FIRST archived expert, not the most recent
    match, sim = archive.best_match(np.array([1.0, 0.0, 0.0]))

    assert match.expert_id == "A-1"
    assert sim > 0.99


def test_hibernate_deep_copies_weights():
    """Archived weights must not track later training of the live expert."""
    archive = ExpertArchive(place=22)
    expert = make_expert(1)
    archive.hibernate("A-1", expert, np.array([0.5, 0.5]), 0, 100)
    archived_weight = archive.experts[0].state_dict["mlp.0.weight"].clone()

    with torch.no_grad():
        expert.mlp[0].weight.add_(1.0)  # keep training the live expert

    assert torch.allclose(archive.experts[0].state_dict["mlp.0.weight"], archived_weight)
    assert not torch.allclose(archive.experts[0].state_dict["mlp.0.weight"], expert.mlp[0].weight)
