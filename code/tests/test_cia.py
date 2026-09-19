import torch

from causal_moe.splitter.cia import cia_alignment_loss, target_closeness_weights
from causal_moe.splitter.dirgnn import DIRGNNSplitter


def test_target_closeness_weights_diagonal_is_one():
    y = torch.tensor([1.0, 5.0, 2.5])
    w = target_closeness_weights(y, bandwidth=1.0)
    assert torch.allclose(torch.diagonal(w), torch.ones(3))


def test_target_closeness_weights_symmetric():
    y = torch.tensor([1.0, 5.0, 2.5, -3.0])
    w = target_closeness_weights(y, bandwidth=2.0)
    assert torch.allclose(w, w.T)


def test_target_closeness_weights_far_targets_near_zero():
    y = torch.tensor([0.0, 100.0])
    w = target_closeness_weights(y, bandwidth=1.0)
    assert w[0, 1].item() < 1e-6


def test_target_closeness_weights_close_targets_near_one():
    y = torch.tensor([1.0, 1.01])
    w = target_closeness_weights(y, bandwidth=1.0)
    assert w[0, 1].item() > 0.99


def test_cia_loss_zero_when_representations_identical():
    hidden = torch.ones(5, 8)  # identical representations for every anchor
    y = torch.tensor([1.0, 2.0, 3.0, 10.0, -5.0])  # wildly different targets
    out = cia_alignment_loss(hidden, y, bandwidth=1.0)
    assert torch.allclose(out.loss, torch.zeros(()), atol=1e-6)


def test_cia_loss_penalizes_divergent_representations_for_close_targets():
    """Two anchors with nearly identical true targets but very different
    hidden representations should produce a large loss -- that's exactly
    the leakage CIA is meant to catch."""
    hidden = torch.stack([
        torch.zeros(8),
        torch.ones(8) * 10.0,  # very different representation
    ])
    y_close = torch.tensor([1.0, 1.01])  # nearly identical targets
    y_far = torch.tensor([1.0, 500.0])  # very different targets

    out_close = cia_alignment_loss(hidden, y_close, bandwidth=1.0)
    out_far = cia_alignment_loss(hidden, y_far, bandwidth=1.0)

    assert out_close.loss > out_far.loss  # same rep-gap, but penalized more when targets should match


def test_cia_loss_excludes_diagonal_self_pairs():
    hidden = torch.randn(4, 6)
    y = torch.randn(4)
    out_excl = cia_alignment_loss(hidden, y, bandwidth=1.0, exclude_diagonal=True)
    out_incl = cia_alignment_loss(hidden, y, bandwidth=1.0, exclude_diagonal=False)
    # including self-pairs (weight=1, distance=0) dilutes the weighted mean
    # differently -- the two must not be silently identical.
    assert not torch.allclose(out_excl.loss, out_incl.loss)


def test_cia_loss_gradient_flows_to_hidden():
    hidden = torch.randn(4, 6, requires_grad=True)
    y = torch.randn(4)
    out = cia_alignment_loss(hidden, y, bandwidth=1.0)
    out.loss.backward()
    assert hidden.grad is not None
    assert (hidden.grad != 0).any()


def test_predict_causal_return_hidden_backward_compatible():
    torch.manual_seed(0)
    model = DIRGNNSplitter(in_channels=3, hidden_channels=8, r=0.5)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.int64)
    x_anchor = torch.randn(3, 3)
    x_env = torch.randn(3, 3)
    split = model.split(x_anchor.unsqueeze(1), edge_index)

    pred_only = model.predict_causal(x_anchor, split, x_env)
    assert isinstance(pred_only, torch.Tensor)

    pred, hidden = model.predict_causal(x_anchor, split, x_env, return_hidden=True)
    assert torch.allclose(pred, pred_only)
    assert hidden.shape == (3, 8)
