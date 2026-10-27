import numpy as np
import pytest
import torch

from causal_moe.causal_graph import validate as V
from causal_moe.causal_graph.prior import apply_prior, build_prior
from causal_moe.causal_graph.season import season_from_amplitude, target_day_mask
from causal_moe.data.candidate_edges import CandidateSourceSet
from causal_moe.splitter.dirgnn import DIRGNNSplitter


def test_phase_randomize_keeps_spectrum_breaks_coupling():
    rng = np.random.default_rng(0)
    x = np.zeros(4096)
    for i in range(1, x.size):  # red noise, like a daily anomaly series
        x[i] = 0.9 * x[i - 1] + rng.standard_normal()
    s = V.phase_randomize(x, rng)
    px = np.abs(np.fft.rfft((x - x.mean()) / x.std())) ** 2
    ps = np.abs(np.fft.rfft(s)) ** 2
    assert np.allclose(px, ps, rtol=1e-3, atol=1e-3 * px.max())
    assert abs(np.corrcoef(x, s)[0, 1]) < 0.2


def test_decoy_report_rate():
    graph = {"0": [{"src_region": -1, "decoy_id": 3, "val": 0.05}, {"src_region": 2, "val": 0.2}],
             "1": [{"src_region": -1, "decoy_id": 3, "val": 0.07}]}
    rep = V.decoy_report(graph, n_decoys=10)
    assert rep["n_decoys_picked"] == 1 and rep["decoy_rate"] == pytest.approx(0.1)
    assert rep["max_decoy_strength"] == pytest.approx(0.07)


def test_b2_speed_mask_is_per_field_and_flags_alias():
    dist = np.array([[0.0, 5.0e6], [5.0e6, 0.0]])  # 5000 km apart
    edges = [
        {"src_region": 1, "field": "u200", "lag": 2, "val": 0.2},  # ~29 m/s: ok for u200
        {"src_region": 1, "field": "sst", "lag": 2, "val": 0.2},  # ~29 m/s: too fast for SST
        {"src_region": 0, "field": "sst", "lag": 1, "val": 0.2},  # own region, distance 0
        {"src_region": 1, "field": "u200", "lag": 58, "val": 0.1},
    ]
    kept = V.apply_b2(edges, 0, dist)
    assert [e["field"] for e in kept] == ["u200", "sst", "u200"]
    assert kept[-1]["flag_oscillation_alias"] and not kept[0]["flag_oscillation_alias"]


def test_b3_budget_excludes_own_region():
    edges = [{"src_region": 5, "val": 0.9}, {"src_region": 1, "val": 0.3},
             {"src_region": 2, "val": -0.5}, {"src_region": 3, "val": 0.1}]
    own, top = V.apply_b3(edges, 5, k=2)
    assert len(own) == 1 and [e["src_region"] for e in top] == [2, 1]


def test_b4_persistence_lag_tolerance():
    drivers = [{"src_region": 21, "field": "sst", "lag": 3}, {"src_region": 20, "field": "u850", "lag": 5}]
    fine = {44: [{"src_region": 43, "field": "sst", "lag": 3}, {"src_region": 40, "field": "u850", "lag": 9}]}
    score = V.apply_b4(drivers, 22, fine)
    assert score == 0.5 and drivers[0]["b4_persists"] and not drivers[1]["b4_persists"]


def test_region_prior_merges_fields_and_lags():
    prior = V.region_prior([{"src_region": 3, "field": "sst", "lag": 3, "val": 0.2},
                            {"src_region": 3, "field": "pw", "lag": 9, "val": -0.4}])
    assert prior[3]["strength"] == pytest.approx(0.4) and len(prior[3]["edges"]) == 2


def test_priors_have_equal_size_and_exclude_target():
    pool = np.arange(10)
    pcmci = {2: 0.3, 7: 0.2, 22: 0.9}  # 22 not in pool
    olr = np.random.default_rng(0).standard_normal((500, 10))
    sizes = {k: build_prior(k, 0, pool, pcmci, olr_train=olr) for k in ("pcmci", "random", "corr")}
    assert all(len(p) == 2 and 0 not in p for p in sizes.values())


def test_apply_prior_modes():
    cs = CandidateSourceSet(target=0, variant="full", source_clusters=np.arange(5))
    prior = {2: 0.4, 3: 0.2}
    restricted, _, _ = apply_prior("restrict", cs, prior)
    assert restricted.source_clusters.tolist() == [0, 2, 3]
    _, bias, _ = apply_prior("init", cs, prior, init_scale=2.0)
    assert bias.tolist() == [0.0, 0.0, 2.0, 1.0, 0.0]
    _, _, mask = apply_prior("regularize", cs, prior)
    assert mask.tolist() == [False, True, False, False, True]


def test_splitter_prior_bias_and_penalty():
    torch.manual_seed(0)
    edge_index = torch.tensor([[0, 1, 2], [0, 0, 0]])
    bias = torch.tensor([0.0, 5.0, 0.0])
    sp = DIRGNNSplitter(in_channels=4, hidden_channels=8, r=0.34, edge_prior_bias=bias)
    x = torch.randn(3, 4)
    out = sp.split(x.unsqueeze(1), edge_index)
    assert out.causal_edge_index[0].tolist() == [1]  # the biased edge wins at init
    assert any(p is sp.edge_bias for p in sp.parameters())

    env = torch.randn(2, 3, 4)
    y = torch.randn(3)
    mask = torch.tensor([False, False, True])
    base = sp.compute_loss(x, y, env, edge_index, 0.0)
    pen = sp.compute_loss(x, y, env, edge_index, 0.0, off_prior_mask=mask, off_prior_weight=1.0)
    assert pen.total_loss.item() == pytest.approx(base.total_loss.item() + pen.split.edge_scores[2].item(), rel=1e-5)


def test_season_from_amplitude_onset_and_margin():
    time = np.arange(np.datetime64("2000-01-01"), np.datetime64("2001-01-01"))
    amp = np.zeros(time.size)
    active = (time >= np.datetime64("2000-06-01")) & (time <= np.datetime64("2000-09-30"))
    amp[active] = 2.0
    in_season, onset, end, fallback = season_from_amplitude(time, amp, margin_days=21)
    assert not fallback
    assert np.datetime64("2000-05-01") <= onset[2000] <= np.datetime64("2000-05-20")
    assert np.datetime64("2000-10-11") <= end[2000] <= np.datetime64("2000-11-05")
    mask = target_day_mask(np.array([np.datetime64("2000-05-30")]), 7, time, in_season)
    assert mask.tolist() == [True]
