import numpy as np
import pytest

from causal_moe.causal_graph.pcmci import LaggedDesign, PCMCIConfig, bh_fdr, pc1, run_target


def _var_data(T=3000, seed=0):
    """Node 0 is driven by node 1 at lag 2 and node 2 at lag 1; node 3 is
    driven by node 1 (common driver, no direct link to 0); node 4 is noise."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((T, 5))
    for t in range(3, T):
        x[t, 1] += 0.6 * x[t - 1, 1]
        x[t, 2] += 0.5 * x[t - 1, 2]
        x[t, 3] += 0.5 * x[t - 1, 1] + 0.3 * x[t - 1, 3]
        x[t, 0] += 0.4 * x[t - 2, 1] + 0.35 * x[t - 1, 2] + 0.2 * x[t - 1, 0]
    return x


def test_recovers_true_parents():
    x = _var_data()
    cfg = PCMCIConfig(tau_max=3, pc_alpha=0.05)
    design = LaggedDesign(x, np.arange(6, x.shape[0]), np.arange(5), cfg)
    links, _ = run_target(design, 0, cfg)
    found = {(l.node, l.lag) for l in links}
    assert {(1, 2), (2, 1), (0, 1)} <= found
    assert not any(l.node in (3, 4) for l in links)


def test_matches_tigramite():
    tigramite = pytest.importorskip("tigramite")
    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI

    x = _var_data(T=2000, seed=1)
    tau_max = 3
    assumptions = {
        j: {(i, -tau): "-?>" for i in range(5) for tau in range(1, tau_max + 1) if j == 0 or i == j}
        for j in range(5)
    }
    pc = PCMCI(dataframe=pp.DataFrame(x), cond_ind_test=ParCorr(), verbosity=0)
    res = pc.run_pcmci(tau_min=1, tau_max=tau_max, pc_alpha=0.05, link_assumptions=assumptions)

    cfg = PCMCIConfig(tau_max=tau_max, pc_alpha=0.05, fdr_alpha=1.0)
    design = LaggedDesign(x, np.arange(2 * tau_max, x.shape[0]), np.arange(5), cfg)
    ours = pc1(design.target(0), design, np.arange(design.X.shape[1]), cfg)
    ours_set = {(int(design.col_node[c]), int(design.col_lag[c])) for c in ours}
    assert ours_set == {(i, -t) for i, t in pc.all_parents[0]}

    links, _ = run_target(design, 0, cfg)
    for l in links:
        assert l.val == pytest.approx(res["val_matrix"][l.node, 0, l.lag], abs=2e-3)
        assert l.p == pytest.approx(res["p_matrix"][l.node, 0, l.lag], rel=0.05, abs=1e-6)


def test_matches_tigramite_two_targets():
    """Targets 0 and 3 may have any parent; a target acting as a source is
    conditioned on its full PC1 parent set (not only its own past)."""
    pytest.importorskip("tigramite")
    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI

    from causal_moe.causal_graph.pcmci import run_targets

    x = _var_data(T=2000, seed=2)
    tau_max, targets = 3, (0, 3)
    assumptions = {
        j: {(i, -tau): "-?>" for i in range(5) for tau in range(1, tau_max + 1) if j in targets or i == j}
        for j in range(5)
    }
    pc = PCMCI(dataframe=pp.DataFrame(x), cond_ind_test=ParCorr(), verbosity=0)
    res = pc.run_pcmci(tau_min=1, tau_max=tau_max, pc_alpha=0.05, link_assumptions=assumptions)

    cfg = PCMCIConfig(tau_max=tau_max, pc_alpha=0.05, fdr_alpha=1.0)
    design = LaggedDesign(x, np.arange(2 * tau_max, x.shape[0]), np.arange(5), cfg)
    out = run_targets(design, list(targets), cfg)
    for t in targets:
        for l in out[t][0]:
            assert l.val == pytest.approx(res["val_matrix"][l.node, t, l.lag], abs=2e-3)


def test_bh_fdr_monotone():
    p = np.array([0.001, 0.04, 0.03, 0.5, 1.0])
    q = bh_fdr(p)
    assert np.all(q >= p)
    assert q[0] == pytest.approx(0.005)
