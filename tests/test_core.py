"""Unit tests for the shared core.  Run:  pytest -q   (from the repo root)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from covidphylo import clock, distances, metrics, qc, seqio, tree as T  # noqa: E402
from covidphylo.simulate import simulate_dataset  # noqa: E402


@pytest.fixture(scope="module")
def sim():
    return simulate_dataset(120, seed=7)


def test_fasta_roundtrip(tmp_path):
    p = tmp_path / "a.fa"
    seqio.write_fasta({"x": "ACGTN", "y": "ACGTA"}, p)
    aln = seqio.Alignment.from_fasta(p)
    assert aln.names == ["x", "y"] and aln.X[0, 4] == seqio.AMBIG
    assert seqio.decode(aln.X[1]) == "ACGTA"


def test_newick_roundtrip_and_root(sim):
    t = sim.tree_subs
    t2 = T.Tree.from_newick(t.to_newick())
    assert metrics.rf_metrics(t, t2)["rf"] == 0
    assert t2.root_edge is not None
    assert abs(t.total_length() - t2.total_length()) < 1e-6


def test_rf_identical_zero_and_known_difference():
    a = T.Tree.from_newick("((A:1,B:1):1,(C:1,D:1):1,(E:1,F:1):1);")
    b = T.Tree.from_newick("((A:1,C:1):1,(B:1,D:1):1,(E:1,F:1):1);")
    assert metrics.rf_metrics(a, a)["rf"] == 0
    assert metrics.rf_metrics(a, b)["rf"] == 4      # 2 of 3 splits differ on each side


def test_prune_keeps_distances(sim):
    t = sim.tree_subs
    keep = t.leaf_names[:40]
    p = t.pruned(keep)
    n1, D1 = t.patristic()
    n2, D2 = p.patristic()
    idx = [n1.index(x) for x in n2]
    assert np.allclose(D1[np.ix_(idx, idx)], D2, atol=1e-9)


def test_patristic_symmetric_zero_diag(sim):
    _, D = sim.tree_subs.patristic()
    assert np.allclose(D, D.T) and np.allclose(np.diag(D), 0)


def test_nj_recovers_additive_tree(sim):
    """NJ is exact on additive distances: feed it the true patristic matrix."""
    names, D = sim.tree_subs.patristic()
    nj = T.neighbor_joining(D, names)
    assert metrics.rf_metrics(sim.tree_subs, nj)["rf"] == 0
    assert metrics.patristic_correlation(sim.tree_subs, nj) > 0.9999


def test_distances_handle_ambiguity():
    aln = seqio.Alignment.from_dict({"a": "ACGTACGT", "b": "ACGTACGA", "c": "ACGNNCGT"})
    mism, comp = distances.mismatch_counts(aln)
    assert mism[0, 1] == 1 and comp[0, 1] == 8
    assert mism[0, 2] == 0 and comp[0, 2] == 6        # two ambiguous sites dropped pairwise
    d = distances.distance_matrix(aln, "jc")
    assert d[0, 1] > 1 / 8                            # JC correction inflates p


def test_dedupe_and_mask():
    aln = seqio.Alignment.from_dict({"a": "ACGT", "b": "ACGT", "c": "ACGA"})
    out, groups = qc.deduplicate(aln)
    assert out.n == 2 and groups["a"] == ["a", "b"]
    assert qc.build_mask(1000, None, 5, 7)[:6] == [0, 1, 2, 3, 4, 993]


def test_clock_recovers_rate(sim):
    nj = T.neighbor_joining(distances.distance_matrix(sim.aln, "jc"), sim.aln.names)
    rooted, fit = clock.find_best_root(nj, sim.tip_dates)
    assert 0.7 * sim.rate < fit.slope < 1.3 * sim.rate
    assert fit.p_value < 1e-3


def test_true_tree_clock_exact(sim):
    _, rtt, t = clock.root_to_tip(sim.tree_subs, sim.tip_dates)
    f = clock.regress(rtt, t)
    assert f.r2 > 0.999 and abs(f.slope - sim.rate) / sim.rate < 0.01


def test_lineage_metrics_bounds(sim):
    auc = metrics.lineage_auc(sim.tree_subs, sim.lineages)
    assert 0.5 < auc <= 1.0
    mono = metrics.strict_monophyly(sim.tree_subs, sim.lineages)
    assert 0 <= mono["fraction"] <= 1


def test_holm_monotone():
    adj = metrics.holm([0.01, 0.04, 0.03])
    assert adj[0] <= adj[2] <= adj[1] <= 1


# ---- variants / EDA / subsampling -----------------------------------------------------------------
def test_variants_and_lineage_defining(sim):
    from covidphylo import variants
    ref = variants.consensus(sim.aln)
    tab = variants.site_table(sim.aln, ref)
    assert (tab["alt_freq"] <= 0.5 + 1e-9).all() and len(tab) == len(sim.aln.variable_columns())
    calls = variants.call_snps(sim.aln, ref)
    assert set(calls["sample"]) <= set(sim.aln.names) and (calls["ref"] != calls["alt"]).all()
    ld = variants.lineage_defining_sites(sim.aln, ref, sim.lineages, min_in=0.9, max_out=0.05, min_n=5)
    assert {"lineage", "pos", "freq_in", "freq_out"} <= set(ld.columns)
    assert (ld["freq_in"] >= 0.9).all() and (ld["freq_out"] <= 0.05).all()


def test_eda_runs(sim, tmp_path):
    import pandas as pd
    from covidphylo import eda
    meta = pd.DataFrame({"name": sim.aln.names, "lineage": [sim.lineages[n] for n in sim.aln.names],
                         "date": [f"{int(sim.tip_dates[n])}-06-15" for n in sim.aln.names]})
    p = eda.eda_panel(sim.aln, meta, tmp_path, mask_sites=list(range(55)))
    assert p.exists() and eda.eda_summary(sim.aln, meta)["n_sequences"] == sim.aln.n


def test_subsample_fasta_script(tmp_path):
    import subprocess
    meta = tmp_path / "m.tsv"
    meta.write_text("strain\tdate\tpango_lineage\n" + "".join(
        f"s{i}\t2021-0{1 + i % 9}-10\t{'B.1' if i % 2 else 'AY.4'}\n" for i in range(40)) + "bad\t2021-XX-XX\tB.1\n")
    fa = tmp_path / "s.fa"
    seqio.write_fasta({f"s{i}": "ACGT" * 5 for i in range(40)} | {"bad": "ACGT"}, fa)
    ref = tmp_path / "ref.fa"
    seqio.write_fasta({"NC_045512.2": "ACGT" * 5}, ref)
    out = tmp_path / "o"
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / "scripts" / "subsample_fasta.py"), "--metadata", str(meta),
                    "--sequences", str(fa), "--reference", str(ref), "--n", "10", "--min-length", "10", "--out-dir", str(out)], check=True)
    names = list(seqio.read_fasta(out / "subsample.fasta"))
    assert names[0] == "NC_045512.2" and len(names) == 11 and "bad" not in names
    import pandas as pd
    assert len(pd.read_csv(out / "subsample_provenance.tsv", sep="\t")) == 10


def test_signflip_wilcoxon_exact_floor_and_zeros():
    from covidphylo.metrics import signflip_wilcoxon
    p, k = signflip_wilcoxon([1, 2, 3, 4, 5])
    assert k == 5 and abs(p - 0.0625) < 1e-12              # smallest attainable p with 5 pairs
    p, k = signflip_wilcoxon([0.1, 0, 0, 0, 0])             # zeros dropped, one pair -> p = 1
    assert k == 1 and p == 1.0
    p, k = signflip_wilcoxon([1] * 10 + [-0.1])
    assert p < 0.01


def test_mean_ci_bounds_and_log():
    from covidphylo.metrics import mean_ci
    m, lo, hi = mean_ci([0.137, 0.02, 0.02, 0.02, 0.02], log=True)
    assert lo > 0 and lo < m < hi
    m, lo, hi = mean_ci([0.9, 1, 1, 0.95, 1], bounds=(0, 1))
    assert hi <= 1.0
