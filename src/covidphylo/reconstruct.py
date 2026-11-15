"""End-to-end reconstruction pipeline (Project B): alignment -> QC -> mask -> tree -> root -> clock -> figures."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import clock, metrics, qc
from .infer import infer_tree
from .plotting import plot_root_to_tip, plot_tree, plt
from .seqio import Alignment


def run_reconstruction(aln: Alignment, dates: dict[str, float], lineages: dict[str, str] | None,
                       outdir: str | Path, method: str = "nj_jc", mask_sites: list[int] | None = None,
                       max_ambig_frac: float = 0.05, dedupe: bool = True, seed: int = 1,
                       n_perm: int = 1000, n_boot: int = 500, truth_rate: float | None = None,
                       verbose: bool = True) -> dict:
    """Run the whole pipeline; write tree, tables, figures and summary.json to ``outdir``."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    log = {"n_input": aln.n, "alignment_length": aln.length, "method": method}

    # 1. mask + QC -----------------------------------------------------------------------------
    if mask_sites:
        aln = aln.mask_sites(mask_sites)
        log["n_masked_sites"] = len(set(mask_sites))
    aln = qc.drop_ambiguous(aln, max_ambig_frac)
    log["n_after_ambig_filter"] = aln.n
    keep_dates = [i for i, n in enumerate(aln.names) if np.isfinite(dates.get(n, np.nan))]
    aln = aln.subset(keep_dates)
    log["n_with_complete_date"] = aln.n
    if dedupe:                                   # keep the earliest-dated member of identical groups
        order = sorted(range(aln.n), key=lambda i: dates[aln.names[i]])
        aln, groups = qc.deduplicate(aln, order)
        log["n_unique_sequences"] = aln.n
        pd.DataFrame([(k, len(v), ";".join(v)) for k, v in groups.items()],
                     columns=["representative", "n_identical", "members"]).to_csv(out / "duplicate_groups.csv", index=False)
    if verbose:
        print(log)

    # 2. tree ------------------------------------------------------------------------------------
    res = infer_tree(method, aln, workdir=out / "work", seed=seed)
    log["inference_wall_s"] = round(res.wall_s, 2)

    # 3. root + clock ----------------------------------------------------------------------------
    rooted, fit = clock.find_best_root(res.tree, dates)
    perm = clock.permutation_test(rooted, dates, n_perm=n_perm, seed=seed)
    boot = clock.bootstrap_clock(rooted, dates, n_boot=n_boot, seed=seed)
    node_dates = clock.date_nodes(rooted, fit)
    log.update(clock_rate=fit.slope, clock_rate_ci95=boot["rate_ci"], tmrca=fit.tmrca,
               tmrca_ci95=boot["tmrca_ci"], r2=fit.r2, p_permutation=perm["p_permutation"], n_tips_in_fit=fit.n)
    if truth_rate is not None:
        log["true_rate"] = truth_rate
    (out / "tree_rooted.nwk").write_text(rooted.to_newick())

    v = rooted.rooted()
    pd.DataFrame({"node": list(node_dates), "date": list(node_dates.values()),
                  "root_distance": [v.depth[k] for k in node_dates],
                  "is_tip": [k in rooted.name for k in node_dates],
                  "name": [rooted.name.get(k, "") for k in node_dates]}).to_csv(out / "node_dates.csv", index=False)

    # 4. lineage annotation -------------------------------------------------------------------------
    if lineages:
        lab = {n: lineages[n] for n in aln.names if n in lineages}
        log["lineage_auc"] = metrics.lineage_auc(rooted, lab)
        log["lineage_strict_monophyly"] = metrics.strict_monophyly(rooted, lab)
        v = rooted.rooted()
        rows = []
        for l in sorted(set(lab.values())):
            members = [n for n in lab if lab[n] == l]
            d = np.array([dates[n] for n in members])
            rows.append(dict(lineage=l, n_tips=len(members), first_date=d.min(), last_date=d.max()))
        pd.DataFrame(rows).sort_values("n_tips", ascending=False).to_csv(out / "lineage_summary.csv", index=False)

    # 5. figures ---------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 7.5))
    plot_tree(rooted, ax, labels=lineages, x="date", node_dates=node_dates)
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(out / "fig_dated_tree.png")
    plt.close(fig)
    names, rtt, t = clock.root_to_tip(rooted, dates)
    fig, ax = plt.subplots(figsize=(4.4, 3.5))
    plot_root_to_tip(t, rtt, fit, ax)
    fig.tight_layout()
    fig.savefig(out / "fig_root_to_tip.png")
    plt.close(fig)

    (out / "summary.json").write_text(json.dumps(log, indent=2, default=float))
    return log
