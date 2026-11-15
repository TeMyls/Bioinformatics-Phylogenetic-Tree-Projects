#!/usr/bin/env python3
"""Fill the dataset-slide blanks from the genomes downloaded by download_variant_genomes.py.

Inputs  (from your earlier scripts):
  <rawdir>/all_genomes.fasta   downloaded genomes
  <rawdir>/variant_map.tsv     accession, variant, pango_lineage
Collection dates are fetched from NCBI (esummary, standard library only).

Outputs (in <rawdir>):
  genome_metadata.tsv   accession, variant, lineage, collection_date, length, n_fraction, qc_pass, duplicate_of
  dataset_summary.md    slide-ready numbers: totals, date range, source, download date,
                        per-variant table, and the QC funnel for the metrics slide

QC here is the same simple screen as the pipeline's first steps: length within --len-tol of 29,903,
fraction of ambiguous bases (anything not A/C/G/T) <= --max-n, then exact-duplicate removal.
Masking of problematic sites happens later, after alignment, so that funnel row stays [fill in].
UNTESTED against the live service (written offline); date fetching can be skipped with --skip-dates.

Usage:
    python scripts/dataset_summary.py --rawdir data/raw --email you@example.com
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
REF_LEN = 29_903


def read_fasta(path):
    name, chunks = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks)
                name, chunks = line[1:].split()[0], []
            else:
                chunks.append(line.strip().upper())
    if name is not None:
        yield name, "".join(chunks)


def fetch_dates(accs, email, api_key, batch=200, retries=5):
    """accession(.version) -> collection_date string ('' if absent)."""
    out = {}
    for i in range(0, len(accs), batch):
        part = accs[i:i + batch]
        params = {"db": "nuccore", "id": ",".join(part), "retmode": "json",
                  "tool": "covid-phylo", "email": email}
        if api_key:
            params["api_key"] = api_key
        data = urllib.parse.urlencode(params).encode()
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(urllib.request.Request(ESUMMARY, data=data), timeout=120) as r:
                    res = json.loads(r.read().decode())["result"]
                break
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
                print(f"  esummary failed ({e}); retrying in {2 ** attempt}s", file=sys.stderr)
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError("esummary failed after retries")
        for uid in res.get("uids", []):
            rec = res[uid]
            keys = rec.get("subtype", "").split("|")
            vals = rec.get("subname", "").split("|")
            meta = dict(zip(keys, vals))
            out[rec.get("accessionversion", uid)] = meta.get("collection_date", "")
        print(f"  dates {min(i + batch, len(accs))}/{len(accs)}")
        time.sleep(0.11 if api_key else 0.35)
    return out


def norm_date(s):
    """Return an ISO-like 'YYYY[-MM[-DD]]' string, or '' if unparseable."""
    s = (s or "").strip()
    if re.fullmatch(r"\d{4}(-\d{2}(-\d{2})?)?", s):
        return s
    for fmt, keep in (("%d-%b-%Y", "%Y-%m-%d"), ("%b-%Y", "%Y-%m"), ("%Y/%m/%d", "%Y-%m-%d")):
        try:
            return dt.datetime.strptime(s, fmt).strftime(keep)
        except ValueError:
            pass
    return ""


def span(dates):
    d = sorted(x for x in dates if x)
    return (d[0], d[-1]) if d else ("", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rawdir", default="data/raw")
    ap.add_argument("--email", default="")
    ap.add_argument("--api-key", default=os.environ.get("NCBI_API_KEY"))
    ap.add_argument("--max-n", type=float, default=0.05, help="max fraction of ambiguous bases")
    ap.add_argument("--len-tol", type=float, default=0.05, help="allowed relative deviation from 29,903")
    ap.add_argument("--skip-dates", action="store_true", help="do not contact NCBI")
    ap.add_argument("--download-date", default=None, help="YYYY-MM-DD (default: file date of all_genomes.fasta)")
    a = ap.parse_args()
    if not a.skip_dates and not a.email:
        sys.exit("--email is required to fetch dates from NCBI (or use --skip-dates)")

    fasta = os.path.join(a.rawdir, "all_genomes.fasta")
    lab = {}
    with open(os.path.join(a.rawdir, "variant_map.tsv"), newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            lab[r["accession"]] = (r["variant"], r["pango_lineage"])
    downloaded = dt.datetime.fromtimestamp(os.path.getmtime(fasta)).date().isoformat()
    dl_date = a.download_date or downloaded

    rows, seen = [], {}
    for name, seq in read_fasta(fasta):
        base = name.split(".")[0]
        variant, lineage = lab.get(name) or lab.get(base) or ("?", "?")
        n_frac = (sum(c not in "ACGT" for c in seq) / len(seq)) if seq else 1.0
        ok_len = abs(len(seq) - REF_LEN) <= a.len_tol * REF_LEN
        qc = ok_len and n_frac <= a.max_n
        h = hashlib.md5(seq.encode()).hexdigest()
        dup = ""
        if qc:
            if h in seen:
                dup = seen[h]
            else:
                seen[h] = name
        rows.append(dict(accession=name, variant=variant, lineage=lineage, length=len(seq),
                         n_fraction=round(n_frac, 5), qc_pass=int(qc), duplicate_of=dup, collection_date=""))

    if not a.skip_dates:
        print(f"fetching collection dates for {len(rows)} genomes")
        dates = fetch_dates([r["accession"] for r in rows], a.email, a.api_key)
        for r in rows:
            r["collection_date"] = norm_date(dates.get(r["accession"], ""))

    cols = ["accession", "variant", "lineage", "collection_date", "length", "n_fraction", "qc_pass", "duplicate_of"]
    with open(os.path.join(a.rawdir, "genome_metadata.tsv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    kept = [r for r in rows if r["qc_pass"] and not r["duplicate_of"]]
    n_all, n_qc = len(rows), sum(r["qc_pass"] for r in rows)
    lo, hi = span(r["collection_date"] for r in kept)
    variants = ["Alpha", "Beta", "Delta", "Omicron", "Other"]
    lines = ["# Dataset summary (fill the slides from this)", "",
             "## Slide 6: dataset tiles",
             f"- N genomes (after QC and de-duplication): **{len(kept)}**",
             f"- Collection dates: **{lo[:4] or '[n/a]'} – {hi[:4] or '[n/a]'}** (exact range {lo or '?'} to {hi or '?'})",
             "- Source: **NCBI Virus (GenBank), SARS-CoV-2 complete genomes**",
             f"- Download date: **{dl_date}**", "",
             "## Slide 6: per-variant table", "",
             "| Variant | Genomes | Collection dates |", "| --- | --- | --- |"]
    for v in variants:
        g = [r for r in kept if r["variant"] == v]
        s, e = span(r["collection_date"] for r in g)
        lines.append(f"| {v} | {len(g)} | {(s + ' – ' + e) if s else 'n/a'} |")
    lines += ["", "## Slide 9: QC funnel", "",
              f"- Drawn from metadata (downloaded): **{n_all}**",
              f"- Pass sequence QC (length within {a.len_tol:.0%}, ambiguous bases <= {a.max_n:.0%}): **{n_qc}**",
              "- After masking problematic sites: [fill in from the alignment step]",
              f"- Unique sequences: **{len(kept)}**", ""]
    missing = [r["accession"] for r in rows if r["variant"] == "?"]
    undated = sum(1 for r in kept if not r["collection_date"])
    if missing:
        lines.append(f"WARNING: {len(missing)} FASTA records not found in variant_map.tsv, e.g. {missing[:3]}")
    if undated and not a.skip_dates:
        lines.append(f"Note: {undated} kept genomes have no usable collection date and are left out of the date ranges.")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(a.rawdir, "dataset_summary.md"), "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
