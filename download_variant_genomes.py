#!/usr/bin/env python3
"""Download the genomes listed in variant_links.tsv (from ncbi_variant_links.py).

Uses NCBI Entrez efetch in batches (standard library only). Writes:
  <outdir>/<Variant>.fasta       one FASTA per variant
  <outdir>/all_genomes.fasta     everything combined
  <outdir>/variant_map.tsv       accession -> variant, lineage (for stratified sampling)

Re-running skips accessions already present in all_genomes.fasta, so it can be resumed.
UNTESTED against the live service (written offline); NCBI allows ~3 requests/s
without an API key, ~10/s with one (--api-key or env NCBI_API_KEY).

Usage:
    python download_variant_genomes.py --links variant_links.tsv --outdir data/raw --email you@example.com
"""
import argparse
import csv
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def accession_from_link(link):
    return link.rstrip("/").rsplit("/", 1)[-1]


def fetch_batch(accs, email, api_key, retries=5):
    params = {"db": "nuccore", "id": ",".join(accs), "rettype": "fasta",
              "retmode": "text", "tool": "covid-phylo", "email": email}
    if api_key:
        params["api_key"] = api_key
    # POST keeps long id lists out of the URL.
    data = urllib.parse.urlencode(params).encode()
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(EFETCH, data=data), timeout=120) as r:
                return r.read().decode()
        except (urllib.error.URLError, TimeoutError) as e:
            wait = 2 ** attempt
            print(f"  request failed ({e}); retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("batch failed after retries: " + ",".join(accs[:3]) + "...")


def split_fasta(text):
    """Yield (accession, record_text) from a multi-FASTA string."""
    for rec in re.split(r"(?m)^(?=>)", text):
        if not rec.strip():
            continue
        m = re.match(r">(\S+)", rec)
        if m:
            yield m.group(1), rec if rec.endswith("\n") else rec + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--links", default="variant_links.tsv")
    ap.add_argument("--outdir", default="data/raw")
    ap.add_argument("--email", required=True, help="NCBI asks for a contact email")
    ap.add_argument("--api-key", default=os.environ.get("NCBI_API_KEY"))
    ap.add_argument("--batch", type=int, default=100)
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    rows = []
    with open(a.links, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            r["accession"] = r.get("accession") or accession_from_link(r["nuccore_link"])
            rows.append(r)
    info = {r["accession"]: r for r in rows}

    combined = os.path.join(a.outdir, "all_genomes.fasta")
    done = set()
    if os.path.exists(combined):
        with open(combined) as fh:
            for line in fh:
                if line.startswith(">"):
                    done.add(line[1:].split()[0])
    todo = [acc for acc in info if acc not in done]
    print(f"{len(info)} accessions listed, {len(done)} already downloaded, {len(todo)} to fetch")

    delay = 0.11 if a.api_key else 0.35
    missing = set(todo)
    with open(combined, "a") as allfh:
        for i in range(0, len(todo), a.batch):
            batch = todo[i:i + a.batch]
            text = fetch_batch(batch, a.email, a.api_key)
            for acc_full, rec in split_fasta(text):
                acc = acc_full.split(".")[0]
                key = acc_full if acc_full in info else acc
                if key not in info:
                    continue
                allfh.write(rec)
                with open(os.path.join(a.outdir, info[key]["variant"] + ".fasta"), "a") as vf:
                    vf.write(rec)
                missing.discard(key)
            print(f"  fetched {min(i + a.batch, len(todo))}/{len(todo)}")
            time.sleep(delay)

    with open(os.path.join(a.outdir, "variant_map.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["accession", "variant", "pango_lineage"])
        for r in rows:
            w.writerow([r["accession"], r["variant"], r["pango_lineage"]])

    if missing:
        print(f"WARNING: {len(missing)} accessions returned no sequence, e.g. {sorted(missing)[:5]}")
    print("done ->", a.outdir)


if __name__ == "__main__":
    main()
