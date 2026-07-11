"""Build the feature matrix from the labelled DDI CSV.

Resolves each drug to SMILES (offline table + PubChem), computes the full
feature vector per pair, and saves an npz with X, y, and feature names ready
for scripts/train.py.

Usage:
    python scripts/build_features.py \
        --labelled data/processed/drugbank_ddi_labelled.csv \
        --out data/processed/features.npz \
        --limit 50000
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from ddi.features.engineer import (
    featurize_pair, feature_names, UnresolvedStructure,
)
from ddi.features.pubchem import resolve_smiles, load_reference_table, reference_table
from ddi.data.knowledge_base import load_reference_cyp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labelled", required=True)
    ap.add_argument("--out", default="data/processed/features.npz")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--allow-network", action="store_true")
    ap.add_argument("--reference", default="data/processed/drug_reference.json",
                    help="DrugBank name->SMILES+CYP JSON from prepare_data.py")
    args = ap.parse_args()

    n_ref = load_reference_table(args.reference)
    if n_ref:
        n_cyp = load_reference_cyp(reference_table())
        print(f"Loaded reference table: {n_ref} drugs "
              f"({n_cyp} with CYP data) from {args.reference}")
    else:
        print("No reference table found — falling back to offline table"
              + (" + PubChem network" if args.allow_network else ""))

    df = pd.read_csv(args.labelled)
    total_pairs = len(df)

    # Drop pairs we cannot structurally featurize BEFORE sampling. Previously
    # these were kept with an all-zero fingerprint block, which fed the model
    # ~19% pure-noise rows carrying real severity labels. Resolve each distinct
    # name once rather than once per row: 1.4M rows, ~20k distinct drugs.
    distinct = pd.unique(
        pd.concat([df["drug_name"], df["partner_name"]]).astype(str))
    ok = {n for n in distinct if resolve_smiles(n, args.allow_network)}
    print(f"Resolvable drugs: {len(ok)} of {len(distinct)} distinct names")
    df = df[df["drug_name"].astype(str).isin(ok)
            & df["partner_name"].astype(str).isin(ok)]

    # DrugBank lists each interaction twice, once from each drug's record. The
    # two rows always carry the same severity label, so they are exact duplicates
    # under a symmetric featurizer -- and sampling both would leak a pair's mirror
    # image from the training fold into the test fold. Keep one direction.
    before = len(df)
    key = [tuple(sorted((str(a), str(b))))
           for a, b in zip(df["drug_name"], df["partner_name"])]
    df = df.assign(_key=key).drop_duplicates("_key").drop(columns="_key")
    print(f"Deduplicated mirrored pairs: {before} -> {len(df)}")
    print(f"Structurally resolvable pairs: {len(df)} of {total_pairs} "
          f"({100 * len(df) / max(total_pairs, 1):.1f}%)")

    if args.limit:
        if args.limit > len(df):
            print(f"WARNING: only {len(df)} resolvable pairs available, "
                  f"requested {args.limit}")
        df = df.sample(min(args.limit, len(df)), random_state=42)

    names = feature_names()
    X, y, pair_names = [], [], []
    dropped = 0
    for i, row in enumerate(df.itertuples(index=False)):
        a, b = str(row.drug_name), str(row.partner_name)
        sa = resolve_smiles(a, args.allow_network)
        sb = resolve_smiles(b, args.allow_network)
        try:
            vec = featurize_pair(sa, sb, a, b)
        except UnresolvedStructure as exc:
            dropped += 1
            continue
        X.append(vec)
        y.append(int(row.severity))
        pair_names.append((a, b))
        if (i + 1) % 1000 == 0:
            print(f"  {i+1} pairs featurized ({dropped} dropped)")

    X = np.vstack(X)
    y = np.array(y)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # pair_names lets downstream code build drug-disjoint (cold-start) splits.
    np.savez_compressed(args.out, X=X, y=y, names=np.array(names),
                        pairs=np.array(pair_names, dtype=object))
    print(f"Saved {X.shape} feature matrix -> {args.out} "
          f"({dropped} pairs dropped as unparseable)")


if __name__ == "__main__":
    main()
