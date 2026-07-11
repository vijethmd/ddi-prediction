"""Prepare the labelled DDI dataset from the DrugBank XML.

Step 1: stream-parse the XML into a pairs CSV.
Step 2: apply the NLP keyword classifier to assign 5-class severity labels.

Usage:
    python scripts/prepare_data.py \
        --xml data/raw/full_database.xml \
        --pairs data/processed/drugbank_ddi.csv \
        --labelled data/processed/drugbank_ddi_labelled.csv
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from ddi.data.parse_drugbank import parse_drugbank, extract_drug_reference
from ddi.data.nlp_labeller import label_dataframe, label_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--pairs", default="data/processed/drugbank_ddi.csv")
    ap.add_argument("--labelled",
                    default="data/processed/drugbank_ddi_labelled.csv")
    ap.add_argument("--reference",
                    default="data/processed/drug_reference.json")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.pairs), exist_ok=True)

    print(f"Parsing DDI pairs from {args.xml} ...")
    n = parse_drugbank(args.xml, args.pairs)
    print(f"Wrote {n} pairs to {args.pairs}")

    print(f"Extracting per-drug SMILES + CYP reference from {args.xml} ...")
    n_ref = extract_drug_reference(args.xml, args.reference)
    print(f"Wrote {n_ref} drug reference entries to {args.reference}")

    print("Applying NLP severity labels ...")
    df = pd.read_csv(args.pairs)
    df = label_dataframe(df)
    df.to_csv(args.labelled, index=False)
    print(f"Wrote labelled dataset to {args.labelled}")

    print("\nClass distribution:")
    for name, stat in label_stats(df["severity"]).items():
        print(f"  {name:<16} {stat['count']:>8}  ({stat['pct']:.1f}%)")


if __name__ == "__main__":
    main()
