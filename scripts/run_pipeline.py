"""One-command pipeline reproducer.

Runs the full path end to end:
    DrugBank XML  ->  labelled pairs + reference JSON  ->  feature matrix  ->  bundle

If the DrugBank XML is not present, the pipeline falls back to the synthetic
dataset so `make pipeline` always produces a trained model bundle. The console
output states clearly which path was taken.

Usage:
    python scripts/run_pipeline.py                 # auto-detect XML, else synthetic
    python scripts/run_pipeline.py --xml path.xml  # force a specific XML
    python scripts/run_pipeline.py --limit 20000   # cap feature rows
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

DEFAULT_XML = os.path.join(ROOT, "data", "raw", "full_database.xml")
PAIRS = os.path.join(ROOT, "data", "processed", "drugbank_ddi.csv")
LABELLED = os.path.join(ROOT, "data", "processed", "drugbank_ddi_labelled.csv")
REFERENCE = os.path.join(ROOT, "data", "processed", "drug_reference.json")
FEATURES = os.path.join(ROOT, "data", "processed", "features.npz")
BUNDLE = os.path.join(ROOT, "models", "ddi_bundle.joblib")


def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nStep failed (exit {result.returncode}). Stopping.")
        sys.exit(result.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=DEFAULT_XML)
    ap.add_argument("--limit", type=int, default=50000)
    ap.add_argument("--coverage", type=float, default=0.95)
    args = ap.parse_args()

    py = sys.executable
    print("=" * 64)
    print("DDI PIPELINE REPRODUCER")
    print("=" * 64)

    if os.path.exists(args.xml):
        print(f"DrugBank XML found: {args.xml}")
        print("Running the REAL data path (fully offline).")

        run([py, os.path.join(HERE, "prepare_data.py"),
             "--xml", args.xml, "--pairs", PAIRS,
             "--labelled", LABELLED, "--reference", REFERENCE])

        run([py, os.path.join(HERE, "build_features.py"),
             "--labelled", LABELLED, "--reference", REFERENCE,
             "--out", FEATURES, "--limit", str(args.limit)])

        run([py, os.path.join(HERE, "train.py"),
             "--data", FEATURES, "--out", BUNDLE,
             "--coverage", str(args.coverage)])
    else:
        print(f"DrugBank XML not found at: {args.xml}")
        print("Falling back to the SYNTHETIC data path.")
        print("(Place the real full_database.xml in data/raw/ for real training.)")

        run([py, os.path.join(HERE, "train.py"),
             "--out", BUNDLE, "--coverage", str(args.coverage)])

    print("\n" + "=" * 64)
    print(f"Pipeline complete. Model bundle -> {BUNDLE}")
    print("Next: `make api` or `make app` to serve predictions.")
    print("=" * 64)


if __name__ == "__main__":
    main()
