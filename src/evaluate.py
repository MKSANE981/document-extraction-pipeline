"""Field-level evaluation of the extractor against the ground-truth labels."""
import json
from pathlib import Path

import pandas as pd

from .extract import extract

FIELDS = ["doc_type", "insurer", "insured_name", "policy_number", "issue_date",
          "period_start", "period_end", "amount_eur"]


def _norm(v):
    if v is None:
        return None
    if isinstance(v, float):
        return round(v, 2)
    return str(v).strip().casefold()


def evaluate(doc_dir="data/documents", labels="data/labels.json"):
    truth = json.loads(Path(labels).read_text(encoding="utf-8"))
    rows = []
    for name, gt in truth.items():
        pred = extract(Path(doc_dir) / name).model_dump(mode="json")
        row = {"file": name, "gt_type": gt["doc_type"]}
        for f in FIELDS:
            row[f] = _norm(pred.get(f)) == _norm(gt.get(f))
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    import sys
    doc_dir = sys.argv[1] if len(sys.argv) > 1 else "data/documents"
    labels = sys.argv[2] if len(sys.argv) > 2 else "data/labels.json"
    df = evaluate(doc_dir, labels)
    per_field = df[FIELDS].mean().round(3)
    print("Field-level accuracy (exact match after normalisation):\n")
    print(per_field.to_string())
    print(f"\nAll fields correct: {df[FIELDS].all(axis=1).mean():.1%} of {len(df)} documents")
    print("\nBy document type:")
    print(df.groupby("gt_type")[FIELDS].mean().round(2).to_string())
    failed = df[~df[FIELDS].all(axis=1)]
    if len(failed):
        print("\nDocuments with at least one wrong field:", ", ".join(failed["file"]))
    df.to_csv(f"{Path(labels).parent}/evaluation_{Path(labels).stem}.csv", index=False)
