# Document Extraction Pipeline (insurance documents, demo)

A small, fully reproducible pipeline that turns PDF insurance documents into validated,
typed records, with an evaluation harness that measures the result field by field.

It is a **demonstration built on synthetic data**: every person, insurer, policy number and
address is invented. It reproduces the *structure* of document-processing work on insurance
paperwork (varied layouts, French dates and amounts, validation, evaluation), not any
client document or professional result.

## Pipeline

```
generate_documents.py   60 synthetic PDFs (3 document types, 3 layouts, varied wording) + ground truth
        ↓
src/extract.py          pdfplumber text -> rule-based field extraction
        ↓
src/normalize.py        French dates ("1er août 2026", "12.03.2026") and amounts ("1 245,50 €") -> typed values
        ↓
src/schema.py           Pydantic model; invalid fields are dropped instead of failing the whole document
        ↓
src/evaluate.py         exact-match accuracy per field, against labels.json
```

Document types: premium notice (`avis_echeance`), motor insurance certificate
(`attestation_auto`), premium receipt (`quittance_prime`).
Extracted fields: document type, insurer, insured name, policy number, issue date, period start,
period end, amount.

## Run it

```bash
pip install -r requirements.txt
python generate_documents.py --n 60 --seed 7 --out data/documents
python -m src.evaluate                                   # in-distribution set
python generate_documents.py --n 60 --seed 11 --shift --out data/documents_shifted
python -m src.evaluate data/documents_shifted data/labels_shifted.json
pytest
```

## Results (exact match after normalisation, 60 documents per set)

| Field | Same wordings as the extractor | Unseen wordings (`--shift`) |
|---|---|---|
| doc_type | 100% | 100% |
| insurer | 100% | 100% |
| policy_number | 100% | 100% |
| insured_name | 100% | 0% |
| issue_date | 100% | 0% |
| period_start | 100% | 0% |
| period_end | 100% | 67% |
| amount_eur | 100% | 33% |
| **All fields correct** | **100%** | **0%** |

## How to read these numbers

- The extractor is a rule-based **baseline that I wrote against the same label vocabulary as the
  generator**, so the left column is an upper bound, not a real-world estimate.
- The right column uses field wordings the extractor was not written for ("Bénéficiaire",
  "À compter du", "Somme due"…). It collapses on every field that depends on a label, which is the
  point: real documents vary far more than this, and label-based rules do not generalise.
- This is the case where a language-model extractor (structured output validated by the same
  Pydantic schema) is worth its cost. It is **not implemented in this repository**; the
  evaluation harness is ready to compare it with the baseline on the same two sets.

## Limits

- Text-based PDFs only: no scans, no OCR.
- Synthetic layouts are simpler than real documents (no tables spanning pages, no stamps).
- Exact-match scoring: a name with a different accent or a date in another format counts as wrong
  unless `normalize.py` handles it.
