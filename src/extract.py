"""Rule-based baseline extractor: PDF text -> InsuranceDocument.

An optional LLM extractor can be plugged in with the same signature (see README); it is
not used for the reported results.
"""
import re
from typing import Optional

import pdfplumber
from pydantic import ValidationError

from .normalize import AMOUNT_RE, DATE_RE, parse_amount, parse_date
from .schema import InsuranceDocument

POLICY = re.compile(r"\b([A-Z]{2}\d{2}-\d{6})\b")
INSURERS = ["Assurances Valmont", "Mutuelle Horizon Ouest", "Cap Sud Assurances"]
START_LABELS = r"(?:Début de période|Prend effet le|Période du|Date d'effet)"
END_LABELS = r"(?:Fin de période|Échéance au|Date d'échéance|\bau)"
INSURED_LABELS = r"(?:Assuré\(e\)|Souscripteur|Titulaire du contrat|Nom de l'assuré)"
ISSUE_LABELS = r"(?:Fait le|Émis le|Date d'émission|Édité le)"
AMOUNT_LABELS = r"(?:Montant TTC|Total à régler|Cotisation annuelle TTC|Prime TTC)"
DOC_KEYWORDS = {"avis_echeance": "AVIS D'ÉCHÉANCE", "attestation_auto": "ATTESTATION D'ASSURANCE",
                "quittance_prime": "QUITTANCE DE PRIME"}


def read_text(path) -> str:
    with pdfplumber.open(path) as pdf:
        return "\n".join((p.extract_text() or "") for p in pdf.pages)


def _after(label: str, pattern: str, text: str) -> Optional[str]:
    m = re.search(label + r"\s*:?\s*" + pattern, text)
    return m.group(1) if m else None


def extract_fields(text: str) -> dict:
    flat = re.sub(r"\s+", " ", text)  # labels can wrap across lines
    doc_type = next((k for k, kw in DOC_KEYWORDS.items() if kw in text.upper()), None)
    insurer = next((i for i in INSURERS if i in text), None)
    policy = POLICY.search(text)
    name = _after(INSURED_LABELS, r"([A-ZÀ-Ý][\w'\- ]+?)(?:\.|\n|\s{2,}|$)", text)
    issue = _after(ISSUE_LABELS, DATE_RE, flat)
    start = _after(START_LABELS, DATE_RE, flat)
    end = _after(END_LABELS, DATE_RE, flat)
    amt = re.search(AMOUNT_LABELS + r"\s*:?\s*(.{0,30})", flat)
    amount = parse_amount(amt.group(1)) if amt else None
    return {
        "doc_type": doc_type,
        "insurer": insurer,
        "insured_name": name.strip() if name else None,
        "policy_number": policy.group(1) if policy else None,
        "issue_date": parse_date(issue) if issue else None,
        "period_start": parse_date(start) if start else None,
        "period_end": parse_date(end) if end else None,
        "amount_eur": amount,
    }


def extract(path) -> InsuranceDocument:
    fields = extract_fields(read_text(path))
    try:
        return InsuranceDocument(**fields)
    except ValidationError:
        # Keep what validates, drop the invalid fields instead of failing the document.
        clean = {}
        for k, v in fields.items():
            try:
                InsuranceDocument(**{k: v})
                clean[k] = v
            except ValidationError:
                pass
        return InsuranceDocument(**clean)
