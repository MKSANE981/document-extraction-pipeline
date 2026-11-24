"""Parsing helpers: French dates and amounts to typed values."""
import re
from datetime import date
from typing import Optional

MONTHS = {m: i + 1 for i, m in enumerate(
    ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
     "septembre", "octobre", "novembre", "décembre"])}

DATE_RE = (r"(\d{1,2}[/.]\d{1,2}[/.]\d{4}|\d{1,2}(?:er)?\s+(?:janvier|février|mars|avril|mai|juin|juillet|"
           r"août|septembre|octobre|novembre|décembre)\s+\d{4})")
AMOUNT_RE = r"(\d{1,3}(?:[ \u202f\u00a0]\d{3})*,\d{2}|\d+,\d{2})\s*(?:€|EUR|euros)"


def parse_date(text: str) -> Optional[date]:
    text = text.strip()
    m = re.fullmatch(r"(\d{1,2})[/.](\d{1,2})[/.](\d{4})", text)
    if m:
        d, mo, y = map(int, m.groups())
        return date(y, mo, d)
    m = re.fullmatch(r"(\d{1,2})(?:er)?\s+([a-zéû]+)\s+(\d{4})", text)
    if m and m.group(2) in MONTHS:
        return date(int(m.group(3)), MONTHS[m.group(2)], int(m.group(1)))
    return None


def parse_amount(text: str) -> Optional[float]:
    m = re.search(AMOUNT_RE, text)
    if not m:
        return None
    return float(re.sub(r"[ \u202f\u00a0]", "", m.group(1)).replace(",", "."))
