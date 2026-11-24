from datetime import date

from src.extract import extract_fields
from src.normalize import parse_amount, parse_date


def test_parse_date_numeric_and_textual():
    assert parse_date("12/03/2026") == date(2026, 3, 12)
    assert parse_date("12.03.2026") == date(2026, 3, 12)
    assert parse_date("1er août 2026") == date(2026, 8, 1)
    assert parse_date("n'importe quoi") is None


def test_parse_amount_variants():
    assert parse_amount("1 245,50 €") == 1245.50
    assert parse_amount("323,53 EUR") == 323.53
    assert parse_amount("48,00 euros") == 48.00
    assert parse_amount("sans montant") is None


def test_label_wrapped_across_lines():
    text = "Assuré(e) : Julie Martin.\nFin de\npériode 12.04.2027. Prime TTC : 99,90 €"
    out = extract_fields(text)
    assert out["period_end"] == date(2027, 4, 12)
    assert out["amount_eur"] == 99.90


def test_missing_fields_are_none():
    out = extract_fields("Document sans aucune information utile")
    assert out["policy_number"] is None and out["amount_eur"] is None
