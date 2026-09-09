import pytest
from pathlib import Path
from src.pipeline import DocumentPipeline
from src.templates import InvoiceTemplate, PersonalFormTemplate

FIXTURES = Path(__file__).parent / "fixtures"


def test_invoice_extraction_from_text():
    pipeline = DocumentPipeline()
    text = """
    Invoice Number: INV-2024-0042
    Date: 2024-11-15
    Vendor: TechSolutions SARL
    Client: GlobalCorp Inc.
    Subtotal: 6800.00 EUR
    VAT (20%): 1360.00 EUR
    Total Due: 8160.00 EUR
    Currency: EUR
    Due Date: 2024-12-15
    """
    result = pipeline.process_text(text, InvoiceTemplate)
    assert result.invoice_number == "INV-2024-0042"
    assert result.total_amount == 8160.00
    assert result.currency == "EUR"
    assert result.vendor_name is not None


def test_form_extraction_from_text():
    pipeline = DocumentPipeline()
    text = """
    Full Name: Jean-Baptiste MARTIN
    Date of Birth: 15/03/1985
    Nationality: French
    Email: jb.martin@example.com
    Phone: +33 6 12 34 56 78
    Occupation: Software Engineer
    Signature: present
    """
    result = pipeline.process_text(text, PersonalFormTemplate)
    assert result.full_name is not None
    assert "MARTIN" in (result.full_name or "")
    assert result.occupation is not None


@pytest.mark.skipif(
    not (FIXTURES / "sample_invoice.png").exists(),
    reason="Run tests/fixtures/generate_fixtures.py first",
)
def test_invoice_from_image():
    pipeline = DocumentPipeline()
    result = pipeline.process(FIXTURES / "sample_invoice.png", InvoiceTemplate)
    assert result.invoice_number is not None
    assert result.total_amount is not None


@pytest.mark.skipif(
    not (FIXTURES / "sample_form.png").exists(),
    reason="Run tests/fixtures/generate_fixtures.py first",
)
def test_form_from_image():
    pipeline = DocumentPipeline()
    result = pipeline.process(FIXTURES / "sample_form.png", PersonalFormTemplate)
    assert result.full_name is not None
