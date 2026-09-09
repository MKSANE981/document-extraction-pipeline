"""Quick demo: extract structured data from a document using a custom template."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import Field
from typing import Optional

from src.pipeline import DocumentPipeline
from src.templates.base import ExtractionTemplate


# Define your own template on the fly
class PurchaseOrderTemplate(ExtractionTemplate):
    po_number: Optional[str] = Field(None, description="Purchase order number")
    supplier: Optional[str] = Field(None, description="Supplier name")
    delivery_date: Optional[str] = Field(None, description="Expected delivery date")
    total_value: Optional[float] = Field(None, description="Total order value")
    currency: Optional[str] = Field(None, description="Currency")


# Built-in templates
from src.templates import InvoiceTemplate, PersonalFormTemplate


def demo_with_text():
    pipeline = DocumentPipeline()

    sample_text = """
    PURCHASE ORDER - PO-2024-789
    Supplier: OfficeSupplies Pro
    Delivery Expected: 2024-12-01
    Total Value: 3250.00 USD

    INVOICE INV-001 | Date: 2024-11-20
    From: OfficeSupplies Pro
    To: Acme Corp
    Total Due: 3250.00 USD
    """

    print("=== Custom Template (Purchase Order) ===")
    result = pipeline.process_text(sample_text, PurchaseOrderTemplate)
    print(result.model_dump_json(indent=2))

    print("\n=== Built-in Invoice Template ===")
    result = pipeline.process_text(sample_text, InvoiceTemplate)
    print(result.model_dump_json(indent=2))


def demo_with_file(file_path: str):
    pipeline = DocumentPipeline()
    result = pipeline.process(file_path, InvoiceTemplate)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    demo_with_text()
