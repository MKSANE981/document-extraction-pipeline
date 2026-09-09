"""Invoice extraction template.

InvoiceTemplate targets standard billing documents: invoices, receipts, credit
notes. All monetary fields are Optional[float]; the LLM returns null for any
field not found in the document.

LineItem is a sub-template for individual line items. It is not used directly
by the pipeline in this version but can be embedded in a custom template.
"""
from pydantic import Field
from typing import Optional
from .base import ExtractionTemplate


class LineItem(ExtractionTemplate):
    description: Optional[str] = Field(None, description="Item description")
    quantity: Optional[float] = Field(None, description="Quantity")
    unit_price: Optional[float] = Field(None, description="Unit price")
    total: Optional[float] = Field(None, description="Line total")


class InvoiceTemplate(ExtractionTemplate):
    invoice_number: Optional[str] = Field(None, description="Invoice number or ID")
    date: Optional[str] = Field(None, description="Invoice date (YYYY-MM-DD if possible)")
    vendor_name: Optional[str] = Field(None, description="Vendor or supplier name")
    vendor_address: Optional[str] = Field(None, description="Vendor address")
    client_name: Optional[str] = Field(None, description="Client or buyer name")
    subtotal: Optional[float] = Field(None, description="Subtotal before tax")
    tax_amount: Optional[float] = Field(None, description="Tax amount")
    total_amount: Optional[float] = Field(None, description="Total amount due")
    currency: Optional[str] = Field(None, description="Currency code (EUR, USD, etc.)")
    due_date: Optional[str] = Field(None, description="Payment due date")
