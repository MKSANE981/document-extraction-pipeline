"""Receipt and expense extraction template."""
from pydantic import Field
from typing import Optional
from .base import ExtractionTemplate


class ReceiptTemplate(ExtractionTemplate):
    merchant: Optional[str] = Field(None, description="Merchant or store name")
    date: Optional[str] = Field(None, description="Transaction date")
    total: Optional[float] = Field(None, description="Total amount paid")
    currency: Optional[str] = Field(None, description="Currency code")
    payment_method: Optional[str] = Field(None, description="Payment method (card, cash, transfer, etc.)")
    vat_number: Optional[str] = Field(None, description="VAT or tax registration number")
    items: Optional[str] = Field(None, description="Summary of purchased items or services")
