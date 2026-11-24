"""Target schema for the extraction step."""
from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field

DocType = Literal["avis_echeance", "attestation_auto", "quittance_prime"]


class InsuranceDocument(BaseModel):
    doc_type: Optional[DocType] = None
    insurer: Optional[str] = None
    insured_name: Optional[str] = None
    policy_number: Optional[str] = Field(default=None, pattern=r"^[A-Z]{2}\d{2}-\d{6}$")
    issue_date: Optional[date] = None
    period_start: Optional[date] = None
    period_end: Optional[date] = None
    amount_eur: Optional[float] = Field(default=None, ge=0)
