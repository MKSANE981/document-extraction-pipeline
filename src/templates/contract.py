"""Contract and mission order extraction template."""
from pydantic import Field
from typing import Optional
from .base import ExtractionTemplate


class ContractTemplate(ExtractionTemplate):
    parties: Optional[str] = Field(None, description="Names of all contracting parties")
    mission_description: Optional[str] = Field(None, description="Description of the mission or services")
    start_date: Optional[str] = Field(None, description="Contract or mission start date")
    end_date: Optional[str] = Field(None, description="Contract or mission end date")
    total_value: Optional[float] = Field(None, description="Total contract value or daily rate")
    currency: Optional[str] = Field(None, description="Currency code")
    payment_terms: Optional[str] = Field(None, description="Payment terms or schedule")
    jurisdiction: Optional[str] = Field(None, description="Governing law or jurisdiction")
    signatory: Optional[str] = Field(None, description="Signatory name or representative")
