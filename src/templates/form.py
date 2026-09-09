"""Personal information form template.

PersonalFormTemplate targets registration forms, KYC forms, and any document
that captures individual identity information. signature_present is a bool
field — the LLM returns true if it sees evidence of a signature in the text.
"""
from pydantic import Field
from typing import Optional
from .base import ExtractionTemplate


class PersonalFormTemplate(ExtractionTemplate):
    full_name: Optional[str] = Field(None, description="Full name of the applicant")
    date_of_birth: Optional[str] = Field(None, description="Date of birth (YYYY-MM-DD)")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    address: Optional[str] = Field(None, description="Full address")
    nationality: Optional[str] = Field(None, description="Nationality")
    occupation: Optional[str] = Field(None, description="Occupation or job title")
    signature_present: Optional[bool] = Field(None, description="Whether a signature is present")
