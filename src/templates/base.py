"""Base extraction template.

ExtractionTemplate is the root Pydantic class all templates inherit from.
It adds schema_prompt(), which converts the model's field metadata into a
compact JSON schema string that gets injected directly into the LLM prompt.

To define a new template, subclass ExtractionTemplate and declare Optional
fields with Field(description=...) — the description is what the LLM uses
to know what to look for in the document text.
"""
import re
from pydantic import BaseModel, model_validator


class ExtractionTemplate(BaseModel):
    """Base class for all extraction templates.

    Subclass this to define what fields to extract from a document.
    Each field description is used as a hint for the LLM.

    Includes a pre-validation coercion step that handles common LLM output
    quirks: monetary strings ("6800.00 EUR" → 6800.0), boolean strings
    ("present"/"yes" → True), so templates are robust even when the model
    ignores the type rules in the prompt.
    """

    @model_validator(mode="before")
    @classmethod
    def _coerce_llm_types(cls, data: dict) -> dict:
        if not isinstance(data, dict):
            return data
        schema = cls.model_json_schema()
        props = schema.get("properties", {})
        for field, info in props.items():
            if field not in data or data[field] is None:
                continue
            val = data[field]
            # Resolve anyOf/oneOf (Optional fields are represented this way)
            ftype = info.get("type") or next(
                (t.get("type") for t in info.get("anyOf", []) if t.get("type") != "null"),
                None,
            )
            if ftype in ("number", "integer") and isinstance(val, str):
                cleaned = re.sub(r"[^\d.\-]", "", val.replace(",", "."))
                try:
                    data[field] = float(cleaned)
                except ValueError:
                    data[field] = None
            elif ftype == "boolean" and isinstance(val, str):
                data[field] = val.strip().lower() in ("true", "yes", "present", "1", "oui", "vrai")
        return data

    @classmethod
    def schema_prompt(cls) -> str:
        """Return a JSON schema string suitable for injection into an LLM prompt."""
        schema = cls.model_json_schema()
        fields = schema.get("properties", {})
        lines = []
        for name, info in fields.items():
            desc = info.get("description", "")
            ftype = info.get("type", "string")
            lines.append(f'  "{name}": <{ftype}>  // {desc}')
        return "{\n" + ",\n".join(lines) + "\n}"
