"""LLM-based structured extraction.

Takes raw OCR text and a Pydantic template, builds a prompt that injects
the field schema, calls a local HuggingFace text-generation model, then
parses the JSON output back into a validated template instance.

Default model: Qwen/Qwen2.5-1.5B-Instruct (1.5B parameters, runs on CPU,
free from HuggingFace). Can be swapped for any instruction-following model.
"""
import json
import re
from typing import Type, TypeVar
from transformers import pipeline
from .templates.base import ExtractionTemplate

T = TypeVar("T", bound=ExtractionTemplate)

PROMPT_TEMPLATE = """You are a document data extraction assistant.

Extract information from the document text below and return a valid JSON object.
Only include fields you find in the text. Use null for missing fields.

STRICT TYPE RULES:
- number fields: return a plain number (e.g. 6800.00), never a string like "6800.00 EUR"
- boolean fields: return true or false, never "present" or "yes"
- string fields: return a plain string

Expected JSON schema:
{schema}

Document text:
\"\"\"
{text}
\"\"\"

Return only the JSON object, no explanation."""


FALLBACK_MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]

_DEFAULT_SUGGESTIONS = {
    "en": [
        "Summarize in 3 bullet points.",
        "Extract all dates and amounts as JSON.",
        "List all parties and their roles.",
        "What are the main obligations?",
        "Convert key info into a markdown table.",
        "Translate key fields into French.",
    ],
    "fr": [
        "Résume en 3 points clés.",
        "Extrais toutes les dates et montants en JSON.",
        "Liste toutes les parties et leurs rôles.",
        "Quelles sont les obligations principales ?",
        "Convertis les infos clés en tableau markdown.",
        "Traduis les champs principaux en anglais.",
    ],
}


class StructuredExtractor:
    """Extract structured data from OCR text using a HuggingFace LLM.

    Tries FALLBACK_MODELS in order until one loads successfully.
    Primary: Qwen2.5-3B-Instruct → Phi-3.5-mini → Qwen2.5-1.5B → TinyLlama.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
        models_to_try = [model_name] + [m for m in FALLBACK_MODELS if m != model_name]
        last_error = None
        for model in models_to_try:
            try:
                self.pipe = pipeline(
                    "text-generation",
                    model=model,
                    do_sample=False,
                )
                self.model_name = model
                return
            except Exception as e:
                last_error = e
                continue
        raise RuntimeError(f"All models failed to load. Last error: {last_error}")

    def analyze_and_suggest(self, text: str, lang: str = "en") -> dict:
        """Identify the document type and generate contextual suggestions."""
        lang_hint = "Answer in English." if lang == "en" else "Réponds en français."
        prompt = (
            f"{lang_hint}\n"
            "Analyze the document excerpt below. Return ONLY a valid JSON object with:\n"
            '- "doc_type": a short label for this document type (e.g. "Resume/CV", "Invoice", "Contract", "Medical Report", "Lease Agreement", "Legal Brief", "Research Paper")\n'
            '- "suggestions": a list of exactly 6 short, specific, actionable instructions tailored to THIS document\n\n'
            f'Document excerpt:\n"""\n{text[:2000]}\n"""\n\nJSON:'
        )
        try:
            output = self.pipe(prompt, max_new_tokens=400, do_sample=False)[0]["generated_text"]
            generated = output[len(prompt):].strip()
            match = re.search(r"\{.*\}", generated, re.DOTALL)
            if match:
                data = json.loads(match.group())
                if "doc_type" in data and isinstance(data.get("suggestions"), list):
                    data["suggestions"] = [str(s) for s in data["suggestions"][:8]]
                    return data
        except Exception:
            pass
        return {"doc_type": "Document", "suggestions": _DEFAULT_SUGGESTIONS.get(lang, _DEFAULT_SUGGESTIONS["en"])}

    def extract(self, text: str, template: Type[T]) -> T:
        """Extract fields defined in `template` from raw OCR text."""
        prompt = PROMPT_TEMPLATE.format(
            schema=template.schema_prompt(),
            text=text[:8000],
        )
        output = self.pipe(prompt, max_new_tokens=1024, do_sample=False)[0]["generated_text"]
        raw_json = self._parse_json(output, prompt)
        return template.model_validate(raw_json)

    def chat(self, text: str, instruction: str) -> str:
        """Answer a free-form instruction about the document text."""
        prompt = (
            "You are a document analysis assistant. "
            "Answer the instruction based solely on the document below.\n\n"
            f"Document:\n\"\"\"\n{text[:8000]}\n\"\"\"\n\n"
            f"Instruction: {instruction}\n\nAnswer:"
        )
        output = self.pipe(prompt, max_new_tokens=1024, do_sample=False)[0]["generated_text"]
        return output[len(prompt):].strip() or "No response generated."

    def _parse_json(self, output: str, prompt: str) -> dict:
        # Strip the prompt echo if the model repeats it
        generated = output[len(prompt):].strip()
        # Find the first JSON block
        match = re.search(r"\{.*\}", generated, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return {}
