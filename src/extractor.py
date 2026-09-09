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
    "Qwen/Qwen2.5-1.5B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "microsoft/phi-2",
]


class StructuredExtractor:
    """Extract structured data from OCR text using a small HuggingFace LLM.

    Tries FALLBACK_MODELS in order until one loads successfully.
    Defaults to Qwen2.5-1.5B-Instruct; falls back to TinyLlama, then Phi-2.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        models_to_try = [model_name] + [m for m in FALLBACK_MODELS if m != model_name]
        last_error = None
        for model in models_to_try:
            try:
                self.pipe = pipeline(
                    "text-generation",
                    model=model,
                    max_new_tokens=512,
                    do_sample=False,
                )
                self.model_name = model
                return
            except Exception as e:
                last_error = e
                continue
        raise RuntimeError(f"All models failed to load. Last error: {last_error}")

    def extract(self, text: str, template: Type[T]) -> T:
        """Extract fields defined in `template` from raw OCR text."""
        prompt = PROMPT_TEMPLATE.format(
            schema=template.schema_prompt(),
            text=text[:3000],  # truncate to avoid context overflow
        )
        output = self.pipe(prompt)[0]["generated_text"]
        raw_json = self._parse_json(output, prompt)
        return template.model_validate(raw_json)

    def chat(self, text: str, instruction: str) -> str:
        """Answer a free-form instruction about the document text."""
        prompt = (
            "You are a document analysis assistant. "
            "Answer the instruction based solely on the document below.\n\n"
            f"Document:\n\"\"\"\n{text[:3000]}\n\"\"\"\n\n"
            f"Instruction: {instruction}\n\nAnswer:"
        )
        output = self.pipe(prompt)[0]["generated_text"]
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
