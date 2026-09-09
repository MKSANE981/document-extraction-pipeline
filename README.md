# Document Extraction Pipeline

A lightweight, modular pipeline for extracting structured data from documents (PDF, images) using free, open-source HuggingFace models — no paid API required.

The pipeline combines an OCR engine to read document text with a small language model to extract fields defined by a user-supplied Pydantic template. This makes it easy to adapt to any document type (invoices, forms, contracts, reports) by simply changing the template.

---

## Architecture

```
Document (PDF or image)
        │
        ▼
┌───────────────┐
│  OCR Engine   │   doctr — db_resnet50 (detection) + crnn_vgg16_bn (recognition)
│  (src/ocr.py) │   Pretrained weights loaded automatically from HuggingFace
└───────┬───────┘
        │  raw text
        ▼
┌─────────────────────┐
│  Structured         │   Qwen2.5-1.5B-Instruct (HuggingFace)
│  Extractor          │   Prompted with the Pydantic schema → returns JSON
│  (src/extractor.py) │   Parsed and validated into the target template
└─────────┬───────────┘
          │  populated Pydantic model
          ▼
     Your template
  (InvoiceTemplate, PersonalFormTemplate, or your own)
```

---

## Project Structure

```
document-extraction-pipeline/
│
├── src/
│   ├── __init__.py              # Public API exports
│   ├── ocr.py                   # OCR engine wrapper (doctr)
│   ├── extractor.py             # LLM-based structured extraction
│   ├── pipeline.py              # Main pipeline combining OCR + extraction
│   └── templates/
│       ├── __init__.py          # Template exports
│       ├── base.py              # ExtractionTemplate base class
│       ├── invoice.py           # Invoice / billing document template
│       └── form.py              # Personal information form template
│
├── tests/
│   ├── fixtures/
│   │   └── generate_fixtures.py # Generates synthetic test images with Pillow
│   └── test_pipeline.py         # Unit and integration tests
│
├── examples/
│   └── demo.py                  # Runnable demo with built-in and custom templates
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quickstart

```bash
pip install -r requirements.txt
```

```python
from src.pipeline import DocumentPipeline
from src.templates import InvoiceTemplate

pipeline = DocumentPipeline()

# From a PDF or image file
result = pipeline.process("my_invoice.pdf", InvoiceTemplate)
print(result.model_dump_json(indent=2))
```

**Example output:**
```json
{
  "invoice_number": "INV-2024-0042",
  "date": "2024-11-15",
  "vendor_name": "TechSolutions SARL",
  "client_name": "GlobalCorp Inc.",
  "subtotal": 6800.0,
  "tax_amount": 1360.0,
  "total_amount": 8160.0,
  "currency": "EUR",
  "due_date": "2024-12-15"
}
```

---

## Define Your Own Template

The pipeline is template-agnostic. Define a Pydantic model that inherits from `ExtractionTemplate` — field descriptions are injected into the LLM prompt as extraction hints.

```python
from pydantic import Field
from typing import Optional
from src.templates.base import ExtractionTemplate
from src.pipeline import DocumentPipeline

class ContractTemplate(ExtractionTemplate):
    parties: Optional[str] = Field(None, description="Names of the contracting parties")
    start_date: Optional[str] = Field(None, description="Contract start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="Contract end date (YYYY-MM-DD)")
    value: Optional[float] = Field(None, description="Total contract value")
    currency: Optional[str] = Field(None, description="Currency code")
    jurisdiction: Optional[str] = Field(None, description="Governing law or jurisdiction")

pipeline = DocumentPipeline()
result = pipeline.process("contract.pdf", ContractTemplate)
print(result.model_dump_json(indent=2))
```

---

## Built-in Templates

| Template | Document type | Key fields |
|---|---|---|
| `InvoiceTemplate` | Invoices, billing | invoice_number, date, vendor, client, subtotal, tax, total, currency, due_date |
| `PersonalFormTemplate` | Registration forms | full_name, date_of_birth, email, phone, address, nationality, occupation |

---

## Generate Test Fixtures

Synthetic test documents (invoice and form images) are generated programmatically using Pillow — no real documents needed.

```bash
python tests/fixtures/generate_fixtures.py
```

---

## Run Tests

```bash
pytest tests/
```

Text-based tests run without OCR or LLM (fast). Image-based tests require fixtures to be generated first.

---

## Configuration

The pipeline accepts optional arguments to swap models:

```python
pipeline = DocumentPipeline(
    ocr_det_arch="db_resnet50",          # doctr detection model
    ocr_reco_arch="crnn_vgg16_bn",       # doctr recognition model
    llm_model="Qwen/Qwen2.5-1.5B-Instruct",  # any HuggingFace text-gen model
)
```

To use a larger model for better extraction quality, replace `llm_model` with any instruction-following model available on HuggingFace (e.g. `mistralai/Mistral-7B-Instruct-v0.3`).

---

## Stack

| Component | Library | Source |
|---|---|---|
| Document OCR | [doctr](https://github.com/mindee/doctr) | HuggingFace pretrained |
| Language model | [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | HuggingFace pretrained |
| Schema validation | [Pydantic v2](https://docs.pydantic.dev/) | — |
| Test fixtures | [Pillow](https://python-pillow.org/) | Synthetic generation |
