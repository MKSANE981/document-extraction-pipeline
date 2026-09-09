---
title: Document Extraction Pipeline
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
---

# Document Chat & Extraction Pipeline

Talk to your documents. Upload a PDF or image, then ask anything in plain language — summarize, extract, restructure, translate, or export in any format. All models run free from HuggingFace, no API key required.

🚀 **[Live demo on HuggingFace Spaces](https://huggingface.co/spaces/mansourkama/document-extraction-pipeline)**

---

## What you can do

| Instruction example | What you get |
|---|---|
| "Summarize in 3 bullet points" | Concise summary |
| "Extract all dates and amounts as JSON" | Structured JSON object |
| "List all parties and their roles" | Named entity list |
| "Convert to a markdown table" | Formatted table |
| "What are the payment terms?" | Direct answer |
| "Translate the key fields to English" | Translated output |

Plus template-based structured extraction (Invoice, Contract, Receipt, Personal Form) that returns a validated Pydantic model.

---

## Architecture

```
Document (PDF or image)
        │
        ▼
┌───────────────┐
│  OCR Engine   │   doctr — db_resnet50 (detection) + crnn_vgg16_bn (recognition)
│  (src/ocr.py) │   Pretrained weights from HuggingFace, no account needed
└───────┬───────┘
        │  raw text
        ▼
┌──────────────────────┐
│  LLM                 │   Qwen2.5-1.5B-Instruct (HuggingFace)
│  (src/extractor.py)  │   ├─ chat()    — free-form instruction → any output format
│                      │   └─ extract() — schema-guided → validated Pydantic model
└──────────────────────┘
```

---

## Project Structure

```
document-extraction-pipeline/
│
├── src/
│   ├── ocr.py                   # OCR engine (doctr)
│   ├── extractor.py             # LLM: chat() + extract() methods
│   ├── pipeline.py              # Orchestrates OCR + LLM; exposes chat_file(), process()
│   └── templates/
│       ├── base.py              # ExtractionTemplate base class (schema_prompt + type coercion)
│       ├── invoice.py           # Invoice / billing
│       ├── contract.py          # Contract / mission order
│       ├── receipt.py           # Receipt / expense
│       └── form.py              # Personal information form
│
├── app.py                       # Gradio web interface (HuggingFace Spaces)
├── tests/
│   ├── fixtures/generate_fixtures.py   # Synthetic test images (Pillow)
│   └── test_pipeline.py               # Unit + integration tests
├── examples/demo.py             # CLI demo
└── requirements.txt
```

---

## Quickstart

```bash
pip install -r requirements.txt
```

**Free-form chat:**
```python
from src.pipeline import DocumentPipeline

pipeline = DocumentPipeline()

# From a file (OCR + LLM)
response = pipeline.chat_file("contract.pdf", "List all parties and the total value.")
print(response)

# From text (LLM only)
response = pipeline.chat(my_text, "Convert to a markdown table.")
print(response)
```

**Structured extraction:**
```python
from src.pipeline import DocumentPipeline
from src.templates import InvoiceTemplate

pipeline = DocumentPipeline()
result = pipeline.process("invoice.pdf", InvoiceTemplate)
print(result.model_dump_json(indent=2))
```

```json
{
  "invoice_number": "INV-2024-0042",
  "date": "2024-11-15",
  "vendor_name": "TechSolutions SARL",
  "total_amount": 8160.0,
  "currency": "EUR"
}
```

---

## Built-in Templates

| Template | Document type | Key fields |
|---|---|---|
| `InvoiceTemplate` | Invoices, billing | invoice_number, date, vendor, client, subtotal, tax, total, currency, due_date |
| `ContractTemplate` | Contracts, mission orders | parties, mission_description, start_date, end_date, total_value, payment_terms |
| `ReceiptTemplate` | Receipts, expense slips | merchant, date, total, currency, payment_method, items |
| `PersonalFormTemplate` | Registration, KYC forms | full_name, date_of_birth, email, phone, address, nationality, occupation |

**Custom template:**
```python
from pydantic import Field
from typing import Optional
from src.templates.base import ExtractionTemplate

class MedicalReportTemplate(ExtractionTemplate):
    patient_name: Optional[str] = Field(None, description="Patient full name")
    diagnosis: Optional[str] = Field(None, description="Main diagnosis")
    physician: Optional[str] = Field(None, description="Treating physician name")
    date: Optional[str] = Field(None, description="Report date")

result = pipeline.process("report.pdf", MedicalReportTemplate)
```

---

## Run Tests

```bash
python tests/fixtures/generate_fixtures.py   # generate synthetic test images
pytest tests/ -v
```

---

## Configuration

```python
pipeline = DocumentPipeline(
    ocr_det_arch="db_resnet50",
    ocr_reco_arch="crnn_vgg16_bn",
    llm_model="Qwen/Qwen2.5-1.5B-Instruct",   # swap for any HF text-gen model
)
```

---

## Stack

| Component | Library |
|---|---|
| Document OCR | [doctr](https://github.com/mindee/doctr) (HuggingFace pretrained) |
| Language model | [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |
| Schema validation | [Pydantic v2](https://docs.pydantic.dev/) |
| Web interface | [Gradio](https://gradio.app/) · deployed on [HuggingFace Spaces](https://huggingface.co/spaces/mansourkama/document-extraction-pipeline) |
| Test fixtures | [Pillow](https://python-pillow.org/) |
