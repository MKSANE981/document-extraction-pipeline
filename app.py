"""Gradio web interface for the document extraction pipeline.

Two input modes:
  - Upload tab: drag-and-drop a PDF or image file
  - Text tab:   paste raw text directly (useful for quick demos)

Models load lazily on first GPU request so the Space starts fast.
"""
import json

# HF Spaces ships a very recent huggingface_hub that removed HfFolder,
# but some gradio builds still import it. Patch it back before gradio loads.
try:
    from huggingface_hub import HfFolder  # noqa: F401
except ImportError:
    import huggingface_hub as _hfh
    import os as _os

    class _HfFolder:
        @staticmethod
        def get_token():
            return _os.environ.get("HF_TOKEN")

    _hfh.HfFolder = _HfFolder

import spaces
import gradio as gr
from src.pipeline import DocumentPipeline
from src.templates import (
    InvoiceTemplate,
    PersonalFormTemplate,
    ContractTemplate,
    ReceiptTemplate,
)

TEMPLATES = {
    "📄 Invoice": InvoiceTemplate,
    "📝 Contract / Mission Order": ContractTemplate,
    "🧾 Receipt / Expense": ReceiptTemplate,
    "👤 Personal Form / ID": PersonalFormTemplate,
}

TEMPLATE_DESCRIPTIONS = {
    "📄 Invoice": "Invoices, billing documents — extracts vendor, client, amounts, dates.",
    "📝 Contract / Mission Order": "Service contracts, mission orders — extracts parties, dates, value, terms.",
    "🧾 Receipt / Expense": "Purchase receipts, expense slips — extracts merchant, total, payment method.",
    "👤 Personal Form / ID": "Registration forms, KYC — extracts name, DOB, email, address, nationality.",
}

_pipeline: DocumentPipeline | None = None


def _get_pipeline() -> DocumentPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = DocumentPipeline()
    return _pipeline


def update_description(template_name):
    return TEMPLATE_DESCRIPTIONS.get(template_name, "")


@spaces.GPU(duration=120)
def extract_from_file(file_path, template_name):
    if file_path is None:
        return json.dumps({"error": "Please upload a PDF or image file."}, indent=2)
    path = file_path if isinstance(file_path, str) else str(file_path)
    try:
        result = _get_pipeline().process(path, TEMPLATES[template_name])
        return result.model_dump_json(indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


@spaces.GPU(duration=60)
def extract_from_text(text, template_name):
    if not text or not text.strip():
        return json.dumps({"error": "Please enter some document text."}, indent=2)
    try:
        result = _get_pipeline().process_text(text, TEMPLATES[template_name])
        return result.model_dump_json(indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


CSS = """
#title { text-align: center; margin-bottom: 0.25rem; }
#subtitle { text-align: center; color: #6b7280; margin-bottom: 1.5rem; font-size: 0.95rem; }
#stack { font-size: 0.85rem; text-align: center; color: #9ca3af; margin-bottom: 1.5rem; }
#extract-btn { background: #f97316 !important; border: none !important; }
#extract-btn:hover { background: #ea6c0a !important; }
#template-desc { font-size: 0.85rem; color: #6b7280; padding: 0.25rem 0; min-height: 1.5rem; }
#output-box textarea { font-family: monospace; font-size: 0.85rem; }
.footer { text-align: center; font-size: 0.8rem; color: #9ca3af; margin-top: 1rem; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=CSS, title="Document Extraction Pipeline") as demo:

    gr.Markdown("# Document Extraction Pipeline", elem_id="title")
    gr.Markdown(
        "Extract structured data from any document — PDF or image — using free HuggingFace models. No API key required.",
        elem_id="subtitle",
    )
    gr.Markdown(
        "**OCR:** doctr (db_resnet50 · crnn_vgg16_bn) &nbsp;·&nbsp; **LLM:** Qwen2.5-1.5B-Instruct &nbsp;·&nbsp; **Validation:** Pydantic v2",
        elem_id="stack",
    )

    with gr.Row():
        with gr.Column(scale=1):
            template_selector = gr.Dropdown(
                choices=list(TEMPLATES.keys()),
                value="📄 Invoice",
                label="Document type",
            )
            template_desc = gr.Markdown("", elem_id="template-desc")
            template_selector.change(
                fn=update_description,
                inputs=template_selector,
                outputs=template_desc,
            )
            # Show initial description
            demo.load(
                fn=lambda: update_description("📄 Invoice"),
                outputs=template_desc,
            )

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("📎 Upload document"):
                    gr.Markdown("Drop a **PDF or image** (PNG, JPG). OCR runs automatically.")
                    file_input = gr.File(
                        label="Document",
                        file_types=[".pdf", ".png", ".jpg", ".jpeg"],
                    )
                    file_btn = gr.Button("Extract →", variant="primary", elem_id="extract-btn")
                    file_output = gr.Textbox(
                        label="Extracted fields (JSON)",
                        lines=18,
                        show_copy_button=True,
                        elem_id="output-box",
                    )
                    file_btn.click(
                        fn=extract_from_file,
                        inputs=[file_input, template_selector],
                        outputs=file_output,
                    )

                with gr.Tab("✏️ Paste text"):
                    gr.Markdown("Paste document text directly — skips OCR, faster for quick tests.")
                    text_input = gr.Textbox(
                        lines=10,
                        placeholder="Invoice Number: INV-2024-0042\nDate: 2024-11-15\nVendor: Acme Corp\nTotal Due: 8160.00 EUR\n...",
                        label="Document text",
                    )
                    text_btn = gr.Button("Extract →", variant="primary", elem_id="extract-btn")
                    text_output = gr.Textbox(
                        label="Extracted fields (JSON)",
                        lines=12,
                        show_copy_button=True,
                        elem_id="output-box",
                    )
                    text_btn.click(
                        fn=extract_from_text,
                        inputs=[text_input, template_selector],
                        outputs=text_output,
                    )

    gr.Markdown(
        '<div class="footer">First request takes ~2 min while models load · '
        '<a href="https://github.com/MKSANE981/document-extraction-pipeline">GitHub</a></div>'
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", show_api=False)
