"""Gradio web interface for the document extraction pipeline.

Two input modes:
  - Upload tab: drag-and-drop a PDF or image file
  - Text tab:   paste raw text directly (useful for quick demos)

Models load lazily on first request so the Space starts fast.
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

import gradio as gr
from src.pipeline import DocumentPipeline
from src.templates import InvoiceTemplate, PersonalFormTemplate

TEMPLATES = {
    "Invoice": InvoiceTemplate,
    "Personal Form": PersonalFormTemplate,
}

_pipeline: DocumentPipeline | None = None


def _get_pipeline() -> DocumentPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = DocumentPipeline()
    return _pipeline


def extract_from_file(file_path, template_name: str) -> str:
    if file_path is None:
        return json.dumps({"error": "Please upload a PDF or image file."}, indent=2)
    # Gradio 5 passes the filepath directly as a string
    path = file_path if isinstance(file_path, str) else str(file_path)
    try:
        result = _get_pipeline().process(path, TEMPLATES[template_name])
        return result.model_dump_json(indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


def extract_from_text(text: str, template_name: str) -> str:
    if not text.strip():
        return json.dumps({"error": "Please enter some text."}, indent=2)
    try:
        result = _get_pipeline().process_text(text, TEMPLATES[template_name])
        return result.model_dump_json(indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


with gr.Blocks(title="Document Extraction Pipeline") as demo:
    gr.Markdown(
        """
        # Document Extraction Pipeline
        Extract structured data from documents using free HuggingFace models —
        no API key required.

        **OCR:** doctr (db_resnet50 + crnn_vgg16_bn) &nbsp;|&nbsp;
        **LLM:** Qwen2.5-1.5B-Instruct &nbsp;|&nbsp;
        **Validation:** Pydantic v2
        """
    )

    template_selector = gr.Dropdown(
        choices=list(TEMPLATES.keys()),
        value="Invoice",
        label="Template",
    )

    with gr.Tabs():
        with gr.Tab("Upload document"):
            gr.Markdown("Upload a **PDF or image** (PNG, JPG). OCR runs automatically.")
            file_input = gr.File(label="Document", file_types=[".pdf", ".png", ".jpg", ".jpeg"])
            file_btn = gr.Button("Extract", variant="primary")
            file_output = gr.Code(language="json", label="Extracted fields")
            file_btn.click(
                fn=extract_from_file,
                inputs=[file_input, template_selector],
                outputs=file_output,
            )

        with gr.Tab("Paste text"):
            gr.Markdown("Paste the document text directly — skips OCR, runs faster.")
            text_input = gr.Textbox(
                lines=10,
                placeholder="Invoice Number: INV-2024-0042\nDate: 2024-11-15\nVendor: Acme Corp\nTotal Due: 8160.00 EUR\n...",
                label="Document text",
            )
            text_btn = gr.Button("Extract", variant="primary")
            text_output = gr.Code(language="json", label="Extracted fields")
            text_btn.click(
                fn=extract_from_text,
                inputs=[text_input, template_selector],
                outputs=text_output,
            )

    gr.Markdown(
        """
        ---
        **Note:** First request takes ~2 minutes while models load.
        Subsequent requests are fast (models stay in memory).

        [GitHub](https://github.com/MKSANE981/document-extraction-pipeline)
        """
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", show_api=False)
