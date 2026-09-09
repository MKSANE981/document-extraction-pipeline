"""Gradio web interface — document chat + structured extraction.

Two modes:
  - Chat: free-form natural language instructions on the document
  - Extract: template-based structured JSON output

Models load lazily on first GPU request.
"""
import json

# ── Compatibility patches (must run before importing gradio) ────────────────

# 1. HfFolder was removed from huggingface_hub ≥0.26 but gradio 4.x imports it
try:
    from huggingface_hub import HfFolder  # noqa: F401
except ImportError:
    import huggingface_hub as _hfh, os as _os  # noqa: E401

    class _HfFolder:
        @staticmethod
        def get_token():
            return _os.environ.get("HF_TOKEN")

    _hfh.HfFolder = _HfFolder

# 2. Pydantic v2 generates "additionalProperties": false in JSON schemas,
#    which gradio 4.x passes as a Python bool to get_type() and crashes.
try:
    import gradio_client.utils as _gcu
    _orig = _gcu._json_schema_to_python_type

    def _safe(schema, defs=None):
        if not isinstance(schema, dict):
            return "Any"
        return _orig(schema, defs)

    _gcu._json_schema_to_python_type = _safe
except Exception:
    pass

# ── Imports ─────────────────────────────────────────────────────────────────
import spaces
import gradio as gr
from src.pipeline import DocumentPipeline
from src.templates import (
    InvoiceTemplate,
    ContractTemplate,
    ReceiptTemplate,
    PersonalFormTemplate,
)

TEMPLATES = {
    "📄 Invoice": InvoiceTemplate,
    "📝 Contract / Mission": ContractTemplate,
    "🧾 Receipt / Expense": ReceiptTemplate,
    "👤 Personal Form": PersonalFormTemplate,
}

EXAMPLES = [
    "Summarize this document in 3 bullet points.",
    "Extract all dates and amounts as a JSON object.",
    "List all parties mentioned with their roles.",
    "Convert the key information into a markdown table.",
    "What are the main obligations of each party?",
    "Extract the payment terms and deadlines.",
    "Translate the key fields into English.",
    "What is the total value and how is it broken down?",
]

_pipeline: DocumentPipeline | None = None


def _get_pipeline() -> DocumentPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = DocumentPipeline()
    return _pipeline


# ── GPU functions ────────────────────────────────────────────────────────────

@spaces.GPU(duration=60)
def run_ocr(file_path):
    """Step 1 — Extract raw text from uploaded document."""
    if file_path is None:
        return "", "⚠ Please upload a file."
    path = file_path if isinstance(file_path, str) else str(file_path)
    try:
        text = _get_pipeline().ocr_only(path)
        chars = len(text)
        return text, f"✓ OCR complete — {chars} characters extracted."
    except Exception as e:
        return "", f"✗ OCR error: {e}"


@spaces.GPU(duration=90)
def run_chat(doc_text, instruction):
    """Step 2 — Answer a free-form instruction on the document text."""
    if not doc_text or not doc_text.strip():
        return "⚠ No document text. Run OCR first, or paste text in the Document tab."
    if not instruction or not instruction.strip():
        return "⚠ Please enter an instruction."
    try:
        return _get_pipeline().chat(doc_text, instruction)
    except Exception as e:
        return f"✗ Error: {e}"


@spaces.GPU(duration=90)
def run_extract(doc_text, template_name):
    """Template-based structured extraction — returns JSON."""
    if not doc_text or not doc_text.strip():
        return '{"error": "No document text. Run OCR first, or paste text."}'
    try:
        result = _get_pipeline().process_text(doc_text, TEMPLATES[template_name])
        return result.model_dump_json(indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


def fill_instruction(example):
    return example


# ── UI ───────────────────────────────────────────────────────────────────────

CSS = """
#title { text-align:center; margin-bottom:.1rem; }
#subtitle { text-align:center; color:#6b7280; font-size:.95rem; margin-bottom:1.5rem; }
#stack-row { text-align:center; font-size:.8rem; color:#9ca3af; margin-bottom:1.5rem; }
.primary-btn { background:#f97316 !important; border:none !important; }
.primary-btn:hover { background:#ea6c0a !important; }
#status-box textarea { font-size:.85rem; color:#6b7280; }
#doc-preview textarea { font-family: monospace; font-size:.82rem; }
#output-area textarea { font-size:.88rem; }
.example-chip button { font-size:.8rem !important; padding:.2rem .6rem !important; }
.footer-txt { text-align:center; font-size:.78rem; color:#9ca3af; margin-top:1rem; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=CSS, title="Document Chat") as demo:

    doc_state = gr.State("")  # stores extracted text across interactions

    gr.Markdown("# Document Chat", elem_id="title")
    gr.Markdown(
        "Talk to your document — extract, structure, summarize, translate, or export in any format.",
        elem_id="subtitle",
    )
    gr.Markdown(
        "**OCR:** doctr &nbsp;·&nbsp; **LLM:** Qwen2.5-1.5B-Instruct &nbsp;·&nbsp; **Validation:** Pydantic v2 &nbsp;·&nbsp; Runs on ZeroGPU (free)",
        elem_id="stack-row",
    )

    with gr.Row():

        # ── Left: document input ────────────────────────────────────────────
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("📎 Upload file"):
                    file_input = gr.File(
                        label="PDF or image (PNG, JPG)",
                        file_types=[".pdf", ".png", ".jpg", ".jpeg"],
                    )
                    ocr_btn = gr.Button("① Extract text (OCR)", variant="primary", elem_classes="primary-btn")
                    ocr_status = gr.Textbox(
                        label="", lines=1, interactive=False, elem_id="status-box"
                    )

                with gr.Tab("✏️ Paste text"):
                    paste_input = gr.Textbox(
                        lines=12,
                        placeholder="Paste document text here...",
                        label="Document text",
                    )
                    paste_btn = gr.Button("Use this text →", variant="secondary")

            with gr.Accordion("📄 Extracted text preview", open=False):
                doc_preview = gr.Textbox(
                    lines=8,
                    interactive=False,
                    label="",
                    elem_id="doc-preview",
                    show_copy_button=True,
                )

        # ── Right: interact ─────────────────────────────────────────────────
        with gr.Column(scale=3):
            with gr.Tabs():

                with gr.Tab("💬 Chat (free-form)"):
                    instruction_input = gr.Textbox(
                        lines=3,
                        placeholder='e.g. "Summarize in 3 bullet points" · "Extract all amounts as JSON" · "Convert to markdown table" · "List all parties"',
                        label="Your instruction",
                    )
                    gr.Markdown("**Quick examples — click to use:**")
                    with gr.Row(elem_classes="example-chip"):
                        for ex in EXAMPLES[:4]:
                            gr.Button(ex, size="sm").click(
                                fn=lambda e=ex: e,
                                outputs=instruction_input,
                            )
                    with gr.Row(elem_classes="example-chip"):
                        for ex in EXAMPLES[4:]:
                            gr.Button(ex, size="sm").click(
                                fn=lambda e=ex: e,
                                outputs=instruction_input,
                            )
                    chat_btn = gr.Button("② Process instruction", variant="primary", elem_classes="primary-btn")
                    chat_output = gr.Textbox(
                        lines=14,
                        label="Response",
                        show_copy_button=True,
                        elem_id="output-area",
                    )
                    chat_btn.click(
                        fn=run_chat,
                        inputs=[doc_state, instruction_input],
                        outputs=chat_output,
                    )

                with gr.Tab("🗂 Extract (structured JSON)"):
                    template_selector = gr.Dropdown(
                        choices=list(TEMPLATES.keys()),
                        value="📄 Invoice",
                        label="Document type",
                    )
                    extract_btn = gr.Button("② Extract fields", variant="primary", elem_classes="primary-btn")
                    extract_output = gr.Textbox(
                        lines=14,
                        label="Extracted fields (JSON)",
                        show_copy_button=True,
                        elem_id="output-area",
                    )
                    extract_btn.click(
                        fn=run_extract,
                        inputs=[doc_state, template_selector],
                        outputs=extract_output,
                    )

    gr.Markdown(
        '<div class="footer-txt">First request takes ~2 min while models load on ZeroGPU · '
        '<a href="https://github.com/MKSANE981/document-extraction-pipeline">GitHub</a></div>'
    )

    # ── Wiring ───────────────────────────────────────────────────────────────
    def _ocr_and_store(file_path):
        text, status = run_ocr(file_path)
        return text, text, status

    ocr_btn.click(
        fn=_ocr_and_store,
        inputs=file_input,
        outputs=[doc_state, doc_preview, ocr_status],
    )

    def _paste_and_store(text):
        if not text or not text.strip():
            return "", "", "⚠ Please enter some text."
        return text, text, f"✓ {len(text)} characters loaded."

    paste_btn.click(
        fn=_paste_and_store,
        inputs=paste_input,
        outputs=[doc_state, doc_preview, ocr_status],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", show_api=False)
