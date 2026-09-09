"""Gradio web interface — document chat + structured extraction.

Two modes:
  - Chat: free-form natural language instructions on the document
  - Extract: template-based structured JSON output

Models load lazily on first GPU request; three fallback LLMs in cascade.
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

# Multilingual example prompts (8 per language)
EXAMPLES = {
    "en": [
        "Summarize in 3 bullet points.",
        "Extract all dates and amounts as JSON.",
        "List all parties and their roles.",
        "Convert key info into a markdown table.",
        "What are the payment terms?",
        "What are the main obligations?",
        "Translate key fields into French.",
        "What is the total value?",
    ],
    "fr": [
        "Résume en 3 points clés.",
        "Extrais toutes les dates et montants en JSON.",
        "Liste toutes les parties et leurs rôles.",
        "Convertis les infos clés en tableau markdown.",
        "Quelles sont les conditions de paiement ?",
        "Quelles sont les obligations principales ?",
        "Traduis les champs principaux en anglais.",
        "Quel est le montant total ?",
    ],
}

# Language instruction prepended to every chat prompt
LANG_PREFIX = {
    "en": "Answer in English.",
    "fr": "Réponds en français.",
}

UI_TEXT = {
    "en": {
        "subtitle": "Upload a document — then ask anything in plain English.",
        "step1": "Load document",
        "tab_upload": "📎 Upload file",
        "tab_paste": "✏️ Paste text",
        "file_label": "PDF or image (PNG, JPG, JPEG)",
        "ocr_btn": "Extract text →",
        "paste_placeholder": "Paste document text here…",
        "paste_btn": "Use this text →",
        "preview_label": "Extracted text preview",
        "step2": "Analyse",
        "tab_chat": "💬 Chat",
        "tab_extract": "🗂 Extract JSON",
        "instr_placeholder": 'e.g. "Summarize in 3 points" · "Extract all amounts as JSON"',
        "instr_label": "Your instruction",
        "examples_title": "Quick examples",
        "chat_btn": "Send →",
        "response_label": "Response",
        "doctype_label": "Document type",
        "extract_btn": "Extract →",
        "json_label": "Extracted fields (JSON)",
        "warn_no_file": "⚠ Please upload a file.",
        "warn_no_text_ocr": "⚠ No document text. Run OCR first, or paste text.",
        "warn_no_instr": "⚠ Please enter an instruction.",
        "warn_no_paste": "⚠ Please enter some text.",
        "ok_ocr": "✓ {} characters extracted.",
        "ok_paste": "✓ {} characters loaded.",
        "err_ocr": "✗ OCR error: {}",
        "err_chat": "✗ Error: {}",
    },
    "fr": {
        "subtitle": "Importez un document — puis posez n'importe quelle question.",
        "step1": "Charger le document",
        "tab_upload": "📎 Importer un fichier",
        "tab_paste": "✏️ Coller du texte",
        "file_label": "PDF ou image (PNG, JPG, JPEG)",
        "ocr_btn": "Extraire le texte →",
        "paste_placeholder": "Coller le texte du document ici…",
        "paste_btn": "Utiliser ce texte →",
        "preview_label": "Aperçu du texte extrait",
        "step2": "Analyser",
        "tab_chat": "💬 Chat",
        "tab_extract": "🗂 Extraction JSON",
        "instr_placeholder": 'ex. "Résume en 3 points" · "Extrais les montants en JSON"',
        "instr_label": "Votre instruction",
        "examples_title": "Exemples rapides",
        "chat_btn": "Envoyer →",
        "response_label": "Réponse",
        "doctype_label": "Type de document",
        "extract_btn": "Extraire →",
        "json_label": "Champs extraits (JSON)",
        "warn_no_file": "⚠ Veuillez importer un fichier.",
        "warn_no_text_ocr": "⚠ Aucun texte. Lancez l'OCR ou collez du texte.",
        "warn_no_instr": "⚠ Veuillez entrer une instruction.",
        "warn_no_paste": "⚠ Veuillez entrer du texte.",
        "ok_ocr": "✓ {} caractères extraits.",
        "ok_paste": "✓ {} caractères chargés.",
        "err_ocr": "✗ Erreur OCR : {}",
        "err_chat": "✗ Erreur : {}",
    },
}

_pipeline: DocumentPipeline | None = None


def _get_pipeline() -> DocumentPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = DocumentPipeline()
    return _pipeline


# ── GPU functions ────────────────────────────────────────────────────────────

@spaces.GPU(duration=60)
def run_ocr(file_path, lang):
    t = UI_TEXT.get(lang, UI_TEXT["en"])
    if file_path is None:
        return "", t["warn_no_file"]
    path = file_path if isinstance(file_path, str) else str(file_path)
    try:
        text = _get_pipeline().ocr_only(path)
        return text, t["ok_ocr"].format(len(text))
    except Exception as e:
        return "", t["err_ocr"].format(e)


@spaces.GPU(duration=90)
def run_chat(doc_text, instruction, lang):
    t = UI_TEXT.get(lang, UI_TEXT["en"])
    if not doc_text or not doc_text.strip():
        return t["warn_no_text_ocr"]
    if not instruction or not instruction.strip():
        return t["warn_no_instr"]
    prefix = LANG_PREFIX.get(lang, "")
    full_instruction = f"{prefix} {instruction}".strip()
    try:
        return _get_pipeline().chat(doc_text, full_instruction)
    except Exception as e:
        return t["err_chat"].format(e)


@spaces.GPU(duration=90)
def run_extract(doc_text, template_name, lang):
    t = UI_TEXT.get(lang, UI_TEXT["en"])
    if not doc_text or not doc_text.strip():
        return json.dumps({"error": t["warn_no_text_ocr"]}, indent=2)
    try:
        result = _get_pipeline().process_text(doc_text, TEMPLATES[template_name])
        return result.model_dump_json(indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


def get_doc_preview(file_path):
    """Render document to a displayable image (CPU, no GPU needed).

    Images are returned as-is. PDFs are rasterised at 1.5× scale using
    pypdfium2 (bundled with doctr) or PyMuPDF as fallback.
    """
    if file_path is None:
        return None
    path = file_path if isinstance(file_path, str) else str(file_path)
    ext = path.rsplit(".", 1)[-1].lower()
    if ext in ("png", "jpg", "jpeg", "webp", "bmp", "tiff"):
        return path
    if ext == "pdf":
        try:
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(path)
            bitmap = pdf[0].render(scale=1.5)
            return bitmap.to_pil()
        except Exception:
            pass
        try:
            import fitz
            from PIL import Image as PILImage
            import io
            pix = fitz.open(path)[0].get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            return PILImage.open(io.BytesIO(pix.tobytes("png")))
        except Exception:
            return None
    return None


# ── UI ───────────────────────────────────────────────────────────────────────

CSS = """
/* Step label */
.step-label {
    display: flex;
    align-items: center;
    gap: .55rem;
    font-weight: 700;
    font-size: .92rem;
    color: #f97316;
    margin-bottom: .85rem;
    letter-spacing: .01em;
}
.step-num {
    width: 1.65rem;
    height: 1.65rem;
    border-radius: 50%;
    background: #f97316;
    color: #fff;
    font-size: .75rem;
    font-weight: 700;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

/* Accent buttons */
.accent-btn button {
    background: #f97316 !important;
    color: #fff !important;
    border: none !important;
    font-weight: 600 !important;
    transition: background .15s !important;
}
.accent-btn button:hover { background: #ea6c0a !important; }

/* Example chips */
.chip-row { display: flex; flex-wrap: wrap; gap: .35rem; margin: .4rem 0 .75rem; }
.chip button {
    font-size: .76rem !important;
    padding: .22rem .65rem !important;
    border-radius: 9999px !important;
    line-height: 1.4 !important;
}

/* Status bar: hide label, shrink textarea */
#status-box { margin-top: .35rem; }
#status-box label { display: none; }
#status-box textarea {
    font-size: .82rem;
    color: #6b7280;
    min-height: 1.6rem !important;
    max-height: 2.5rem !important;
    resize: none;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* Preview */
#doc-preview textarea { font-family: monospace; font-size: .79rem; color: #6b7280; }

/* Output areas */
#chat-output textarea, #extract-output textarea { font-size: .87rem; }

/* Compact lang selector */
#lang-selector { max-width: 160px; }
#lang-selector label { font-size: .78rem; margin-bottom: .1rem; }
#lang-selector select { padding: .25rem .5rem; font-size: .83rem; }

/* Footer */
.footer-txt { text-align: center; font-size: .77rem; color: #9ca3af; margin-top: .75rem; }
.footer-txt a { color: #f97316; text-decoration: none; }

/* Section divider */
.section-divider { border: none; border-top: 1px solid #e5e7eb; margin: .5rem 0 1rem; }

/* Document viewer */
#doc-viewer { border-radius: 8px; overflow: hidden; }
#doc-viewer img { object-fit: contain; max-height: 380px; width: 100%; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=CSS, title="Document Chat") as demo:

    lang_state = gr.State("en")
    doc_state  = gr.State("")

    # ── Header ────────────────────────────────────────────────────────────────
    with gr.Row(equal_height=True):
        with gr.Column(scale=5):
            gr.Markdown("# Document Chat")
            subtitle_md = gr.Markdown(UI_TEXT["en"]["subtitle"])
        with gr.Column(scale=1, min_width=155, elem_id="lang-selector"):
            lang_selector = gr.Dropdown(
                choices=[("🇬🇧 English", "en"), ("🇫🇷 Français", "fr")],
                value="en",
                label="Language / Langue",
                container=True,
                interactive=True,
            )

    # ── Step 1: Document input ────────────────────────────────────────────────
    with gr.Group():
        step1_label = gr.HTML(
            '<div class="step-label">'
            '<span class="step-num">1</span>'
            f'{UI_TEXT["en"]["step1"]}'
            '</div>'
        )
        with gr.Row(equal_height=False):

            # Left: controls ──────────────────────────────────────────────
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.Tab(UI_TEXT["en"]["tab_upload"]) as tab_upload:
                        file_input = gr.File(
                            label=UI_TEXT["en"]["file_label"],
                            file_types=[".pdf", ".png", ".jpg", ".jpeg"],
                        )
                        ocr_btn = gr.Button(
                            UI_TEXT["en"]["ocr_btn"],
                            variant="primary",
                            elem_classes="accent-btn",
                        )

                    with gr.Tab(UI_TEXT["en"]["tab_paste"]) as tab_paste:
                        paste_input = gr.Textbox(
                            lines=6,
                            placeholder=UI_TEXT["en"]["paste_placeholder"],
                            label="",
                            show_label=False,
                        )
                        paste_btn = gr.Button(
                            UI_TEXT["en"]["paste_btn"],
                            variant="secondary",
                        )

                ocr_status = gr.Textbox(
                    value="",
                    label="",
                    interactive=False,
                    elem_id="status-box",
                )
                with gr.Accordion(UI_TEXT["en"]["preview_label"], open=False) as preview_acc:
                    doc_preview = gr.Textbox(
                        lines=6,
                        interactive=False,
                        label="",
                        elem_id="doc-preview",
                        show_copy_button=True,
                    )

            # Right: document viewer ───────────────────────────────────────
            with gr.Column(scale=1):
                doc_image = gr.Image(
                    label="Document viewer",
                    interactive=False,
                    show_download_button=False,
                    show_label=True,
                    elem_id="doc-viewer",
                    height=380,
                )

    # ── Step 2: Analyse ───────────────────────────────────────────────────────
    with gr.Group():
        step2_label = gr.HTML(
            '<div class="step-label">'
            '<span class="step-num">2</span>'
            f'{UI_TEXT["en"]["step2"]}'
            '</div>'
        )
        with gr.Tabs():

            # Chat ─────────────────────────────────────────────────────────
            with gr.Tab(UI_TEXT["en"]["tab_chat"]):
                instruction_input = gr.Textbox(
                    lines=2,
                    placeholder=UI_TEXT["en"]["instr_placeholder"],
                    label=UI_TEXT["en"]["instr_label"],
                )
                examples_title = gr.Markdown(f"**{UI_TEXT['en']['examples_title']}**")
                chips = []
                with gr.Row(elem_classes="chip-row"):
                    for i in range(4):
                        btn = gr.Button(
                            EXAMPLES["en"][i],
                            size="sm",
                            elem_classes="chip",
                        )
                        chips.append(btn)
                with gr.Row(elem_classes="chip-row"):
                    for i in range(4, 8):
                        btn = gr.Button(
                            EXAMPLES["en"][i],
                            size="sm",
                            elem_classes="chip",
                        )
                        chips.append(btn)

                chat_btn = gr.Button(
                    UI_TEXT["en"]["chat_btn"],
                    variant="primary",
                    elem_classes="accent-btn",
                )
                chat_output = gr.Textbox(
                    lines=10,
                    label=UI_TEXT["en"]["response_label"],
                    show_copy_button=True,
                    elem_id="chat-output",
                )
                chat_btn.click(
                    fn=run_chat,
                    inputs=[doc_state, instruction_input, lang_state],
                    outputs=chat_output,
                )

            # Extract ──────────────────────────────────────────────────────
            with gr.Tab(UI_TEXT["en"]["tab_extract"]):
                template_selector = gr.Dropdown(
                    choices=list(TEMPLATES.keys()),
                    value="📄 Invoice",
                    label=UI_TEXT["en"]["doctype_label"],
                )
                extract_btn = gr.Button(
                    UI_TEXT["en"]["extract_btn"],
                    variant="primary",
                    elem_classes="accent-btn",
                )
                extract_output = gr.Textbox(
                    lines=10,
                    label=UI_TEXT["en"]["json_label"],
                    show_copy_button=True,
                    elem_id="extract-output",
                )
                extract_btn.click(
                    fn=run_extract,
                    inputs=[doc_state, template_selector, lang_state],
                    outputs=extract_output,
                )

    gr.HTML(
        '<div class="footer-txt">'
        'OCR: <b>doctr</b> &nbsp;·&nbsp; LLM: <b>Qwen2.5-1.5B</b> + fallbacks &nbsp;·&nbsp; ZeroGPU &nbsp;·&nbsp;'
        '<a href="https://github.com/mansourkama/document-extraction-pipeline">GitHub</a>'
        '</div>'
    )

    # ── Event wiring ──────────────────────────────────────────────────────────

    def _ocr_and_store(file_path, lang):
        text, status = run_ocr(file_path, lang)
        return text, text, status

    def _paste_and_store(text, lang):
        t = UI_TEXT.get(lang, UI_TEXT["en"])
        if not text or not text.strip():
            return "", "", t["warn_no_paste"]
        return text, text, t["ok_paste"].format(len(text))

    def _change_lang(lang):
        t = UI_TEXT.get(lang, UI_TEXT["en"])
        examples = EXAMPLES.get(lang, EXAMPLES["en"])
        chip_updates = [gr.update(value=ex) for ex in examples]
        return (
            lang,
            t["subtitle"],
            f'<div class="step-label"><span class="step-num">1</span>{t["step1"]}</div>',
            f'<div class="step-label"><span class="step-num">2</span>{t["step2"]}</div>',
            f"**{t['examples_title']}**",
            *chip_updates,
        )

    def _chip_click(lang, idx):
        return EXAMPLES.get(lang, EXAMPLES["en"])[idx]

    # Show document preview as soon as a file is selected (no GPU needed)
    file_input.change(
        fn=get_doc_preview,
        inputs=file_input,
        outputs=doc_image,
    )

    ocr_btn.click(
        fn=_ocr_and_store,
        inputs=[file_input, lang_state],
        outputs=[doc_state, doc_preview, ocr_status],
    )
    paste_btn.click(
        fn=_paste_and_store,
        inputs=[paste_input, lang_state],
        outputs=[doc_state, doc_preview, ocr_status],
    )

    lang_selector.change(
        fn=_change_lang,
        inputs=lang_selector,
        outputs=[
            lang_state,
            subtitle_md,
            step1_label,
            step2_label,
            examples_title,
            *chips,
        ],
    )

    # Wire each chip: on click, read current language, return the right example
    for i, chip in enumerate(chips):
        chip.click(
            fn=lambda lang, _i=i: EXAMPLES.get(lang, EXAMPLES["en"])[_i],
            inputs=lang_state,
            outputs=instruction_input,
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", show_api=False)
