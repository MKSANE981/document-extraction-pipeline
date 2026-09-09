"""Gradio web interface — document chat + structured extraction.

Layout: two-column — persistent document viewer on the left,
upload controls + chat/extract steps on the right.

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

LANG_PREFIX = {
    "en": "Answer in English.",
    "fr": "Réponds en français.",
}

UI_TEXT = {
    "en": {
        "subtitle": "Upload a document, then ask anything in plain language.",
        "doc_panel": "Document",
        "drop_hint": "Upload a file to preview it here.",
        "step1": "Load document",
        "tab_upload": "📎 Upload",
        "tab_paste": "✏️ Paste text",
        "file_label": "PDF or image (PNG, JPG, JPEG)",
        "ocr_btn": "Extract text →",
        "paste_placeholder": "Paste document text here…",
        "paste_btn": "Use this text →",
        "preview_label": "Extracted text",
        "step2": "Analyse",
        "tab_chat": "💬 Chat",
        "tab_extract": "🗂 Extract JSON",
        "instr_placeholder": 'e.g. "Summarize in 3 points" · "Extract amounts as JSON"',
        "instr_label": "Your instruction",
        "examples_title": "Quick examples",
        "chat_btn": "Send →",
        "response_label": "Response",
        "doctype_label": "Document type",
        "extract_btn": "Extract →",
        "json_label": "Extracted fields (JSON)",
        "warn_no_file": "⚠ Please upload a file.",
        "warn_no_text": "⚠ No document text — run OCR first, or paste text.",
        "warn_no_instr": "⚠ Please enter an instruction.",
        "warn_no_paste": "⚠ Please enter some text.",
        "ok_ocr": "✓ {n} characters extracted.",
        "ok_paste": "✓ {n} characters loaded.",
        "err_ocr": "✗ OCR error: {e}",
        "err_chat": "✗ Error: {e}",
    },
    "fr": {
        "subtitle": "Importez un document, puis posez n'importe quelle question.",
        "doc_panel": "Document",
        "drop_hint": "Importez un fichier pour le voir ici.",
        "step1": "Charger le document",
        "tab_upload": "📎 Importer",
        "tab_paste": "✏️ Coller du texte",
        "file_label": "PDF ou image (PNG, JPG, JPEG)",
        "ocr_btn": "Extraire le texte →",
        "paste_placeholder": "Coller le texte du document ici…",
        "paste_btn": "Utiliser ce texte →",
        "preview_label": "Texte extrait",
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
        "warn_no_text": "⚠ Aucun texte — lancez l'OCR ou collez du texte.",
        "warn_no_instr": "⚠ Veuillez entrer une instruction.",
        "warn_no_paste": "⚠ Veuillez entrer du texte.",
        "ok_ocr": "✓ {n} caractères extraits.",
        "ok_paste": "✓ {n} caractères chargés.",
        "err_ocr": "✗ Erreur OCR : {e}",
        "err_chat": "✗ Erreur : {e}",
    },
}

MAX_PREVIEW_PAGES = 20

_pipeline: DocumentPipeline | None = None


def _get_pipeline() -> DocumentPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = DocumentPipeline()
    return _pipeline


# ── Document preview (CPU — fires immediately on file select) ────────────────

def get_doc_preview(file_path):
    """Rasterise every page of the document to PIL images.

    Returns a list of (PIL Image, caption) tuples for gr.Gallery.
    Images pass through; PDFs are rendered at 2× via pypdfium2 / PyMuPDF.
    """
    if file_path is None:
        return []
    path = file_path if isinstance(file_path, str) else str(file_path)
    ext = path.rsplit(".", 1)[-1].lower()

    if ext in ("png", "jpg", "jpeg", "webp", "bmp", "tiff"):
        return [(path, "")]

    if ext == "pdf":
        pages = []
        try:
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(path)
            total = len(pdf)
            n = min(total, MAX_PREVIEW_PAGES)
            for i in range(n):
                bm = pdf[i].render(scale=2.0)
                caption = f"Page {i + 1} / {total}" if total > 1 else ""
                pages.append((bm.to_pil(), caption))
            return pages
        except Exception:
            pass
        try:
            import fitz
            from PIL import Image as PILImage
            import io
            doc = fitz.open(path)
            total = len(doc)
            n = min(total, MAX_PREVIEW_PAGES)
            for i in range(n):
                pix = doc[i].get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                img = PILImage.open(io.BytesIO(pix.tobytes("png")))
                caption = f"Page {i + 1} / {total}" if total > 1 else ""
                pages.append((img, caption))
            return pages
        except Exception:
            return []

    return []


# ── GPU functions ────────────────────────────────────────────────────────────

@spaces.GPU(duration=60)
def run_ocr(file_path, lang):
    t = UI_TEXT.get(lang, UI_TEXT["en"])
    if file_path is None:
        return "", t["warn_no_file"]
    path = file_path if isinstance(file_path, str) else str(file_path)
    try:
        text = _get_pipeline().ocr_only(path)
        return text, t["ok_ocr"].format(n=len(text))
    except Exception as e:
        return "", t["err_ocr"].format(e=e)


@spaces.GPU(duration=90)
def run_chat(doc_text, instruction, lang):
    t = UI_TEXT.get(lang, UI_TEXT["en"])
    if not doc_text or not doc_text.strip():
        return t["warn_no_text"]
    if not instruction or not instruction.strip():
        return t["warn_no_instr"]
    prefix = LANG_PREFIX.get(lang, "")
    try:
        return _get_pipeline().chat(doc_text, f"{prefix} {instruction}".strip())
    except Exception as e:
        return t["err_chat"].format(e=e)


@spaces.GPU(duration=90)
def run_extract(doc_text, template_name, lang):
    t = UI_TEXT.get(lang, UI_TEXT["en"])
    if not doc_text or not doc_text.strip():
        return json.dumps({"error": t["warn_no_text"]}, indent=2)
    try:
        result = _get_pipeline().process_text(doc_text, TEMPLATES[template_name])
        return result.model_dump_json(indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


# ── UI ───────────────────────────────────────────────────────────────────────

CSS = """
/* ── Step labels ─────────────────────────────────────── */
.step-label {
    display: flex; align-items: center; gap: .5rem;
    font-weight: 700; font-size: .9rem; color: #f97316;
    margin-bottom: .7rem; letter-spacing: .01em;
}
.step-num {
    width: 1.55rem; height: 1.55rem; border-radius: 50%;
    background: #f97316; color: #fff;
    font-size: .72rem; font-weight: 700;
    display: inline-flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}

/* ── Accent buttons ───────────────────────────────────── */
.accent-btn button {
    background: #f97316 !important; color: #fff !important;
    border: none !important; font-weight: 600 !important;
    transition: background .15s !important;
}
.accent-btn button:hover { background: #ea6c0a !important; }

/* ── Example chips ────────────────────────────────────── */
.chip-row { display: flex; flex-wrap: wrap; gap: .3rem; margin: .35rem 0 .6rem; }
.chip button {
    font-size: .74rem !important; padding: .2rem .6rem !important;
    border-radius: 9999px !important; line-height: 1.4 !important;
}

/* ── Status bar ───────────────────────────────────────── */
#status-box { margin-top: .3rem; }
#status-box label { display: none; }
#status-box textarea {
    font-size: .8rem; color: #6b7280;
    min-height: 1.4rem !important; max-height: 2rem !important;
    resize: none; border: none !important;
    background: transparent !important; box-shadow: none !important;
    padding: 0 !important;
}

/* ── Document viewer panel ────────────────────────────── */
#doc-panel-title {
    font-size: .82rem; font-weight: 600; letter-spacing: .06em;
    text-transform: uppercase; color: #9ca3af;
    margin-bottom: .5rem; padding-bottom: .4rem;
    border-bottom: 1px solid #334155;
}
#doc-viewer {
    border-radius: 8px;
    border: 1px solid #334155;
    overflow: hidden;
}
/* make gallery images display at full width, white bg (document on dark bg) */
#doc-viewer .thumbnail-item { background: #fff !important; border-radius: 0 !important; }
#doc-viewer .thumbnail-item img { object-fit: contain !important; background: #fff; }

/* ── Compact lang selector ────────────────────────────── */
#lang-wrap { display: flex; justify-content: flex-end; align-items: center; }
#lang-wrap label { font-size: .78rem; }

/* ── Text preview ─────────────────────────────────────── */
#doc-preview textarea { font-family: monospace; font-size: .78rem; color: #9ca3af; }

/* ── Output boxes ─────────────────────────────────────── */
#chat-out textarea, #json-out textarea { font-size: .86rem; }

/* ── Footer ───────────────────────────────────────────── */
.footer { text-align: center; font-size: .75rem; color: #6b7280; margin-top: .75rem; }
.footer a { color: #f97316; text-decoration: none; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=CSS, title="Document Chat") as demo:

    lang_state = gr.State("en")
    doc_state  = gr.State("")

    # ── Header ────────────────────────────────────────────────────────────────
    with gr.Row(equal_height=True):
        with gr.Column(scale=6):
            gr.Markdown("# Document Chat")
            subtitle_md = gr.Markdown(UI_TEXT["en"]["subtitle"])
        with gr.Column(scale=1, min_width=160, elem_id="lang-wrap"):
            lang_selector = gr.Dropdown(
                choices=[("🇬🇧 English", "en"), ("🇫🇷 Français", "fr")],
                value="en",
                label="Language / Langue",
                container=True,
                interactive=True,
            )

    # ── Main two-column layout ────────────────────────────────────────────────
    with gr.Row(equal_height=False):

        # LEFT — persistent document viewer ───────────────────────────────────
        with gr.Column(scale=5, min_width=320):
            gr.HTML('<div id="doc-panel-title">Document</div>')
            doc_viewer = gr.Gallery(
                label="",
                show_label=False,
                columns=1,
                rows=1,
                height=600,
                object_fit="contain",
                elem_id="doc-viewer",
                show_share_button=False,
                show_download_button=False,
                preview=True,
            )

        # RIGHT — steps ───────────────────────────────────────────────────────
        with gr.Column(scale=6):

            # Step 1 ──────────────────────────────────────────────────────────
            with gr.Group():
                step1_label = gr.HTML(
                    f'<div class="step-label"><span class="step-num">1</span>'
                    f'{UI_TEXT["en"]["step1"]}</div>'
                )
                with gr.Tabs():
                    with gr.Tab(UI_TEXT["en"]["tab_upload"]):
                        file_input = gr.File(
                            label=UI_TEXT["en"]["file_label"],
                            file_types=[".pdf", ".png", ".jpg", ".jpeg"],
                        )
                        ocr_btn = gr.Button(
                            UI_TEXT["en"]["ocr_btn"],
                            variant="primary",
                            elem_classes="accent-btn",
                        )
                    with gr.Tab(UI_TEXT["en"]["tab_paste"]):
                        paste_input = gr.Textbox(
                            lines=5,
                            placeholder=UI_TEXT["en"]["paste_placeholder"],
                            label="", show_label=False,
                        )
                        paste_btn = gr.Button(
                            UI_TEXT["en"]["paste_btn"],
                            variant="secondary",
                        )

                ocr_status = gr.Textbox(
                    value="", label="", interactive=False, elem_id="status-box"
                )
                with gr.Accordion(UI_TEXT["en"]["preview_label"], open=False):
                    doc_preview = gr.Textbox(
                        lines=5, interactive=False,
                        label="", elem_id="doc-preview", show_copy_button=True,
                    )

            # Step 2 ──────────────────────────────────────────────────────────
            with gr.Group():
                step2_label = gr.HTML(
                    f'<div class="step-label"><span class="step-num">2</span>'
                    f'{UI_TEXT["en"]["step2"]}</div>'
                )
                with gr.Tabs():

                    # Chat ─────────────────────────────────────────────────
                    with gr.Tab(UI_TEXT["en"]["tab_chat"]):
                        instruction_input = gr.Textbox(
                            lines=2,
                            placeholder=UI_TEXT["en"]["instr_placeholder"],
                            label=UI_TEXT["en"]["instr_label"],
                        )
                        examples_title = gr.Markdown(
                            f"**{UI_TEXT['en']['examples_title']}**"
                        )
                        chips = []
                        with gr.Row(elem_classes="chip-row"):
                            for i in range(4):
                                btn = gr.Button(
                                    EXAMPLES["en"][i], size="sm",
                                    elem_classes="chip",
                                )
                                chips.append(btn)
                        with gr.Row(elem_classes="chip-row"):
                            for i in range(4, 8):
                                btn = gr.Button(
                                    EXAMPLES["en"][i], size="sm",
                                    elem_classes="chip",
                                )
                                chips.append(btn)

                        chat_btn = gr.Button(
                            UI_TEXT["en"]["chat_btn"],
                            variant="primary", elem_classes="accent-btn",
                        )
                        chat_output = gr.Textbox(
                            lines=10, label=UI_TEXT["en"]["response_label"],
                            show_copy_button=True, elem_id="chat-out",
                        )
                        chat_btn.click(
                            fn=run_chat,
                            inputs=[doc_state, instruction_input, lang_state],
                            outputs=chat_output,
                        )

                    # Extract JSON ─────────────────────────────────────────
                    with gr.Tab(UI_TEXT["en"]["tab_extract"]):
                        template_selector = gr.Dropdown(
                            choices=list(TEMPLATES.keys()),
                            value="📄 Invoice",
                            label=UI_TEXT["en"]["doctype_label"],
                        )
                        extract_btn = gr.Button(
                            UI_TEXT["en"]["extract_btn"],
                            variant="primary", elem_classes="accent-btn",
                        )
                        extract_output = gr.Textbox(
                            lines=10, label=UI_TEXT["en"]["json_label"],
                            show_copy_button=True, elem_id="json-out",
                        )
                        extract_btn.click(
                            fn=run_extract,
                            inputs=[doc_state, template_selector, lang_state],
                            outputs=extract_output,
                        )

    gr.HTML(
        '<div class="footer">'
        'OCR: <b>doctr</b> · LLM: <b>Qwen2.5-1.5B</b> + fallbacks · ZeroGPU · '
        '<a href="https://github.com/mansourkama/document-extraction-pipeline">GitHub</a>'
        '</div>'
    )

    # ── Wiring ────────────────────────────────────────────────────────────────

    # Preview fires immediately on file select — CPU, no GPU wait
    file_input.change(
        fn=get_doc_preview,
        inputs=file_input,
        outputs=doc_viewer,
    )

    def _ocr_and_store(file_path, lang):
        text, status = run_ocr(file_path, lang)
        return text, text, status

    def _paste_and_store(text, lang):
        t = UI_TEXT.get(lang, UI_TEXT["en"])
        if not text or not text.strip():
            return "", "", t["warn_no_paste"]
        return text, text, t["ok_paste"].format(n=len(text))

    def _change_lang(lang):
        t = UI_TEXT.get(lang, UI_TEXT["en"])
        examples = EXAMPLES.get(lang, EXAMPLES["en"])
        return (
            lang,
            t["subtitle"],
            f'<div class="step-label"><span class="step-num">1</span>{t["step1"]}</div>',
            f'<div class="step-label"><span class="step-num">2</span>{t["step2"]}</div>',
            f"**{t['examples_title']}**",
            *[gr.update(value=ex) for ex in examples],
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
        outputs=[lang_state, subtitle_md, step1_label, step2_label, examples_title, *chips],
    )
    for i, chip in enumerate(chips):
        chip.click(
            fn=lambda lang, _i=i: EXAMPLES.get(lang, EXAMPLES["en"])[_i],
            inputs=lang_state,
            outputs=instruction_input,
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", show_api=False)
