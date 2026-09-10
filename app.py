"""Gradio web interface — document chat + structured extraction."""
import json

# ── Compatibility patches ────────────────────────────────────────────────────
try:
    from huggingface_hub import HfFolder  # noqa: F401
except ImportError:
    import huggingface_hub as _hfh, os as _os  # noqa: E401
    class _HfFolder:
        @staticmethod
        def get_token():
            return _os.environ.get("HF_TOKEN")
    _hfh.HfFolder = _HfFolder

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

import spaces
import gradio as gr
from src.pipeline import DocumentPipeline
from src.templates import InvoiceTemplate, ContractTemplate, ReceiptTemplate, PersonalFormTemplate

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

LANG_PREFIX = {"en": "Answer in English.", "fr": "Réponds en français."}

UI = {
    "en": {
        "subtitle": "Upload any document · ask anything · extract, summarize, translate",
        "step1": "Load document", "step2": "Analyse",
        "tab_upload": "📎 Upload", "tab_paste": "✏️ Paste",
        "file_label": "PDF or image (PNG, JPG, JPEG)",
        "ocr_btn": "Extract text →", "paste_btn": "Use this text →",
        "paste_ph": "Paste document text here…",
        "preview_lbl": "Extracted text",
        "tab_chat": "💬 Chat", "tab_extract": "🗂 Extract JSON",
        "instr_lbl": "Instruction",
        "instr_ph": 'e.g. "Summarize in 3 bullet points" · "Extract all amounts"',
        "examples": "Quick examples", "chat_btn": "Send →",
        "resp_lbl": "Response", "type_lbl": "Document type",
        "extract_btn": "Extract →", "json_lbl": "Extracted fields (JSON)",
        "no_file": "⚠ Upload a file first.",
        "no_text": "⚠ No text yet — run OCR or paste text.",
        "no_instr": "⚠ Enter an instruction.",
        "no_paste": "⚠ Paste some text first.",
        "ok_ocr": "✓ {n} characters extracted",
        "ok_paste": "✓ {n} characters loaded",
        "err_ocr": "✗ OCR error: {e}",
        "err_chat": "✗ {e}",
    },
    "fr": {
        "subtitle": "Importez un document · posez n'importe quelle question · extrayez, résumez, traduisez",
        "step1": "Charger le document", "step2": "Analyser",
        "tab_upload": "📎 Importer", "tab_paste": "✏️ Coller",
        "file_label": "PDF ou image (PNG, JPG, JPEG)",
        "ocr_btn": "Extraire le texte →", "paste_btn": "Utiliser ce texte →",
        "paste_ph": "Collez le texte du document ici…",
        "preview_lbl": "Texte extrait",
        "tab_chat": "💬 Chat", "tab_extract": "🗂 Extraction JSON",
        "instr_lbl": "Instruction",
        "instr_ph": 'ex. "Résume en 3 points" · "Extrais les montants"',
        "examples": "Exemples rapides", "chat_btn": "Envoyer →",
        "resp_lbl": "Réponse", "type_lbl": "Type de document",
        "extract_btn": "Extraire →", "json_lbl": "Champs extraits (JSON)",
        "no_file": "⚠ Importez un fichier d'abord.",
        "no_text": "⚠ Aucun texte — lancez l'OCR ou collez du texte.",
        "no_instr": "⚠ Entrez une instruction.",
        "no_paste": "⚠ Collez d'abord du texte.",
        "ok_ocr": "✓ {n} caractères extraits",
        "ok_paste": "✓ {n} caractères chargés",
        "err_ocr": "✗ Erreur OCR : {e}",
        "err_chat": "✗ {e}",
    },
}

MAX_PAGES = 20
_pipeline: DocumentPipeline | None = None

def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = DocumentPipeline()
    return _pipeline

def get_doc_preview(file_path):
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
            for i in range(min(total, MAX_PAGES)):
                bm = pdf[i].render(scale=2.0)
                pages.append((bm.to_pil(), f"Page {i+1} / {total}" if total > 1 else ""))
            return pages
        except Exception:
            pass
        try:
            import fitz
            from PIL import Image as PILImage
            import io
            doc = fitz.open(path)
            total = len(doc)
            for i in range(min(total, MAX_PAGES)):
                pix = doc[i].get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                img = PILImage.open(io.BytesIO(pix.tobytes("png")))
                pages.append((img, f"Page {i+1} / {total}" if total > 1 else ""))
            return pages
        except Exception:
            return []
    return []

@spaces.GPU(duration=60)
def run_ocr(file_path, lang):
    t = UI.get(lang, UI["en"])
    if file_path is None:
        return "", t["no_file"]
    path = file_path if isinstance(file_path, str) else str(file_path)
    try:
        text = _get_pipeline().ocr_only(path)
        return text, t["ok_ocr"].format(n=len(text))
    except Exception as e:
        return "", t["err_ocr"].format(e=e)

@spaces.GPU(duration=90)
def run_chat(doc_text, instruction, lang):
    t = UI.get(lang, UI["en"])
    if not doc_text or not doc_text.strip():
        return t["no_text"]
    if not instruction or not instruction.strip():
        return t["no_instr"]
    prefix = LANG_PREFIX.get(lang, "")
    try:
        return _get_pipeline().chat(doc_text, f"{prefix} {instruction}".strip())
    except Exception as e:
        return t["err_chat"].format(e=e)

@spaces.GPU(duration=90)
def run_extract(doc_text, template_name, lang):
    t = UI.get(lang, UI["en"])
    if not doc_text or not doc_text.strip():
        return json.dumps({"error": t["no_text"]}, indent=2)
    try:
        result = _get_pipeline().process_text(doc_text, TEMPLATES[template_name])
        return result.model_dump_json(indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


# ── Design system ─────────────────────────────────────────────────────────────
#
# Palette: deep navy ground / blue-tinted cards / sky-blue accent / white document
# Strategy: embrace Gradio's natural dark rendering; override body + block
# backgrounds through both our CSS tokens AND aggressive svelte-class selectors.
# The document viewer keeps a white background so pages look like paper.

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Tokens ───────────────────────────────────────────────── */
:root {
    --bg:          #070C18;
    --surface:     #0D1526;
    --surface2:    #162038;
    --border:      #1C2E50;
    --border-hi:   #2A4070;
    --text:        #DDE6FF;
    --muted:       #5C7399;
    --accent:      #4D9EFF;
    --accent-h:    #2D84F0;
    --accent-glow: rgba(77,158,255,.15);
    --accent-ring: rgba(77,158,255,.35);
    --ok:          #34D399;
    --err:         #FB7185;
    --radius:      10px;
    --doc-bg:      #FFFFFF;
}

/* ── Page foundation ──────────────────────────────────────── */
html { color-scheme: dark; }
body, .gradio-container, .main, .wrap {
    background: var(--bg) !important;
    font-family: 'Inter', ui-sans-serif, system-ui, sans-serif !important;
    color: var(--text) !important;
}

/* ── Force dark on ALL Gradio Svelte components ───────────── */
[class*="svelte-"] {
    color: var(--text) !important;
}
.block, .form, .tabitem, .tab-item,
.input-text, .output-text, .wrap-inner,
.file-preview, .upload-container, .file-uploader,
.file-preview-holder {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}
textarea, input[type="text"], input[type="search"], select {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Tab nav ──────────────────────────────────────────────── */
.tabs > .tab-nav {
    background: var(--surface) !important;
    border-bottom: 1px solid var(--border) !important;
}
.tabs > .tab-nav button {
    color: var(--muted) !important;
    border-bottom: 2px solid transparent !important;
    font-size: .82rem !important;
    font-weight: 500 !important;
    transition: color .15s, border-color .15s !important;
}
.tabs > .tab-nav button.selected {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

/* ── Step cards ───────────────────────────────────────────── */
#step1, #step2 {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1.1rem 1.25rem !important;
    margin-bottom: .7rem !important;
    box-shadow: 0 4px 24px rgba(0,0,0,.35) !important;
}
#step1 .block, #step2 .block {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
}
#step1 textarea, #step2 textarea,
#step1 input, #step2 input {
    background: var(--surface2) !important;
}

/* ── Step label ───────────────────────────────────────────── */
.step-label {
    display: flex; align-items: center; gap: .55rem;
    font-size: .72rem; font-weight: 700; letter-spacing: .1em;
    text-transform: uppercase; color: var(--accent);
    margin-bottom: .9rem;
}
.step-num {
    width: 1.55rem; height: 1.55rem; border-radius: 50%;
    background: var(--accent); color: #fff;
    font-size: .7rem; font-weight: 800;
    display: inline-flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    box-shadow: 0 0 0 4px var(--accent-glow), 0 0 16px var(--accent-glow);
}

/* ── Document viewer panel ────────────────────────────────── */
#viewer-col {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden !important;
    box-shadow: 0 4px 24px rgba(0,0,0,.35) !important;
}
#viewer-hdr {
    background: var(--surface2);
    border-bottom: 1px solid var(--border);
    padding: .65rem 1rem;
    font-size: .68rem; font-weight: 700; letter-spacing: .1em;
    text-transform: uppercase; color: var(--muted);
    display: flex; align-items: center; gap: .45rem;
}
/* Gallery inside viewer — pages look like paper */
#doc-viewer {
    background: #1a1a1a !important;
    border: none !important;
    border-radius: 0 !important;
}
#doc-viewer .thumbnail-item {
    background: var(--doc-bg) !important;
    border-radius: 4px !important;
    box-shadow: 0 2px 16px rgba(0,0,0,.6) !important;
    margin: 8px auto !important;
    max-width: calc(100% - 16px) !important;
}
#doc-viewer .thumbnail-item img {
    object-fit: contain !important;
    background: var(--doc-bg) !important;
}

/* ── Language toggle (Radio) ──────────────────────────────── */
#lang-toggle { display: flex; justify-content: flex-end; align-items: center; }
#lang-toggle .block { background: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important; }
#lang-toggle .wrap { background: transparent !important; border: none !important; gap: .3rem; flex-wrap: nowrap; }
#lang-toggle label.svelte-1gfknih, #lang-toggle .choice-label,
#lang-toggle [data-testid="radio-label"], #lang-toggle label {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 9999px !important;
    padding: .28rem .75rem !important;
    font-size: .76rem !important; font-weight: 600 !important;
    color: var(--muted) !important; cursor: pointer !important;
    transition: all .15s !important;
}
#lang-toggle input[type="radio"]:checked ~ label,
#lang-toggle .selected label,
#lang-toggle [aria-checked="true"] label {
    background: var(--accent-glow) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* ── Primary button ───────────────────────────────────────── */
.go-btn button {
    background: var(--accent) !important;
    color: #fff !important; border: none !important;
    font-weight: 600 !important; font-family: 'Inter', sans-serif !important;
    letter-spacing: .01em !important;
    box-shadow: 0 2px 12px var(--accent-glow) !important;
    transition: background .15s, box-shadow .15s, transform .1s !important;
}
.go-btn button:hover {
    background: var(--accent-h) !important;
    box-shadow: 0 4px 20px var(--accent-ring) !important;
    transform: translateY(-1px) !important;
}
.go-btn button:active { transform: translateY(0) !important; }

/* ── Example chips ────────────────────────────────────────── */
.chip-row { display: flex; flex-wrap: wrap; gap: .3rem; margin: .35rem 0 .65rem; }
.chip button {
    font-size: .72rem !important; padding: .22rem .62rem !important;
    border-radius: 9999px !important; line-height: 1.5 !important;
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--muted) !important;
    transition: all .12s !important; white-space: nowrap !important;
}
.chip button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: var(--accent-glow) !important;
}

/* ── Status bar ───────────────────────────────────────────── */
#status-bar { margin-top: .25rem; }
#status-bar label { display: none !important; }
#status-bar textarea {
    font-size: .78rem !important; min-height: 1.3rem !important;
    max-height: 1.6rem !important; resize: none !important;
    border: none !important; background: transparent !important;
    box-shadow: none !important; padding: 0 !important;
    color: var(--muted) !important;
}

/* ── Extracted text accordion ─────────────────────────────── */
#text-preview textarea {
    font-family: 'JetBrains Mono', 'Menlo', monospace !important;
    font-size: .76rem !important; color: var(--muted) !important;
}

/* ── Output boxes ─────────────────────────────────────────── */
#chat-out textarea {
    font-size: .86rem !important; line-height: 1.65 !important;
}
#json-out textarea {
    font-family: 'JetBrains Mono', 'Menlo', monospace !important;
    font-size: .8rem !important; line-height: 1.6 !important;
}

/* ── Header ───────────────────────────────────────────────── */
#app-head { padding: 1.25rem 0 .85rem; border-bottom: 1px solid var(--border); margin-bottom: 1rem; }
#app-head h1 {
    font-size: 1.45rem; font-weight: 800; color: var(--text);
    letter-spacing: -.025em; margin: 0; line-height: 1.15;
}
#app-head p {
    font-size: .82rem; color: var(--muted); margin: .3rem 0 0; line-height: 1.5;
}

/* ── Accordion ────────────────────────────────────────────── */
.label-wrap, details > summary {
    color: var(--muted) !important; font-size: .8rem !important;
}

/* ── Dropdown ─────────────────────────────────────────────── */
#template-sel .wrap, #template-sel select {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

/* ── Footer ───────────────────────────────────────────────── */
#footer {
    text-align: center; font-size: .72rem; color: var(--muted);
    padding: .75rem 0 .5rem;
    border-top: 1px solid var(--border); margin-top: .5rem;
}
#footer a { color: var(--accent); text-decoration: none; }
#footer a:hover { text-decoration: underline; }
"""

# ── Build UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono"),
    ).set(
        body_background_fill="#070C18",
        block_background_fill="#0D1526",
        input_background_fill="#162038",
        block_border_color="#1C2E50",
        block_border_width="1px",
        body_text_color="#DDE6FF",
        body_text_color_subdued="#5C7399",
        button_primary_background_fill="#4D9EFF",
        button_primary_background_fill_hover="#2D84F0",
        button_primary_text_color="#FFFFFF",
        button_secondary_background_fill="#162038",
        button_secondary_border_color="#1C2E50",
        button_secondary_text_color="#DDE6FF",
        block_shadow="0 4px 24px rgba(0,0,0,.35)",
        block_radius="10px",
        input_radius="8px",
        checkbox_background_color="#162038",
        checkbox_border_color="#1C2E50",
    ),
    css=CSS,
    title="Document Chat",
) as demo:

    lang_state = gr.State("en")
    doc_state  = gr.State("")

    # ── Header ────────────────────────────────────────────────────────────────
    with gr.Row(equal_height=True, elem_id="app-head"):
        with gr.Column(scale=5):
            gr.Markdown("# Document Chat")
            subtitle_md = gr.Markdown(UI["en"]["subtitle"])
        with gr.Column(scale=1, min_width=140, elem_id="lang-toggle"):
            lang_radio = gr.Radio(
                choices=[("🇬🇧 EN", "en"), ("🇫🇷 FR", "fr")],
                value="en",
                label="",
                container=False,
                interactive=True,
            )

    # ── Two-column body ───────────────────────────────────────────────────────
    with gr.Row(equal_height=False):

        # LEFT — document viewer ───────────────────────────────────────────────
        with gr.Column(scale=5, min_width=300, elem_id="viewer-col"):
            gr.HTML(
                '<div id="viewer-hdr">'
                '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>'
                'Document preview'
                '</div>'
            )
            doc_viewer = gr.Gallery(
                label="", show_label=False,
                columns=1, rows=1, height=562,
                object_fit="contain",
                elem_id="doc-viewer",
                show_share_button=False,
                show_download_button=False,
                preview=True,
            )

        # RIGHT — steps ───────────────────────────────────────────────────────
        with gr.Column(scale=6):

            # Step 1 ──────────────────────────────────────────────────────────
            with gr.Group(elem_id="step1"):
                step1_lbl = gr.HTML(
                    f'<div class="step-label"><span class="step-num">1</span>'
                    f'{UI["en"]["step1"]}</div>'
                )
                with gr.Tabs():
                    with gr.Tab(UI["en"]["tab_upload"]):
                        file_input = gr.File(
                            label=UI["en"]["file_label"],
                            file_types=[".pdf", ".png", ".jpg", ".jpeg"],
                        )
                        ocr_btn = gr.Button(
                            UI["en"]["ocr_btn"],
                            variant="primary", elem_classes="go-btn",
                        )
                    with gr.Tab(UI["en"]["tab_paste"]):
                        paste_input = gr.Textbox(
                            lines=5, placeholder=UI["en"]["paste_ph"],
                            label="", show_label=False,
                        )
                        paste_btn = gr.Button(
                            UI["en"]["paste_btn"], variant="secondary",
                        )

                ocr_status = gr.Textbox(
                    value="", label="", interactive=False, elem_id="status-bar"
                )
                with gr.Accordion(UI["en"]["preview_lbl"], open=False):
                    doc_preview = gr.Textbox(
                        lines=5, interactive=False,
                        label="", elem_id="text-preview", show_copy_button=True,
                    )

            # Step 2 ──────────────────────────────────────────────────────────
            with gr.Group(elem_id="step2"):
                step2_lbl = gr.HTML(
                    f'<div class="step-label"><span class="step-num">2</span>'
                    f'{UI["en"]["step2"]}</div>'
                )
                with gr.Tabs():

                    with gr.Tab(UI["en"]["tab_chat"]):
                        instr = gr.Textbox(
                            lines=2,
                            placeholder=UI["en"]["instr_ph"],
                            label=UI["en"]["instr_lbl"],
                        )
                        ex_title = gr.Markdown(f"**{UI['en']['examples']}**")
                        chips = []
                        with gr.Row(elem_classes="chip-row"):
                            for i in range(4):
                                b = gr.Button(EXAMPLES["en"][i], size="sm", elem_classes="chip")
                                chips.append(b)
                        with gr.Row(elem_classes="chip-row"):
                            for i in range(4, 8):
                                b = gr.Button(EXAMPLES["en"][i], size="sm", elem_classes="chip")
                                chips.append(b)
                        chat_btn = gr.Button(
                            UI["en"]["chat_btn"], variant="primary", elem_classes="go-btn"
                        )
                        chat_out = gr.Textbox(
                            lines=10, label=UI["en"]["resp_lbl"],
                            show_copy_button=True, elem_id="chat-out",
                        )
                        chat_btn.click(
                            fn=run_chat,
                            inputs=[doc_state, instr, lang_state],
                            outputs=chat_out,
                        )

                    with gr.Tab(UI["en"]["tab_extract"]):
                        tpl_sel = gr.Dropdown(
                            choices=list(TEMPLATES.keys()),
                            value="📄 Invoice",
                            label=UI["en"]["type_lbl"],
                            elem_id="template-sel",
                        )
                        extract_btn = gr.Button(
                            UI["en"]["extract_btn"], variant="primary", elem_classes="go-btn"
                        )
                        json_out = gr.Textbox(
                            lines=10, label=UI["en"]["json_lbl"],
                            show_copy_button=True, elem_id="json-out",
                        )
                        extract_btn.click(
                            fn=run_extract,
                            inputs=[doc_state, tpl_sel, lang_state],
                            outputs=json_out,
                        )

    gr.HTML(
        '<div id="footer">'
        'OCR · <b>doctr</b> &ensp;LLM · <b>Qwen2.5-1.5B</b> + fallbacks &ensp;GPU · <b>ZeroGPU</b>'
        '&ensp;<a href="https://github.com/mansourkama/document-extraction-pipeline">GitHub ↗</a>'
        '</div>'
    )

    # ── Wiring ────────────────────────────────────────────────────────────────

    file_input.change(fn=get_doc_preview, inputs=file_input, outputs=doc_viewer)

    def _ocr(fp, lang):
        text, status = run_ocr(fp, lang)
        return text, text, status

    def _paste(text, lang):
        t = UI.get(lang, UI["en"])
        if not text or not text.strip():
            return "", "", t["no_paste"]
        return text, text, t["ok_paste"].format(n=len(text))

    def _change_lang(lang):
        t = UI.get(lang, UI["en"])
        examples = EXAMPLES.get(lang, EXAMPLES["en"])
        return (
            lang, t["subtitle"],
            f'<div class="step-label"><span class="step-num">1</span>{t["step1"]}</div>',
            f'<div class="step-label"><span class="step-num">2</span>{t["step2"]}</div>',
            f"**{t['examples']}**",
            *[gr.update(value=ex) for ex in examples],
        )

    ocr_btn.click(fn=_ocr, inputs=[file_input, lang_state], outputs=[doc_state, doc_preview, ocr_status])
    paste_btn.click(fn=_paste, inputs=[paste_input, lang_state], outputs=[doc_state, doc_preview, ocr_status])
    lang_radio.change(
        fn=_change_lang, inputs=lang_radio,
        outputs=[lang_state, subtitle_md, step1_lbl, step2_lbl, ex_title, *chips],
    )
    for i, chip in enumerate(chips):
        chip.click(
            fn=lambda lang, _i=i: EXAMPLES.get(lang, EXAMPLES["en"])[_i],
            inputs=lang_state, outputs=instr,
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", show_api=False)
