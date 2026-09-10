"""Gradio web interface — document chat + structured extraction.

Layout: two-column — persistent document viewer left, steps right.
Models load lazily on first GPU request; three fallback LLMs in cascade.
"""
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

# ── Imports ──────────────────────────────────────────────────────────────────
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
        "subtitle": "Upload any document and talk to it — extract, summarize, translate.",
        "step1": "Load document", "step2": "Analyse",
        "tab_upload": "📎 Upload", "tab_paste": "✏️ Paste",
        "file_label": "PDF or image (PNG, JPG, JPEG)",
        "ocr_btn": "Extract text →", "paste_btn": "Use this text →",
        "paste_ph": "Paste document text here…",
        "preview_lbl": "Extracted text",
        "tab_chat": "💬 Chat", "tab_extract": "🗂 Extract JSON",
        "instr_lbl": "Instruction", "instr_ph": 'e.g. "Summarize in 3 bullet points"',
        "examples": "Quick examples", "chat_btn": "Send →",
        "resp_lbl": "Response", "type_lbl": "Document type",
        "extract_btn": "Extract →", "json_lbl": "Extracted fields (JSON)",
        "no_file": "⚠ Please upload a file.",
        "no_text": "⚠ No document text — run OCR first or paste text.",
        "no_instr": "⚠ Please enter an instruction.",
        "no_paste": "⚠ Please enter some text.",
        "ok_ocr": "✓ {n} characters extracted",
        "ok_paste": "✓ {n} characters loaded",
        "err_ocr": "✗ OCR error: {e}",
        "err_chat": "✗ Error: {e}",
    },
    "fr": {
        "subtitle": "Importez un document et dialoguez avec lui — extraire, résumer, traduire.",
        "step1": "Charger le document", "step2": "Analyser",
        "tab_upload": "📎 Importer", "tab_paste": "✏️ Coller",
        "file_label": "PDF ou image (PNG, JPG, JPEG)",
        "ocr_btn": "Extraire le texte →", "paste_btn": "Utiliser ce texte →",
        "paste_ph": "Collez le texte du document ici…",
        "preview_lbl": "Texte extrait",
        "tab_chat": "💬 Chat", "tab_extract": "🗂 Extraction JSON",
        "instr_lbl": "Instruction", "instr_ph": 'ex. "Résume en 3 points clés"',
        "examples": "Exemples rapides", "chat_btn": "Envoyer →",
        "resp_lbl": "Réponse", "type_lbl": "Type de document",
        "extract_btn": "Extraire →", "json_lbl": "Champs extraits (JSON)",
        "no_file": "⚠ Veuillez importer un fichier.",
        "no_text": "⚠ Aucun texte — lancez l'OCR ou collez du texte.",
        "no_instr": "⚠ Veuillez entrer une instruction.",
        "no_paste": "⚠ Veuillez entrer du texte.",
        "ok_ocr": "✓ {n} caractères extraits",
        "ok_paste": "✓ {n} caractères chargés",
        "err_ocr": "✗ Erreur OCR : {e}",
        "err_chat": "✗ Erreur : {e}",
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
                pages.append((bm.to_pil(), f"Page {i+1}/{total}" if total > 1 else ""))
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
                pages.append((img, f"Page {i+1}/{total}" if total > 1 else ""))
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


# ── Design ────────────────────────────────────────────────────────────────────

CSS = """
/* ── Tokens ─────────────────────────────────────────────────── */
:root {
    --bg:       #F0F4FF;
    --surface:  #FFFFFF;
    --surface2: #F5F8FF;
    --border:   #D4DCEE;
    --text:     #0F172A;
    --muted:    #5A6A85;
    --accent:   #2563EB;
    --accent-h: #1D4ED8;
    --accent-t: rgba(37,99,235,.08);
    --ok:       #059669;
    --err:      #DC2626;
    --radius:   12px;
    --shadow:   0 2px 12px rgba(15,23,42,.07), 0 1px 3px rgba(15,23,42,.04);
    --shadow-lg:0 8px 32px rgba(15,23,42,.12), 0 2px 8px rgba(15,23,42,.06);
}
@media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
        --bg:       #0B1120;
        --surface:  #111827;
        --surface2: #1A2540;
        --border:   #243352;
        --text:     #E2E8F5;
        --muted:    #7A90B3;
        --accent:   #60A5FA;
        --accent-h: #3B82F6;
        --accent-t: rgba(96,165,250,.1);
        --ok:       #34D399;
        --err:      #F87171;
        --shadow:   0 2px 12px rgba(0,0,0,.4);
        --shadow-lg:0 8px 32px rgba(0,0,0,.5);
    }
}
:root[data-theme="dark"] {
    --bg:       #0B1120;
    --surface:  #111827;
    --surface2: #1A2540;
    --border:   #243352;
    --text:     #E2E8F5;
    --muted:    #7A90B3;
    --accent:   #60A5FA;
    --accent-h: #3B82F6;
    --accent-t: rgba(96,165,250,.1);
    --ok:       #34D399;
    --err:      #F87171;
    --shadow:   0 2px 12px rgba(0,0,0,.4);
    --shadow-lg:0 8px 32px rgba(0,0,0,.5);
}

/* ── Page ──────────────────────────────────────────────────────── */
body, .gradio-container {
    background: var(--bg) !important;
    color: var(--text) !important;
}

/* ── Step cards ────────────────────────────────────────────────── */
#step1, #step2 {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    box-shadow: var(--shadow) !important;
    padding: 1.25rem !important;
    margin-bottom: .75rem !important;
}

/* ── Step label ────────────────────────────────────────────────── */
.step-label {
    display: flex; align-items: center; gap: .5rem;
    font-weight: 700; font-size: .88rem;
    color: var(--accent); margin-bottom: .85rem;
    letter-spacing: .02em; text-transform: uppercase;
}
.step-num {
    width: 1.6rem; height: 1.6rem; border-radius: 50%;
    background: var(--accent); color: #fff;
    font-size: .72rem; font-weight: 800;
    display: inline-flex; align-items: center; justify-content: center;
    flex-shrink: 0; box-shadow: 0 2px 8px var(--accent-t);
}

/* ── Document viewer panel ─────────────────────────────────────── */
#viewer-panel {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    box-shadow: var(--shadow) !important;
    overflow: hidden !important;
    display: flex; flex-direction: column;
}
#viewer-header {
    padding: .7rem 1rem;
    border-bottom: 1px solid var(--border);
    background: var(--surface2);
    display: flex; align-items: center; gap: .5rem;
    font-size: .78rem; font-weight: 600;
    letter-spacing: .06em; text-transform: uppercase;
    color: var(--muted);
}
#viewer-icon { font-size: 1rem; }

/* Gallery inside the viewer */
#doc-viewer {
    border: none !important;
    border-radius: 0 !important;
    background: #f9f9f9 !important;
    flex: 1;
}
#doc-viewer .thumbnail-item {
    background: #fff !important;
    border-radius: 4px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,.15) !important;
    margin: 4px !important;
}
#doc-viewer .thumbnail-item img { object-fit: contain !important; }

/* Empty state hint */
#viewer-hint {
    text-align: center; padding: 3rem 2rem;
    color: var(--muted); font-size: .88rem;
    display: flex; flex-direction: column; align-items: center; gap: .75rem;
}
#viewer-hint svg { opacity: .3; }

/* ── Language selector ─────────────────────────────────────────── */
#lang-box {
    display: flex; justify-content: flex-end; align-items: center; gap: .4rem;
}
#lang-box label { display: none !important; }
#lang-box select, #lang-box .wrap {
    font-size: .8rem !important;
    padding: .3rem .6rem !important;
    border-radius: 20px !important;
    border: 1px solid var(--border) !important;
    background: var(--surface) !important;
    color: var(--text) !important;
    min-width: 130px !important;
    max-width: 145px !important;
    box-shadow: none !important;
    cursor: pointer;
}

/* ── Buttons ───────────────────────────────────────────────────── */
.send-btn button, .send-btn button:focus {
    background: var(--accent) !important;
    color: #fff !important; border: none !important;
    font-weight: 600 !important; letter-spacing: .01em !important;
    box-shadow: 0 2px 8px var(--accent-t) !important;
    transition: background .15s, box-shadow .15s !important;
}
.send-btn button:hover {
    background: var(--accent-h) !important;
    box-shadow: 0 4px 16px var(--accent-t) !important;
}

/* ── Example chips ─────────────────────────────────────────────── */
.chip-row { display: flex; flex-wrap: wrap; gap: .3rem; margin: .4rem 0 .65rem; }
.chip button {
    font-size: .73rem !important; padding: .22rem .65rem !important;
    border-radius: 9999px !important; line-height: 1.4 !important;
    border: 1px solid var(--border) !important;
    background: var(--surface2) !important; color: var(--muted) !important;
    transition: all .12s !important;
}
.chip button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: var(--accent-t) !important;
}

/* ── Status bar ────────────────────────────────────────────────── */
#status-bar { margin-top: .3rem; }
#status-bar label { display: none !important; }
#status-bar textarea {
    font-size: .8rem !important; color: var(--muted) !important;
    min-height: 1.3rem !important; max-height: 1.8rem !important;
    resize: none !important; border: none !important;
    background: transparent !important; box-shadow: none !important;
    padding: 0 !important;
}

/* ── Extracted text accordion ──────────────────────────────────── */
#text-preview textarea {
    font-family: 'Menlo', 'Consolas', monospace;
    font-size: .78rem !important; color: var(--muted) !important;
    background: var(--surface2) !important;
}

/* ── Output areas ──────────────────────────────────────────────── */
#chat-out textarea, #json-out textarea {
    font-size: .86rem !important;
    background: var(--surface2) !important;
}

/* ── Title area ────────────────────────────────────────────────── */
#app-title h1 {
    font-size: 1.55rem !important; font-weight: 800 !important;
    color: var(--text) !important; margin: 0 !important;
    letter-spacing: -.02em;
}
#app-sub p { color: var(--muted) !important; font-size: .88rem !important; margin: .2rem 0 0 !important; }

/* ── Footer ────────────────────────────────────────────────────── */
#footer {
    text-align: center; font-size: .74rem; color: var(--muted);
    margin-top: .75rem; padding: .5rem;
    border-top: 1px solid var(--border);
}
#footer a { color: var(--accent); text-decoration: none; }

/* ── Misc cleanup ──────────────────────────────────────────────── */
.gr-prose p { color: var(--text) !important; }
"""

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="blue", neutral_hue="slate"),
    css=CSS,
    title="Document Chat",
) as demo:

    lang_state = gr.State("en")
    doc_state  = gr.State("")

    # ── Header ────────────────────────────────────────────────────────────────
    with gr.Row(equal_height=True):
        with gr.Column(scale=6):
            gr.Markdown("# Document Chat", elem_id="app-title")
            subtitle_md = gr.Markdown(UI["en"]["subtitle"], elem_id="app-sub")
        with gr.Column(scale=1, min_width=150, elem_id="lang-box"):
            lang_sel = gr.Dropdown(
                choices=[("🇬🇧 English", "en"), ("🇫🇷 Français", "fr")],
                value="en", label="", container=False, interactive=True,
            )

    # ── Two-column body ───────────────────────────────────────────────────────
    with gr.Row(equal_height=False):

        # LEFT — document viewer ───────────────────────────────────────────────
        with gr.Column(scale=5, min_width=300, elem_id="viewer-panel"):
            gr.HTML(
                '<div id="viewer-header">'
                '<span id="viewer-icon">📄</span>'
                'Document preview'
                '</div>'
            )
            doc_viewer = gr.Gallery(
                label="", show_label=False,
                columns=1, rows=1, height=560,
                object_fit="contain",
                elem_id="doc-viewer",
                show_share_button=False,
                show_download_button=False,
                preview=True,
            )

        # RIGHT — steps ───────────────────────────────────────────────────────
        with gr.Column(scale=6):

            # ── Step 1 ────────────────────────────────────────────────────────
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
                            UI["en"]["ocr_btn"], variant="primary",
                            elem_classes="send-btn",
                        )
                    with gr.Tab(UI["en"]["tab_paste"]):
                        paste_input = gr.Textbox(
                            lines=5, placeholder=UI["en"]["paste_ph"],
                            label="", show_label=False,
                        )
                        paste_btn = gr.Button(UI["en"]["paste_btn"], variant="secondary")

                ocr_status = gr.Textbox(
                    value="", label="", interactive=False, elem_id="status-bar"
                )
                with gr.Accordion(UI["en"]["preview_lbl"], open=False):
                    doc_preview = gr.Textbox(
                        lines=5, interactive=False,
                        label="", elem_id="text-preview", show_copy_button=True,
                    )

            # ── Step 2 ────────────────────────────────────────────────────────
            with gr.Group(elem_id="step2"):
                step2_lbl = gr.HTML(
                    f'<div class="step-label"><span class="step-num">2</span>'
                    f'{UI["en"]["step2"]}</div>'
                )
                with gr.Tabs():

                    # Chat ───────────────────────────────────────────────────
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
                            UI["en"]["chat_btn"], variant="primary", elem_classes="send-btn"
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

                    # Extract JSON ────────────────────────────────────────────
                    with gr.Tab(UI["en"]["tab_extract"]):
                        tpl_sel = gr.Dropdown(
                            choices=list(TEMPLATES.keys()),
                            value="📄 Invoice",
                            label=UI["en"]["type_lbl"],
                        )
                        extract_btn = gr.Button(
                            UI["en"]["extract_btn"], variant="primary", elem_classes="send-btn"
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
        'OCR: <b>doctr</b> &nbsp;·&nbsp; LLM: <b>Qwen2.5-1.5B</b> + fallbacks'
        '&nbsp;·&nbsp; ZeroGPU &nbsp;·&nbsp;'
        '<a href="https://github.com/mansourkama/document-extraction-pipeline">GitHub</a>'
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
    lang_sel.change(
        fn=_change_lang, inputs=lang_sel,
        outputs=[lang_state, subtitle_md, step1_lbl, step2_lbl, ex_title, *chips],
    )
    for i, chip in enumerate(chips):
        chip.click(
            fn=lambda lang, _i=i: EXAMPLES.get(lang, EXAMPLES["en"])[_i],
            inputs=lang_state, outputs=instr,
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", show_api=False)
