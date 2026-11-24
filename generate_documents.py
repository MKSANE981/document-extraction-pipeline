"""Generate synthetic French insurance documents (PDF) with ground-truth labels.

Everything is fictional: people, insurers, policy numbers and addresses are invented.
Layouts, wording and formats vary from one document to the next so that the extraction
step cannot rely on a single template.

Usage:
    python generate_documents.py --n 60 --seed 7 --out data/documents
"""
import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas

FIRST = ["Camille", "Mamadou", "Julie", "Antoine", "Fatou", "Nicolas", "Awa", "Pierre", "Sarah",
         "Ibrahima", "Claire", "Thomas", "Aminata", "Laurent", "Marie", "Moussa", "Elodie", "Hugo"]
LAST = ["Martin", "Diop", "Bernard", "Ndiaye", "Dubois", "Sow", "Lefebvre", "Fall", "Moreau",
        "Gueye", "Girard", "Sarr", "Roux", "Ba", "Fournier", "Thiam", "Lambert", "Cisse"]
STREETS = ["rue des Lilas", "avenue de la Gare", "impasse du Moulin", "rue Victor Hugo",
           "boulevard des Acacias", "chemin des Vignes", "place de la Mairie"]
CITIES = [("35000", "Rennes"), ("44000", "Nantes"), ("29200", "Brest"), ("56100", "Lorient"),
          ("22000", "Saint-Brieuc"), ("49000", "Angers")]
INSURERS = ["Assurances Valmont", "Mutuelle Horizon Ouest", "Cap Sud Assurances"]
MONTHS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
          "septembre", "octobre", "novembre", "décembre"]
DOC_TYPES = ["avis_echeance", "attestation_auto", "quittance_prime"]


def fmt_date(d, style):
    if style == 0:
        return d.strftime("%d/%m/%Y")
    if style == 1:
        return f"{d.day} {MONTHS[d.month - 1]} {d.year}"
    return d.strftime("%d.%m.%Y")


def fmt_amount(x, style):
    s = f"{x:,.2f}".replace(",", " ").replace(".", ",")
    return [f"{s} €", f"{s} EUR", f"{s} euros"][style]


def make_record(rng, doc_type):
    first, last = rng.choice(FIRST), rng.choice(LAST)
    start = date(2026, 1, 1) + timedelta(days=rng.randint(0, 330))
    end = start + timedelta(days=364)
    return {
        "doc_type": doc_type,
        "insured_name": f"{first} {last.upper()}",
        "insurer": rng.choice(INSURERS),
        "policy_number": f"{rng.choice(['AU', 'MR', 'HB'])}{rng.randint(10, 99)}-{rng.randint(100000, 999999)}",
        "issue_date": (start - timedelta(days=rng.randint(5, 40))).isoformat(),
        "period_start": start.isoformat(),
        "period_end": end.isoformat(),
        "amount_eur": round(rng.uniform(48, 1480), 2) if doc_type != "attestation_auto" else None,
        "_address": f"{rng.randint(1, 98)} {rng.choice(STREETS)}, {'%s %s' % rng.choice(CITIES)}",
    }


LABELS = {
    "policy": ["N° de contrat", "Contrat n°", "Référence police", "Votre numéro de contrat"],
    "insured": ["Assuré(e)", "Souscripteur", "Titulaire du contrat", "Nom de l'assuré"],
    "amount": ["Montant TTC", "Total à régler", "Cotisation annuelle TTC", "Prime TTC"],
    "start": ["Début de période", "Prend effet le", "Période du", "Date d'effet"],
    "end": ["Fin de période", "Échéance au", "au", "Date d'échéance"],
    "issued": ["Fait le", "Émis le", "Date d'émission", "Édité le"],
}

INTRO = {
    "avis_echeance": [
        "Madame, Monsieur,\nVotre contrat arrive prochainement à échéance. Nous vous remercions de votre fidélité et vous "
        "invitons à régler votre cotisation avant la date indiquée ci-dessous.",
        "Bonjour,\nVoici l'avis d'échéance de votre contrat. Si vous avez déjà réglé cette somme, merci de ne pas tenir "
        "compte de ce courrier.",
    ],
    "attestation_auto": [
        "Nous soussignés, attestons que le contrat désigné ci-dessous garantit la responsabilité civile du véhicule "
        "de l'assuré, conformément aux articles L211-1 et suivants du Code des assurances.",
        "Cette attestation est délivrée pour servir et valoir ce que de droit. Elle ne constitue pas une preuve de "
        "paiement de la cotisation.",
    ],
    "quittance_prime": [
        "Nous reconnaissons avoir reçu de la part de l'assuré désigné ci-dessous la somme correspondant à la prime "
        "afférente à la période indiquée.",
        "Quittance de paiement. Cette somme a bien été encaissée ; ce document tient lieu de justificatif.",
    ],
}
TITLES = {"avis_echeance": "AVIS D'ÉCHÉANCE", "attestation_auto": "ATTESTATION D'ASSURANCE AUTOMOBILE",
          "quittance_prime": "QUITTANCE DE PRIME"}


def wrap(c, text, x, y, width, leading=14, font=("Helvetica", 10)):
    c.setFont(*font)
    for para in text.split("\n"):
        line = ""
        for word in para.split():
            test = (line + " " + word).strip()
            if c.stringWidth(test, *font) > width:
                c.drawString(x, y, line)
                y -= leading
                line = word
            else:
                line = test
        c.drawString(x, y, line)
        y -= leading
    return y


SHIFTED = {
    "policy": ["Référence du contrat", "Dossier n°", "Police d'assurance n°"],
    "insured": ["Bénéficiaire", "Client", "Assuré principal"],
    "amount": ["Somme due", "Coût annuel", "Montant à payer"],
    "start": ["À compter du", "Valable à partir du", "Début de garantie"],
    "end": ["Jusqu'au", "Valable jusqu'au", "Fin de garantie"],
    "issued": ["Établi le", "Daté du", "Émission"],
}


def render(rec, rng, path, shift=False):
    layout = rng.randint(0, 2)       # 0: label/value lines, 1: table-like, 2: sentence form
    dstyle, astyle = rng.randint(0, 2), rng.randint(0, 2)
    lab = {k: rng.choice(v) for k, v in (SHIFTED if shift else LABELS).items()}
    c = canvas.Canvas(str(path), pagesize=A4)
    W, H = A4
    c.setFont("Helvetica-Bold", 13)
    c.drawString(2 * cm, H - 2.2 * cm, rec["insurer"])
    c.setFont("Helvetica", 8)
    c.drawString(2 * cm, H - 2.7 * cm, "Service clients, 12 quai de la Fosse, 44000 Nantes. Tél. 02 40 00 00 00")
    c.setFont("Helvetica", 10)
    c.drawString(12 * cm, H - 4 * cm, rec["insured_name"].title().replace(" ", " ", 1))
    c.drawString(12 * cm, H - 4.5 * cm, rec["_address"])
    c.setFont("Helvetica-Bold", 14)
    c.drawString(2 * cm, H - 6.2 * cm, TITLES[rec["doc_type"]])
    y = wrap(c, rng.choice(INTRO[rec["doc_type"]]), 2 * cm, H - 7.2 * cm, 16 * cm)
    y -= 12
    d = lambda iso: fmt_date(date.fromisoformat(iso), dstyle)
    amount = fmt_amount(rec["amount_eur"], astyle) if rec["amount_eur"] is not None else None
    pairs = [(lab["policy"], rec["policy_number"]), (lab["insured"], rec["insured_name"].title()),
             (lab["start"], d(rec["period_start"])), (lab["end"], d(rec["period_end"]))]
    if amount:
        pairs.append((lab["amount"], amount))
    if layout == 0:
        for k, v in pairs:
            c.setFont("Helvetica-Bold", 10); c.drawString(2 * cm, y, f"{k} :")
            c.setFont("Helvetica", 10); c.drawString(7.5 * cm, y, v); y -= 18
        y -= 8
        c.drawString(2 * cm, y, f"{lab['issued']} {d(rec['issue_date'])}")
    elif layout == 1:
        c.setFont("Helvetica-Bold", 9); c.drawString(2 * cm, y, "Désignation"); c.drawString(9 * cm, y, "Détail")
        c.line(2 * cm, y - 3, 18 * cm, y - 3); y -= 18
        c.setFont("Helvetica", 10)
        for k, v in pairs:
            c.drawString(2 * cm, y, k); c.drawString(9 * cm, y, v); y -= 16
        y -= 14
        c.drawString(2 * cm, y, f"{lab['issued']} {d(rec['issue_date'])}, à {rec['_address'].split(', ')[1].split(' ', 1)[1]}.")
    else:
        text = (f"{lab['insured']} : {rec['insured_name'].title()}. {lab['policy']} {rec['policy_number']}. "
                f"{lab['start']} {d(rec['period_start'])}, {lab['end']} {d(rec['period_end'])}.")
        if amount:
            text += f" {lab['amount']} : {amount}."
        text += f" {lab['issued']} {d(rec['issue_date'])}."
        y = wrap(c, text, 2 * cm, y, 16 * cm, leading=16, font=("Helvetica", 11))
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(2 * cm, 2 * cm, "Document généré à titre de démonstration. Toutes les données sont fictives.")
    c.save()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/documents")
    ap.add_argument("--shift", action="store_true", help="use field wordings the extractor was not written for")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    labels = {}
    for i in range(a.n):
        rec = make_record(rng, DOC_TYPES[i % 3])
        name = f"doc_{i:03d}.pdf"
        render(rec, rng, out / name, a.shift)
        labels[name] = {k: v for k, v in rec.items() if not k.startswith("_")}
    (out.parent / ("labels_shifted.json" if a.shift else "labels.json")).write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{a.n} documents written to {out} and ground truth to {out.parent / ('labels_shifted.json' if a.shift else 'labels.json')}")


if __name__ == "__main__":
    main()
