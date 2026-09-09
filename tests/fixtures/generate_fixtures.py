"""Generate synthetic test documents using Pillow. Run once to create fixtures."""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent


def make_invoice(path: Path) -> None:
    img = Image.new("RGB", (800, 1000), color="white")
    draw = ImageDraw.Draw(img)

    lines = [
        ("INVOICE", (300, 40), 28),
        ("Invoice Number: INV-2024-0042", (50, 120), 16),
        ("Date: 2024-11-15", (50, 150), 16),
        ("Due Date: 2024-12-15", (50, 180), 16),
        ("", (50, 210), 14),
        ("FROM:", (50, 240), 14),
        ("TechSolutions SARL", (50, 265), 16),
        ("12 Rue de la Paix, 75001 Paris, France", (50, 290), 14),
        ("", (50, 320), 14),
        ("TO:", (50, 350), 14),
        ("GlobalCorp Inc.", (50, 375), 16),
        ("100 Main Street, New York, NY 10001, USA", (50, 400), 14),
        ("", (50, 430), 14),
        ("DESCRIPTION                    QTY    UNIT PRICE    TOTAL", (50, 460), 13),
        ("-" * 75, (50, 480), 13),
        ("Data pipeline development       1      4500.00       4500.00 EUR", (50, 500), 13),
        ("Model fine-tuning (2 days)      2       800.00       1600.00 EUR", (50, 520), 13),
        ("API integration & testing       1       700.00        700.00 EUR", (50, 540), 13),
        ("-" * 75, (50, 560), 13),
        ("Subtotal:                                             6800.00 EUR", (50, 590), 14),
        ("VAT (20%):                                           1360.00 EUR", (50, 615), 14),
        ("TOTAL DUE:                                           8160.00 EUR", (50, 645), 18),
        ("", (50, 680), 14),
        ("Payment method: Bank transfer", (50, 710), 13),
        ("IBAN: FR76 1234 5678 9012 3456 7890 123", (50, 730), 13),
    ]

    for text, pos, size in lines:
        try:
            font = ImageFont.truetype("arial.ttf", size)
        except Exception:
            font = ImageFont.load_default()
        draw.text(pos, text, fill="black", font=font)

    img.save(path)
    print(f"Created: {path}")


def make_form(path: Path) -> None:
    img = Image.new("RGB", (800, 900), color="white")
    draw = ImageDraw.Draw(img)

    lines = [
        ("PERSONAL INFORMATION FORM", (220, 40), 22),
        ("", (50, 80), 14),
        ("Full Name:      Jean-Baptiste MARTIN", (50, 110), 15),
        ("Date of Birth:  15/03/1985", (50, 145), 15),
        ("Nationality:    French", (50, 180), 15),
        ("Email:          jb.martin@example.com", (50, 215), 15),
        ("Phone:          +33 6 12 34 56 78", (50, 250), 15),
        ("Address:        15 Rue de la Republique", (50, 285), 15),
        ("                75001 Paris, France", (50, 310), 15),
        ("Occupation:     Software Engineer", (50, 345), 15),
        ("", (50, 390), 14),
        ("Signature: ___________________________", (50, 430), 14),
        ("Date:      15 / 11 / 2024", (50, 465), 14),
    ]

    for text, pos, size in lines:
        try:
            font = ImageFont.truetype("arial.ttf", size)
        except Exception:
            font = ImageFont.load_default()
        draw.text(pos, text, fill="black", font=font)

    img.save(path)
    print(f"Created: {path}")


if __name__ == "__main__":
    make_invoice(OUTPUT_DIR / "sample_invoice.png")
    make_form(OUTPUT_DIR / "sample_form.png")
    print("All fixtures generated.")
