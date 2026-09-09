"""OCR engine wrapper using doctr (Mindee).

Reads PDF and image files using two pretrained HuggingFace models:
  - db_resnet50: text detection (locates word bounding boxes)
  - crnn_vgg16_bn: text recognition (reads the words)

No API key or account needed — weights download automatically on first run.
"""
from pathlib import Path
from doctr.io import DocumentFile
from doctr.models import ocr_predictor


class DocumentOCR:
    """OCR engine backed by doctr (HuggingFace-based models).

    Supports PDF and image inputs. Uses db_resnet50 + crnn_vgg16_bn by default,
    both available as free pretrained weights from HuggingFace.
    """

    def __init__(self, det_arch: str = "db_resnet50", reco_arch: str = "crnn_vgg16_bn"):
        self.model = ocr_predictor(
            det_arch=det_arch,
            reco_arch=reco_arch,
            pretrained=True,
        )

    def extract_text(self, file_path: str | Path) -> str:
        """Run OCR on a PDF or image file and return the full text."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path.suffix.lower() == ".pdf":
            doc = DocumentFile.from_pdf(str(path))
        else:
            doc = DocumentFile.from_images(str(path))

        result = self.model(doc)
        return self._to_text(result)

    def _to_text(self, result) -> str:
        lines = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    words = [w.value for w in line.words]
                    lines.append(" ".join(words))
        return "\n".join(lines)
