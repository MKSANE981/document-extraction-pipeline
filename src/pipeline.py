"""Main pipeline: document → OCR text → structured Pydantic model.

Combines DocumentOCR (src/ocr.py) and StructuredExtractor (src/extractor.py)
into a single two-step call. The template controls which fields are extracted —
swap it out to handle any document type without changing the pipeline itself.

Two entry points:
  process(file_path, template)      — full pipeline (OCR + extraction)
  process_text(text, template)      — extraction only (skips OCR; fast for tests)
"""
from pathlib import Path
from typing import Type, TypeVar
from .ocr import DocumentOCR
from .extractor import StructuredExtractor
from .templates.base import ExtractionTemplate

T = TypeVar("T", bound=ExtractionTemplate)


class DocumentPipeline:
    """End-to-end pipeline: document file → OCR text → structured extraction.

    Usage:
        pipeline = DocumentPipeline()
        result = pipeline.process("invoice.pdf", InvoiceTemplate)
        print(result.model_dump())
    """

    def __init__(
        self,
        ocr_det_arch: str = "db_resnet50",
        ocr_reco_arch: str = "crnn_vgg16_bn",
        llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    ):
        self.ocr = DocumentOCR(det_arch=ocr_det_arch, reco_arch=ocr_reco_arch)
        self.extractor = StructuredExtractor(model_name=llm_model)

    def process(self, file_path: str | Path, template: Type[T]) -> T:
        """Process a document and return a populated template instance."""
        text = self.ocr.extract_text(file_path)
        return self.extractor.extract(text, template)

    def process_text(self, text: str, template: Type[T]) -> T:
        """Skip OCR and extract directly from pre-existing text (useful for testing)."""
        return self.extractor.extract(text, template)
