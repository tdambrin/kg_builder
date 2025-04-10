from functools import cache
from pathlib import Path
from typing import Dict, List

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PipelineOptions,
    VlmPipelineOptions,
    smoldocling_vlm_mlx_conversion_options,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling_core.types import DoclingDocument

from kg_builder.config import DATA_FOLDER
from kg_builder.parser.models import DocumentLocator


class DocumentParser:
    """
    Simple document parser expecting list of DocumentLocator that can be different file-formats
    """  # noqa: E501

    def __init__(
        self, document_locators: DocumentLocator | List[DocumentLocator]
    ):
        self.document_locators = (
            document_locators
            if isinstance(document_locators, list)
            else [document_locators]
        )
        self.converted_documents: Dict[str, DoclingDocument] = {}

    @property
    @cache
    def pipeline_options(self) -> PipelineOptions:
        """
        Parsing options
        """
        # Use experimental VlmPipeline
        pipeline_options = VlmPipelineOptions()

        # If True, text from backend is used instead of generated text
        pipeline_options.force_backend_text = False

        # for Apple Silicon
        pipeline_options.vlm_options = smoldocling_vlm_mlx_conversion_options
        return pipeline_options

    @property
    def document_converter(self) -> DocumentConverter:
        """
        Docling converter
        """
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=self.pipeline_options,
                ),
                InputFormat.IMAGE: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=self.pipeline_options,
                ),
            }
        )

    @property
    def corpus(self) -> str:
        """
        Markdown export of all parsed documents
        :return:
        """
        return "\n".join(
            doc.export_to_markdown()
            for doc in self.converted_documents.values()
        )

    def parse(
        self,
        save: bool = True,
        override: bool = True,
        out_directory: Path | str = DATA_FOLDER / "out",
    ) -> Dict[str, DoclingDocument]:
        """
        Parse all document locators

        Args:
            save (bool): save the result to filesystem
            override (bool): override existing results
            out_directory (Path): save location

        Returns:

        """
        if not self.document_locators:
            raise ValueError("No document locator provided")

        converter = self.document_converter
        local_converted_docs = {}
        for d_locator in self.document_locators:
            converted = converter.convert(d_locator.uri)
            local_converted_docs.update({d_locator.name: converted.document})

        self.converted_documents.update(local_converted_docs)

        if save:
            self.save_converted(
                local_converted_docs,
                override=override,
                out_directory=out_directory,
            )

        return local_converted_docs

    def load_parsed(self, cache_dir: Path):
        self.converted_documents = {
            doc.name: self.document_converter.convert(
                cache_dir / f"{doc.name}.md"
            ).document
            for doc in self.document_locators
        }

    @staticmethod
    def save_converted(
        documents: Dict[str, DoclingDocument],
        override: bool = True,
        out_directory: Path | str = DATA_FOLDER / "out",
    ):
        """
        Export list of docs to markdown on filesystem

        Args:
            documents ([DoclingDocument]): converted documents
            override (bool): override existing results
            out_directory (Path): save location

        Returns:

        """
        out_directory = (
            out_directory
            if isinstance(out_directory, Path)
            else Path(out_directory)
        )
        for id_, document in documents.items():
            if not override and (out_directory / f"{id_}.md").exists():
                continue
            document.save_as_markdown(filename=out_directory / f"{id_}.md")
