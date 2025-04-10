# Knowledge Graph Builder
## From multi format documents with target ontology

![Python](https://img.shields.io/badge/python-3.13-blue)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)


This repo sits on top of a few libraries and third parties to build a simple, yet to be improved, framework to build a Neo4j knowledge graph from a data store of multi-format documents.

## Dependencies
- Docling: Document Converter to translate everything (PDF, CSV, docx, images, etc) into markdown
- Neo4j server: To create/amend the knowledge graph
- neo4j-graphrag[openai]: for LLM-based NER/Relation extractions
- neo4j-graphrag: for Graph RAG QA
