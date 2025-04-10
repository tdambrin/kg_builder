from typing import List

from kg_builder.builder import KGBuilderFromDocuments
from kg_builder.config import DATA_FOLDER, DATA_STORE_CONF
from kg_builder.parser import DocumentLocator, DocumentParser

# -- Read config --
documents_config: List[DocumentLocator] = [
    DocumentLocator(**conf_object)
    for conf_object in DATA_STORE_CONF["documents"]
]

# -- Load store and parse --
USE_CACHE = True  # False
doc_parser = DocumentParser(document_locators=documents_config)

if USE_CACHE:
    parsed_docs = doc_parser.load_parsed(cache_dir=DATA_FOLDER / "out")
else:
    parsed_docs = doc_parser.parse(
        save=True, override=False, out_directory=DATA_FOLDER / "out"
    )

# Ontology
entities = ["Person", "Article", "Scientific_Topic", "Total_Citations"]
relations = ["AUTHOR_OF", "CITES", "INTERESTED_IN", "HAS"]
potential_schema = [
    (
        "Person",
        "AUTHOR_OF",
        "Article",
    ),
    ("Person", "CITES", "Article"),
    ("Person", "INTERESTED_IN", "Scientific_Topic"),
    ("Person", "HAS", "Total_Citations"),
]

# Build KG

builder: KGBuilderFromDocuments = KGBuilderFromDocuments(
    entities=entities,
    relations=relations,
    potential_schema=potential_schema,
    local=False,
)

result = builder.build(doc_parser.corpus)
print(result)
