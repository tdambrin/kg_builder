import asyncio
from typing import Any, List, Tuple

from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

from kg_builder.builder.connectors import (
    get_embedder,
    get_llm,
    get_neo4j_driver,
)


class KGBuilderFromDocuments:
    """
    Simple Knowledge graph builder with a backend of
        - Neo4J server for the graph
        - OpenAI API for NER/etc
    """

    def __init__(
        self,
        entities: List[str],
        relations: List[str],
        potential_schema: List[Tuple[str, str, str]],
        local: bool = True,
    ):
        self._driver = get_neo4j_driver()
        self._llm = get_llm(local=local)
        self._embedder = get_embedder(local=local)

        self.kg_pipeline = SimpleKGPipeline(
            llm=self._llm,
            driver=self._driver,
            embedder=self._embedder,
            entities=entities,
            relations=relations,
            potential_schema=potential_schema,
            on_error="IGNORE",
            from_pdf=False,  # todo: check vs docling
        )

    def __del__(self):
        self._driver.close()

    def build(self, text: str) -> Any:
        result = asyncio.run(self.kg_pipeline.run_async(text=text))
        return result.result
