from neo4j import Driver, GraphDatabase
from neo4j_graphrag.embeddings import (
    Embedder,
    OllamaEmbeddings,
    OpenAIEmbeddings,
)
from neo4j_graphrag.llm import LLMInterface, OllamaLLM, OpenAILLM

from kg_builder.config import (
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USERNAME,
    OPENAI_API_KEY,
)


def get_neo4j_driver() -> Driver:
    return GraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )


def get_embedder(local: bool = True) -> Embedder:
    if local:
        return OllamaEmbeddings(model="zephyr")
    return OpenAIEmbeddings(
        model="text-embedding-3-large", api_key=OPENAI_API_KEY
    )


def get_llm(local: bool = True) -> LLMInterface:
    if local:
        return OllamaLLM(
            model_name="zephyr",
            model_params={
                "max_tokens": 2000,
                "response_format": {"type": "json_object"},
                "temperature": 0,
            },
        )
    return OpenAILLM(
        model_name="gpt-4o",
        model_params={
            "max_tokens": 2000,
            "response_format": {"type": "json_object"},
            "temperature": 0,
        },
        api_key=OPENAI_API_KEY,
    )
