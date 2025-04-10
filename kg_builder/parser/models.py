from dataclasses import dataclass


@dataclass
class DocumentLocator:
    name: str
    uri: str
