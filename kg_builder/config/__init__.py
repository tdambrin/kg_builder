from pathlib import Path
from typing import Any, Dict, List

import yaml

PKG_ROOT = Path(__file__).parent.parent
HERE = Path(__file__).parent
DATA_FOLDER = PKG_ROOT / "data"

GLOBAL_CONF_PATH = HERE / "conf.yml"
if not GLOBAL_CONF_PATH.exists():
    raise FileNotFoundError(
        f"You need to add a {GLOBAL_CONF_PATH.absolute()} file with Neo4J and OpenAI credentials."
    )

with open(HERE / "conf.yml", "r", encoding="utf-8") as f:
    _conf: Dict[str, Any] = yaml.safe_load(f)

# Neo4J driver
NEO4J_URI = _conf["NEO4J_URI"]
NEO4J_USERNAME = _conf["NEO4J_USERNAME"]
NEO4J_PASSWORD = _conf["NEO4J_PASSWORD"]

# Open AI API
OPENAI_API_KEY = _conf["OPENAI_API_KEY"]

with open(HERE / "data_store.yml", "r", encoding="utf-8") as f:
    DATA_STORE_CONF: Dict = yaml.safe_load(f)
