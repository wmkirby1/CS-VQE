import utils.cs_vqe_tools as cs_tools
import json
from typing import List, Dict, Any
from zquantum.core.typing import AnyPath

SCHEMA_VERSION = "zapata-v1"


def save_list(array: List, filename: AnyPath, artifact_name: str = ""):
    """Save expectation values to a file.
    Args:
        array (list): the list to be saved
        file (str or file-like object): the name of the file, or a file-like object
        artifact_name (str): optional argument to specify the schema name
    """
    dictionary: Dict[str, Any] = {}
    dictionary["schema"] = SCHEMA_VERSION + "-" + artifact_name + "-list"
    dictionary["list"] = array

    with open(filename, "w") as f:
        f.write(json.dumps(dictionary, indent=2))


def extract_noncon(ham, search_time=10, criterion='weight'):
    terms_noncon = cs_tools.greedy_dfs(ham, search_time, criterion)[-1]
    save_list(terms_noncon, 'terms_noncon.json')