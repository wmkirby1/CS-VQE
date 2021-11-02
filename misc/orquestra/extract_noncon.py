import utils.cs_vqe_tools as cs_tools
import json
from typing import List, Dict, Any
from zquantum.core.typing import AnyPath
from zquantum.core.utils import save_list


def extract_noncon(ham, search_time=10, criterion='weight'):
    terms_noncon = cs_tools.greedy_dfs(ham, search_time, criterion)[-1]
    save_list(terms_noncon, 'terms_noncon.json')