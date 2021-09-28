import utils.cs_vqe_tools as cs_tools
from zquantum.core import save_list

def extract_noncon(ham, search_time=10, criterion='weight'):
    terms_noncon = cs_tools.greedy_dfs(ham, search_time, criterion)[-1]
    save_list(terms_noncon, 'terms_noncon.json')