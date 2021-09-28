import utils.cs_vqe_tools as cs_tools

def extract_noncon(ham, search_time=10, criterion='weight'):
    terms_noncon = cs_tools.greedy_dfs(ham, search_time, criterion)[-1]
    return terms_noncon