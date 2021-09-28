import utils.cs_vqe_tools as cs_tools
from utils.json_tools import *

def extract_noncon(ham, search_time=10, criterion='weight'):
    terms_noncon = cs_tools.greedy_dfs(ham, search_time, criterion)[-1]
    data_dict = {}
    data_dict['noncon'] = terms_noncon
    print(data_dict)
    save_json(data_dict, 'noncon.json')