import json

def get_ham_data(speciesname):
    file = 'ham_data'
    with open('utils/orquestra/hamiltonians/'+file+'.json', 'r') as json_file:
        ham_data = json.load(json_file)

    ham   = ham_data[speciesname]['ham']
    uccsd = ham_data[speciesname]['uccsd']
    num_qubits = ham_data[speciesname]['num_qubits']

    return speciesname, ham, uccsd, num_qubits