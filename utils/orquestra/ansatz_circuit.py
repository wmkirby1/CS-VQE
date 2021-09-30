import json
import cs_vqe_classes.cs_vqe_circuit as cs_circ
import utils.qonversion_tools as qonvert
from qeqiskit.conversions import import_from_qiskit
from zquantum.core.circuits import save_circuit
from zquantum.core.openfermion import save_qubit_operator
from zquantum.core.utils import save_list


def ansatz_circuit(ham, terms_noncon, anz_op, num_qubits, num_sim_q):

    with open(terms_noncon, 'r') as json_file:
        terms_noncon = (json.load(json_file))['list']

    mol_circ = cs_circ.cs_vqe_circuit(hamiltonian=ham,
                                    terms_noncon=terms_noncon,
                                    num_qubits=num_qubits)

    # output reduced Hamiltonian
    ham_red = (mol_circ.ham_reduced)[num_sim_q]
    ham_red_q = qonvert.dict_to_QubitOperator(ham_red)
    save_qubit_operator(ham_red_q, "ham_red.json")
    
    # output Ansatz circuit
    anz_circ = mol_circ.build_circuit(anz_op, num_sim_q)
    anz_circ_zq = import_from_qiskit(anz_circ)
    save_circuit(anz_circ_zq, "ansatz_circuit.json")

    # output the initial parameter values
    init_params = list(mol_circ.init_params(anz_op, num_sim_q))
    save_list(init_params, 'init_params.json')