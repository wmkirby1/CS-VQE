import json
import cs_vqe_classes.cs_vqe_circuit as cs_circ
from qeqiskit.conversions import import_from_qiskit
from zquantum.core.circuits import save_circuit

def ansatz_circuit(ham, terms_noncon, anz_op, num_qubits, num_sim_q):

    with open(terms_noncon, 'r') as json_file:
        terms_noncon = (json.load(json_file))['list']

    mol_circ = cs_circ.cs_vqe_circuit(hamiltonian=ham,
                                    terms_noncon=terms_noncon,
                                    num_qubits=num_qubits)

    anz_circ = mol_circ.build_circuit(anz_op, num_sim_q)
    anz_circ_zq = import_from_qiskit(anz_circ)
    save_circuit(anz_circ_zq, "ansatz_circuit.json")