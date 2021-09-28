import cs_vqe_classes.cs_vqe_circuit as cs_circ
from qusetta import Qiskit
from zquantum.core.circuits import (save_circuit,import_from_cirq)

def ansatz_circuit(ham, terms_noncon, anz_op, num_qubits, num_sim_q):
    print(terms_noncon)
    mol_circ = cs_circ.cs_vqe_circuit(hamiltonian=ham,
                                    terms_noncon=terms_noncon,
                                    num_qubits=num_qubits)

    anz_circ = mol_circ.build_circuit(anz_op, num_sim_q)
    anz_circ_cirq = Qiskit.to_cirq(anz_circ)
    anz_circ_zquantum = import_from_cirq(anz_circ_cirq) 
    save_circuit(anz_circ_zquantum, "circuit.json")