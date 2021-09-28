import cs_vqe_classes.cs_vqe_circuit as cs_circ
from zquantum.core.circuits import (
    Circuit,
    load_circuit,
    load_circuitset,
    save_circuit,
    save_circuitset,
    import_from_qiskit
)

def ansatz_circuit(ham, terms_noncon, anz_op, num_qubits, num_sim_q):
    
    mol_circ = cs_circ.cs_vqe_circuit(hamiltonian=ham,
                                    terms_noncon=terms_noncon,
                                    num_qubits=num_qubits)

    anz_circ = mol_circ.build_circuit(anz_op, num_sim_q)
    anz_circ_zquantum = import_from_qiskit(anz_circ) 
    save_circuit(anz_circ_zquantum, "circuit.json")

    return anz_circ_zquantum