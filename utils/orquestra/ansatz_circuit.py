import cs_vqe_classes.cs_vqe_circuit as cs_circ

def ansatz_circuit(ham, terms_noncon, anz_op, num_qubits, num_sim_q):
    
    mol_circ = cs_circ.cs_vqe_circuit(hamiltonian=ham,
                                    terms_noncon=terms_noncon,
                                    num_qubits=num_qubits)

    anz_circ = mol_circ.build_circuit(anz_op, num_sim_q)

    return anz_circ