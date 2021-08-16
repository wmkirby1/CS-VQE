print('Importing...')
import cs_vqe_classes.cs_vqe_circuit as cs_circ
import utils.cs_vqe_tools as cs_tools
import utils.molecule_tools as mol
print('Finished imports')
def cs_vqe_test(bond_len, multiplicity, charge, basis, rot_A, num_sim_q, atom1=None, atom2=None, atom3=None):
    """
    """
    print('checkpoint 1')
    #construct molecule and related parameters
    atoms = [a for a in [atom1, atom2, atom3] if a is not None]
    molecule = mol.construct_molecule(atoms, bond_len, multiplicity, charge, basis)
    ham   = molecule['hamiltonian']
    uccsd = molecule['uccsdansatz']
    num_qubits = molecule['num_qubits']
    num_electrons = molecule['num_electrons']
    print('checkpoint 2', molecule['speciesname'])
    #find noncontextual subset and initialise cs_vqe circuit instance
    terms_noncon = cs_tools.greedy_dfs(ham, 3, criterion='weight')[-1]
    mol_circ = cs_circ.cs_vqe_circuit(hamiltonian=ham,
                                    terms_noncon=terms_noncon,
                                    num_qubits=num_qubits,
                                    num_electrons=num_electrons, 
                                    rot_A=rot_A)
    
    # run CS-VQE
    cs_vqe_results = mol_circ.CS_VQE(uccsd, num_sim_q)
    
    return cs_vqe_results
