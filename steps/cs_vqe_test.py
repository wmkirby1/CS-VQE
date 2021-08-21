print('Importing...')
import cs_vqe_classes.cs_vqe_circuit as cs_circ
import utils.cs_vqe_tools as cs_tools
from utils.molecule_tools import get_molecule
print('Finished imports')
def cs_vqe_test(speciesname):
    """
    """
    print('checkpoint 1')
    #construct molecule and related parameters
    molecule = get_molecule(speciesname)
    ham   = molecule['hamiltonian']
    uccsd = molecule['uccsdansatz']
    num_qubits = molecule['num_qubits']
    print('checkpoint 2', molecule['speciesname'])
    #find noncontextual subset and initialise cs_vqe circuit instance
    terms_noncon = cs_tools.greedy_dfs(ham, 3, criterion='weight')[-1]
    mol_circ = cs_circ.cs_vqe_circuit(hamiltonian=ham,
                                    terms_noncon=terms_noncon,
                                    num_qubits=num_qubits,
                                    rot_A=True)
    
    # run CS-VQE
    cs_vqe_results = mol_circ.CS_VQE(uccsd, num_sim_q)
    
    return cs_vqe_results