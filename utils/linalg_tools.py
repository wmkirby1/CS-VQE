import utils.qonversion_tools as qonvert
import utils.bit_tools as bit
#import cs_vqe.circuit as cs_circ
#from openfermion.ops import QubitOperator
#from openfermion.linalg import LinearQubitOperator, get_sparse_operator, get_ground_state
import numpy as np
import scipy
import math


def get_ground_state(sparse_operator, initial_guess=None):
    """Compute lowest eigenvalue and eigenstate.
    Args:
        sparse_operator (LinearOperator): Operator to find the ground state of.
        initial_guess (ndarray): Initial guess for ground state.  A good
            guess dramatically reduces the cost required to converge.
    Returns
    -------
        eigenvalue:
            The lowest eigenvalue, a float.
        eigenstate:
            The lowest eigenstate in scipy.sparse csc format.
    """
    values, vectors = scipy.sparse.linalg.eigsh(sparse_operator,
                                                k=1,
                                                v0=initial_guess,
                                                which='SA',
                                                maxiter=1e7)

    order = np.argsort(values)
    values = values[order]
    vectors = vectors[:, order]
    eigenvalue = values[0]
    eigenstate = vectors[:, 0]
    return eigenvalue, eigenstate.T


def factor_int(n):
    """Finds factorisation of n closest to a square (for optimal plot layout)
    """
    val = math.ceil(math.sqrt(n))
    val2 = int(n/val)
    while val2 * val != float(n):
        val -= 1
        val2 = int(n/val)
    # order the factors
    if val > val2:
        val, val2 = val2, val

    return val, val2


def random_vector(n):
    """
    """
    components = [np.random.normal() for i in range(n)]
    r = math.sqrt(sum(x*x for x in components))
    v = [x/r for x in components]
    return v


def random_complex_unit():
    """
    """
    rand_vec = random_vector(2)
    x = rand_vec[0]
    y = rand_vec[1]
    
    return x + y*1j


def random_complex_vector(n, order=False):  
    """
    """
    return [random_complex_unit()*a for a in random_vector(n)]


def expectation(op, state, num_qubits):
    """
    """
    assert(type(op) == dict)
    #if type(op) == dict:
    #    op = qonvert.dict_to_QubitOperator(op)
    
    state = np.array(state)
    #O = LinearQubitOperator(op, num_qubits)
    O_mat = qonvert.dict_to_WeightedPauliOperator(op).to_matrix()
    expect = np.conjugate(state).dot(O_mat).dot(state)
    
    return expect.real


def eigenstate_projector(A, num_qubits, eval=+1):
    """
    """
    I_op = np.identity(2**num_qubits)
    A_op = qonvert.dict_to_WeightedPauliOperator(A).to_matrix()
    projector = (I_op+eval*A_op)/2
    
    return np.matrix(projector)


def noncon_projector(nc_state, sim_indices, num_qubits):
    """
    """
    projector = 1

    for i in range(num_qubits):
        if i in sim_indices:
            tensor_factor = np.identity(2)
        else:
            nc_index = int(nc_state[i])
            basis_state = np.zeros(2)
            basis_state[nc_index] = 1
            tensor_factor = np.outer(basis_state, basis_state)
        projector = np.kron(projector, tensor_factor)
    
    return np.matrix(projector)


def pauli_matrix(pauli):
    num_qubits = len(pauli)
    single_paulis ={'I': np.matrix(np.identity(num_qubits)),
                    'X': np.matrix([[0, 1],
                                    [1, 0]]),
                    'Y': np.matrix([[0,-1.j],
                                    [1.j, 0]]),
                    'Z': np.matrix([[1, 0],
                                    [0,-1]])}
    
    pauli_matrix = 1
    for p in pauli:
        pauli_matrix = np.kron(pauli_matrix, single_paulis[p])

    return pauli_matrix


def exp_pauli(pauli, param):
    num_qubits = len(pauli)
    I_mat = np.matrix(np.identity(2**num_qubits))
    p_mat = pauli_matrix(pauli)

    return np.cos(param)*I_mat + 1.j*np.sin(param)*p_mat


def apply_projections(psi, proj_list=[]):
    """Applied left to right
    """
    # apply projections
    for p in proj_list:
        psi = np.dot(p, psi)
    # renormalise
    psi_conj = np.conjugate(psi)
    norm = np.sqrt(np.dot(psi_conj, psi))
    psi = psi/norm

    return psi


def project_hamiltonian(hamiltonian, terms_noncon, num_qubits):
    mol = cs_circ.cs_vqe_circuit(hamiltonian, terms_noncon, num_qubits)
    A = mol.A
    qubit_nums = range(1, num_qubits+1)
    
    gs_true = []
    gs_proj = []

    for n_q in qubit_nums:
        ham_red = mol.ham_reduced[n_q-1]
        ham_mat = qonvert.dict_to_WeightedPauliOperator(ham_red).to_matrix()
        #ham_red_q = qonvert.dict_to_QubitOperator(ham_red)
        #ham_mat = np.matrix(get_sparse_operator(ham_red_q, n_q).toarray())
        gs_true.append(get_ground_state(ham_mat)[0])
        
        A_red = mol.reduce_anz_terms(A, n_q)
        eig_mat = np.matrix(eigenstate_projector(A_red, n_q))
        ham_proj = eig_mat*ham_mat*eig_mat.H
        gs_proj.append(get_ground_state(ham_proj)[0])

    return {'qubit_nums':list(qubit_nums),
            'gs_true':gs_true,
            'gs_proj':gs_proj,
            'diff':[a-b for a, b in zip(gs_proj, gs_true)]}


def hf_state(occupied_orbitals, n_qubits):
    """Function to produce a basis state in the occupation number basis.
    Args:
        occupied_orbitals(list): A list of integers representing the indices
            of the occupied orbitals in the desired basis state
        n_qubits(int): The total number of qubits
    Returns:
        basis_vector(sparse): The basis state as a sparse matrix
    """
    one_index = sum(2**(n_qubits - 1 - i) for i in occupied_orbitals)
    basis_vector = np.zeros(2**n_qubits, dtype=float)
    basis_vector[one_index] = 1
    return basis_vector