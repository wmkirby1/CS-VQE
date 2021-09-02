import utils.qonversion_tools as qonvert
import utils.bit_tools as bit
import cs_vqe_classes.cs_vqe_circuit as cs_circ
from openfermion.ops import QubitOperator
from openfermion.linalg import LinearQubitOperator, get_sparse_operator, get_ground_state
import numpy as np
import math


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
    if type(op) == dict:
        op = qonvert.dict_to_QubitOperator(op)
    
    state = np.array(state)
    conj_state = np.conjugate(state)
    O = LinearQubitOperator(op, num_qubits)
    
    O_state = O.matvec(state)
    expect = conj_state.dot(O_state)
    
    return expect


def eigenstate_projector(A, num_qubits, eval=+1):
    """
    """
    I_op = QubitOperator.identity()
    A_op = qonvert.dict_to_QubitOperator(A)
    projector = get_sparse_operator((I_op+eval*A_op)/2, n_qubits=num_qubits).toarray()
    
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
        ham_red_q = qonvert.dict_to_QubitOperator(ham_red)
        ham_mat = np.matrix(get_sparse_operator(ham_red_q, n_q).toarray())
        gs_true.append(get_ground_state(ham_mat)[0])
        
        A_red = mol.reduce_anz_terms(A, n_q)
        eig_mat = np.matrix(eigenstate_projector(A_red, n_q))
        ham_proj = eig_mat*ham_mat*eig_mat.H
        gs_proj.append(get_ground_state(ham_proj)[0])

    return {'qubit_nums':list(qubit_nums),
            'gs_true':gs_true,
            'gs_proj':gs_proj,
            'diff':[a-b for a, b in zip(gs_proj, gs_true)]}
