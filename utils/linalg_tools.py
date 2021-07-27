import utils.qonversion_tools as qonvert
import utils.bit_tools as bit
from openfermion.ops import QubitOperator
from openfermion.linalg import LinearQubitOperator, get_sparse_operator
import numpy as np
import math


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


def eigenstate_projector(A, num_qubits):
    """
    """
    I_op = QubitOperator.identity()
    A_op = qonvert.dict_to_QubitOperator(A)
    projector = get_sparse_operator((A_op+I_op)/2, n_qubits=num_qubits).toarray()
    
    return projector


def noncon_projector(nc_state, Z_indices, num_qubits):
    """
    """
    num_gen_rem = len(Z_indices)
    Z_complement = list(set(range(num_qubits)) - set(Z_indices))
    fixed_qubits = [nc_state[i] for i in Z_complement]
    fixed_index = bit.bin_to_int(''.join(fixed_qubits))
    fixed_state = np.array([0 for i in range(2**(num_qubits-num_gen_rem))])
    fixed_state[fixed_index] = 1
    a = np.identity(2**num_gen_rem)
    b = np.outer(fixed_state, fixed_state)
    projector = np.kron(a, b)

    return projector


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