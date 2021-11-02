import itertools
from itertools import chain, combinations
from copy import deepcopy
import math
import numpy as np
from openfermion.linalg import LinearQubitOperator
from openfermion.ops import FermionOperator, QubitOperator
import utils.cs_vqe_tools as c
import utils.qonversion_tools as qonvert


def bin_to_int(bits):
    bit_string = deepcopy(bits)
    
    if type(bit_string) == str:
        bit_string = [int(b) for b in bit_string]
        
    for index, b in enumerate(bit_string):
        bit_string[index] = b * 2 ** (len(bit_string)-index-1)
    
    return sum(bit_string)


def int_to_bin(integer, num_qubits):
    if integer >= 2**num_qubits:
        raise ValueError('Input integer larger than specified number of bits.')
    bin_str=bin(integer)[2:]
    leading_0 = ''.join(['0' for i in range(num_qubits-len(bin_str))])
    return leading_0 + bin_str


def A_action(molecule, num_qubits, basis_index, rot=False):
    """This will be computed programmatically from A operator in the future***
    """
    B = list(itertools.product([0,1], repeat=num_qubits))
    b1 = list(B[basis_index])
    b2 = deepcopy(b1)
    i1 = bin_to_int(b1)
    
    if molecule == 'H2O':
        b2[5] = (b2[5]+1)%2
        i2 = bin_to_int(b2)
        parity = b1[9]+b1[8]+b1[7]+b1[6]+b1[4]
        Z_loc  = b1[5]
        
    elif molecule == 'HeH+':
        if not rot:
            b2[0] = (b2[0]+1)%2
            b2[6] = (b2[6]+1)%2
            i2 = bin_to_int(b2)
            parity = 1+sum(b1)
            Z_loc  = b1[6]
        else:
            b2[6] = (b2[6]+1)%2
            i2 = bin_to_int(b2)
            parity = b1[1]+b1[2]+b1[3]+b1[4]+b1[5]
            Z_loc  = b1[6]
    else:
        raise ValueError('Molecule is not recognised.')
    
    return i1, i2, parity, Z_loc


def add_eigenstate(molecule, r1, r2, index, num_qubits, theta=0, custom_amp=None, rot=False):
    """
    """
    i1, i2, parity, Z_loc = A_action(molecule, num_qubits, index, rot)
    amp_ratio = (1 + r2 * (-1)**Z_loc) / (r1 * (-1)**(parity))
    t = np.arctan(amp_ratio)
    #print(q4, t)
    #print(i1, ':', np.sin(t), i2, ':', np.cos(t))
    psi = [0 for i in range(2**num_qubits)]
    
    if custom_amp is None:   
        psi[i1] = np.sin(t)*np.exp(1j*theta)
        psi[i2] = np.cos(t)*np.exp(1j*theta)
    else:
        psi[i1] = custom_amp[0]
        psi[i2] = custom_amp[1]
        
    return np.array(psi)


def expectation(op, state, num_qubits):
    assert(type(op)==QubitOperator)
    
    state = np.array(state)
    conj_state = np.conjugate(state)
    O = LinearQubitOperator(op, num_qubits)
    
    O_state = O.matvec(state)
    expect = conj_state.dot(O_state)
    
    return expect


def discard_generator(ham_noncon, ham_context, generators):
    new_ham_noncon = deepcopy(ham_noncon)
    new_ham_context = deepcopy(ham_context)
    
    Z_indices = [g.index('Z') for g in generators]
    removed=[]
    
    for index in Z_indices:
        for p in ham_noncon:
            if p not in removed:
                if p[index] == 'Z':
                    new_ham_context[p] = ham_noncon[p]
                    del new_ham_noncon[p]
                    removed.append(p)
            
    return new_ham_noncon, new_ham_context


def rotate_operator(rotations, op):
    rot_op = {}
    
    for p in op.keys():
        p_ref = deepcopy(p)
        parity = 1
        coeff = op[p]
        for r in rotations:
            rotate_p = c.apply_rotation(r, p)
            p = list(rotate_p.keys())[0]
            parity *= rotate_p[p]
        
        rot_op[p] = parity * coeff
        
    return rot_op

def rotate_hamiltonian(rotations, ham, ham_noncon, ham_context):
    
    rot_ham={}
    rot_ham_noncon={}
    rot_ham_context={}

    for p in ham.keys():
        p_ref = deepcopy(p)
        parity = 1
        coeff = ham[p]
        for r in rotations:
            rotate_p = c.apply_rotation(r, p)
            p = list(rotate_p.keys())[0]
            parity *= rotate_p[p]
        
        rot_ham[p] = parity * coeff
        if p_ref in ham_noncon.keys():
            rot_ham_noncon[p] = parity * coeff
        else:
            rot_ham_context[p] = parity * coeff
            
    return rot_ham, rot_ham_noncon, rot_ham_context


def rotate_state(rotations, state, num_qubits):
    
    rot_state = deepcopy(state)
    
    for r in rotations:
        r_op = QubitOperator('', 1/np.sqrt(2)) - qonvert.dict_to_QubitOperator({r[1]: 1/np.sqrt(2)*1j}, num_qubits)
        r_op = LinearQubitOperator(r_op, num_qubits)
        rot_state = r_op.matvec(rot_state)
        
    return rot_state


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def qubit_map(molecule, num_qubits, rot=False):
    qubit_map={}
    B = list(itertools.product([0,1], repeat=num_qubits))
    for i in range(2**(num_qubits)):
        i1, i2 = A_action(molecule, num_qubits, i, rot)[0:2]
        b1 = int_to_bin(i1, num_qubits)
        b2 = int_to_bin(i2, num_qubits)
        qubit_map[i1] = [(i1, b1), (i2, b2)]
    return qubit_map


def find_eigenstate_indices(initial, removed_Z_indices, include_complement=False, num_qubits = None, molecule=None, rot=False):
    indices = []
    index_powerset = list(powerset(removed_Z_indices))
    
    for comb in index_powerset:
        initial_ref = list(deepcopy(initial))
        for c in comb:
            initial_ref[c] = str((int(initial_ref[c])+1)%2)
        indices.append(bin_to_int(''.join(initial_ref)))
    
    # Complement is simply the negative state, so is the same eigenvector
    if include_complement:
        indices_ref = deepcopy(indices)
        for i in indices_ref:
            maps_to = A_action(molecule, num_qubits, basis_index=i, rot=rot)[1]
            indices.append(maps_to)
            
    return indices


def random_vector(n):
    components = [np.random.normal() for i in range(n)]
    r = math.sqrt(sum(x*x for x in components))
    v = [x/r for x in components]
    return v


def random_complex_unit():
        rand_vec = random_vector(2)
        x = rand_vec[0]
        y = rand_vec[1]
        
        return x + y*1j
    
def expectation_optimiser(molecule, ham_n, ham_c, r1, r2, amps, initial_state, 
                          Z_indices, num_qubits, rotations=None, include_complement = False, rot = False):
    """
    """
    eigenstate_indices = find_eigenstate_indices(initial_state, Z_indices, include_complement, num_qubits, molecule, rot)

    psi = np.array([0 for i in range(2**num_qubits)], dtype=complex)
    for index, i in enumerate(eigenstate_indices):
        psi += (amps[index])*add_eigenstate(molecule=molecule, r1=r1, r2=r2, theta=0, index=i, num_qubits=num_qubits, rot=rot)
    
    if rotations is not None:
        psi = rotate_state(rotations, psi, num_qubits)
    
    expect_noncon = expectation(ham_n, psi, num_qubits)
    expect_context = expectation(ham_c, psi, num_qubits)
    
    return expect_noncon, expect_context