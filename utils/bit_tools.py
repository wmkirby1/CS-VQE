from copy import deepcopy
from itertools import chain, combinations, product


def binary_strings(N):
    bins = product([0, 1], repeat=N)
    bin_strings = [''.join([str(i) for i in b]) for b in bins]
    return bin_strings


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


def parity(init_state, indices):
    return sum([int(init_state[i]) for i in indices])


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def unconstrain(initial, removed_Z_indices):
    indices = []
    index_powerset = list(powerset(removed_Z_indices))
    
    for comb in index_powerset:
        initial_ref = list(deepcopy(initial))
        for c in comb:
            initial_ref[c] = str((int(initial_ref[c])+1)%2)
        indices.append(bin_to_int(''.join(initial_ref)))
    
    return indices