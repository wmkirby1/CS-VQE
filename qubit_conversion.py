from openfermion.ops import QubitOperator
from qiskit.quantum_info.operators.symplectic.pauli import Pauli
from qiskit.aqua.operators.legacy import WeightedPauliOperator
from qiskit.opflow.primitive_ops import PauliOp

def QubitOperator_to_dict(op, num_qubits):
    assert(type(op) == QubitOperator)
    op_dict = {}
    term_dict = op.terms
    terms = list(term_dict.keys())

    for t in terms:    
        letters = ['I' for i in range(num_qubits)]
        for i in t:
            letters[i[0]] = i[1]
        p_string = ''.join(letters)        
        op_dict[p_string] = term_dict[t]
         
    return op_dict

def dict_to_QubitOperator(op, num_qubits):
    assert(type(op) == dict)
    p_strings = list(op.keys())
    out = QubitOperator()
    
    for p in p_strings:
        p_str = ''
        for index, p_single in enumerate(p):
            if p_single != 'I':
                p_str += (p_single + str(index) + ' ')
        out += QubitOperator(p_str, op[p])
    
    return out

def WeightedPauliOperator_to_dict(op):
    assert(type(op) == WeightedPauliOperator)
    op_dict = {(p[1]).to_label():p[0] for p in op.paulis}
    
    return op_dict

def dict_to_WeightedPauliOperator(op):
    assert(type(op) == dict)
    return sum([PauliOp(Pauli(p), red_ham[p]) for p in op.keys()])
