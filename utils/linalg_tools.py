from openfermion.ops import QubitOperator
from openfermion.linalg import LinearQubitOperator
import numpy as np
import math


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


def random_complex_vector(n, order=False):  
    return [random_complex_unit()*a for a in random_vector(n)]


def expectation(op, state, num_qubits):
    assert(type(op)==QubitOperator)
    
    state = np.array(state)
    conj_state = np.conjugate(state)
    O = LinearQubitOperator(op, num_qubits)
    
    O_state = O.matvec(state)
    expect = conj_state.dot(O_state)
    
    return expect