from openfermion.ops import QubitOperator
from openfermion.linalg import LinearQubitOperator
import numpy as np


def expectation(op, state, num_qubits):
    assert(type(op)==QubitOperator)
    
    state = np.array(state)
    conj_state = np.conjugate(state)
    O = LinearQubitOperator(op, num_qubits)
    
    O_state = O.matvec(state)
    expect = conj_state.dot(O_state)
    
    return expect