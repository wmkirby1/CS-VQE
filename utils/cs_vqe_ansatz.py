from qiskit.aqua.algorithms import NumPyEigensolver#, VQE
from qiskit.algorithms import VQE
import matplotlib.pyplot as plt
import numpy as np
import itertools
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.chemistry.components.initial_states import HartreeFock
from qiskit.circuit.library import EfficientSU2
from qiskit.aqua.components.optimizers import COBYLA, SPSA, SLSQP
from qiskit.aqua.operators import Z2Symmetries
from qiskit import IBMQ, BasicAer, Aer
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry import FermionicOperator
from qiskit.aqua import QuantumInstance
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
from qiskit.providers.aer.noise import NoiseModel
from qiskit.visualization import plot_histogram
from qiskit.quantum_info.operators.symplectic.pauli import Pauli
from qiskit.aqua.operators.legacy import WeightedPauliOperator
from qiskit.circuit.library import TwoLocal
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.utils import QuantumInstance, algorithm_globals
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, IBMQ, execute
from qiskit import execute, Aer
from qiskit.providers.ibmq import IBMQBackend, least_busy
from qiskit.tools.visualization import circuit_drawer
from qiskit.providers.ibmq import least_busy
from IPython.display import clear_output
import utils.cs_vqe as c
import copy
from qiskit.opflow import X, Z, I
from qiskit.opflow.primitive_ops import PauliOp

import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, IBMQ, execute
from qiskit import execute, Aer
from qiskit.providers.ibmq import IBMQBackend, least_busy
from qiskit.tools.visualization import circuit_drawer
from qiskit.providers.ibmq import least_busy
from IPython.display import clear_output


def connect_to_ibm(simulator: bool = False) -> IBMQBackend:
    """Connect to IBMQ chip and return backend.

    Parameters
    ----------
    simulator : bool, optional
        Whether to use the simulator or the least busy IBMQ chip, by default False

    Returns
    -------
    IBMQBackend
        Backend object of the device used to run calculations
    """
    # load individual IBMQ account
    #if len(IBMQ.stored_account()) == 0:
    with open("token.txt", "r") as token_f:
        token = token_f.read()
    IBMQ.save_account(token, overwrite=True)
    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')

    # select backend
    if simulator:
        backend = Aer.get_backend("qasm_simulator")
    else:
        small_devices = provider.backends(filters=lambda x: not x.configuration().simulator)
        backend = least_busy(small_devices)

    print(backend)

    return backend


def ontic_prob(ep_state, ontic_state):
    
    if ep_state[0] != ontic_state[0]:
        return 0
    
    else:
        prod = 1
        for index, r in enumerate(ep_state[1]):
            f = 1/2 * abs(r + ontic_state[1][index])
            prod *= f
        
        return prod    

    
def epistemic_dist(ep_state):
    size_G = len(ep_state[0])
    size_Ci = len(ep_state[1])
    size_R = size_G + size_Ci
    
    ep_prob = {}
    
    ontic_states = list(itertools.product([1, -1], repeat=size_R))
    
    for o in ontic_states:
        o_state = [list(o[0:size_G]), list(o[size_G:size_R])]
        o_prob = ontic_prob(ep_state, o_state)
        
        if o_prob != 0:
            ep_prob[o] = o_prob
    
    return ep_prob


def exp_P(p_string, control=None, circ=None, rot=0):
    
    num_qubits = len(p_string)
    
    #index X, Y, Z in the string of Paulis
    p_index = {}
    for index, p in enumerate(p_string):
        if p not in p_index:
            p_index[p] = [index]
        else:
            p_index[p] += [index]

    #initiate quantum circuit object
    if circ is None:
        circ = QuantumCircuit(num_qubits)
        circ.barrier()
    
    #Rotate X and Y Paulis into Z basis
    for q in p_index.keys():
        if q == 'X':
            #rotate X to Z
            for i in p_index['X']:
                circ.h(i)
            #circ.barrier()
        
        elif q == 'Y':
            #rotate Y to Z
            for i in p_index['Y']:
                circ.sdg(i)
                circ.h(i)
            #circ.barrier()
            
        else:
            pass
    
    #Evaluate parity of remaining Z qubits
    if 'I' in p_index:
        #return blank circuit if all qubits are identity
        if p_index['I'] == list(range(num_qubits)):
            return circ
        #Index qubits which are non identity
        else:
            non_I = list(set(range(num_qubits)) - set(p_index['I']))
            num_Z = range(len(non_I)-1)
    else:
        non_I = list(range(num_qubits))
        num_Z = range(num_qubits - 1)
    
    non_I = sorted(non_I)
    
    #cascade of CNOT gates between adjacent non-identity qubits
    for i in num_Z:
        circ.cx(non_I[i], non_I[i+1])
    
    #apply the rotation
    if control is None:
        circ.rz(2*rot, non_I[-1])
    else:
        circ.crz(2*rot, control, non_I[-1])
        
    #reverse cascade of CNOT gates between adjacent non-identity qubits
    for i in num_Z:
        circ.cx(non_I[len(num_Z)-i-1], non_I[len(num_Z)-i])
        
    #circ.barrier()
    
    #Rotate X and Y Paulis into Z basis
    for q in p_index.keys():
        if q == 'X':
            #rotate X to Z
            for i in p_index['X']:
                circ.h(i)
            #circ.barrier()
        
        elif q == 'Y':
            #rotate Y to Z
            for i in p_index['Y']:
                circ.h(i)
                circ.s(i)
            #circ.barrier()
            
        else:
            pass
        
    circ.barrier()
    
    return circ


def construct_ansatz(init_state=[], paulis=[], params=[], rots=[], circ=None, trot_order=1) -> QuantumCircuit:
    """
    init_state: list of qubit positions that should have value 1 (apply X). By default all 0.
    paulis: list of Pauli strings, applied left to right
    params: list of angles by which to rotate each exponentiated Pauli, leave empty to fill with parameters for optimisation in VQE
    rots: list of rotations from CS-VQE, applied left to right
    """
    if params == []:
        #parameters to be optimised in VQE routine
        param_chars = ['α','β','γ','δ','ε','ζ','η','θ','ι','κ','λ','μ','ν','ξ','ο','π','ρ','ς','σ','τ','υ','φ','χ','ψ','ω']
        params = [] 
        for comb in list(itertools.combinations(param_chars, 3)):
            char_str = ''.join(comb)
            params.append(Parameter(char_str))

    #initiate quantum state (usually Hartree Fock)
    if circ is None:
        circ = QuantumCircuit(len(paulis[0]))
        for q in init_state:
            circ.x(q)

    #applies the ansatz
    for t in range(trot_order):
        for index, p in enumerate(paulis):
            t_index = t*len(paulis) + index
            circ += exp_P(p_string = p, rot = params[index]/trot_order)
    
    #rotates in accordance with CS-VQE routine
    for r in rots:
        circ += exp_P(p_string = r[1], rot = r[0])
      
    return circ


def construct_reduced_hamiltonian(ham, terms_noncon=[], num_qubits = 0, out_raw_dict = False) -> PauliOp:
    """Determine reduced Hamiltonian on minimal contextual subset
    
    """
    #restructure WeightedPauliOperator ham from Qiskit list [[complex, Pauli]] -> dict {str:complex}
    if str(type(ham)) == "<class 'qiskit.aqua.operators.legacy.weighted_pauli_operator.WeightedPauliOperator'>":
        Paulis = ham.paulis
        ham = {(p[1]).to_label():p[0] for p in Paulis}
        
    #Leave if already in correct format
    elif str(type(ham)) == "<class 'dict'>":
        pass
    
    #Reserved for alternative formats
    else:
        raise Exception("Unrecognised Hamiltonian Format")
            
    #Find largest noncontextual subset
    if terms_noncon == []:
        terms_noncon = c.greedy_dfs(ham, 10, criterion='weight')[-1]
    
    ham_noncon = {p:ham[p] for p in terms_noncon}
    
    #Contrusct epistricted model
    model = c.quasi_model(ham_noncon)
    print(model)
    fn_form = c.energy_function_form(ham_noncon, model)
    gs_noncon = c.find_gs_noncon(ham_noncon)
    ep_state = gs_noncon[1]
    
    rotations, diagonal_set, vals = c.diagonalize_epistemic(model,fn_form,ep_state)
    #print(rotations)
    
    if num_qubits == 0:
        return [], gs_noncon[0]
    
    #elif num_qubits == len(model[0][0]):
    #    return sum([PauliOp(Pauli(k), ham[k]) for k in ham.keys()]), gs_noncon[0]
    
    else:
        #Determine contextual subspace Hamiltonians
        order = list(range(len(model[0][0]))) #this can be user-specified in future
        order_ref = copy.deepcopy(order) #since get_reduced_hamiltonians empties original order
        reduced_hamiltonians = c.get_reduced_hamiltonians(ham, model, fn_form, ep_state, order)
        red_ham = reduced_hamiltonians[num_qubits]

        if out_raw_dict == True:
            return red_ham, gs_noncon[0]
        else:
            #return WeightedPauliOperator([[red_ham[k], Pauli(k)] for k in red_ham.keys()]), gs_noncon[0]
            return sum([PauliOp(Pauli(k), red_ham[k]) for k in red_ham.keys()]), gs_noncon[0]

        
def CS_VQE(ham, terms_noncon, num_qubits, ansatz=None, num_orbitals=0, num_particles=0, shift=0, 
           backend=BasicAer.get_backend("statevector_simulator"), optimizer = SLSQP(maxiter=10000)):
    """
    """
    red_ham, gs_approx = construct_reduced_hamiltonian(ham, terms_noncon, num_qubits)
    print(red_ham)
    if num_qubits == 0:
        return gs_approx + shift#, exact_result
    
    else:
        exact = NumPyEigensolver(red_ham).run()
        exact_result = float(np.real(exact.eigenvalues)) + shift
        
        if ansatz is None:
            ent_map = list(itertools.combinations(range(num_qubits), 2))
            ansatz = TwoLocal(num_qubits, 'ry', 'cx', ent_map, reps=3, insert_barriers=True)
            
        seed = 50
        algorithm_globals.random_seed = seed
        qi = QuantumInstance(Aer.get_backend('statevector_simulator'), seed_transpiler=seed, seed_simulator=seed)
    
        vqe = VQE(ansatz, optimizer=optimizer, quantum_instance=qi)
        vqe_run    = vqe.compute_minimum_eigenvalue(operator=red_ham)
        vqe_result = vqe_run.optimal_value + shift

        return vqe_result, exact_result