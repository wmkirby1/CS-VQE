from numpy.lib.npyio import _save_dispatcher
from cs_vqe_classes.cs_vqe import cs_vqe
from cs_vqe_classes.eigenstate import eigenstate
import utils.bit_tools as bit
import utils.cs_vqe_tools as c_tools
import utils.circuit_tools as circ
import utils.qonversion_tools as qonvert
import utils.linalg_tools as la
from copy import deepcopy
import numpy as np
import itertools

from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit.utils import QuantumInstance
from qiskit.aqua.components.optimizers import (SLSQP, COBYLA, SPSA, AQGD, L_BFGS_B, P_BFGS,
                                                NELDER_MEAD, CG, ADAM, POWELL, TNC, GSLS,
                                                NFT, IMFIL, BOBYQA, SNOBFIT)
from qiskit.algorithms import VQE
from qiskit import Aer
#from openfermion.linalg import get_ground_state, jw_configuration_state

import numpy as np
import matplotlib.pyplot as plt

from qiskit import Aer
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.algorithms import VQE, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit.library import TwoLocal
from qiskit.opflow import I, X, Z
import os
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.test.mock import FakeVigo


def jw_configuration_state(occupied_orbitals, n_qubits):
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


class cs_vqe_circuit():
    """
    """
    # class variables for storing VQE results
    cs_vqe_results = {}
    counts = []
    values = []
    errors = []
    # if True adds additional qubit to circuit
    ancilla = False
    red_anz_drop = False
    
    def __init__(self, hamiltonian, terms_noncon, num_qubits, order=None, rot_G=True, rot_A=True):
        #occ_orb = list(set(range(num_qubits))-set(range(int(num_qubits/2))))
        for index, i in enumerate(jw_configuration_state(range(int(num_qubits/2)), num_qubits)):
            if i == 1:
                self.HF_config = bit.int_to_bin(index, num_qubits)
        
        self.hamiltonian = hamiltonian
        self.num_qubits = num_qubits
        self.rot_G = rot_G
        self.rot_A = rot_A

        # epistricted model and reduced Hamiltonian
        cs = cs_vqe(hamiltonian, terms_noncon, num_qubits, rot_G, rot_A)
        self.ham_rotations = cs.rotations()
        generators = cs.generators()

        # defined here so only executed once instead of at each call
        self.gs_noncon_energy = cs.gs_noncon_energy()
        self.true_gs = cs.true_gs()[0]
        self.G = generators[0]
        self.A = generators[1] 

        if rot_A:
            self.X_index = list(self.A.keys())[0].find('Z')
        else:
            for p in self.A.keys():
                if 'X' in list(p):
                    self.X_index = p.index('X')
        self.X_qubit = num_qubits-1-self.X_index

        ham_noncon = {}
        for t in terms_noncon:
            ham_noncon[t] = hamiltonian[t]

        if order is None:
            order = c_tools.csvqe_approximations_heuristic(hamiltonian, ham_noncon, num_qubits, self.true_gs)[3]
        self.order, self.ham_reduced = cs.reduced_hamiltonian(order)

        # +1-eigenstate parameters
        self.init_state = cs.init_state()
        eig = eigenstate(self.A, bit.bin_to_int(self.init_state), num_qubits)
        self.index_paulis = eig.P_index(q_pos=True)
        if not rot_A:
            #self.t1, self.t2 = eig.t_val(alt=True)
            self.r1, self.r2 = self.A.values()
            self.A1, self.A2 = self.A.keys()

    def sim_qubits(self, num_sim_q, complement=False):
        """
        """
        q_pos_ord = [self.num_qubits-1-q for q in self.order]
        if complement:
            sim_indices = self.order[num_sim_q:]
            sim_qubits = q_pos_ord[num_sim_q:]
        else:
            sim_indices = self.order[0:num_sim_q]
            sim_qubits = q_pos_ord[0:num_sim_q]
        # if ordering left to right
        if sim_indices != []:
            sim_qubits, sim_indices = zip(*sorted(list(zip(sim_qubits, sim_indices)), key=lambda x:x[1]))

        return sim_qubits, sim_indices


    def reduced_parity_bits(self, num_sim_q):
        """
        """
        IZ1_red = [q for q in self.index_paulis['Z1'] if q in self.sim_qubits(num_sim_q)[0]]
        IZ2_red = [q for q in self.index_paulis['Z2'] if q in self.sim_qubits(num_sim_q)[0]]
        return IZ1_red, IZ2_red


    def lost_parity(self, Z_qubits, num_sim_q):
        """
        """
        lost_parity = 0

        for i in self.sim_qubits(num_sim_q, complement=True)[0]:
            if i in Z_qubits:
                lost_parity += int(self.init_state[self.num_qubits-1-i])

        return lost_parity%2
    

    def reference_state(self, input_ham=None):
        """
        """
        reference_bits={}

        if input_ham is None:
            for g in self.G.keys():
                reference_bits[self.num_qubits-1-g.find('Z')] = self.G[g]
            if self.rot_A:
                a = list(self.A.keys())[0]
                reference_bits[self.num_qubits-1-a.find('Z')] = self.A[a]
            blank_state = list(np.zeros(self.num_qubits, dtype=int))
            for q in reference_bits.keys():
                index = self.num_qubits-1-q
                if round(reference_bits[q]) == -1:
                    blank_state[index] = 1
            ref_string = ''.join([str(i) for i in blank_state])

        else:
            #ham_q = qonvert.dict_to_QubitOperator(input_ham)
            ham_mat = qonvert.dict_to_WeightedPauliOperator(input_ham).to_matrix()
            gs_vec = la.get_ground_state(ham_mat)[1]
            n_q = len(list(input_ham.keys())[0])
            #gs = get_ground_state(get_sparse_operator(ham_q, n_q).toarray())
            #gs_vec = gs[1]
            amp_list = [abs(a)**2 for a in list(gs_vec)]
            sig_amp_list = sorted([(str(index), a) for index, a in enumerate(amp_list) if a > 0.001], key=lambda x:x[1])
            sig_amp_list.reverse()
            reference_str=bit.int_to_bin(int(sig_amp_list[0][0]), n_q)
            for index, b in enumerate(reference_str):
                reference_bits[n_q-1-index] = int(b)
            ref_string = ''.join([str(i) for i in reference_bits.values()])

        return ref_string

    
    def qubit_map(self, num_sim_q):
        """
        """
        q_map={}
        sim_qubits = self.sim_qubits(num_sim_q)[0]
        for q in sim_qubits:
            try:
                q_map[q] = num_sim_q - 1 - (sim_qubits).index(q)
            except:
                pass    

        return q_map

   
    def reduce_anz_terms(self, anz_terms, num_sim_q, fix_X_qubit=False):
        """
        """
        if type(anz_terms)!=dict:
            anz_terms = {t:0 for t in anz_terms}
        #if self.rot_G:
        #    anz_terms = c_tools.rotate_operator(self.rotations, anz_terms)
        
        anz_terms_reduced = {}
        for p in anz_terms.keys():
            blank_op = ['I' for i in range(num_sim_q)]
            for i, sim_i in enumerate(self.sim_qubits(num_sim_q)[1]):
                if fix_X_qubit:
                    if sim_i != self.X_index:
                        blank_op[i] = p[sim_i]
                else:
                    blank_op[i] = p[sim_i]
            
            #if set(blank_op) != {'I'}:
            t = ''.join(blank_op)
            if self.ancilla:
                t = 'I' + t
            if t in anz_terms_reduced.keys():
                anz_terms_reduced[t] = anz_terms_reduced[t] + anz_terms[p]
            else:
                anz_terms_reduced[t] = anz_terms[p]

        return anz_terms_reduced

    
    def reduce_rotations(self, rotations, num_sim_q):
        rot = [tuple(r) for r in deepcopy(rotations)]
        rot_red = []
        for r, p in rot:
            blank_op = ['I' for i in range(num_sim_q)]
            for i, sim_i in enumerate(self.sim_qubits(num_sim_q)[1]):
                blank_op[i] = p[sim_i]
            if r == 'pi/2':
                r = np.pi/4
            else:
                r=r/2
            if set(blank_op) != {'I'}:
                t = ''.join(blank_op)
                if self.ancilla:
                    t = 'I' + t
                rot_red.append((t, r))

        return rot_red

    
    def project_anz_terms(self, anz_terms, num_sim_q):
        """
        """
        sim_indices = list(self.sim_qubits(num_sim_q)[1])
        sim_complement = list(set(range(self.num_qubits))-set(sim_indices))
        anz_rot = c_tools.rotate_operator(self.ham_rotations, anz_terms)
        nc_state = self.reference_state()
        proj_anz = {}
        
        drop_op = []

        for p in anz_rot.keys():
            proj_pauli = [p[i] for i in sim_complement]
            if ('X' not in proj_pauli) and ('Y' not in proj_pauli):
                parity = int(len([i for i in sim_complement if nc_state[i]=='1' and p[i]=='Z'])%2)
                #print(parity)
                sgn = 1-2*parity
                #print(p, sgn)
                sim_pauli_list = [p[i] for i in sim_indices]
                sim_pauli = ''.join(sim_pauli_list)
                if set(sim_pauli) != {'I'}:
                    if sim_pauli in proj_anz.keys():
                        proj_anz[sim_pauli] = proj_anz[sim_pauli] + sgn*anz_rot[p]
                    else:
                        proj_anz[sim_pauli] = sgn*anz_rot[p]
            else:
                drop_op.append(p)
        prod=''.join(['I' for i in range(self.num_qubits)])
        s_coeff=1
        c_coeff=1
        sgn = 1
        for p in drop_op:
            s_coeff *= np.sin(anz_rot[p])
            c_coeff *= np.cos(anz_rot[p])
            prod, new_sgn = c_tools.pauli_mult(prod, p)
            sgn *= new_sgn

        drop_pauli = ''.join([prod[i] for i in sim_indices])
        print(drop_pauli)
        if set(drop_pauli) != {'I'}:
            self.red_anz_drop = True
            proj_anz[drop_pauli] = 0
        else:
            self.red_anz_drop = False
        return proj_anz, drop_pauli

    

    ################################## circuit blocks #################################
    ## Below are circuit components that may be selected in the build_circuit method ##
    ###################################################################################

    def gs_check_block(self, qc, num_sim_q):
        """
        """
        ham_red = self.ham_reduced[num_sim_q]
        ham_mat = qonvert.dict_to_WeightedPauliOperator(ham_red).to_matrix()
        gs_vector = la.get_ground_state(ham_mat)[1]
        qc.initialize(gs_vector)


    def ref_state_block(self, qc, num_sim_q, ref_type='nc_gs'):
        """
        """
        q_map = self.qubit_map(num_sim_q)
        sim_qubits = self.sim_qubits(num_sim_q)[0]
        
        if ref_type=='most significant':
            reference_state = self.reference_state(self.hamiltonian)
        elif ref_type=='nc_gs':
            reference_state = self.reference_state()
        elif ref_type=='HF':
            reference_state = self.HF_config
        else:
            raise ValueError('Invalid reference state type - must be "most significant", "nc_gs" or "HF".')
        
        for q in sim_qubits:
            q_index = self.num_qubits-1-q
            if reference_state[q_index] == '1':
                qc.x(q_map[q])


    def anz_block(self, anz_terms, qc, num_sim_q):
        """
        """
        q_map = self.qubit_map(num_sim_q)

        if anz_terms is not None:
            anz_terms_reduced = self.project_anz_terms(anz_terms, num_sim_q)[0]
            if anz_terms_reduced == {}:
                # because VQE requires at least one parameter...
                qc.rz(Parameter('a'), 0)
            else:
                qc = circ.circ_from_paulis(paulis=list(anz_terms_reduced.keys()), circ=qc, trot_order=2, dup_param=False)
        else:
            qc += TwoLocal(num_sim_q, 'ry', 'cx', 'full', reps=2, insert_barriers=True)


    def swap_entgl_block(self, qc, num_sim_q):
        """requires ancilla
        """
        assert(self.ancilla == True)

        q_map = self.qubit_map(num_sim_q)
        qc.cx(q_map[self.X_qubit], num_sim_q)
        qc.cx(num_sim_q, q_map[self.X_qubit])


    def rot_ham_block(self, qc, num_sim_q, inverse=False):
        """
        """
        ham_rot = self.reduce_rotations(self.ham_rotations, num_sim_q)
        if inverse:
            ham_rot.reverse()
        for p, r in ham_rot:
            if inverse:
                qc = circ.exp_P(p, circ=qc, rot=-r)
            else:
                qc = circ.exp_P(p, circ=qc, rot=r)


    def rot_A_block(self, qc, num_sim_q, inverse=False):
        """
        """
        diag_A = []
        rot = c_tools.pauli_mult(self.A1, self.A2)
        t = np.arctan(self.r1/self.r2)
        diag_A.append([t, rot[0]])
        single_A = c_tools.rotate_operator([[t*(rot[1]*1j).real, rot[0]]], self.A)
        Z_indices = [g.find('Z') for g in self.G]
        single_A_indices = [i for i, p in enumerate(list(single_A.keys())[-1]) if p == 'Z']
        ind_Z = [i for i in single_A_indices if i not in Z_indices]
        for i in ind_Z:    
            blank_op = ['I' for i in range(self.num_qubits)]
            blank_op[i] = 'Y'
            rot = ''.join(blank_op)
            diag_A.append(['pi/2', rot])
            rotated = c_tools.pauli_mult(rot, list(single_A.keys())[-1])[0]
            blank_op = ['I' for i in range(self.num_qubits)]
            for j in single_A_indices:
                if rotated[j]=='Z':
                    blank_op[j] = 'Z'
                elif rotated[j] == 'X':
                    blank_op[j] = 'Y'
            rot = ''.join(blank_op)
            diag_A.append(['pi/2', rot])

        A_rot = self.reduce_rotations(diag_A, num_sim_q)
        if inverse:
            A_rot.reverse()
        for p, r in A_rot:
            if inverse:
                qc = circ.exp_P(p, circ=qc, rot=-r)
            else:
                qc = circ.exp_P(p, circ=qc, rot=r)


    def A_eig_block(self, qc, num_sim_q):
        """
        """
        assert(self.ancilla == True)

        q_map = self.qubit_map(num_sim_q)
        Q = self.r1/(1-self.r2)

        qc.cx(q_map[self.X_qubit], num_sim_q)
        qc.cry(2*np.arctan(Q), num_sim_q, q_map[self.X_qubit])
        qc.x(num_sim_q)
        qc.cry(2*np.arctan(-1/Q), num_sim_q, q_map[self.X_qubit])


    def parity_cascade_block(self, qc, Z_qubits, num_sim_q, inner_circ=None):
        """
        """
        assert(self.ancilla == True)

        q_map = self.qubit_map(num_sim_q)
        sim_qubits = self.sim_qubits(num_sim_q)[0]
        #index_paulis_reduced = self.index_paulis_reduced(num_sim_q)
        lost_parity = self.lost_parity(Z_qubits, num_sim_q)
        
        if lost_parity:
            qc.x(num_sim_q)
            
        cascade_bits = [q_map[q] for q in Z_qubits if q in q_map.keys()]

        #store parity in ancilla qubit (8) via CNOT cascade
        qc = circ.cascade(cascade_bits+[num_sim_q], circ=qc)

        if inner_circ is None:
            qc.cz(num_sim_q, q_map[self.X_qubit])
        else:
            inner_circ(qc, num_sim_q)
        
        #reverse CNOT cascade
        qc = circ.cascade(cascade_bits+[num_sim_q], circ=qc, reverse=True)

        if lost_parity:
            qc.x(num_sim_q)

    
    ####################### circuit builder and VQE functionality #####################
    ###################################################################################

    def full_uccsd(self, anz_terms):
        q_map = self.qubit_map(self.num_qubits)
        qc = QuantumCircuit(self.num_qubits)
        for q in self.HF_config:
            qc.x(q_map[int(q)])
        qc = circ.circ_from_paulis(paulis=list(anz_terms.keys()), circ=qc, trot_order=2, dup_param=False)
        return qc
    
    def build_circuit(self, anz_terms, num_sim_q):
        """
        """
        #ham_red = self.ham_reduced[num_sim_q]
        #ham_mat = qonvert.dict_to_WeightedPauliOperator(ham_red).to_matrix()
        #gs_vector = get_ground_state(ham_mat)[1]
        #amp_list = [abs(a)**2 for a in list(gs_vector)]
        #sig_amp_list = sorted([(str(index), a) for index, a in enumerate(amp_list) if a > 0.001], key=lambda x:x[1])
        #sig_amp_list.reverse()
        #X, Y = zip(*sig_amp_list)
        #fig, ax = plt.subplots()
        #ax.bar(X, Y)
        #print(plt.show())

        self.ancilla = False

        q_map = self.qubit_map(num_sim_q)
        sim_qubits = self.sim_qubits(num_sim_q)[0]

        if self.ancilla:
            dim = num_sim_q + 1
        else:
            dim = num_sim_q

        qc = QuantumCircuit(dim)
        self.ref_state_block(qc, num_sim_q)
        self.anz_block(anz_terms, qc, num_sim_q)

        #self.gs_check_block(qc, num_sim_q)
        #for q in sim_qubits:
        #    if q in [5, 2]:
        #        qc.x(q_map[q])
        #params = ['α','β','γ','δ','ε','ζ','η','θ','ι','κ','λ','μ','ν','ξ','ο','π','ρ','ς','σ','τ','υ','φ','χ','ψ','ω']
        #for q in range(num_sim_q):
        #    qc.rx(Parameter(params[q]), q)
        #self.rot_ham_block(qc, num_sim_q)
        

        # qiskit variational_algorithm struggling with only one parameter
        if qc.num_parameters == 1:
            qc.rz(Parameter('b'), 0)
        #print(qc.draw())
        return qc


    def store_intermediate_result(self, eval_count, parameters, mean, std):
        """
        """
        self.counts.append(eval_count)
        self.values.append(mean)
        self.errors.append(std)


    def CS_VQE(self, anz_terms=None, num_sim_q=None, ham=None, optimizer=IMFIL(maxiter=10000), check_A=False, noise=False):
        """
        """
        self.counts.clear()
        self.values.clear()
        self.errors.clear()

        qc = self.build_circuit(anz_terms, num_sim_q)
        #init_anz_params = np.array([0 for i in range(num_sim_q)])
        #bounds = np.array([(-np.pi/2, np.pi/2) for i in range(num_sim_q)])

        if anz_terms is not None:
            anz_red, drop_pauli = self.project_anz_terms(anz_terms, num_sim_q)
            if anz_red != {}:
                init_anz_params = np.array([(anz_red[p]).imag for p in anz_red.keys() if set(p)!={'I'}])
                if len(init_anz_params) != qc.num_parameters:
                    init_anz_params = np.append(init_anz_params, 0)  
            else:
                init_anz_params = np.zeros(qc.num_parameters)
        else:
            drop_pauli = None

        bounds = np.array([(p-np.pi/8, p+np.pi/8) for p in init_anz_params])
        qc.parameter_bounds = bounds

        seed = 42
        algorithm_globals.random_seed = seed
        if not noise:
            backend = Aer.get_backend('statevector_simulator')
            qi = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed)

        else:
            device_backend = FakeVigo()
            noise_model = None
            backend = Aer.get_backend('qasm_simulator')
            os.environ['QISKIT_IN_PARALLEL'] = 'TRUE'
            device = QasmSimulator.from_backend(device_backend)
            coupling_map = device.configuration().coupling_map
            noise_model = NoiseModel.from_backend(device)
            basis_gates = noise_model.basis_gates
            qi = QuantumInstance(backend=backend, 
                                seed_simulator=seed, 
                                seed_transpiler=seed,
                                coupling_map=coupling_map, 
                                noise_model=noise_model)
        
        # print status update
        sim_qubits = self.sim_qubits(num_sim_q)[0]
        ancilla_string = ''
        num_sim_print = num_sim_q
        if self.ancilla:
            ancilla_string = ' + ancilla'
            num_sim_print += 1

        status = '*   Performing %i-qubit CS-VQE over qubit positions %s ...' % (num_sim_print, str(list(sim_qubits))[1:-1])
        print(status+ancilla_string)

        if ham is None:
            ham = self.ham_reduced[num_sim_q]

        if self.ancilla:
            dim = num_sim_q + 1
            input_ham={}
            for p in ham.keys():
                p_add_q = 'I' + p
                input_ham[p_add_q] = ham[p]
        else:
            dim = num_sim_q
            input_ham = ham
        
        # check expectation value of A (if 1 then in +1 eigenspace)
        # uses BFGS optimizer for speed (not accuracy!)
        vqe = VQE(qc, initial_point=init_anz_params, optimizer=L_BFGS_B(maxiter=1000), quantum_instance=qi)
        A_red = self.reduce_anz_terms(self.A, num_sim_q)
        A_red_q = qonvert.dict_to_WeightedPauliOperator(A_red)
        A_vqe_run = vqe.compute_minimum_eigenvalue(operator=A_red_q)
        A_expct = A_vqe_run.optimal_value
        if check_A:
            print('Expectation value of A:', A_expct)
        
        # simulate the input Hamiltonian

        vqe = VQE(qc, initial_point=init_anz_params, optimizer=optimizer, callback=self.store_intermediate_result, quantum_instance=qi)
        vqe_input_ham = qonvert.dict_to_WeightedPauliOperator(input_ham)
        gs_red = la.get_ground_state(vqe_input_ham.to_matrix())
        target_energy = gs_red[0]
        vqe_run = vqe.compute_minimum_eigenvalue(operator=vqe_input_ham)

        counts = deepcopy(self.counts)
        values = deepcopy(self.values)
        errors = deepcopy(self.errors)

        # compute target energy for projected hamiltonian
        ham_mat = np.matrix(vqe_input_ham.to_matrix())
        eig_mat = np.matrix(la.eigenstate_projector(A_red, dim))
        ham_proj = eig_mat*ham_mat*eig_mat.H
        proj_energy = la.get_ground_state(ham_proj)[0]

        return {'num_sim_q':num_sim_q,
                'sim_qubits':sim_qubits,
                'result':vqe_run.optimal_value,
                'target':target_energy,
                'projected_target':proj_energy,
                'A_expct':A_expct,
                'drop_pauli':drop_pauli,
                'counts':counts,
                'values':values,
                'errors':errors}


    def run_cs_vqe(self, anz_terms=None, max_sim_q=None, min_sim_q=0, optimizer=IMFIL(maxiter=10000), iters=1, check_A=False, noise=False):
        """
        """
        if max_sim_q is None:
            max_sim_q = self.num_qubits
        if max_sim_q>self.num_qubits:
            print('WARNING: specified maximum number of qubits to simulate exceeds total', '\n')
            max_sim_q = self.num_qubits

        rows, cols = la.factor_int(max_sim_q-min_sim_q)
        if rows == 1:
            grid_pos = list(range(cols))
        else:
            grid_pos = list(itertools.product(range(rows), range(cols)))

        self.cs_vqe_results = { 'rows':rows, 
                                'cols':cols,
                                'grid_pos':grid_pos,
                                'gs_noncon_energy':self.gs_noncon_energy, 
                                'true_gs':self.true_gs, 
                                'num_qubits':self.num_qubits,
                                'X_index':self.X_index}

        for index, grid in enumerate(grid_pos):
            num_sim_q = index+1+min_sim_q
            vqe_iters=[]
            for i in range(iters):
                vqe_run = self.CS_VQE(anz_terms, num_sim_q, optimizer=optimizer, check_A=check_A, noise=noise)
                vqe_iters.append(vqe_run)
                print('**  Contextual target:', round(vqe_run['target'], 15), '| VQE result:', round(vqe_run['result'], 15))
                error = vqe_run['result']-vqe_run['target']
                if error < 0.0016:
                    print('*** Succesfully converged on CS-VQE target energy')
                    break
                else:
                    print('*** Did not converge on contextual target | Error = ', round(error, 5))
            print(' ')
            vqe_iters = sorted(vqe_iters, key=lambda x:x['result'])
            self.cs_vqe_results[grid] = vqe_iters[0]
