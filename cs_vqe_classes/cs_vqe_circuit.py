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

from qiskit.circuit.parameter import Parameter
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit.utils import QuantumInstance
from qiskit.aqua.components.optimizers import SLSQP, COBYLA, SPSA, AQGD
from qiskit.algorithms import VQE
from qiskit import Aer
from openfermion.linalg import get_sparse_operator, get_ground_state

import numpy as np
import pylab

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
device_backend = FakeVigo()


class cs_vqe_circuit():
    """
    """
    # class variables for storing VQE results
    cs_vqe_results = {}
    counts = []
    values = []
    # if True adds additional qubit to circuit
    ancilla = False
    
    def __init__(self, hamiltonian, terms_noncon, num_qubits, order=None, rot_G=True, rot_A=False):
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
        else:
            ham_q = qonvert.dict_to_QubitOperator(input_ham)
            n_q = len(list(input_ham.keys())[0])
            gs = get_ground_state(get_sparse_operator(ham_q, n_q).toarray())
            gs_vec = gs[1]
            amp_list = [abs(a)**2 for a in list(gs_vec)]
            sig_amp_list = sorted([(str(index), a) for index, a in enumerate(amp_list) if a > 0.001], key=lambda x:x[1])
            sig_amp_list.reverse()
            reference_str=bit.int_to_bin(int(sig_amp_list[0][0]), n_q)
            for index, b in enumerate(reference_str):
                reference_bits[n_q-1-index] = int(b)

        return reference_bits

    
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
            
            if set(blank_op) != {'I'}:
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
    

    ################################## circuit blocks #################################
    ## Below are circuit components that may be selected in the build_circuit method ##
    ###################################################################################

    def ref_state_block(self, qc, num_sim_q):
        """
        """
        q_map = self.qubit_map(num_sim_q)
        sim_qubits = self.sim_qubits(num_sim_q)[0]
        reference_state = self.reference_state(self.hamiltonian)
        
        for q in sim_qubits:
            if reference_state[q] == 1:
                qc.x(q_map[q])


    def anz_block(self, anz_terms, qc, num_sim_q):
        anz_terms_reduced = self.reduce_anz_terms(anz_terms, num_sim_q)
        q_map = self.qubit_map(num_sim_q)
        
        if anz_terms_reduced == {}:
            # because VQE requires at least one parameter...
            qc.rz(Parameter('a'), q_map[self.X_qubit])
        else:
            qc = circ.circ_from_paulis(paulis=list(anz_terms_reduced.keys()), circ=qc, trot_order=2, dup_param=False)
            #qc += TwoLocal(num_sim_q, 'ry', 'cx', 'full', reps=2, insert_barriers=True)
            #qc.reset(num_sim_q)


    def swap_entgl_block(self, qc, num_sim_q):
        """requires ancilla
        """
        assert(self.ancilla == True)

        q_map = self.qubit_map(num_sim_q)
        qc.cx(q_map[self.X_qubit], num_sim_q)
        qc.cx(num_sim_q, q_map[self.X_qubit])


    def rot_ham_block(self, qc, num_sim_q, inverse=False):
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

    def build_circuit(self, anz_terms, num_sim_q, trot_order=2):
        """
        """
        self.ancilla = True

        q_map = self.qubit_map(num_sim_q)
        sim_qubits = self.sim_qubits(num_sim_q)[0]

        if self.ancilla:
            dim = num_sim_q + 1
        else:
            dim = num_sim_q

        qc = QuantumCircuit(dim)
        
        self.ref_state_block(qc, num_sim_q)
        self.anz_block(anz_terms, qc, num_sim_q)
        self.rot_ham_block(qc, num_sim_q)
        self.rot_A_block(qc, num_sim_q, inverse=True)
        
        #qc.reset(q_map[self.X_qubit])
        
        self.swap_entgl_block(qc, num_sim_q)
        qc.x(q_map[self.X_qubit])

        self.rot_A_block(qc, num_sim_q)
        
        #self.A_eig_block(qc, num_sim_q)
        #self.parity_cascade_block(qc, self.index_paulis['Z1'], num_sim_q)
        #qc.reset(num_sim_q)

        #print(qc.draw())
        return qc


    def store_intermediate_result(self, eval_count, parameters, mean, std):
        """
        """
        self.counts.append(eval_count)
        self.values.append(mean)


    def CS_VQE(self, anz_terms, num_sim_q, ham=None, optimizer=SLSQP(maxiter=500), check_A=False):
        """
        """
        self.counts.clear()
        self.values.clear()

        init_anz_params = list(-p.imag for p in self.reduce_anz_terms(anz_terms, num_sim_q).values())
        if init_anz_params == []:
            init_anz_params = [0]

        #device_backend = FakeVigo()
        backend = Aer.get_backend('statevector_simulator')
        seed = 50
        #noise_model = None
        #os.environ['QISKIT_IN_PARALLEL'] = 'TRUE'
        #device = QasmSimulator.from_backend(device_backend)
        #coupling_map = device.configuration().coupling_map
        #noise_model = NoiseModel.from_backend(device)
        #basis_gates = noise_model.basis_gates

        #algorithm_globals.random_seed = seed
        qi = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed)
                            #coupling_map=coupling_map, noise_model=noise_model)
        
        qc = self.build_circuit(anz_terms, num_sim_q)

        # print status update
        sim_qubits = self.sim_qubits(num_sim_q)[0]
        ancilla_string = ''
        num_sim_print = num_sim_q
        if self.ancilla:
            ancilla_string = ' + ancilla'
            num_sim_print += 1

        status = '*Performing %i-qubit CS-VQE over qubit positions %s' % (num_sim_print, str(list(sim_qubits))[1:-1])
        print(status+ancilla_string)

        if ham is None:
            ham = self.ham_reduced[num_sim_q-1]

        if self.ancilla:
            dim = num_sim_q + 1
            input_ham={}
            for p in ham.keys():
                p_add_q = 'I' + p
                input_ham[p_add_q] = ham[p]
        else:
            dim = num_sim_q
            input_ham = ham
        
        A_red = self.reduce_anz_terms(self.A, num_sim_q)
        
        # check expectation value of A (if 1 then in +1 eigenspace)
        vqe = VQE(qc, initial_point=init_anz_params, optimizer=optimizer, quantum_instance=qi)
        A_red_q = qonvert.dict_to_WeightedPauliOperator(A_red)
        A_vqe_run = vqe.compute_minimum_eigenvalue(operator=A_red_q)
        A_expct = A_vqe_run.optimal_value
        print('Expectation value of A:', A_expct)
        
        # simulate the input Hamiltonian
        vqe = VQE(qc, initial_point=init_anz_params, optimizer=optimizer, callback=self.store_intermediate_result, quantum_instance=qi)
        vqe_input_ham = qonvert.dict_to_WeightedPauliOperator(input_ham)
        input_ham_q = qonvert.dict_to_QubitOperator(input_ham)
        gs_red = get_ground_state(get_sparse_operator(input_ham_q, dim).toarray())
        target_energy = gs_red[0]
        vqe_run = vqe.compute_minimum_eigenvalue(operator=vqe_input_ham)

        counts = deepcopy(self.counts)
        values = deepcopy(self.values)

        # compute target energy for projected hamiltonian
        ham_mat = np.matrix(get_sparse_operator(input_ham_q, dim).toarray())
        eig_mat = np.matrix(la.eigenstate_projector(A_red, dim))
        ham_proj = eig_mat*ham_mat*eig_mat.H
        proj_energy = get_ground_state(ham_proj)[0]

        return {'num_sim_q':num_sim_q,
                'result':vqe_run.optimal_value,
                'target':target_energy,
                'projected_target':proj_energy,
                'A_expct':A_expct,
                'counts':counts,
                'values':values}


    def run_cs_vqe(self, anz_terms, max_sim_q, min_sim_q=0, iters=1, check_A=False):
        """
        """
        if max_sim_q>self.num_qubits:
            print('*** specified maximum number of qubits to simulate exceeds total ***')
            max_sim_q = self.num_qubits

        rows, cols = la.factor_int(max_sim_q-min_sim_q)
        if rows == 1:
            grid_pos = range(cols)
        else:
            grid_pos = list(itertools.product(range(rows), range(cols)))

        self.cs_vqe_results = {'rows':rows, 
                                'cols':cols,
                                'grid_pos':grid_pos,
                                'gs_noncon_energy':self.gs_noncon_energy, 
                                'true_gs':self.true_gs, 
                                'num_qubits':self.num_qubits}

        for index, grid in enumerate(grid_pos):
            num_sim_q = index+1+min_sim_q
            vqe_iters=[]
            for i in range(iters):
                vqe_run = self.CS_VQE(anz_terms, num_sim_q, check_A=check_A)
                vqe_iters.append(vqe_run)
                if vqe_run['result']-vqe_run['projected_target'] < 1e-3:
                    print('Reached target energy in fewer iterations than specified')
                    break
            vqe_iters = sorted(vqe_iters, key=lambda x:x['result'])
            self.cs_vqe_results[grid] = vqe_iters[0]