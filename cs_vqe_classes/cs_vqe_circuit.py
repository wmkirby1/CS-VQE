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
    counts = []
    values = []
    
    def __init__(self, hamiltonian, terms_noncon, num_qubits, order, rot_G=True, rot_A=False, rot_override=False):
        self.hamiltonian = hamiltonian
        self.num_qubits = num_qubits
        self.order = deepcopy(order)
        self.rot_G = rot_G
        self.rot_A = rot_A

        # epistricted model and reduced Hamiltonian
        cs = cs_vqe(hamiltonian, terms_noncon, num_qubits, rot_G, rot_A)
        self.all_rotations = cs.rotations()
        if rot_A:
            self.G_rotations = cs.rotations(rot_override=True)
            self.A_rotations = [r for r in self.all_rotations if r not in self.G_rotations]
        
        if rot_override:
            cs = cs_vqe(hamiltonian, terms_noncon, num_qubits, rot_G, rot_A=False)

        generators = cs.generators()
        # defined here so only executed once instead of at each call
        self.gs_noncon_energy = cs.gs_noncon_energy()
        self.true_gs = cs.true_gs()[0]
        self.G = generators[0]
        self.A = generators[1]

        ordertemp=[6,0,1,2,3,4,5,7]
        self.ham_reduced = cs.reduced_hamiltonian(ordertemp)

        # +1-eigenstate parameters
        self.init_state = cs.init_state()
        eig = eigenstate(self.A, bit.bin_to_int(self.init_state), num_qubits)
        self.index_paulis = eig.P_index(q_pos=True)
        if rot_override:
            self.t1, self.t2 = eig.t_val(alt=True)
            self.r1, self.r2 = self.A.values()


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


    def index_paulis_reduced(self, num_sim_q):
        """
        """
        P_indices = [q for q in self.index_paulis['Z1'] if q in self.sim_qubits(num_sim_q)[0]]
        return P_indices 

    
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


    def lost_parity(self, num_sim_q):
        """
        """
        lost_parity = 0
        for i in self.sim_qubits(num_sim_q, complement=True)[1]:
            lost_parity += int(self.init_state[i])

        return lost_parity%2

    
    def reduce_anz_terms(self, anz_terms, num_sim_q):
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
                #if sim_i != 6:
                    blank_op[i] = p[sim_i]
            
            if set(blank_op) != {'I'}:
                t = ''+''.join(blank_op)
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
                rot_red.append((''+''.join(blank_op), r))

        return rot_red
                
    
    def build_circuit(self, anz_terms, num_sim_q, trot_order=2):
        """
        """
        anz_terms_reduced = self.reduce_anz_terms(anz_terms, num_sim_q)
        sim_qubits, sim_indices = self.sim_qubits(num_sim_q)
        reference_state = self.reference_state(self.hamiltonian)
        q_map = self.qubit_map(num_sim_q)
        print('*Performing CS-VQE over the following qubit positions:', sim_qubits)
        
        qc = QuantumCircuit(num_sim_q)
        Q = self.r1/(1+self.r2)     
           
        for q in sim_qubits:
            if reference_state[q] == 1:
                qc.x(q_map[q])

        #G_rot = self.reduce_rotations(self.G_rotations, num_sim_q)       
        #G_rot.reverse()
        #for p, r in G_rot:
        #    qc = circ.exp_P(p, circ=qc, rot=-r)
        
        if anz_terms_reduced == {}:
            # because VQE requires at least one parameter...
            qc.rz(Parameter('a'), q_map[1])
        else:
            qc = circ.circ_from_paulis(paulis=list(anz_terms_reduced.keys()), circ=qc, trot_order=trot_order, dup_param=False)
            #qc += TwoLocal(num_sim_q, 'ry', 'cx', 'full', reps=2, insert_barriers=True)
            #qc.reset(num_sim_q)

        
        #qc.cx(q_map[1], num_sim_q)
        #qc.cry(2*np.arctan(1/Q), num_sim_q, q_map[1])
        #qc.x(num_sim_q)
        #qc.cry(2*np.arctan(-Q), num_sim_q, q_map[1])

        #index_paulis_reduced = self.index_paulis_reduced(num_sim_q)
        #lost_parity = self.lost_parity(num_sim_q)
        #if lost_parity:
        #    qc.x(num_sim_q)
            
        #cascade_bits = []
        #for i in index_paulis_reduced:
        #    cascade_bits.append(num_sim_q - 1 - sim_qubits.index(i))

        ##store parity in ancilla qubit (8) via CNOT cascade
        #qc = circ.cascade(cascade_bits+[num_sim_q], circ=qc)

        #qc.cz(num_sim_q, q_map[1])
        
        ##reverse CNOT cascade
        #qc = circ.cascade(cascade_bits+[num_sim_q], circ=qc, reverse=True)

        #if lost_parity:
        #    qc.x(num_sim_q)

        #A_rot = self.reduce_rotations(self.A_rotations, num_sim_q)
        #for p, r in A_rot:
        #    qc = circ.exp_P(p, circ=qc, rot=r)

        #qc.x(num_sim_q)
        #qc.cx(q_map[1], num_sim_q)
        #qc.cx(num_sim_q, q_map[1])

        for i in range(num_sim_q-1):
            qc.cx(i, i+1)
        for i in range(num_sim_q-2):
            qc.cx(num_sim_q-1-i, num_sim_q-2-i)

        if num_sim_q > 1:
            qc.cx(q_map[0], q_map[1])
        
        #qc.x(q_map[1])

        #A_rot.reverse()
        #for p, r in A_rot:
        #    qc = circ.exp_P(p, circ=qc, rot=-r)
        A_rot = self.reduce_rotations([[-0.04018297694234569, 'IZZZZZYI']], num_sim_q)
        for p, r in A_rot:
            qc = circ.exp_P(p, circ=qc, rot=r)

        #qc.rx(Parameter('b'), q_map[1])
        #for q in sim_qubits:
        #    try:
        #        if reference_state[q] == -1:
        #            qc.x(q_map[q])
        #    except:
        #        pass

        #qc.x(q_map[1])
        
        #qc.x(num_sim_q)
        #qc.cx(q_map[1], num_sim_q)
        
        #qc.reset(num_sim_q)

        #qc.cx(num_sim_q, q1_pos)

        #qc.reset(num_sim_q)
        #qc.reset(num_sim_q)

        #r1 = list(self.A.values())[0]
        #r2 = list(self.A.values())[1]
        #A_terms_reduced=[]
        #for p in self.A.keys():
        #    blank_op = ['I' for i in range(num_sim_q)]
        #    for i, sim_i in enumerate(sim_indices):
        #        blank_op[i] = p[sim_i]
        #    A_terms_reduced.append('I'+''.join(blank_op))
        #print(A_terms_reduced)
        #qc = circ.circ_from_paulis(paulis=A_terms_reduced, params=[np.pi*r1/4, np.pi*r2/4], circ=qc, trot_order=4)
        
        #print(qc.draw())
        return qc


    def store_intermediate_result(self, eval_count, parameters, mean, std):
        """
        """
        self.counts.append(eval_count)
        self.values.append(mean)


    def CS_VQE(self, anz_terms, num_sim_q, ham=None, optimizer=SPSA(maxiter=200)):
        """
        """
        self.counts.clear()
        self.values.clear()
        
        init_anz_params = list(-p.imag for p in self.reduce_anz_terms(anz_terms, num_sim_q).values())
        if init_anz_params == []:
            init_anz_params = [0]

        #qi = QuantumInstance(Aer.get_backend('unitary_simulator'))

        backend = Aer.get_backend('aer_simulator')
        seed = 170
        noise_model = None
        #device = QasmSimulator.from_backend(device_backend)
        #coupling_map = device.configuration().coupling_map
        #noise_model = NoiseModel.from_backend(device)
        #basis_gates = noise_model.basis_gates

        #print(noise_model)
        #print()

        algorithm_globals.random_seed = seed
        qi = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed)
                            #coupling_map=coupling_map, noise_model=noise_model,)
        
        qc = self.build_circuit(anz_terms, num_sim_q)

        #vqe = VQE(qc, optimizer=optimizer, initial_point=init_anz_params, callback=self.store_intermediate_result, quantum_instance=qi)
        vqe = VQE(qc, optimizer=optimizer, callback=self.store_intermediate_result, quantum_instance=qi)
        if ham is None:
            ham = self.ham_reduced[num_sim_q-1]

        h_add_anc={}
        for p in ham.keys():
            p_add_q = '' + p
            h_add_anc[p_add_q] = ham[p]
        #print(h_add_anc)
        vqe_input_ham = qonvert.dict_to_WeightedPauliOperator(h_add_anc)
        ham_red_q = qonvert.dict_to_QubitOperator(h_add_anc)
        gs_red = get_ground_state(get_sparse_operator(ham_red_q, num_sim_q).toarray())
        target_energy = gs_red[0]

        vqe_run = vqe.compute_minimum_eigenvalue(operator=vqe_input_ham)

        counts = deepcopy(self.counts)
        values = deepcopy(self.values)

        return {'num_sim_q':num_sim_q,
                'result':vqe_run.optimal_value,
                'target':target_energy,
                'counts':counts,
                'values':values}


    def run_cs_vqe(self, anz_terms, max_sim_q, min_sim_q=0, iters=1):
        """
        """
        rows, cols = la.factor_int(max_sim_q-min_sim_q)
        if rows == 1:
            grid_pos = range(cols)
        else:
            grid_pos = list(itertools.product(range(rows), range(cols)))

        cs_vqe_results = {'rows':rows, 
                        'cols':cols,
                        'grid_pos':grid_pos,
                        'gs_noncon_energy':self.gs_noncon_energy, 
                        'true_gs':self.true_gs, 
                        'num_qubits':self.num_qubits}

        for index, grid in enumerate(grid_pos):
            num_sim_q = index+1+min_sim_q
            vqe_iters=[]
            for i in range(iters):
                vqe_iters.append(self.CS_VQE(anz_terms, num_sim_q))
            vqe_iters = sorted(vqe_iters, key=lambda x:x['result'])
            cs_vqe_results[grid] = vqe_iters[0]
        
        return cs_vqe_results