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


class cs_vqe_circuit():
    """
    """
    # class variables for storing VQE results
    counts = []
    values = []
    
    def __init__(self, hamiltonian, terms_noncon, num_qubits, order, rot_G=True, rot_A=False):
        self.num_qubits = num_qubits
        self.order = deepcopy(order)
        self.rot_G = rot_G
        # epistricted model and reduced Hamiltonian
        cs = cs_vqe(hamiltonian, terms_noncon, num_qubits, rot_G, rot_A)
        generators = cs.generators()
        # defined here so only executed once instead of at each call
        ordertemp=[6,0,1,2,3,4,5,7]
        self.ham_reduced = cs.reduced_hamiltonian(ordertemp)
        self.gs_noncon_energy = cs.gs_noncon_energy()
        self.true_gs = cs.true_gs()[0]
        self.G = generators[0]
        self.A = generators[1]
        
        self.reference_qubits={}
        for g in self.G.keys():
            self.reference_qubits[num_qubits-1-g.find('Z')] = self.G[g]
        if rot_A:
            a = list(self.A.keys())[0]
            self.reference_qubits[num_qubits-1-a.find('Z')] = self.A[a]

        self.rotations = cs.rotations()
        self.init_state = cs.init_state()
        self.fixed_qubit = list(self.A.keys())[0].find('X')
        # +1-eigenstate parameters
        eig = eigenstate(self.A, bit.bin_to_int(self.init_state), num_qubits)
        self.index_paulis = eig.P_index(q_pos=True)
        if not rot_A:
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
        if self.rot_G:
            if type(anz_terms)!=dict:
                anz_terms = {t:1 for t in anz_terms}
            anz_terms = c_tools.rotate_operator(self.rotations, anz_terms)
        
        anz_terms_reduced = []
        for p in anz_terms.keys():
            blank_op = ['I' for i in range(num_sim_q)]
            for i, sim_i in enumerate(self.sim_qubits(num_sim_q)[1]):
                #if sim_i != 6:
                    blank_op[i] = p[sim_i]
            if set(blank_op) != {'I'}:
                anz_terms_reduced.append(''+''.join(blank_op))

        return anz_terms_reduced

    
    #def reduce_rotations(self):

    
    def build_circuit(self, anz_terms, num_sim_q, trot_order=2):
        """
        """
        anz_terms_reduced = self.reduce_anz_terms(anz_terms, num_sim_q)
        sim_qubits, sim_indices = self.sim_qubits(num_sim_q)
        index_paulis_reduced = self.index_paulis_reduced(num_sim_q)
        lost_parity = self.lost_parity(num_sim_q)
        qc = QuantumCircuit(num_sim_q)
        q1_pos = num_sim_q - 1 - (sim_qubits).index(1)
        print('*Performing CS-VQE over the following qubit positions:', sim_qubits)
        #Q = self.r1/(1+self.r2)
        #t2 = (1-self.r2)/self.r1        
   
        for q in sim_qubits:
            q_pos = num_sim_q - 1 - (sim_qubits).index(q)
            if self.reference_qubits[q] == -1:
                qc.x(q_pos)

        if set(anz_terms_reduced) == {'II'} or anz_terms_reduced == []:
            # because VQE requires at least one parameter...
            qc.rz(Parameter('a'), q1_pos)
        else:
            qc = circ.circ_from_paulis(paulis=list(set(anz_terms_reduced)), circ=qc, trot_order=trot_order, dup_param=False)
            #qc += TwoLocal(num_sim_q+1, 'ry', 'cx', 'linear', reps=2, insert_barriers=True)
            #qc.reset(num_sim_q)

        #qc.x(q1_pos)
        #qc.cx(q1_pos, num_sim_q)
        #qc.cry(2*np.arctan(1/Q), num_sim_q, q1_pos)
        #qc.x(num_sim_q)
        #qc.cry(2*np.arctan(-Q), num_sim_q, q1_pos)
        
        #qc.x(num_sim_q)
        #qc.cx(q1_pos, num_sim_q)
        
        #qc.reset(num_sim_q)
                
        #if lost_parity:
        #    qc.x(num_sim_q)
            
        #cascade_bits = []
        #for i in index_paulis_reduced:
        #    cascade_bits.append(num_sim_q - 1 - sim_qubits.index(i))

        ##store parity in ancilla qubit (8) via CNOT cascade
        #qc = circ.cascade(cascade_bits+[num_sim_q], circ=qc)

        #qc.cz(num_sim_q, q1_pos)
        
        ##reverse CNOT cascade
        #qc = circ.cascade(cascade_bits+[num_sim_q], circ=qc, reverse=True)

        #if lost_parity:
        #    qc.x(num_sim_q)

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
        
        print(qc.draw())
        return qc


    def store_intermediate_result(self, eval_count, parameters, mean, std):
        """
        """
        self.counts.append(eval_count)
        self.values.append(mean)


    def CS_VQE(self, anz_terms, num_sim_q, ham=None, optimizer=SLSQP(maxiter=10000)):
        """
        """
        self.counts.clear()
        self.values.clear()

        qi = QuantumInstance(Aer.get_backend('statevector_simulator'))
        qc = self.build_circuit(anz_terms, num_sim_q)
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