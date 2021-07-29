from cs_vqe_classes.cs_vqe import cs_vqe
from cs_vqe_classes.eigenstate import eigenstate
import utils.bit_tools as bit
import utils.cs_vqe_tools as c_tools
import utils.circuit_tools as circ
import utils.qonversion_tools as qonvert
from copy import deepcopy
import numpy as np

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
    
    def __init__(self, hamiltonian, terms_noncon, num_qubits, order, rot=True):
        self.num_qubits = num_qubits
        self.order = deepcopy(order)
        self.rot = rot
        # epistricted model and reduced Hamiltonian
        cs = cs_vqe(hamiltonian, terms_noncon, num_qubits)
        generators = cs.generators(rot=True)
        # defined here so only executed once instead of at each call
        ordertemp=[6,0,1,2,3,4,5,7]
        self.ham_reduced = cs.reduced_hamiltonian(ordertemp)
        self.gs_noncon_energy = cs.gs_noncon_energy()
        self.true_gs = cs.true_gs()[0]
        self.G = generators[0]
        self.A = generators[1]
        self.rotations = cs.rotations()
        self.init_state = cs.init_state(rot=rot)
        self.fixed_qubit = list(self.A.keys())[0].find('X')
        # +1-eigenstate parameters
        eig = eigenstate(self.A, bit.bin_to_int(self.init_state), num_qubits)
        self.index_paulis = eig.P_index(q_pos=True)
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
        if self.rot:
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
                anz_terms_reduced.append('I'+''.join(blank_op))

        return anz_terms_reduced

    
    def build_circuit(self, anz_terms, num_sim_q, trot_order=2):
        """
        """
        anz_terms_reduced = self.reduce_anz_terms(anz_terms, num_sim_q)
        sim_qubits, sim_indices = self.sim_qubits(num_sim_q)
        index_paulis_reduced = self.index_paulis_reduced(num_sim_q)
        lost_parity = self.lost_parity(num_sim_q)
        qc = QuantumCircuit(num_sim_q+1)
        q1_pos = num_sim_q - 1 - (sim_qubits).index(1)
        print('*Performing CS-VQE over the following qubit positions:', sim_qubits)
        Q = self.r1/(1+self.r2)
        #t2 = (1-self.r2)/self.r1

        #for i in sim_qubits:
        #    if i in [0,1,2,3]:
        #        qc.x(num_sim_q - 1 - (sim_qubits).index(i))
        
        # only applies for HeH+, needs to be generalised for init_state
        #try:
        #    q4_pos = num_sim_q - 1 - (sim_qubits).index(4)
        #    qc.x(q4_pos)
        #except:
        #    print('q4 not simulated')

        if set(anz_terms_reduced) == {'II'} or anz_terms_reduced == []:
            # because VQE requires at least one parameter...
            qc.rz(Parameter('a'), q1_pos)
        else:
            #qc = circ.circ_from_paulis(paulis=list(set(anz_terms_reduced)), circ=qc, trot_order=trot_order, dup_param=False)
            qc += TwoLocal(num_sim_q+1, 'ry', 'cx', 'linear', reps=2, insert_barriers=True)
            #qc.reset(num_sim_q)

        qc.cx(q1_pos, num_sim_q)
        qc.cry(2*np.arctan(1/Q), num_sim_q, q1_pos)
        qc.x(num_sim_q)
        qc.cry(2*np.arctan(-Q), num_sim_q, q1_pos)
        #qc.x(num_sim_q)
        #qc.cx(q1_pos, num_sim_q)
        
        qc.reset(num_sim_q)
                
        if lost_parity:
            qc.x(num_sim_q)
            
        cascade_bits = []
        for i in index_paulis_reduced:
            cascade_bits.append(num_sim_q - 1 - sim_qubits.index(i))

        #store parity in ancilla qubit (8) via CNOT cascade
        qc = circ.cascade(cascade_bits+[num_sim_q], circ=qc)

        qc.cz(num_sim_q, q1_pos)
        
        #reverse CNOT cascade
        qc = circ.cascade(cascade_bits+[num_sim_q], circ=qc, reverse=True)

        if lost_parity:
            qc.x(num_sim_q)

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
            p_add_q = 'I' + p
            h_add_anc[p_add_q] = ham[p]

        vqe_input_ham = qonvert.dict_to_WeightedPauliOperator(h_add_anc)
        ham_red_q = qonvert.dict_to_QubitOperator(h_add_anc)
        gs_red = get_ground_state(get_sparse_operator(ham_red_q, num_sim_q+1).toarray())
        target_energy = gs_red[0]

        vqe_run = vqe.compute_minimum_eigenvalue(operator=vqe_input_ham)

        return vqe_run.optimal_value, target_energy, self.counts, self.values

        