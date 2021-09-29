import utils.cs_vqe_tools as c
import utils.qonversion_tools as qonvert
#from openfermion.linalg import get_ground_state
import utils.linalg_tools as la
from copy import deepcopy
import numpy as np
import itertools
import matplotlib.pyplot as plt

class cs_vqe:
    """Class for constructing instances of CS-VQE Hamiltonians

    Attributes
    ----------
    ham : dict
        Dictionary of Hamiltnonian terms (Pauli strings) and corresponding coefficients
    terms_noncon : list
        Noncontextual subset of Hamiltonian terms
    num_qubits : int
        The number of qubits in the Hamiltonian

    Methods
    -------
    ham_noncon
        Return the noncontextual Hamiltnoian
    gs_noncon
        Noncontextual ground state parameter setting and energy
    gs_noncon_energy
        Noncontextual ground state energy
    ep_state
        Noncontextual ground state parameter setting
    ep_dist
        Probability distribution corresponding with the epistemic state
    model
        generate the epistricted model
    fn_form

    rotations
        Determine necessary rotations such the commuting noncontextual generators consist of single Pauli Z
    get_ham
        Retrieve full, noncontextual or contextual Hamiltonian with or without rotations applied
    generators
        Retrieve commuting noncontextual generators and observable A with or without rotations applied
    move_generator
        Manually remove generators and all operators in their support from noncontextual to contextual Hamiltonian
    reduced_hamiltonian
        Generate the contextual subspace Hamiltonians for a given number of qubits
    true_gs
        Obtain true ground state via linear algebra (inefficient)
    true_gs_hist
        Plot histogram of basis state probability weightings in true ground state
    init_state
        Determing the reference state of the ansatz
    """
    def __init__(self, ham, terms_noncon, num_qubits, rot_G=True, rot_A=False):
        assert(type(ham)==dict)
        self.ham = ham
        self.terms_noncon = terms_noncon
        self.num_qubits = num_qubits
        self.rot_A = rot_A
        self.rot_G = rot_G


    # required for the following methods - use get_ham to retrieve hamiltonians in practice
    def ham_noncon(self):
        """Return the noncontextual Hamiltnoian

        Returns
        -------
        dict
            Dictionary of noncontextual Hamiltnonian terms (Pauli strings) and corresponding coefficients
        """
        print(self.terms_noncon)
        return {t:self.ham[t] for t in self.terms_noncon}


    def gs_noncon(self):
        """Noncontextual ground state parameter setting and energy

        Returns
        -------
        list
        """
        return c.find_gs_noncon(self.ham_noncon())
    

    def gs_noncon_energy(self):
        """Noncontextual ground state energy
        """
        return (self.gs_noncon())[0]


    def ep_state(self):
        """Noncontextual ground state parameter setting
        """
        return (self.gs_noncon())[1]


    def ep_dist(self):
        """Probability distribution corresponding with the epistemic state

        Returns
        -------
        """
        ep = self.ep_state()
        size_G = len(ep[0])
        size_Ci = len(ep[1])
        size_R = size_G + size_Ci
        
        ep_prob = {}
        
        ontic_states = list(itertools.product([1, -1], repeat=size_R))
        
        for o in ontic_states:
            o_state = [list(o[0:size_G]), list(o[size_G:size_R])]
            o_prob = c.ontic_prob(ep, o_state)
            
            if o_prob != 0:
                ep_prob[o] = o_prob
        
        return ep_prob


    def model(self):
        """generate the epistricted model

        Returns
        -------
        """
        return c.quasi_model(self.ham_noncon())


    def fn_form(self):
        """

        Returns
        -------
        """
        return c.energy_function_form(self.ham_noncon(), self.model())


    def rotations(self, rot_override=False):
        """Determine necessary rotations such the commuting noncontextual generators consist of single Pauli Z

        Returns
        -------
        """
        if not rot_override:
            return (c.diagonalize_epistemic(self.model(),self.fn_form(),self.ep_state(),rot_A=self.rot_A))[0]
        else:
            return (c.diagonalize_epistemic(self.model(),self.fn_form(),self.ep_state(),rot_A=False))[0]

    # get the noncontextual and contextual Hamiltonians
    def get_ham(self, h_type='full'):
        """Retrieve full, noncontextual or contextual Hamiltonian with or without rotations applied
        
        Paramters
        ---------
        h_type: str optional
            allowed value are 'full', 'noncon', 'context' for corresponding Hamiltonian to be returned
        rot: bool optional
            Specifies either unrotated or rotated Hamiltonian

        Returns
        -------
        dict
            returns the full, noncontextual or contextual Hamiltonian specified by h_type
        """
        if h_type == 'full':
            ham_ref = self.ham
        elif h_type == 'noncon':
            ham_ref = {t:self.ham[t] for t in self.terms_noncon}
        elif h_type == 'context':
            ham_ref = {t:self.ham[t] for t in self.ham.keys() if t not in self.terms_noncon}
        else:
            raise ValueError('Invalid value given for h_type: must be full, noncon or context')
        
        if self.rot_G:
            ham_ref = c.rotate_operator(self.rotations(), ham_ref)

        return ham_ref


    # get generators and observable A
    def generators(self):
        """Retrieve commuting noncontextual generators and observable A with or without rotations applied
        
        Paramters
        ---------
        rot: bool optional
            Specifies either unrotated or rotated Hamiltonian

        Returns
        -------
        set
            Generators and observable A in form (dict(G), dict(A_obsrv))
        """
        ep  = self.ep_state()
        mod = self.model()

        G_list  = {g:ep[0][index] for index, g in enumerate(mod[0])}
        A_obsrv = {Ci1:ep[1][index] for index, Ci1 in enumerate(mod[1])}

        if self.rot_G:
            G_list  = c.rotate_operator(self.rotations(), G_list)
            A_obsrv = c.rotate_operator(self.rotations(), A_obsrv)

        return G_list, A_obsrv


    def move_generator(self, rem_gen):
        """Manually remove generators and all operators in their support from noncontextual to contextual Hamiltonian
        
        Paramters
        ---------
        rem_gen: list
            list of generators (Paulis strings) to remove from noncontextual Hamiltonian
        rot: bool optional
            Specifies either unrotated or rotated Hamiltonian

        Returns
        -------
        set 
            In form (new_ham_noncon, new_ham_context)
        """
        return c.discard_generator(self.get_ham(h_type='noncon'), self.get_ham(h_type='context'), rem_gen)


    def reduced_hamiltonian(self, order=None, num_sim_q=None):
        """Generate the contextual subspace Hamiltonians for a given number of qubits
        
        Parameters
        ----------
        order: list optional
            list of integers specifying order in which to remove qubits
        sim_qubits : int optional
            number of qubits in final Hamiltonian

        Returns
        -------
        dict
            reduced Hamiltonian for number of qubits specified. Returns all if sim_qubits==None.
        """
        if order is None:
            order = list(range(self.num_qubits))
        order_ref = deepcopy(order)
        
        ham_red = c.get_reduced_hamiltonians(self.ham,self.model(),self.fn_form(),self.ep_state(),order_ref,self.rot_A)
        
        if num_sim_q is None:
            return ham_red
        else:
            return ham_red[num_sim_q-1]


    def true_gs(self, rot_override=False):
        """Obtain true ground state via linear algebra (inefficient)

        Parameters
        ----------
        rot: bool optional
            Specifies either unrotated or rotated Hamiltonian

        Returns
        -------
        list
            (true gs energy, true gs eigenvector)
        """
        if not rot_override:
            #ham_q = qonvert.dict_to_QubitOperator(self.get_ham())
            ham_mat = qonvert.dict_to_WeightedPauliOperator(self.get_ham()).to_matrix()

        else:
            #ham_q = qonvert.dict_to_QubitOperator(self.ham)
            ham_mat = qonvert.dict_to_WeightedPauliOperator(self.ham).to_matrix()
            
        gs = la.get_ground_state(ham_mat)

        return gs


    def true_gs_hist(self, threshold, rot_override=False):
        """Plot histogram of basis state probability weightings in true ground state

        Parameters
        ----------
        threshold: float
            minimum probability threshold to include in plot, i.e. 1e-n
        rot: bool optional
            Specifies either unrotated or rotated Hamiltonian

        Returns
        -------
        Figure
            histogram of probabilities
        """
        gs_vec = (self.true_gs(rot_override))[1]

        amp_list = [abs(a)**2 for a in list(gs_vec)]
        sig_amp_list = sorted([(str(index), a) for index, a in enumerate(amp_list) if a > threshold], key=lambda x:x[1])
        sig_amp_list.reverse()

        XY = list(zip(*sig_amp_list))
        X = XY[0]
        Y = XY[1]
        Y_log = [np.log10(a) for a in Y]

        fig = plt.figure(figsize=(15, 6), dpi=300)

        plt.grid(zorder=0)
        plt.bar(X, Y, zorder=2, label='Probability of observing basis state')
        plt.bar(X, Y_log, zorder=3, label = 'log (base 10) of probability')
        plt.xticks(rotation=90)
        plt.title('Probability weighting of basis states in the true ground state (above %s)' % str(threshold))
        plt.xlabel('Basis state index')
        plt.legend()

        return fig


    # corresponds with the Hartree-Fock state
    def init_state(self):
        """
        TODO - should work for non-rotated generators too
        """
        G = self.generators()[0]
        zeroes = list(''.zfill(self.num_qubits))
        
        for g in G.keys():
            if G[g] == -1:
                Z_index = g.find('Z')
                zeroes[Z_index] = '1'

        return ''.join(zeroes)
