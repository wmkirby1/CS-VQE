import utils.cs_vqe as c
import utils.eigenstate_generator as eig
import utils.cs_vqe_ansatz as c_anz
import utils.qubit_conversion as qonvert
import numpy as np
import itertools

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
    init_state
        Determing the reference state of the ansatz
    """
    def __init__(self, ham, terms_noncon, num_qubits):
        assert(type(ham)==dict)
        self.ham = ham
        self.terms_noncon = terms_noncon
        self.num_qubits = num_qubits

    # required for the following methods - use get_ham to retrieve hamiltonians in practice
    def ham_noncon(self):
        """Return the noncontextual Hamiltnoian

        Returns
        -------
        dict
            Dictionary of noncontextual Hamiltnonian terms (Pauli strings) and corresponding coefficients
        """
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
            o_prob = c_anz.ontic_prob(ep, o_state)
            
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

    def rotations(self):
        """Determine necessary rotations such the commuting noncontextual generators consist of single Pauli Z

        Returns
        -------
        """
        return (c.diagonalize_epistemic(self.model(),self.fn_form(),self.ep_state()))[0]


    # get the noncontextual and contextual Hamiltonians
    def get_ham(self, h_type='full', rot=False):
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
        
        if rot:
            ham_ref = eig.rotate_operator(self.rotations(), ham_ref)    
        
        return ham_ref


    # get generators and observable A
    def generators(self, rot=False):
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

        if rot:
            G_list  = eig.rotate_operator(self.rotations(), G_list)
            A_obsrv = eig.rotate_operator(self.rotations(), A_obsrv)

        return G_list, A_obsrv


    def move_generator(self, rem_gen, rot=False):
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
        return eig.discard_generator(self.get_ham(h_type='noncon',rot=rot), self.get_ham(h_type='context',rot=rot), rem_gen)


    def reduced_hamiltonian(self, order=None, sim_qubits=None):
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

        ham_red = c.get_reduced_hamiltonians(self.ham,self.model(),self.fn_form(),self.ep_state(),order)
        
        if sim_qubits is None:
            return ham_red
        else:
            return ham_red[sim_qubits-1]


    # corresponds with the Hartree-Fock state
    def init_state(self, rot=True):
        """
        TODO - should work for non-rotated generators too
        """
        G = self.generators(rot)[0]
        zeroes = list(''.zfill(self.num_qubits))
        
        for g in G.keys():
            if G[g] == -1:
                Z_index = g.find('Z')
                zeroes[Z_index] = '1'

        return ''.join(zeroes)
