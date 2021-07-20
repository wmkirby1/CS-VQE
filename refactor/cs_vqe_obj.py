import utils.cs_vqe as c
import utils.eigenstate_generator as eig
import utils.cs_vqe_ansatz as c_anz
import utils.qubit_conversion as qonvert
import numpy as np

class cs_vqe:
    """
    Base class for CS-VQE
    """
    def __init__(self, ham, terms_noncon, num_qubits):
        assert(type(ham)==dict)
        self.ham = ham
        self.terms_noncon = terms_noncon
        self.num_qubits = num_qubits

    # required for the following methods - use get_ham to retrieve hamiltonians in practice
    def ham_noncon(self):
        return {t:self.ham[t] for t in self.terms_noncon}


    # noncontextual ground state parameter setting and energy
    def gs_noncon(self):
        return c.find_gs_noncon(self.ham_noncon())
    
    def gs_noncon_energy(self):
        return (self.gs_noncon())[0]

    def ep_state(self):
        return (self.gs_noncon())[1]


    # generate the epistricted model and generator rotations
    def model(self):
        return c.quasi_model(self.ham_noncon())

    def fn_form(self):
        return c.energy_function_form(self.ham_noncon(), self.model())

    def rotations(self):
        return (c.diagonalize_epistemic(self.model(),self.fn_form(),self.ep_state()))[0]


    # get the noncontextual and contextual Hamiltonians
    def get_ham(self, h_type='full', rot=False):
        """
        - returns the full, noncontextual or contextual Hamiltonian
          given respectively by h_type = 'full', 'noncon', 'context'
        - Can also speicfy whether it should be rotated
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
        """
        returns (dict(G), dict(A_obsrv))
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
        """
        returns tuple (new_ham_noncon, new_ham_context)
        """
        return eig.discard_generator(self.get_ham(h_type='noncon',rot=rot), self.get_ham(h_type='context',rot=rot), rem_gen)


    def reduced_hamiltonian(self, order=None, sim_qubits=None):
        if order is None:
            order = list(range(self.num_qubits))

        ham_red = c.get_reduced_hamiltonians(self.ham,self.model(),self.fn_form(),self.ep_state(),order)
        
        if sim_qubits is None:
            return ham_red
        else:
            return ham_red[sim_qubits-1]


    # ************Separate everything below into child class***********
    # *often* corresponds with the Hartree-Fock state
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

    # parameter that guarantees correct amplitudes for +1-eigenstates
    def t_val(self):
        ep = self.ep_state()
        r1 = ep[1][0]
        r2 = ep[1][1]
        amp_ratio = (1 + r2) / (r1)
        return np.arctan(amp_ratio)