import numpy as np
from utils.bit_tools import parity, int_to_bin


class eigenstate:
    """Class for constructing the n-th +1-eigenstate of A

    Attributes
    ----------
    A : dict
        Dictionary containg two items A = \{P_1:r_1, P_2:r_2\}}
    n : int
        The eigenstate index
    num_qubits : int
        The number of qubits in the Hamiltonian

    Methods
    -------
    P_index
        Indexes the qubit positions acted upon by each Pauli operator X, Y, Z
    t_val
        Calculates the eigenstate parameter t that satisfies the required amplitude constraint
    construct
        Constructs the eigenstate, stored as a numpy array
    """
    def __init__(self, A, n, num_qubits):
        self.A = A
        self.n = n
        self.num_qubits = num_qubits


    def P_index(self, q_pos=False):
        """Indexes the qubit positions acted upon by each Pauli operator X, Y, Z

        Parameters
        ----------
        q_pos: bool optional
            indices of qubits so compatible with qiskit

        Returns
        -------
        dict
            Dictionary of qubit indices acted upon by a Pauli P. Accessed via keys 'Pj' where j=1,2.
        """
        P_index={}
        for P in ['X', 'Y', 'Z']:
            for index, a in enumerate(self.A.keys()):
                index_key = '%s%i' % (P, index+1)
                offset = 0
                if q_pos:
                    offset = self.num_qubits-1
                P_index[index_key] = [abs(index-offset) for index, p in enumerate(list(a)) if p==P]

        
        return P_index


    def t_val(self, alt=False):
        """Calculates the eigenstate parameter t that satisfies the required amplitude constraint

        Returns
        -------
        float
            The eigenstate parameter t
        """
        r1 = list(self.A.values())[0]
        r2 = list(self.A.values())[1]
        P_index = self.P_index()
        init_state = int_to_bin(self.n, self.num_qubits)
        
        # compute the parities of relevant qubit subsets
        sgn  = (-1)**(parity(init_state, P_index['Z1'] + P_index['Y1']) + len(P_index['Y1'])/2) # check the initial minus
        # calculate the quotient constraint on +1-eigenstates of A
        quotient = r1 / (1 - r2*(-1)**(parity(init_state, P_index['Z2'])))
        alt_quotient = r1 / (1 - r2*(-1)**(1 + parity(init_state, P_index['Z2'])))
        # define t such that |psi_n> := sin(t)|b_n> + cos(t)|b_n'> is a +1-eigenstate of A
        t1 = sgn * np.arctan(quotient)
        t2 = sgn * np.arctan(alt_quotient)

        if alt:
            return t1, t2
        else:
            return t1

    
    def construct(self):
        """Constructs the eigenstate, stored as a numpy array

        Returns
        -------
        numpy.array
            Normalised +1-eigenstate of A, a column vector of 2**num_qubits elements
        """
        # initiate blank state as a list
        psi = [0 for i in range(2**self.num_qubits)]
        P_index = self.P_index()
        t = self.t_val()
        
        # binary representation of the eigenstate index
        init_state = int_to_bin(self.n, self.num_qubits)
        # determine the index of the basis vector that is paired with the initial state
        n_prime    = self.n + sum([((-1)**int(init_state[i])) * 2**(self.num_qubits-1 - i) for i in P_index['X1'] + P_index['Y1']])
        
        # corresponding entries in psi are set as necessary
        psi[self.n]  = np.sin(t)
        psi[n_prime] = np.cos(t)

        return np.array(psi)