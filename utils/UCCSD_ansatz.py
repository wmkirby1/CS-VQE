from openfermion.ops import FermionOperator
from openfermion.transforms import bravyi_kitaev
from openfermion.transforms import jordan_wigner

def Get_UCCSD_ia_terms(n_electrons, n_orbitals):
    """
    Get ia excitation terms as fermionic creation and annihilation operators for UCCSD.
    ia terms are standard single excitation terms (aka only occupied -> unoccupied transitions allowed)

    eqn:  T1 = ∑_{i ∈ occupied} ∑_{A ∈ unoccupied}  t_{A}^{i} a†_{A} a_{i}

    
    #TODO can add method to get pqrs terms
    #TODO these are all non-degenerate excitations which can possibly non-zero, including nocc->nocc, occ->occ, and spin-flips.
    #TODO EXPENSIVE, but will likely  get a slightly better answer.
    
    Args:
        n_electrons (int): number of electrons
        n_orbitals (int): number of orbitals

    returns:
        Sec_Quant_CC_ia_ops (list): list of FermionOperators (openfermion.ops._fermion_operator.FermionOperator)
        theta_parameters (list): list of theta values (parameterisation of excitation amplitudes)

     ** Example **

     n_electrons=2
     n_orbitals=4
     Sec_Quant_CC_ops, theta_parameters = Get_ia_terms(n_electrons, n_orbitals)

     Sec_Quant_CC_ops=  [
                         -1.0[0 ^ 2] + 1.0[2 ^ 0],            # -(a†0 a2) + (a†2 a0)
                         -1.0[1 ^ 3] + 1.0[3 ^ 1],            # -(a†1 a3) + (a†3 a1)
                        ]
    theta_parameters = [0,0,0]
    

    """
    Sec_Quant_CC_ia_ops = []  # second quantised single e- CC operators
    theta_parameters_ia = []

    # single_amplitudes and double_amplitudes from Get_CCSD_Amplitudes Hamiltonian function!
    orbitals_index = range(0, n_orbitals)

    alph_occs = [k for k in orbitals_index if k % 2 == 0 and k < n_electrons]  # spin up occupied
    beta_occs = [k for k in orbitals_index if k % 2 == 1 and k < n_electrons]  # spin down UN-occupied
    alph_noccs = [k for k in orbitals_index if k % 2 == 0 and k >= n_electrons]  # spin down occupied
    beta_noccs = [k for k in orbitals_index if k % 2 == 1 and k >= n_electrons]  # spin up UN-occupied

    # SINGLE electron excitation: spin UP transition
    for i in alph_occs:
        for a in alph_noccs:
            one_elec = FermionOperator(((a, 1), (i, 0))) - FermionOperator(((i, 1), (a, 0)))

            theta_parameters_ia.append(0)
            Sec_Quant_CC_ia_ops.append(one_elec)

    # SINGLE electron excitation: spin DOWN transition
    for i in beta_occs:
        for a in beta_noccs:
            # NO filtering
            one_elec = FermionOperator(((a, 1), (i, 0))) - FermionOperator(((i, 1), (a, 0)))

            theta_parameters_ia.append(0)
            Sec_Quant_CC_ia_ops.append(one_elec)

    return Sec_Quant_CC_ia_ops, theta_parameters_ia


def Get_UCCSD_ijab_terms(n_electrons, n_orbitals):
    """
    Get ijab excitation terms as fermionic creation and annihilation operators for UCCSD.
    ijab terms are standard double excitation terms (aka only occupied -> unoccupied transitions allowed)
    #TODO can add method to get pqrs terms
    #TODO these are all non-degenerate excitations which can possibly non-zero, including nocc->nocc, occ->occ, and spin-flips.
    #TODO EXPENSIVE, but will likely  get a slightly better answer.


    eqn:  T2 = ∑_{i,j ∈ occupied} ∑_{A,B ∈ unoccupied}  t_{AB}^{ij} a†_{A} a†_{B} a_{i} a_{j}

          T2 = ∑_{i=0}^{n_electrons-1} ∑_{j=0}^{i-1} ∑_{A=n_electrons}^{n_orbitals-1} ∑_{B=0}^{A-1}  t_{A}^{i} a†_{A} a_{i}

    Args:
        n_electrons (int): number of electrons
        n_orbitals (int): number of orbitals

    returns:
        Sec_Quant_CC_ijab_ops (list): list of FermionOperators (openfermion.ops._fermion_operator.FermionOperator)
        theta_parameters (list): list of theta values (parameterisation of excitation amplitudes)

     
     ** Example **

     n_electrons=2
     n_orbitals=4
     Sec_Quant_CC_ops, theta_parameters = Get_ijab_terms(n_electrons, n_orbitals)

     Sec_Quant_CC_ops=  [
                            -1.0[0 ^ 1 ^ 2 3] + 1.0 [3^ 2^ 1 0]  # -(a†0 a†1 a2 a3) + a†3 a†2 a1 a0)
                        ]
    theta_parameters = [0]
    """
    Sec_Quant_CC_ijab_ops = []  # second quantised two e- CC operators
    theta_parameters_ijab = []

    for i in range(n_electrons):
        for j in range(n_electrons):
            if (i > j):
                for A in range(n_electrons, n_orbitals):
                    for B in range(n_electrons, n_orbitals):
                        if (A > B):
                            if (i % 2 == 0) and (j % 2 == 0) and (A % 2 == 0) and (B % 2 == 0):  # UP UP --> UP UP
                                op =  FermionOperator(((A, 1), (B, 1), (i, 0), (j, 0)))
                                op_dag = FermionOperator(((j, 1), (i, 1), (B, 0), (A, 0)))
                                two_elec = op - op_dag
                            elif (i % 2 != 0) and (j % 2 != 0) and (A % 2 != 0) and (B % 2 != 0):  # DOWN DOWN --> DOWN DOWN
                                op =  FermionOperator(((A, 1), (B, 1), (i, 0), (j, 0)))
                                op_dag = FermionOperator(((j, 1), (i, 1), (B, 0), (A, 0)))
                                two_elec = op - op_dag
                            elif (i % 2 == 0) and (j % 2 != 0) and (A % 2 == 0) and (B % 2 != 0):  # UP DOWN --> UP DOWN
                                op =  FermionOperator(((A, 1), (B, 1), (i, 0), (j, 0)))
                                op_dag = FermionOperator(((j, 1), (i, 1), (B, 0), (A, 0)))
                                two_elec = op - op_dag
                            elif (i % 2 != 0) and (j % 2 == 0) and (A % 2 != 0) and (B % 2 == 0):  # DOWN UP --> DOWN UP
                                op =  FermionOperator(((A, 1), (B, 1), (i, 0), (j, 0)))
                                op_dag = FermionOperator(((j, 1), (i, 1), (B, 0), (A, 0)))
                                two_elec = op - op_dag
                            else:
                                continue  # spin forbidden transitions!

                            theta_parameters_ijab.append(0)
                            Sec_Quant_CC_ijab_ops.append(two_elec)


    return Sec_Quant_CC_ijab_ops, theta_parameters_ijab


def Fermi_ops_to_qubit_ops(List_Fermi_Ops, transformation='JW'):

        """
        Takes list of fermionic excitation operators and returns JW/BK transforms of each
        term and appends it to a list yielding a list of QubitOperators.


        Args:
            List_Fermi_Ops (list): List of fermionic operators
            transformation (str): defines fermion to qubit transformation type.


        returns:
            List_Qubit_Ops (list): List of QubitOperators (openfermion.ops._qubit_operator.QubitOperator)
                                   under JW/BK transform.

        ** Example **

         List_Fermi_Ops=        [
                                   -(a†0 a2) + (a†2 a0),
                                   -(a†1 a3) + (a†3 a1),
                                   -(a†0 a†1 a2 a3) + a†3 a†2 a1 a0)
                                ]
        out = Fermi_ops_to_qubit_ops(List_Fermi_Ops, transformation='JW')
        print (out)
        >> 
            [
                -0.5j [X0 Z1 Y2] + 0.5j [Y0 Z1 X2],
                -0.5j [X1 Z2 Y3] + 0.5j [Y1 Z2 X3],
                0.125j [X0 X1 X2 Y3] + 0.125j [X0 X1 Y2 X3] + -0.125j [X0 Y1 X2 X3] + 0.125j [X0 Y1 Y2 Y3] +
                -0.125j [Y0 X1 X2 X3] + 0.125j [Y0 X1 Y2 Y3] + -0.125j [Y0 Y1 X2 Y3] + -0.125j [Y0 Y1 Y2 X3]
            ]



        """

        List_Qubit_Ops = []

        if transformation == 'JW':
            for OP in List_Fermi_Ops:
                JW_OP = jordan_wigner(OP)
                List_Qubit_Ops.append(JW_OP)

        elif transformation == 'BK':
            for OP in List_Fermi_Ops:
                BK_OP = bravyi_kitaev(OP)
                List_Qubit_Ops.append(BK_OP)

        else:
            raise ValueError('unknown transformation: {}'.format(transformation))

        return List_Qubit_Ops


def UCCSD_single_trotter_step(transformation, List_FermiOps_ia, List_FermiOps_ijab):

    """
    Performs single trotter step approximation of UCCSD anstaz.
        U = exp [ t02 (a†2a0−a†0a2) + t13(a†3a1−a†1a3) +t0123 (a†3a†2a1a0−a†0a†1a2a3) ]
        becomes
        U=exp [t02(a†2a0−a†0a2)] × exp [t13(a†3a1−a†1a3)] × exp [t0123(a†3a†2a1a0−a†0a†1a2a3)]

    Args:
        transformation(str) : defines fermion to qubit transformation type.
        List_FermiOps_ia(list) : List of fermionic ia operators
        List_FermiOps_ijab(list) : List of fermionic ijab operators
    Returns
        Second_Quant_CC_single_Trot_list_ia(list): List of Qubit ia operators 
        Second_Quant_CC_single_Trot_list_ijab (list): List of Qubit ijab operators 


    Takes list of UCCSD fermionic excitation operators:

                [
                   -(a†0 a2) + (a†2 a0),
                   -(a†1 a3) + (a†3 a1),
                   -(a†0 a†1 a2 a3) + a†3 a†2 a1 a0)
                ]
    and returns JW transform of each term and appends it to a list yielding a list of QubitOperators
    performing UCCSD.

    [
        -0.5j [X0 Z1 Y2] + 0.5j [Y0 Z1 X2],
        -0.5j [X1 Z2 Y3] + 0.5j [Y1 Z2 X3],
        0.125j [X0 X1 X2 Y3] + 0.125j [X0 X1 Y2 X3] + -0.125j [X0 Y1 X2 X3] + 0.125j [X0 Y1 Y2 Y3] +
        -0.125j [Y0 X1 X2 X3] + 0.125j [Y0 X1 Y2 Y3] + -0.125j [Y0 Y1 X2 Y3] + -0.125j [Y0 Y1 Y2 X3]
    ]

    returns:
        Second_Quant_CC_JW_OP_list (list): List of QubitOperators (openfermion.ops._qubit_operator.QubitOperator)
                                           under JW transform. Each performs a UCCSD excitation.

    """
    ##ia
    Second_Quant_CC_single_Trot_list_ia= Fermi_ops_to_qubit_ops(List_FermiOps_ia,
                                       transformation=transformation)
    Second_Quant_CC_single_Trot_list_ijab = Fermi_ops_to_qubit_ops(List_FermiOps_ijab,
                                                    transformation=transformation)

    return Second_Quant_CC_single_Trot_list_ia, Second_Quant_CC_single_Trot_list_ijab