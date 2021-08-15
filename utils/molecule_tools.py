import numpy as np
from collections import Counter
import utils.qonversion_tools as qonvert
import openfermion
import openfermionpyscf
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from openfermion.ops import FermionOperator, QubitOperator
from openfermion.transforms import jordan_wigner, bravyi_kitaev
from openfermion.transforms import get_fermion_operator
from openfermion.circuits import (uccsd_singlet_get_packed_amplitudes,
                               uccsd_singlet_generator)

def construct_molecule(atoms, bond_len, multiplicity, charge, basis):
    num_atoms = len(atoms)
    mult = Counter(atoms)
    speciesname = ''
    for a in mult.keys():
        speciesname += (a+str(mult[a])+'_')
    speciesname += str(basis)
    
    geometry = []
    for index, a in enumerate(atoms):
        blank_coord = list(np.zeros(3))
        if index != 0:
            blank_coord[index] = bond_len
        geometry.append((a, tuple(blank_coord)))

    molecule_data = MolecularData(geometry, basis, multiplicity, charge, description=speciesname)
    delete_input = True
    delete_output = True

    molecule = run_pyscf(molecule_data,run_scf=1,run_mp2=1,run_cisd=1,run_ccsd=1,run_fci=1)
    num_electrons = molecule.n_electrons
    num_qubits = 2*molecule.n_orbitals

    # construct hamiltonian
    ham_f = get_fermion_operator(molecule.get_molecular_hamiltonian())
    ham_q = jordan_wigner(ham_f)
    ham = qonvert.QubitOperator_to_dict(ham_q, num_qubits)
    
    # construct UCCSD Ansatz
    ccsd_single_amps = molecule.ccsd_single_amps
    ccsd_double_amps = molecule.ccsd_double_amps
    packed_amps = uccsd_singlet_get_packed_amplitudes(ccsd_single_amps,  ccsd_double_amps, num_qubits, num_electrons)
    ucc_op = uccsd_singlet_generator(packed_amps, num_qubits, num_electrons)

    ucc_q = jordan_wigner(ucc_op)
    ucc = qonvert.QubitOperator_to_dict(ucc_q, num_qubits)

    return {'speciesname':speciesname,
            'num_qubits':num_qubits,
            'num_electrons':num_electrons,
            'hamiltonian':ham,
            'uccsdansatz':ucc}
    