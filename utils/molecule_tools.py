import numpy as np
import json
from collections import Counter
import utils.qonversion_tools as qonvert
from itertools import combinations
import openfermion
import openfermionpyscf
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from openfermion.transforms import taper_off_qubits
from openfermion.ops import FermionOperator, QubitOperator
from openfermion.linalg import get_sparse_operator, get_ground_state
from openfermion.transforms import jordan_wigner, bravyi_kitaev
from openfermion.transforms import get_fermion_operator
from openfermion.circuits import (uccsd_singlet_get_packed_amplitudes,
                               uccsd_singlet_generator)


def find_tapering(ham_q, num_qubits, taper_num):
    full_energy = get_ground_state(get_sparse_operator(ham_q, num_qubits))[0]
    n_q_tap = num_qubits-taper_num
    single_Z = ['Z'+str(i) for i in range(num_qubits)]
    taper_stabs = [[QubitOperator(z) for z in Z_stab] for Z_stab in combinations(single_Z, taper_num)]

    taper_results=[]
    for stabs in taper_stabs:
        ham_tap_q = taper_off_qubits(ham_q, stabs)
        taper_energy = get_ground_state(get_sparse_operator(ham_tap_q, n_q_tap))[0]
        taper_results.append((taper_energy, stabs))
        if taper_energy - full_energy < 0.0016:
            print("Tapering successful")
            break

    return sorted(taper_results, key=lambda x:x[0])[0][1]


def construct_molecule(atoms, coords, multiplicity, charge, basis, taper_num=0):
    """
    """
    num_atoms = len(atoms)
    mult = Counter(atoms)
    speciesname = ''
    for a in mult.keys():
        speciesname += (a+str(mult[a])+'_')
    speciesname += str(basis)
    
    geometry = []
    for index, a in enumerate(atoms):
        geometry.append((a, coords[index]))
    
    molecule_data = MolecularData(geometry, basis, multiplicity, charge, description=speciesname)
    delete_input = True
    delete_output = True

    molecule = run_pyscf(molecule_data,run_scf=1,run_mp2=1,run_cisd=1,run_ccsd=1,run_fci=1)
    num_electrons = molecule.n_electrons
    num_qubits = 2*molecule.n_orbitals

    # construct hamiltonian
    ham_f = get_fermion_operator(molecule.get_molecular_hamiltonian())
    ham_q = jordan_wigner(ham_f)
    
    # construct UCCSD Ansatz
    ccsd_single_amps = molecule.ccsd_single_amps
    ccsd_double_amps = molecule.ccsd_double_amps
    packed_amps = uccsd_singlet_get_packed_amplitudes(ccsd_single_amps,  ccsd_double_amps, num_qubits, num_electrons)
    ucc_op = uccsd_singlet_generator(packed_amps, num_qubits, num_electrons)
    ucc_q = jordan_wigner(ucc_op)
    
    if taper_num:
        optimal_tapering = find_tapering(ham_q, num_qubits, taper_num)
        ham_q = taper_off_qubits(ham_q, optimal_tapering) 
        ucc_q = taper_off_qubits(ucc_q, optimal_tapering)
        num_qubits = num_qubits-taper_num

    ham = qonvert.QubitOperator_to_dict(ham_q, num_qubits)
    ucc = qonvert.QubitOperator_to_dict(ucc_q, num_qubits)

    return {'speciesname':speciesname,
            'num_qubits': num_qubits,
            'hamiltonian':ham,
            'uccsdansatz':ucc}


def get_molecule(speciesname, taper_num=0):
    """
    """
    file = 'molecule_data'
    with open('data/'+file+'.json', 'r') as json_file:
        molecule_data = json.load(json_file)

    atoms, bond_len, coords, multiplicity, charge, basis = molecule_data[speciesname].values()
    mol_out = construct_molecule(atoms, coords, multiplicity, charge, basis, taper_num)
    
    return mol_out
