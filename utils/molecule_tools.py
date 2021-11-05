import numpy as np
import json
from collections import Counter
import utils.qonversion_tools as qonvert
import utils.bit_tools as bit
import utils.linalg_tools as la
import itertools
# OpenFermion libraries
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from openfermion.ops import FermionOperator, QubitOperator
from openfermion.transforms import jordan_wigner, bravyi_kitaev
from openfermion.transforms import get_fermion_operator
from openfermion.circuits import (uccsd_singlet_get_packed_amplitudes,
                               uccsd_singlet_generator)
# Qiskit libraries
from qiskit_nature.operators.second_quantization.fermionic_op import FermionicOp
from qiskit_nature.drivers import UnitsType, Molecule
from qiskit_nature.drivers.second_quantization import ElectronicStructureDriverType, ElectronicStructureMoleculeDriver
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper, ParityMapper
from qiskit_nature.circuit.library.initial_states.hartree_fock import hartree_fock_bitstring


def construct_molecule(atoms, coords, charge, multiplicity, basis, ucc_threshold=None, taper=False, sym_sector=None):
    """
    """
    # generate speciesname string from atoms
    mult = Counter(atoms)
    speciesname = ''
    for a in mult.keys():
        speciesname += (a+str(mult[a])+'_')
    speciesname += str(basis)
    
    # ensure correct format for geometry
    geometry = []
    for index, a in enumerate(atoms):
        geometry.append((a, coords[index]))
    
    # construct with PySCF through OpenFermion for actual molecular calculations (including coupled cluster amps)
    molecule_data = MolecularData(geometry, basis, multiplicity, charge, description=speciesname)
    delete_input = True
    delete_output = True
    molecule = run_pyscf(molecule_data, run_ccsd=1)
    num_electrons =molecule.n_electrons # total number of particles
    num_particles =(molecule.get_n_alpha_electrons(),
                    molecule.get_n_beta_electrons()) # number of alpha (spin up) and beta (spin down) electrons
    num_qubits = 2*molecule.n_orbitals
    # construct second quantised Hamiltonian:
    ham_2ndQ = get_fermion_operator(molecule.get_molecular_hamiltonian())
    #hf_config_bool = hartree_fock_bitstring(num_particles=num_particles,
    #                                        num_spin_orbitals=num_qubits)
    hf_config_bool = [True for i in range(num_electrons)] + [False for i in range(num_qubits-num_electrons)] # in jordan-wigner
    
    # construct second quantised UCCSD operator:
    if ucc_threshold is not None:
        # determine most significant uccsd terms
        hf_config = ''.join([str(int(b)) for b in hf_config_bool])
        ham_mat = qonvert.dict_to_WeightedPauliOperator(qonvert.QubitOperator_to_dict(jordan_wigner(ham_2ndQ), num_qubits)).to_matrix()
        gsnrg, gsvec = la.get_ground_state(ham_mat)
        gsprobs = [abs(amp)**2 for amp in gsvec]
        bstates = [bit.int_to_bin(i, num_qubits) for i in range(2**num_qubits)]
        sig_state_order = [s for s in sorted(list(zip(gsprobs, bstates)), key=lambda x:-x[0]) if s[0]>ucc_threshold]
        reduccsd = FermionOperator()

        for amp, state in sig_state_order:
            exCite = [index for index, b in enumerate(zip(state, hf_config)) if len(set(b))==2]
            if len(exCite)==2:
                reduccsd += FermionOperator([(exCite[0],1),(exCite[1],0)])
                reduccsd -= FermionOperator([(exCite[1],1),(exCite[0],0)])
            if len(exCite)==4:
                order1 = [2, 0, 3, 1]
                order2 = [3, 1, 2, 0]
                reduccsd += FermionOperator([(cite, (index+1)%2) for index, cite in enumerate([exCite[i] for i in order1])])
                reduccsd -= FermionOperator([(cite, (index+1)%2) for index, cite in enumerate([exCite[i] for i in order1][::-1])])
                reduccsd += FermionOperator([(cite, (index+1)%2) for index, cite in enumerate([exCite[i] for i in order2])])
                reduccsd -= FermionOperator([(cite, (index+1)%2) for index, cite in enumerate([exCite[i] for i in order2][::-1])])
        ucc_2ndQ=reduccsd
    else:
        ccsd_single_amps = molecule.ccsd_single_amps
        ccsd_double_amps = molecule.ccsd_double_amps
        packed_amps = uccsd_singlet_get_packed_amplitudes(ccsd_single_amps,  ccsd_double_amps, num_qubits, num_electrons)
        ucc_2ndQ = uccsd_singlet_generator(packed_amps, num_qubits, num_electrons)


    if not taper:
        ham_q = jordan_wigner(ham_2ndQ)
        ucc_q = jordan_wigner(ucc_2ndQ)
        ham = qonvert.QubitOperator_to_dict(ham_q, num_qubits)
        ucc = qonvert.QubitOperator_to_dict(ucc_q, num_qubits)
        hf_config = ''.join([str(int(b)) for b in hf_config_bool])
        num_tapered=0
        ham_mat = qonvert.dict_to_WeightedPauliOperator(ham).to_matrix()
        true_gs_nrg, true_gs_vec = la.get_ground_state(ham_mat)

    else:
        # now we duplicate this electronic structure problem in Qiskit Nature for Z2 symmetry identification
        molecule_qiskit = Molecule(geometry=geometry, charge=charge, multiplicity=multiplicity) 
        driver = ElectronicStructureMoleculeDriver(molecule_qiskit, basis=basis, driver_type=ElectronicStructureDriverType.PYSCF)
        es_problem = ElectronicStructureProblem(driver)
        second_q_op = es_problem.second_q_ops()
        # Map Hamiltonian and UCCSD operators from OpenFermion to Qiskit Nature representation
        ham_2ndQ_mapped = qonvert.fermionic_openfermion_to_qiskit(ham_2ndQ, num_qubits)
        ucc_2ndQ_mapped = qonvert.fermionic_openfermion_to_qiskit(ucc_2ndQ, num_qubits)
        # Determine tapering stabilisers and (hopefully) correct sector with Qiskit Nature:
        qubit_converter = QubitConverter(JordanWignerMapper(), z2symmetry_reduction='auto')
        ham_ref = qubit_converter.convert(ham_2ndQ_mapped) # stores the Z2 symmetries in qubit_converter 
        taper_qubits = qubit_converter.z2symmetries.sq_list
        hf_config = ''.join([str(int(b)) for index,b in enumerate(hf_config_bool) if index not in taper_qubits])
        Z2sym = qubit_converter.z2symmetries.symmetries
        if sym_sector is None:
            Z2ref = es_problem.symmetry_sector_locator(qubit_converter.z2symmetries) #try to find the correct sector
        else:
            Z2ref = sym_sector
        num_tapered = len(Z2ref)

        # list all possible sectors
        sectors = []
        for c in list(itertools.combinations_with_replacement([+1, -1], len(Z2ref))):
            sectors+=set(itertools.permutations(c))
        # order by hamming distance from the reference sector
        sectors_order=[]
        for s in sectors:
            ham_dist=0
            for a,b in zip(Z2ref, s):
                if a!=b:
                    ham_dist+=1
            sectors_order.append((s, ham_dist))
        sectors=[a for a,b in sorted(sectors_order, key=lambda x:x[1])]

        print('Attempting to taper %i --> %i qubits' % (num_qubits, num_qubits-num_tapered))
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        true_gs_nrg, true_gs_vec = la.get_ground_state(ham_ref.to_spmatrix())
        pretap=true_gs_nrg
        for Z2sec in sectors:
            # Perform Jordan-Wigner transformation and taper
            qubit_taper = QubitConverter(JordanWignerMapper(), z2symmetry_reduction=Z2sec)
            ham_tap = qubit_taper.convert(ham_2ndQ_mapped)
            postap=la.get_ground_state(ham_tap.to_spmatrix())[0]

            if postap-pretap<1e-6:
                print('Energies match in sector %s, tapering successful!\n' % str(Z2sec))
                break
            else:
                print('Energy mismatch with target problem in sector %s, trying another...' % str(Z2sec))

        ucc_q = QubitConverter(JordanWignerMapper()).convert(ucc_2ndQ_mapped)
        ucc_tap = qubit_taper.z2symmetries.taper(ucc_q)

        num_qubits = ham_tap.num_qubits
        ham = {op:coeff for op, coeff in qonvert.PauliOp_to_dict(ham_tap).items() if abs(coeff)>1e-15}
        ucc = {op:coeff for op, coeff in qonvert.PauliOp_to_dict(ucc_tap).items() if abs(coeff)>1e-15}
    
    return {'speciesname':speciesname,
            'num_qubits': num_qubits,
            'hamiltonian':ham,
            'uccsdansatz':ucc,
            'hf_config':  hf_config,
            'num_tapered':num_tapered,
            'true_gs_nrg':true_gs_nrg,
            'true_gs_vec':true_gs_vec}


def get_molecule(speciesname, taper=False):
    """
    """
    file = 'molecule_data'
    with open('data/'+file+'.json', 'r') as json_file:
        molecule_data = json.load(json_file)

    atoms, coords, multiplicity, charge, basis, sym_sector = molecule_data[speciesname].values()
    if taper:
        if sym_sector=="None":
            sym_sector=None
            print('*** sector not specified in molecule data, searching now ***')
        else:
            print('*** sector saved, will check tapered ground state energy matches target problem ***')
    
    mol_out = construct_molecule(atoms, coords, charge, multiplicity, basis, taper, sym_sector)
    
    return mol_out
