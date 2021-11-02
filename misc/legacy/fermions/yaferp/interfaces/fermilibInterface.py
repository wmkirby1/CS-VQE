'''
Created on 23 Oct 2017

@author: andrew
'''

import openfermion
import openfermionpsi4
import openfermionpyscf
import openfermion.utils
import openfermion.hamiltonians
import os
import numpy
import pickle as cPickle
from yaferp.general import optDict, PATHS

def fermiLibGeometryFromFile(filepath):
    geometry = openfermion.hamiltonians._molecular_data.geometry_from_file(filepath)
    return geometry

class FermiLibMolecule:
    
    def __init__(self,geometry,basis,multiplicity,charge,description='',filename='',symmetry='c1'):
        if os.path.isfile(str(geometry)):
            geometry = fermiLibGeometryFromFile(geometry)
        if filename:
            thisMolecularData = openfermion.hamiltonians.MolecularData(
                geometry, basis, multiplicity,charge,description,filename=filename)
        else:
            thisMolecularData = openfermion.hamiltonians.MolecularData(
                geometry, basis, multiplicity,charge,description)
        self.molecularData = thisMolecularData
        self.name = thisMolecularData.name + thisMolecularData.description
        self.symmetry = symmetry
    def runPSI4(self,runSCF=1,runMP2=0,runCISD=0,runCCSD=0,runFCI=0,verbose=0,delete_input=0,delete_output=0):
        self.molecularData = openfermionpsi4.run_psi4(self.molecularData,runSCF,runMP2,runCISD,runCCSD,runFCI,verbose,delete_input,delete_output,point_group=self.symmetry)
        return self

    def runPySCF(self,runSCF=1,runMP2=0,runCISD=0,runCCSD=0,runFCI=0,verbose=0):
        self.molecularData = openfermionpyscf.run_pyscf(self.molecularData,runSCF,runMP2,runCISD,runCCSD,runFCI,verbose)
        return self


    def integrals(self):
        return self.molecularData.get_integrals()

    def numMOs(self):
        return self.molecularData.n_qubits

    def save(self):
        self.molecularData.save()
    def hamiltonianTerms(self,cutoff=1e-14):
        
            '''code adapted from FermiLibPsi4Plugin (https://github.com/ProjectQ-Framework/FermiLib-Plugin-Psi4)'''
            n_qubits = len(self.integrals()[0][0])*2
            # Initialize Hamiltonian coefficients.
            one_body_coefficients = numpy.zeros((n_qubits, n_qubits))
            two_body_coefficients = numpy.zeros((n_qubits, n_qubits,
                                                 n_qubits, n_qubits))
            one_body_integrals = self.integrals()[0]
            two_body_integrals = self.integrals()[1]
            # Loop through integrals.
            for p in range(n_qubits // 2):
                for q in range(n_qubits // 2):
            
                    # Populate 1-body coefficients. Require p and q have same spin.
                    one_body_coefficients[2 * p, 2 * q] = one_body_integrals[p, q]
                    one_body_coefficients[2 * p + 1, 2 *
                                          q + 1] = one_body_integrals[p, q]
            
                    # Continue looping to prepare 2-body coefficients.
                    for r in range(n_qubits // 2):
                        for s in range(n_qubits // 2):
            
                            # Require p,s and q,r to have same spin. Handle mixed
                            # spins.
                            two_body_coefficients[2 * p, 2 * q + 1, 2 * r + 1,
                                                  2 * s] = (
                                two_body_integrals[p, q, r, s])
                            two_body_coefficients[2 * p + 1, 2 * q, 2 * r,
                                                  2 * s + 1] = (
                                two_body_integrals[p, q, r, s])
            
                            # Avoid having two electrons in same orbital. Handle
                            # same spins.
                            if p != q and r != s:
                                two_body_coefficients[2 * p, 2 * q, 2 * r,
                                                      2 * s] = (
                                    two_body_integrals[p, q, r, s])
                                two_body_coefficients[2 * p + 1, 2 * q + 1,
                                                      2 * r + 1, 2 * s + 1] = (
                                    two_body_integrals[p, q, r, s] )
            one_body_coefficients[
            numpy.absolute(one_body_coefficients) < cutoff] = 0.
            two_body_coefficients[
            numpy.absolute(two_body_coefficients) < cutoff] = 0.
            
            clive = [one_body_coefficients,two_body_coefficients]
            numSpinOrbitals = len(clive[0][0])
            return(numSpinOrbitals,clive[0],clive[1])
    def energyFCI(self):
        return self.molecularData.fci_energy   
    def energyNuclear(self):
        return self.molecularData.nuclear_repulsion
    def energyCCSD(self):
        return self.molecularData.ccsd_energy
    def energyMP2(self):
        return self.molecularData.mp2_energy
    def saveIntegrals(self,cutoff=1e-14,protocol='cpickle'):
        filePath = PATHS.INTEGRALS_DIR + str(cutoff) + '/' + self.name + '.ints'
        with open(filePath,'wb') as f:
            if protocol=='cpickle':
                cPickle.dump(self.hamiltonianTerms(cutoff),f,protocol=cPickle.HIGHEST_PROTOCOL)
        return
    def energySCF(self):
        return self.molecularData.hf_energy
        
        
    def electronicHamiltonian(self,boolJWorBK,cutoff=1e-14,verbose=False):
        integrals = self.hamiltonianTerms(cutoff)
        #dave = fermions.electronicHamiltonian(integrals[0],boolJWorBK,integrals[1],integrals[2],False,verbose)
        #dave = fermions.oplistRemoveNegligibles(dave,cutoff)
        dave = optDict.electronicHamiltonian(integrals[0], boolJWorBK, integrals[1], integrals[2], False, verbose, cutoff)
        self.hamiltonianCutoff = cutoff
        return dave

    def gradient(self,otherFLM,boolJWorBK,cutoff=1e-14):
        '''gradient - note the other FLM is in the NEGATIVE direction'''
        myIntegrals = self.hamiltonianTerms(cutoff)
        theirIntegrals = otherFLM.hamiltonianTerms(cutoff)
        oneElectronIntegralsDifference = myIntegrals[1] - theirIntegrals[1]
        twoElectronIntegralsDifference = myIntegrals[2] - theirIntegrals[2]
        dave = optDict.electronicHamiltonian(myIntegrals[0], boolJWorBK, oneElectronIntegralsDifference, twoElectronIntegralsDifference, False, False, cutoff)
        self.hamiltonianCutoff = cutoff
        return dave



    def saveHamiltonian(self,boolJWorBK,overrideCutoff=None,filePath=None,protocol='cpickle',verbose=False):
        DEFAULT_CUTOFF = 1e-14
        if overrideCutoff != None:
            cutoff = overrideCutoff
        else:
            try:
                cutoff = self.hamiltonianCutoff
            except AttributeError:
                cutoff = DEFAULT_CUTOFF
                
        if filePath==None:
            if not boolJWorBK:
                filePath = PATHS.OPLIST_DIR + str(cutoff) + '/JW/' + self.name + '.ham'
            else:
                filePath = PATHS.OPLIST_DIR + str(cutoff) + '/BK/' + self.name + '.ham'
        with open(filePath,'wb') as f:
            if protocol=='cpickle':
                cPickle.dump(self.electronicHamiltonian(boolJWorBK, cutoff,verbose=verbose),f,protocol=cPickle.HIGHEST_PROTOCOL)
        return boolJWorBK
#from memory_profiler import profile

#@profile
def thing():
    ethMol = FermiLibMolecule('/home/andrew/scr/methane.xyz','STO-3G',1,0,'meth')
    ethMol.runPSI4()
    ethMol.saveHamiltonian(0)
    return



'''
def electronicHamiltonianFromFermionOperator(fermionOperator,boolJWorBK,cutoff=1e-12,verbose=False):
    openfermionTerms = fermionOperator.terms
    result = optDict.optDict()
    for thisTerm in openfermionTerms:
        for'''