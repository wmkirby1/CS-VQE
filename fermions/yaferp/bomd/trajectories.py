'''code to generate molecular reactive bomd from openfermion/psi4/forest/etc.
adapted from william simon's work'''
#import pyquil
from yaferp.interfaces import cirqVQE, fermilibInterface
from yaferp.general import sparseFermions, fermions
import datetime
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from yaferp.bomd.GeometrySpec import GeometrySpec

from yaferp.bomd.VerletIntegrator import VerletIntegrator
from yaferp.bomd import IndependentAdaptiveIntegrator, ResourceEstimator, AdaptiveIntegrator
from yaferp.bomd.trajectoriesConstants import DT, OPENFERMION_DATA_DIR,GRADIENT_PRECISION

import copy
from yaferp.misc import noncontextuality_quasi_quantized


def h2Geometry(bondlength):
    return [('H', (0., 0., 0.)), ('H', (0., 0., bondlength))]

def h2Cartesian(x1,x2):
    return [('H', (0., 0., x1)), ('H', (0., 0., x2))]
h2Masses = [1.,1.]

def colinearDH2JacobiGeometry(q1,q2):
    return [('H',(0.,0.,0.)),('H',(0.,0.,q1)),('D',(0.,0.,q2 + 0.5*q1))]

def colinearH3JacobiGeometry(q1,q2):
    return [('H',(0.,0.,0.)),('H',(0.,0.,q1)),('H',(0.,0.,q2 + 0.5*q1))]
def colinearH3JacobiCoordinates(x1,x2):
    return [x1,(x2- 0.5*x1)]

def colinearH3Cartesian(x1,x2,x3):
    return [('H',(0.,0.,x1)),('H',(0.,0.,x2)),('H',(0.,0.,x3))]

def H3CartesianToJacobi(x):
    return [x[1]-x[0],x[2]-(0.5*(x[1]+x[0]))]

def H3ValenceToJacobi(r1,r2):
    return [r1, r2 + 0.5*r1]

def H3JacobiToValence(x1,x2):
    return [x1,x2 - (0.5*x1)]
#def colinearH3LabFrame(x1,q1,q2):
#    return [('H',(0.,0.,x1)),('H',(0.,0.,q1 + x1)),('H',(0.,0.,q2 + ))]

def symmetricH2(r):
    return [('H', (0., 0., 0.)), ('H', (0., 0., r))]

def symmetricH3(r):
    return [('H', (0., 0., 0.)), ('H', (0., 0., r)),('H', (0., 0., r))]

def hChain(n):
    assert n > 0
    def result(r):
        return [('H', (0., 0., float(r) * float(i))) for i in range(n)]
    return result



DH2JacobiMasses = [0.5,1.]
H3JacobiMasses = [0.5,(2./3.)]
H3CartesianMasses = [1.,1.,1.]
H2JacobiMasses = [0.5]

TEST_GEOMETRY_SPEC = GeometrySpec(colinearH3JacobiGeometry,
                                  [0.71, 1.555],
                                  [0., 0.],
                                  H3JacobiMasses,
                                  multiplicity=2)

   # def centreOfMass(self):


def openFermionGradientFunction(initialGeometry,plusGeometry,minusGeometry,resourceEstimator=None):
    #flmGeometries = [x.cartesian for x in [plusGeometry,minusGeometry]]
    flmGeometries = [plusGeometry,minusGeometry]
    flms = [fermilibInterface.FermiLibMolecule(x.cartesian, "STO-3G", x.multiplicity, x.charge, str(x.cartesian), str(x.cartesian)) for x in flmGeometries]
    for flm in flms:
        flm.runPySCF(runFCI=1)
    energies = [x.energyFCI() for x in flms]
    energyDifference = energies[0] - energies[1]
    return energyDifference

def runQuantumChemistry(flm,package='pyscf',runCCSD=0,runFCI=0,runMP2=0):
    if package.lower() == 'psi4':
        flm.runPSI4(runCCSD=runCCSD,runFCI=runFCI,runMP2=runMP2)
    elif package.lower() == 'pyscf':
        flm.runPySCF(runCCSD=runCCSD,runFCI=runFCI,runMP2=runMP2)
    else:
        raise ValueError('unrecognised quantum chemistry package')
    return

def openFermionHartreeFock(initialGeometry,
                           plusGeometry,
                           minusGeometry,
                           resourceEstimator=None,
                           openFermionDataDir=OPENFERMION_DATA_DIR,
                           quantumChemPackage='pyscf'):
    os.chdir(openFermionDataDir)
    flmGeometries = [plusGeometry,minusGeometry]
    flms = [fermilibInterface.FermiLibMolecule(x.cartesian,
                                               "STO-3G",
                                               x.multiplicity,
                                               x.charge,
                                               str(x.cartesian),
                                               str(x.cartesian)) for x in flmGeometries]
    for flm in flms:
        runQuantumChemistry(flm,quantumChemPackage)

    energies = [x.energySCF() for x in flms]
    energyDifference = energies[0] - energies[1]
    return energyDifference


def openFermionHartreeFockEnergy(geometry,
                          resourceEstimator=None,
                          openFermionDataDir=OPENFERMION_DATA_DIR,
                          quantumChemPackage='pyscf',
                                 includeNuclear=False):
    os.chdir(openFermionDataDir)
    flm = fermilibInterface.FermiLibMolecule(geometry.cartesian,
                                             "STO-3G",
                                             geometry.multiplicity,
                                             geometry.charge,
                                             str(geometry),
                                             str(geometry))


    runQuantumChemistry(flm,quantumChemPackage)
    energy = flm.energySCF()
    return energy

def openFermionMP2(initialGeometry,
                   plusGeometry,
                   minusGeometry,
                   resourceEstimator=None,
                   openFermionDataDir=OPENFERMION_DATA_DIR,
                   quantumChemPackage='pyscf'):
    os.chdir(openFermionDataDir)
    quantumChemPackage='psi4'
    flmGeometries = [plusGeometry,minusGeometry]
    flms = [fermilibInterface.FermiLibMolecule(x.cartesian,
                                               "STO-3G",
                                               x.multiplicity,
                                               x.charge,
                                               str(x),
                                               str(x)) for x in flmGeometries]

    for flm in flms:
        runQuantumChemistry(flm,quantumChemPackage,runMP2=True)
    energies = [x.energyMP2() for x in flms]
    energyDifference = energies[0] - energies[1]
    return energyDifference


def openFermionMP2Energy(geometry,
                          resourceEstimator=None,
                          openFermionDataDir=OPENFERMION_DATA_DIR,
                          quantumChemPackage='psi4',
                         includeNuclear=False):
    quantumChemPackage='psi4'
    os.chdir(openFermionDataDir)
    flm = fermilibInterface.FermiLibMolecule(geometry.cartesian,
                                             "STO-3G",
                                             geometry.multiplicity,
                                             geometry.charge,
                                             str(geometry),
                                             str(geometry))


    runQuantumChemistry(flm,quantumChemPackage,runMP2=True)
    energy = flm.energyMP2()
    return energy

def openFermionCCSD(initialGeometry,
                    plusGeometry,
                    minusGeometry,
                    resourceEstimator=None,
                    openFermionDataDir=OPENFERMION_DATA_DIR,
                    quantumChemPackage='pyscf'):
    os.chdir(openFermionDataDir)
    flmGeometries = [plusGeometry,minusGeometry]
    flms = [fermilibInterface.FermiLibMolecule(x.cartesian,
                                               "STO-3G",
                                               x.multiplicity,
                                               x.charge,
                                               str(x),
                                               str(x)) for x in flmGeometries]

    for flm in flms:
        runQuantumChemistry(flm,quantumChemPackage,runCCSD=True)
    energies = [x.energyCCSD() for x in flms]
    energyDifference = energies[0] - energies[1]
    return energyDifference

def openFermionCCSDEnergy(geometry,
                    resourceEstimator=None,
                    openFermionDataDir=OPENFERMION_DATA_DIR,
                    quantumChemPackage='pyscf'):
    os.chdir(openFermionDataDir)
    flm = fermilibInterface.FermiLibMolecule(geometry.cartesian,
                                               "STO-3G",
                                             geometry.multiplicity,
                                             geometry.charge,
                                             str(geometry),
                                             str(geometry))


    runQuantumChemistry(flm,quantumChemPackage,runCCSD=True)
    energy = flm.energyCCSD()
    return energy


def openFermionPotentialFunction(geometry,resourceEstimator=None,
                                 openFermionDataDir=OPENFERMION_DATA_DIR,
                                 quantumChemPackage='pyscf',
                                 includeNuclear=False):
    os.chdir(openFermionDataDir)
    flmGeometry = geometry.cartesian
    flm = fermilibInterface.FermiLibMolecule(flmGeometry, "STO-3G", geometry.multiplicity, geometry.charge, str(geometry), str(geometry))
    runQuantumChemistry(flm,quantumChemPackage,runFCI=True)
    energy = flm.energyFCI()
    if includeNuclear == True:
        energy += flm.energyNuclear()
    return energy

def noncontextualPotentialFunction(geometry,quantumChemPackage='pyscf',includeNuclear=True):
    flmGeometry = geometry.cartesian
    flm = fermilibInterface.FermiLibMolecule(flmGeometry, "STO-3G", geometry.multiplicity, geometry.charge,
                                             str(geometry), str(geometry))
    runQuantumChemistry(flm, quantumChemPackage)
    oplist= flm.electronicHamiltonian(0,1e-12)
    hamDict = noncontextuality_quasi_quantized.oplist2ham_dict(oplist)
    ncHamDict = noncontextuality_quasi_quantized.toNCHamiltonian_fast(hamDict,
                                                                      weighting=None,
                                                                      strategy='greedy',
                                                                      step_size=1,
                                                                      show_progress=False)
    ncOplist = noncontextuality_quasi_quantized.ham_dict2Oplist(ncHamDict)
    eigval,eigvec = sparseFermions.getTrueEigensystem(ncOplist)
    if includeNuclear:
        eigval += flm.energyNuclear()
    return eigval
import contextuality.contextuality as dsfgsdfg
def noncontextualPotentialFunction2(geometry,quantumChemPackage='pyscf',includeNuclear=True):
    flmGeometry = geometry.cartesian
    flm = fermilibInterface.FermiLibMolecule(flmGeometry, "STO-3G", geometry.multiplicity, geometry.charge,
                                             str(geometry), str(geometry))
    runQuantumChemistry(flm, quantumChemPackage)
    oplist= flm.electronicHamiltonian(0,1e-12)
    hamDict = noncontextuality_quasi_quantized.oplist2ham_dict(oplist)
    ncHamDict = dsfgsdfg.greedy_dfs_hamiltonian(hamDict,2)[1]
    ncOplist = noncontextuality_quasi_quantized.ham_dict2Oplist(ncHamDict)
    eigval,eigvec = sparseFermions.getTrueEigensystem(ncOplist)
    if includeNuclear:
        eigval += flm.energyNuclear()
    return eigval

def noncontextualPotentialFunction3(geometry,quantumChemPackage='pyscf',includeNuclear=True):
    flmGeometry = geometry.cartesian
    flm = fermilibInterface.FermiLibMolecule(flmGeometry, "STO-3G", geometry.multiplicity, geometry.charge,
                                             str(geometry), str(geometry))
    runQuantumChemistry(flm, quantumChemPackage)
    oplist= flm.electronicHamiltonian(0,1e-12)
    hamDict = noncontextuality_quasi_quantized.oplist2ham_dict(oplist)
    ncHamDict = dsfgsdfg.greedy_dfs_hamiltonian(hamDict,60)[1]
    ncOplist = noncontextuality_quasi_quantized.ham_dict2Oplist(ncHamDict)
    eigval,eigvec = sparseFermions.getTrueEigensystem(ncOplist)
    if includeNuclear:
        eigval += flm.energyNuclear()
    return eigval


def nucRep(geometry,quantumChemPackage='pyscf',includeNuclear=True):
    flmGeometry = geometry.cartesian
    flm = fermilibInterface.FermiLibMolecule(flmGeometry, "STO-3G", geometry.multiplicity, geometry.charge,
                                             str(geometry), str(geometry))
    runQuantumChemistry(flm, quantumChemPackage)

    return flm.energyNuclear()



def _runPySCFWrapper_(flm):
    #print(flm)
    flm.runPySCF()
    nucRep = flm.energyNuclear() #clunky hack - something in openfermion isn't pickleable so
    #print(flm)
    return nucRep


def SCRAP_openFermionOplistExpectationGradient(initialGeometry,plusGeometry,minusGeometry,returnOplists=False):
    flmGeometries = [initialGeometry,plusGeometry,minusGeometry]
    #flmGeometries = [x.cartesian for x in [initialGeometry,plusGeometry,minusGeometry]]
    #manager = multiprocessing.Manager()
    flms = [fermilibInterface.FermiLibMolecule(x.cartesian, "STO-3G", x.multiplicity, x.charge, str(x.cartesian), str(x.cartesian)) for x in flmGeometries]
    #flms = manager.list([fermilibInterface.FermiLibMolecule(x,"STO-3G",1,1,str(x),str(x)) for x in flmGeometries])
    #rawflms = [fermilibInterface.FermiLibMolecule(x,"STO-3G",1,1,str(x),str(x)) for x in flmGeometries]
    #print(flms)
    #print(flms.__repr__())
    #pool = multiprocessing.Pool(processes=len(flms))
    #print(gc.get_count())
    for flm in flms:
        flm.runPySCF()
    #nucreps = pool.map(_runPySCFWrapper_,flms)
    #print(flms)
    #print(flms.__repr__())
    originalOplist = flms[0].electronicHamiltonian(0,cutoff=1e-12)
    oplists = [x.electronicHamiltonian(0,cutoff=1e-12) for x in flms]
    fullOplist = oplists[1] + fermions.coefficient(-1., oplists[2])
    gradientOplist = fermions.oplistRemoveNegligibles(fermions.simplify(fullOplist), 1e-12)
    #gradientOplist = flms[1].gradient(flms[2],0,1e-12)
    eigenvalue,eigenstate = sparseFermions.getTrueEigensystem(originalOplist)
    result = sparseFermions.oplistExpectation(gradientOplist, eigenstate)
    #print(result)
    #print(flms[0].energyNuclear())
    #print(flms[1].energyNuclear)
    nucRepulsion = flms[1].energyNuclear() - flms[2].energyNuclear()
    #nucRepulsion = nucreps[1] - nucreps[2]
    #pool.close()
    #pool.join()
    if returnOplists:
        return (result[0,0] + nucRepulsion,[originalOplist,gradientOplist])
    return result[0,0] + nucRepulsion

def openFermionOplistExpectationGradient(initialGeometry, plusGeometry, minusGeometry, resourceEstimator=None, openFermionDataDir=OPENFERMION_DATA_DIR):
    os.chdir(openFermionDataDir)
    flmGeometries = [initialGeometry,plusGeometry,minusGeometry]
    flms = [fermilibInterface.FermiLibMolecule(x.cartesian, "STO-3G", x.multiplicity, x.charge, str(x.cartesian), str(x.cartesian)) for x in flmGeometries]
    for flm in flms:
        flm.runPySCF()
    originalOplist = flms[0].electronicHamiltonian(0,cutoff=1e-12)
    oplists = [x.electronicHamiltonian(0,cutoff=1e-12) for x in flms]
    fullOplist = oplists[1] + fermions.coefficient(-1., oplists[2])
    gradientOplist = fermions.oplistRemoveNegligibles(fermions.simplify(fullOplist), 1e-12)
    eigenvalue,eigenstate = sparseFermions.getTrueEigensystem(originalOplist)
    result = sparseFermions.oplistExpectation(gradientOplist, eigenstate)
    nucRepulsion = flms[1].energyNuclear() - flms[2].energyNuclear()
    #if returnOplists:
    #    return (result[0,0] + nucRepulsion,originalOplist,gradientOplist)
    return result[0,0] + nucRepulsion


def openFermionFullVQEFiniteDifference(initialGeometry, plusGeometry, minusGeometry, resourceEstimator=None, openFermionDataDir=OPENFERMION_DATA_DIR, quantumChemPackage='pyscf'):
    os.chdir(openFermionDataDir)
    flmGeometries = [plusGeometry,minusGeometry]
    vqeResults = [cirqVQE.vqe(x.cartesian,
                             'STO-3G',
                              initialGeometry.multiplicity,
                              initialGeometry.charge,
                              str(initialGeometry.cartesian),
                              initialGeometry.numElectrons,
                              quantumChemPackage = quantumChemPackage,
                              resourceEstimator=resourceEstimator) for x in flmGeometries]
    energies = [x.fun for x in vqeResults]
    energyDifference = energies[0] - energies[1]
    return energyDifference

def openFermionPartitionedGradientVQE(initialGeometry, plusGeometry, minusGeometry, quantumChemPackage='pyscf', openFermionDataDir=OPENFERMION_DATA_DIR, resourceEstimator=None):
    os.chdir(openFermionDataDir)
    flmGeometries = [plusGeometry,minusGeometry]
    flms = [fermilibInterface.FermiLibMolecule(x.cartesian, "STO-3G", x.multiplicity, x.charge, str(x.cartesian), str(x.cartesian)) for x in flmGeometries]
    for flm in flms:
        flm.runPySCF()
    oplists = [x.electronicHamiltonian(0,cutoff=1e-12) for x in flms]

    fullOplist = oplists[0] + fermions.coefficient(-1., oplists[1])
    gradientOplist = fermions.oplistRemoveNegligibles(fermions.simplify(fullOplist), 1e-12)
    nucRepulsion = flms[0].energyNuclear() - flms[1].energyNuclear()
    result = cirqVQE.partitionedVQEExpectation(gradientOplist,
                                    initialGeometry.cartesian,
                                    "STO-3G",
                                    initialGeometry.multiplicity,
                                    initialGeometry.charge,
                                    str(initialGeometry.cartesian),
                                    initialGeometry.numElectrons,
                                    quantumChemPackage=quantumChemPackage,
                                    resourceEstimator=resourceEstimator)
    return result + nucRepulsion

def openFermionTaperedPartitionedGradientVQE(initialGeometry, plusGeometry, minusGeometry, quantumChemPackage='pyscf', openFermionDataDir=OPENFERMION_DATA_DIR, resourceEstimator=None,hfStateIndex = None,
                                             hfStateOplist = None):
    os.chdir(openFermionDataDir)
    flmGeometries = [plusGeometry,minusGeometry]
    flms = [fermilibInterface.FermiLibMolecule(x.cartesian, "STO-3G", x.multiplicity, x.charge, str(x.cartesian), str(x.cartesian)) for x in flmGeometries]
    for flm in flms:
        flm.runPySCF()
    oplists = [x.electronicHamiltonian(0,cutoff=1e-12) for x in flms]

    fullOplist = oplists[0] + fermions.coefficient(-1., oplists[1])
    gradientOplist = fermions.oplistRemoveNegligibles(fermions.simplify(fullOplist), 1e-12)
    nucRepulsion = flms[0].energyNuclear() - flms[1].energyNuclear()
    result = cirqVQE.partitionedTaperedVQEExpectation(gradientOplist,
                                               initialGeometry.cartesian,
                                               "STO-3G",
                                               initialGeometry.multiplicity,
                                               initialGeometry.charge,
                                               str(initialGeometry.cartesian),
                                               initialGeometry.numElectrons,
                                               quantumChemPackage=quantumChemPackage,
                                               resourceEstimator=resourceEstimator,
                                               hfStateIndex=hfStateIndex,
                                               hfStateOplist=hfStateOplist)
    return result + nucRepulsion








def openFermionGradientVQE(initialGeometry, plusGeometry, minusGeometry, quantumChemPackage='pyscf', openFermionDataDir=OPENFERMION_DATA_DIR, resourceEstimator=None):
    os.chdir(openFermionDataDir)
    flmGeometries = [plusGeometry,minusGeometry]
    flms = [fermilibInterface.FermiLibMolecule(x.cartesian, "STO-3G", x.multiplicity, x.charge, str(x.cartesian), str(x.cartesian)) for x in flmGeometries]
    for flm in flms:
        flm.runPySCF()
    oplists = [x.electronicHamiltonian(0,cutoff=1e-12) for x in flms]

    fullOplist = oplists[0] + fermions.coefficient(-1., oplists[1])
    gradientOplist = fermions.oplistRemoveNegligibles(fermions.simplify(fullOplist), 1e-12)
    nucRepulsion = flms[0].energyNuclear() - flms[1].energyNuclear()
    result = cirqVQE.vqeExpectation(gradientOplist,
                                    initialGeometry.cartesian,
                                    "STO-3G",
                                    initialGeometry.multiplicity,
                                    initialGeometry.charge,
                                    str(initialGeometry.cartesian),
                                    initialGeometry.numElectrons,
                                    quantumChemPackage=quantumChemPackage,
                                    resourceEstimator=resourceEstimator)
    return result + nucRepulsion

def openFermionEnergyVQE(geometry,quantumChemPackage='pyscf',openFermionDataDir=OPENFERMION_DATA_DIR,resourceEstimator=None):
    os.chdir(openFermionDataDir)
    flm = fermilibInterface.FermiLibMolecule(geometry.cartesian, "STO-3G", geometry.multiplicity, geometry.charge, str(geometry), str(geometry))
    flm.runPySCF()
    oplist = flm.electronicHamiltonian(0,cutoff=1e-12)
    nucRepulsion = flm.energyNuclear()
    result = cirqVQE.vqeExpectation(oplist,
                                    geometry.cartesian,
                                    "STO-3G",
                                    geometry.multiplicity,
                                    geometry.charge,
                                    str(geometry),
                                    geometry.numElectrons,
                                    quantumChemPackage=quantumChemPackage,
                                    resourceEstimator=resourceEstimator)
    return result + nucRepulsion

def run(initialGeometry,gradientFunction,estimateResources=False,countSampling=False,**kwargs):
    #integrator = VerletIntegrator(initialGeometry,potentialFunction)
    if estimateResources:
        resourceEstimator = ResourceEstimator.ResourceEstimator(countSampling=countSampling)
    else:
        resourceEstimator = None
    integrator = VerletIntegrator(initialGeometry, gradientFunction, resourceEstimator=resourceEstimator,**kwargs)
    while True:
        if estimateResources:
            yield (copy.deepcopy(integrator),copy.deepcopy(integrator.resourceEstimator))
        else:
            yield copy.deepcopy(integrator)
        integrator.evolve()

def runAdaptive(initialGeometry,
                badGradientFunction,
                goodGradientFunction,
                estimateResources=False,
                countSampling=False,
                gradientPrecision=GRADIENT_PRECISION,
                testerFunction = AdaptiveIntegrator.compareGradients,
                badEnergyFunction=None,
                goodEnergyFunction=None):
    if estimateResources:
        resourceEstimator = ResourceEstimator.ResourceEstimator(countSampling=countSampling)
    else:
        resourceEstimator = None
    integrator = AdaptiveIntegrator.AdaptiveIntegrator(initialGeometry,
                                                       badGradientFunction,
                                                       goodGradientFunction,
                                                       resourceEstimator=resourceEstimator,
                                                       gradientPrecision=gradientPrecision,
                                                       testerFunction=testerFunction,
                                                       badEnergyFunction=badEnergyFunction,
                                                       goodEnergyFunction=goodEnergyFunction)
    while True:
        if estimateResources:
            yield (copy.deepcopy(integrator),copy.deepcopy(integrator.resourceEstimator),integrator.step)
        else:
            yield (copy.deepcopy(integrator),integrator.step)
        integrator.evolve()


def runIndependent(initialGeometry,
                   badGradientFunction,
                   goodGradientFunction,
                   estimateResources=False,
                   countSampling=False,
                   gradientPrecision=GRADIENT_PRECISION,
                   testerFunction = AdaptiveIntegrator.compareGradients,
                   badEnergyFunction=None,
                   goodEnergyFunction=None):
    if estimateResources:
        resourceEstimator = ResourceEstimator.ResourceEstimator(countSampling=countSampling)
    else:
        resourceEstimator = None
    integrator = IndependentAdaptiveIntegrator.IndependentAdaptiveIntegrator(initialGeometry,
                                                                             badGradientFunction,
                                                                             goodGradientFunction,
                                                                             resourceEstimator=resourceEstimator,
                                                                             gradientPrecision=gradientPrecision,
                                                                             testerFunction=testerFunction,
                                                                             badEnergyFunction=badEnergyFunction,
                                                                             goodEnergyFunction=goodEnergyFunction)
    while True:
        if estimateResources:
            yield (copy.deepcopy(integrator),copy.deepcopy(integrator.resourceEstimator),integrator.step)
        else:
            yield (copy.deepcopy(integrator),integrator.step)
        integrator.evolve()


#def runLab(initialGeometry,)







'''
class Timestep:
    def __init__(self,
                 geometrySpec,
                 numElectrons,
                 numQubits,
                 charge,
                 multiplicity,
                 boolJWorBK,
                 precision,
                 time,
                 description
                 ):
        self.geometry = geometrySpec
        self.NUMELECTRONS = numElectrons
        self.NUMQUBITS = numQubits
        self.CHARGE = charge
        self.MULTIPLICITY = multiplicity
        self.BOOLJWORBK = boolJWorBK
        self.PRECISION = precision
        self.DESCRIPTION = description
        self.DELTAT = deltaT
        self.time = 0.

        self.oplist =


    def doOpenFermionStuff(self):
        oplist, uccOperator, energies = vqe.openFermionUCC(self.geometry.cartesian,
                                                           self.BASIS,
                                                           self.MULTIPLICITY,
                                                           self.CHARGE,
                                                           self.descrption + str(self.time),
                                                           numQubits=self.NUMQUBITS,
                                                           numElectrons=self.NUMELECTRONS,
                                                           boolJWorBK=self.BOOLJWORBK,
                                                           precision=self.PRECISION,
                                                           filename='{}_{}'.format(self.DESCRIPTION,self.time)
                                                           )
        
    

def vqeEnergy(geometry,basis,multiplicity,charge,description,numQubits,numElectrons,boolJWorBK,precision):
    oplist, uccOperator, energies = vqe.openFermionUCC(geometry,
                                                       basis,
                                                       multiplicity,
                                                       charge,
                                                       description,
                                                       numQubits=numQubits,
                                                       numElectrons=numElectrons,
                                                       boolJWorBK=boolJWorBK,
                                                       precision=precision,
                                                       filename=description)
    finalPOD = vqe.parameterisedOptdictFromOpenfermionUCC(uccOperator, numQubits, boolJWorBK)
    uccParameterisedOplist, startingValues = finalPOD.optimise().toOplist()
    testProg = pyquil.Program()
    testProg, symbolicParameters = vqe.ansatz(uccParameterisedOplist, testProg, numElectrons)
    pyquilHamiltonian = vqe.oplistToListPyquil(oplist)
    result = scipy.optimize.minimize(vqe.simulatorObjectiveFunction, args=(testProg, pyquilHamiltonian,symbolicParameters,list(startingValues.keys())),x0=list(startingValues.values()))
    resultPlusNucRep = result.fun + energies['NUCLEAR']
    return resultPlusNucRep,oplist
'''

'''SCRATCH'''

def profiler():
    initialH2Geometry = GeometrySpec(h2Geometry, [0.74], -0.005, h2Masses)
    simulator = run(initialH2Geometry, openFermionOplistExpectationGradient)
    for i in range(100):
        thing = next(simulator)
        print(thing.a)
    return

def tester(its=500,test='H2'):
    starttime = datetime.datetime.now()
    accs = []
    vels = []
    xs = []
    t = 0.
    ts = []
    x2s = []
    carts = []
    if test == 'H3':
        initialGeometry = GeometrySpec(colinearH3JacobiGeometry, [0.71, 1.555], [0., -0.0], H3JacobiMasses, multiplicity=2)
    elif test == 'H2':
        initialGeometry = GeometrySpec(h2Geometry, [1.], [0.0], [0.5])
    else:
        raise ValueError('Invalid test')
    simulator = run(initialGeometry,openFermionOplistExpectationGradient)
    for i in range(its):
        thing = next(simulator)
        #accs.append(thing.a[0])
        #vels.append(thing.v[0])
        xs.append(thing.x[0])
        #x2s.append(thing.x[1])
        t += DT
        ts.append(t)
        carts.append(thing.baseGeometry.cartesian)
    endtime = datetime.datetime.now()
    print('{}'.format(endtime-starttime))
    return (xs,x2s,carts)

def compareGradientFunctions(its=200,
                             geometryFunction=colinearH3JacobiGeometry,
                             startingCoords=[0.71,1.555],
                             startingVels=[0.,0.],
                             masses=H3JacobiMasses,
                             multiplicity=2,
                             gradientFunctions=[openFermionOplistExpectationGradient,openFermionGradientFunction],
                             returnResources=False,
                             countSampling=False):
    REPORT_FREQUENCY = datetime.timedelta(seconds=60)
    lastReport = datetime.datetime.now()
    startTime = lastReport
    results = []
    fullNumTrials = its * len(gradientFunctions)
    doneTrials = 0
    for thisGradientFunction in gradientFunctions:
        result = []
        initialGeometry = GeometrySpec(geometryFunction, startingCoords, startingVels, masses, multiplicity)
        simulator = run(initialGeometry,thisGradientFunction,returnResources,countSampling)

        for _ in range(its+1):
            thing = next(simulator)
            result.append(thing)
            doneTrials += 1
            if datetime.datetime.now() - lastReport > REPORT_FREQUENCY:
                fracDone = doneTrials/float(fullNumTrials)
                estTime = (datetime.datetime.now() - startTime) * ((1./fracDone) - 1.)
                print('{}  {}% DONE, EST TIME TO COMPLETE: {}'.format(datetime.datetime.now(),fracDone*100,estTime))
                lastReport = datetime.datetime.now()
        results.append(result)
    return(results)

def compareGeometries(its=200,
                      geometryFunctions=[colinearH3JacobiGeometry,colinearH3Cartesian],
                      startingCoords=[[0.71,1.555],[0., 0.71, 1.91]],
                      startingVels=[[0.,0.],[0.,0.,0.]],
                      masses=[H3JacobiMasses,H3CartesianMasses],
                      multiplicities=[2,2],
                      gradientFunction=openFermionOplistExpectationGradient):

    REPORT_FREQUENCY = datetime.timedelta(seconds=60)
    lastReport = datetime.datetime.now()
    startTime = lastReport
    results = []
    fullNumTrials = (its+1) * len(geometryFunctions)

    doneTrials = 0
    for i,thisGeometryFunction in enumerate(geometryFunctions):
        result = []
        initialGeometry = GeometrySpec(thisGeometryFunction, startingCoords[i], startingVels[i], masses[i], multiplicities[i])
        simulator = run(initialGeometry,gradientFunction)
        for _ in range(its+1):
            thing = next(simulator)
            result.append(thing.x)
            doneTrials += 1
            if datetime.datetime.now() - lastReport > REPORT_FREQUENCY:
                fracDone = doneTrials / float(fullNumTrials)
                estTime = (datetime.datetime.now() - startTime) * ((1. / fracDone) - 1.)
                print('{} {} of {} ({}%) DONE, EST TIME TO COMPLETE: {}'.format(datetime.datetime.now(), doneTrials, fullNumTrials, fracDone * 100, estTime))
                lastReport = datetime.datetime.now()
        results.append(result)
    return (results)



def plotCartesian(cartesianGeometry,bounds=5.):
    matplotlib.pyplot.cla()
    xs = [thing[1][0] for thing in cartesianGeometry]
    ys = [thing[1][2] for thing in cartesianGeometry]
    zs = [thing[1][3] for thing in cartesianGeometry]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    Axes3D.scatter(xs, ys, zs)
    ax.axes.set_xlim

import numpy
from tqdm import tqdm

def openFermionPotentialAndHamiltonianFunction(geometry,resourceEstimator=None,
                                 openFermionDataDir=OPENFERMION_DATA_DIR,
                                 quantumChemPackage='pyscf',
                                 includeNuclear=False):
    os.chdir(openFermionDataDir)
    flmGeometry = geometry.cartesian
    flm = fermilibInterface.FermiLibMolecule(flmGeometry, "STO-3G", geometry.multiplicity, geometry.charge, str(geometry), str(geometry))
    runQuantumChemistry(flm,quantumChemPackage,runFCI=True)
    energy = flm.energyFCI()
    if includeNuclear == True:
        energy += flm.energyNuclear()
    return [energy,flm.electronicHamiltonian(0,1e-12)]

def openFermionHartreeFockEnergyAndHamiltonian(geometry,
                                 resourceEstimator=None,
                                 openFermionDataDir=OPENFERMION_DATA_DIR,
                                 quantumChemPackage='pyscf',
                                 includeNuclear=False):
    os.chdir(openFermionDataDir)
    flm = fermilibInterface.FermiLibMolecule(geometry.cartesian,
                                             "STO-3G",
                                             geometry.multiplicity,
                                             geometry.charge,
                                             str(geometry),
                                             str(geometry))


    runQuantumChemistry(flm,quantumChemPackage)
    energy = flm.energySCF()
    return [energy,flm.electronicHamiltonian(0,1e-12)]

def noncontextualPotentialFunctionAndHamiltonian(geometry,quantumChemPackage='pyscf',includeNuclear=True):
    flmGeometry = geometry.cartesian
    flm = fermilibInterface.FermiLibMolecule(flmGeometry, "STO-3G", geometry.multiplicity, geometry.charge,
                                             str(geometry), str(geometry))
    runQuantumChemistry(flm, quantumChemPackage)
    oplist= flm.electronicHamiltonian(0,1e-12)
    hamDict = noncontextuality_quasi_quantized.oplist2ham_dict(oplist)
    ncHamDict = noncontextuality_quasi_quantized.toNCHamiltonian_fast(hamDict,
                                                                      weighting=None,
                                                                      strategy='greedy',
                                                                      step_size=1,
                                                                      show_progress=False)
    ncOplist = noncontextuality_quasi_quantized.ham_dict2Oplist(ncHamDict)
    eigval,eigvec = sparseFermions.getTrueEigensystem(ncOplist)
    if includeNuclear:
        eigval += flm.energyNuclear()
    return [eigval,ncOplist]


def potentialEnergySurface1D(initialGeometry,potentialFunction=openFermionPotentialFunction,xVals=None,explicitNuclear=False):
    result = [0] * len(xVals)
    for xi, x in enumerate(tqdm(xVals)):
        coordinates = [x]
        thisGeometry = initialGeometry.copy()
        thisGeometry.update(coordinates)
        try:
            thisEnergy = potentialFunction(thisGeometry,includeNuclear=explicitNuclear)
        except:
            thisEnergy = numpy.NaN
        #thisEnergy = potentialFunction(thisGeometry,includeNuclear=explicitNuclear)
        result[xi] = thisEnergy
    return result

def potentialEnergySurface2D(initialGeometry=TEST_GEOMETRY_SPEC,potentialFunction=openFermionPotentialFunction,xVals=None,yVals=None,explicitNuclear=False):
    result = numpy.zeros((len(xVals), len(yVals)))
    for xi, x in enumerate(tqdm(xVals)):
        for yi, y in enumerate(yVals):
            coordinates = [x,y]
            thisGeometry = initialGeometry.copy()
            thisGeometry.update(coordinates)
            try:
                thisEnergy = potentialFunction(thisGeometry,includeNuclear=explicitNuclear)
            except:
                thisEnergy = numpy.NaN
            result[xi,yi] = thisEnergy
    return result

def potentialEnergySurfaceValence(initialGeometry=TEST_GEOMETRY_SPEC,potentialFunction=openFermionPotentialFunction,valenceToJacobiFunction=H3ValenceToJacobi,xVals=None,yVals=None,explicitNuclear=False):
    result = numpy.zeros((len(xVals), len(yVals)))
    for xi, x in enumerate(tqdm(xVals)):
        for yi, y in enumerate(yVals):
            coordinates = [x,y]
            jacCoordinates = valenceToJacobiFunction(*coordinates)
            thisGeometry = initialGeometry.copy()
            thisGeometry.update(jacCoordinates)
            try:
                thisEnergy = potentialFunction(thisGeometry,includeNuclear=explicitNuclear)
            except:
                thisEnergy = numpy.NaN
            result[xi,yi] = thisEnergy
    return result