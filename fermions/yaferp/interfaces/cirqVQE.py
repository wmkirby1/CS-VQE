import cirq
from yaferp.interfaces import fermilibInterface
import openfermion
from yaferp.general import optDict
import copy
from yaferp.circuits import circuit
import numpy
import scipy
import random
from yaferp.misc import tapering
from yaferp.vqe import termRecombination
from yaferp.orderings import directOrdering

def jwHartreeFock(qubits,numElectrons):
    for i in range(numElectrons):
        yield cirq.X(qubits[i])

def uccOperator(qubits, uccParameterisedOplist):
    ourCircuit = circuit.oplistToCircuit(uccParameterisedOplist)
    return ourCircuit.toCirq(qubits)

def ansatz(qubits,numElectrons,uccParameterisedOplist):
    circuit = cirq.Circuit()
    circuit.append(jwHartreeFock(qubits,numElectrons))
    circuit.append(uccOperator(qubits,uccParameterisedOplist))
    return circuit

def pauliStringBasisChange(qubits,pauliString):
    circuit = cirq.Circuit()
    for i, pauli in enumerate(reversed(pauliString)):
        if pauli == 1:
            circuit.append(cirq.H(qubits[i]))
        elif pauli == 2:
            circuit.append(cirq.rx((-1. * numpy.pi)/2.)(qubits[i]))

    return circuit


def vqeTermCircuit(qubits,opterm,numElectrons,ansatzOplist,onlyAnsatz=False,onlyHamiltonian=False,noHFStep=False):
    pauliString = opterm[1]
    cirquit = cirq.Circuit()
    if not onlyHamiltonian:
        if not noHFStep:
            cirquit.append(jwHartreeFock(qubits,numElectrons))
        cirquit.append(uccOperator(qubits,ansatzOplist))
    if not onlyAnsatz:
        cirquit.append(pauliStringBasisChange(qubits,pauliString))
    return cirquit

def vqeOplistCircuits(qubits,oplist,numElectrons,ansatzOplist,onlyAnsatz=False,onlyHamiltonian=False,noHFStep=False):
    if onlyAnsatz:
        result = vqeTermCircuit(qubits,oplist,numElectrons,ansatzOplist,True,False,noHFStep)
    else:
        result = [vqeTermCircuit(qubits,opterm,numElectrons,ansatzOplist,False,onlyHamiltonian,noHFStep) for opterm in oplist]
    return result

def parameterisedCircuitSimulate(circuit,parameters,qubits=None,dtype=numpy.complex128,initialState=None):
    simulator = cirq.Simulator(dtype=dtype)
    resolver = cirq.ParamResolver(parameters)
    if qubits is not None:
        result = simulator.simulate(circuit,resolver,initial_state=initialState,qubit_order=qubits)
    else:
        result = simulator.simulate(circuit,resolver,initial_state=initialState)
    return result.state_vector()
''' UCC STUFF (TODO:REFACTOR THIS INTO A SEPARATE MODULE) '''


def runQuantumChemistry(flm,package='pyscf',runCCSD=0,runFCI=0):
    if package.lower() == 'psi4':
        flm.runPSI4(runCCSD=runCCSD,runFCI=runFCI)
    elif package.lower() == 'pyscf':
        flm.runPySCF(runCCSD=runCCSD,runFCI=runFCI)
    else:
        raise ValueError('unrecognised quantum chemistry package')
    return


def openFermionUCC(geometry,
                   basis,
                   multiplicity,
                   charge,
                   description,
                   #numQubits,
                   numElectrons,
                   boolJWorBK=0,
                   precision=1e-12,
                   filename=None,
                   runFCI=0,
                   quantumChemPackage='pyscf'):
    id = random.getrandbits(128)
    flm = fermilibInterface.FermiLibMolecule(geometry, basis, multiplicity, charge, '{}_{}'.format(description, id), filename='{}_{}'.format(description, id))
    runQuantumChemistry(flm,quantumChemPackage,runCCSD=1,runFCI=runFCI)
    oplist = flm.electronicHamiltonian(boolJWorBK,precision)
    numQubits = len(oplist[0][1])
    molecularHamiltonian = flm.molecularData.get_molecular_hamiltonian()
    singleAmps = flm.molecularData.ccsd_single_amps
    doubleAmps = flm.molecularData.ccsd_double_amps
   # uccOperator1 = openfermion.utils.uccsd_generator(singleAmps, doubleAmps)
    '''
    packedAmplitudes = openfermion.utils.uccsd_singlet_get_packed_amplitudes(
        singleAmps,
        doubleAmps,
        numQubits,
        numElectrons)
    uccOperator2 = openfermion.utils.uccsd_singlet_generator(
        packedAmplitudes,
        numQubits,
        numElectrons,
        anti_hermitian=False)
    uccOperator = openfermion.normal_ordered(uccOperator2)
    '''
    uccOperator = openfermion.normal_ordered(openfermion.utils.uccsd_generator(
        singleAmps,
        doubleAmps,
        anti_hermitian=False))
    energies = {'FCI':flm.energyFCI(),'NUCLEAR':flm.energyNuclear(),'CCSD':flm.energyCCSD()}
    return oplist,uccOperator,energies

def conjugateUCCTerm(uccTerm):
    uccTermPart1 = tuple(reversed(tuple([(x[0],1-x[1]) for x in uccTerm])))
    return uccTermPart1

def parameterisedOptdictFromOpenfermionUCC(ofUCC,numQubits,boolJWorBK):
    listPODs = []
    for thisFermionicTerm in ofUCC.terms:
        coefficient = ofUCC.terms[thisFermionicTerm]
        workingOpDict = optDict.ferm_op(thisFermionicTerm[0][0], numQubits, thisFermionicTerm[0][1], boolJWorBK).opDict

        for thisOperator in thisFermionicTerm[1:]:
            fred = optDict.ferm_op(thisOperator[0], numQubits, thisOperator[1], boolJWorBK).opDict
            workingOpDict = workingOpDict.product(fred)
        hermetianOptDict = optDict.parameterisedOptDict(workingOpDict, coefficient)
        conjugateTerm = conjugateUCCTerm(thisFermionicTerm)
        workingConjugateOpDict = optDict.ferm_op(conjugateTerm[0][0], numQubits, conjugateTerm[0][1], boolJWorBK).opDict

        for thisOperator in conjugateTerm[1:]:
            fred = optDict.ferm_op(thisOperator[0], numQubits, thisOperator[1], boolJWorBK).opDict
            workingConjugateOpDict = workingConjugateOpDict.product(fred)
        antihermetianOptDict = optDict.parameterisedOptDict(workingConjugateOpDict, coefficient, negated=True)

        hermetianOptDict = hermetianOptDict.sumSkippingRelabel(antihermetianOptDict)

        listPODs.append(hermetianOptDict)

    finalPOD = copy.deepcopy(listPODs[0])
    for thisPOD in listPODs[1:]:
        finalPOD.sum(thisPOD)

    return finalPOD

def expectationValueComputationalBasis(state,measuredQubitIndices):
    #obvs this will blow up.
    result = 0
    measuredQubitMask = 0
    for thisIndex in measuredQubitIndices:
        measuredQubitMask = measuredQubitMask | 1 << thisIndex

    cooState = scipy.sparse.coo_matrix(state)
    for index,coefficient in zip(cooState.row,cooState.data):
        maskedIndex = index & measuredQubitMask
        parity = bin(maskedIndex).count("1") % 2  #this would be the point i stopped trying to do clever bit-level hacks in python
        multiplier = 1. - 2.*parity
        result += multiplier * (abs(coefficient)**2.)

    return result

def expectationValueOpterm(state,opterm):
    pauliString = opterm[1]
    involvedIndices = [i for i,x in enumerate(pauliString) if x != 0]
    expectation = expectationValueComputationalBasis(state,involvedIndices) * opterm[0]
    return expectation

def vqeObjectiveFunction(parameterValues,parameterSymbols,ansatzCircuit,hamiltonianCircuits,qubits,oplist,resourceEstimator=None):
    #print([type(x) for x in parameterSymbols])
    parameters = {parameterSymbols[i]:parameterValues[i] for i in range(len(parameterValues))}
    #print(parameters)
    ansatzState = parameterisedCircuitSimulate(ansatzCircuit,parameters,qubits=qubits)
    #print(parameters)
    finalStates = [numpy.transpose(numpy.mat(parameterisedCircuitSimulate(thisCircuit,parameters,initialState=ansatzState,qubits=qubits))) for thisCircuit in hamiltonianCircuits]
    expectations = [expectationValueOpterm(finalStates[i],oplist[i]) for i in range(len(oplist))]
    energy = sum(expectations)

    if not resourceEstimator is None:
        for i,x in enumerate(hamiltonianCircuits):
            resourceEstimator.addCirquit(x,opterm=oplist[i])
            resourceEstimator.addCirquit(ansatzCircuit,repeatSamples=True)

    return energy

def vqe(geometry,
        basis='STO-3G',
        multiplicity = 1,
        charge = 0,
        description = '',
        numElectrons = None,
        quantumChemPackage = 'pyscf',
        runFCI=0,
        boolJWorBK=0,
        resourceEstimator=None,
        objectiveFunction=vqeObjectiveFunction):

    oplist, uccOperator, energies = openFermionUCC(geometry,
                                                   basis,
                                                   multiplicity,
                                                   charge,
                                                   description,
                                                   numElectrons,
                                                   runFCI=runFCI,
                                                   quantumChemPackage=quantumChemPackage)
    numQubits = len(oplist[0][1])
    pod = parameterisedOptdictFromOpenfermionUCC(uccOperator,numQubits,boolJWorBK)
    qubits = [cirq.LineQubit(i) for i in range(numQubits)]
    ansatzOplist, rawParameters = pod.toOplist()
    parameterNames = list(rawParameters.keys())
    initialParameters = list(rawParameters.values())
    ansatzCircuit = vqeOplistCircuits(qubits,oplist,numElectrons,ansatzOplist,True)
    hamiltonianCircuits = vqeOplistCircuits(qubits,oplist,numElectrons,ansatzOplist,onlyHamiltonian=True)
    result = scipy.optimize.minimize(objectiveFunction,
                            x0=initialParameters,
                            args=(parameterNames,ansatzCircuit,hamiltonianCircuits,qubits,oplist,resourceEstimator),
                            method='L-BFGS-B')
                            #options={'fatol': 0.00001,'xatol': 0.00001})
    return result


def vqeExpectation(oplist,
                   geometry,
                   basis='STO-3G',
                   multiplicity = 1,
            charge = 0,
             description = '',
             numElectrons = None,
             quantumChemPackage = 'pyscf',
             runFCI=0,
             boolJWorBK=0,
                   resourceEstimator=None):
    '''take an oplist and give the expectation value of the oplist with the UCCSD ground state'''
    vqeResult = vqe(geometry,
                 basis,
                 multiplicity,
                 charge,
                 description,
                 numElectrons,
                 quantumChemPackage,
                 runFCI,
                 boolJWorBK,
                    resourceEstimator)

    numQubits = len(oplist[0][1])
    qubits = [cirq.LineQubit(i) for i in range(numQubits)]
    _, uccOperator, energies = openFermionUCC(geometry,
                                                   basis,
                                                   multiplicity,
                                                   charge,
                                                   description,
                                                   numElectrons,
                                                   runFCI=runFCI,
                                                   quantumChemPackage=quantumChemPackage)
    pod = parameterisedOptdictFromOpenfermionUCC(uccOperator,numQubits,boolJWorBK)
    ansatzOplist, rawParameters = pod.toOplist()
    parameterNames = list(rawParameters.keys())
    ansatzCircuit = vqeOplistCircuits(qubits,oplist,numElectrons,ansatzOplist,onlyAnsatz=True)
    optimisedParameters = vqeResult.x
    hamiltonianCircuits = vqeOplistCircuits(qubits,oplist,numElectrons,ansatzOplist,onlyHamiltonian=True)
    result = vqeObjectiveFunction(optimisedParameters,parameterNames,ansatzCircuit,hamiltonianCircuits,qubits,oplist)
    #if not resourceEstimator is None:
    #    for i,x in enumerate(hamiltonianCircuits):
    #        resourceEstimator.addCirquit(x,opterm=oplist[i])
    #        resourceEstimator.addCirquit(ansatzCircuit,repeatSamples=True)
    return result

def taperedHFCirquit(qubits,oplist):
    pauliString = oplist[0][1]
    print(pauliString)
    for i,x in enumerate(pauliString):
        if x == 1:
            yield cirq.X(qubits[i])
        elif x == 2:
            yield cirq.Y(qubits[i])
        elif x == 3:
            print('3')
            yield cirq.Z(qubits[i])

def taperedVQE(geometry,
               basis='STO-3G',
               multiplicity = 1,
               charge = 0,
               description = '',
               numElectrons = None,
               quantumChemPackage = 'pyscf',
               runFCI=0,
               boolJWorBK=0,
               resourceEstimator=None,
               objectiveFunction=vqeObjectiveFunction,
               hfStateIndex = None,
               hfStateOplist = None):

    oplist, uccOperator, energies = openFermionUCC(geometry,
                                                           basis,
                                                           multiplicity,
                                                           charge,
                                                           description,
                                                           numElectrons,
                                                           runFCI=runFCI,
                                                           quantumChemPackage=quantumChemPackage)

    kernelBasis = tapering.oplistParityKernel(oplist)
    taperedOplist = tapering.taperOplist(oplist,hfStateIndex)

    origNumQubits = len(oplist[0][1])
    numQubits = len(taperedOplist[0][1])
    qubits = [cirq.LineQubit(i) for i in range(numQubits)]


    #print(uccOperator)
    pod = parameterisedOptdictFromOpenfermionUCC(uccOperator,origNumQubits,boolJWorBK)
    ansatzOplist, rawParameters = pod.toOplist()
    taperedAnsatzOplist = tapering.taperOplist(ansatzOplist,hfStateIndex,kernelBasis)

    prepCirquit = cirq.Circuit()
    #taperedHFCirquit(qubits,taperedhfOplist)
    prepCirquit.append(taperedHFCirquit(qubits,hfStateOplist))

    ansatzCircuit = vqeOplistCircuits(qubits,taperedOplist,numElectrons,taperedAnsatzOplist,True,False,True)
    prepCirquit.append(ansatzCircuit)

    parameterNames = list(rawParameters.keys())
    initialParameters = list(rawParameters.values())
    #ansatzCircuit = vqeOplistCircuits(qubits,taperedOplist,numElectrons,taperedAnsatzOplist,True)
    hamiltonianCircuits = vqeOplistCircuits(qubits,taperedOplist,numElectrons,taperedAnsatzOplist,onlyHamiltonian=True)
    result = scipy.optimize.minimize(objectiveFunction,
                                     x0=initialParameters,
                                     args=(parameterNames,prepCirquit,hamiltonianCircuits,qubits,taperedOplist,resourceEstimator),
                                     method='L-BFGS-B')
    #options={'fatol': 0.00001,'xatol': 0.00001})
    return result

def partitionedExpectationValue(state,coefficient,indicesToMeasure):
    expectation = expectationValueComputationalBasis(state,indicesToMeasure) * coefficient
    return expectation

def partitionedVQEObjectiveFunction(parameterValues,parameterSymbols,ansatzCircuit,hamiltonianCircuits,qubits,listIndicesToMeasure,listCoefficients,resourceEstimator=None,offset=0.):
    parameters = {parameterSymbols[i]:parameterValues[i] for i in range(len(parameterValues))}
    ansatzState = parameterisedCircuitSimulate(ansatzCircuit,parameters)
    finalStates = [numpy.transpose(numpy.mat(parameterisedCircuitSimulate(thisCircuit,parameters,initialState=ansatzState,qubits=qubits))) for thisCircuit in hamiltonianCircuits]
    expectations = [partitionedExpectationValue(finalStates[i],listCoefficients[i],listIndicesToMeasure[i]) for i in range(len(hamiltonianCircuits))]
    energy = sum(expectations) + offset
    if not resourceEstimator is None:
        for i,x in enumerate(hamiltonianCircuits):
            resourceEstimator.addCirquit(x,opterm=[listCoefficients[i],None])
            resourceEstimator.addCirquit(ansatzCircuit,repeatSamples=True)
    return energy

def partitionedVQEOplistCircuits(qubits,oplist,numElectrons,ansatzOplist,onlyAnsatz=False,onlyHamiltonian=False,noHFStep=False):
    if onlyAnsatz:
        result = vqeTermCircuit(qubits,oplist,numElectrons,ansatzOplist,True,False,noHFStep)
        return result
    else:
        numQubits = len(oplist[0][1])
        colouredOplist = directOrdering.greedyColourOplist(oplist,1)
        colouringData = termRecombination.optimalCircuitsFromAnticommutingSets(colouredOplist) #fullGammas,fullCircuits,fullUndoableIndices,fullListIndicesToMeasure, offset
        circuits = [x.toCirq(qubits,set_to_true_if_you_are_panicking=True) for x in colouringData[1]]
        newListIndicesToMeasure = [[(numQubits-y) -1 for y in x] for x in colouringData[3]] #clunky hack fix - our qubit indexing is reversed from cirq's
        return colouringData[0],circuits,newListIndicesToMeasure,colouringData[4] #fullGammas,fullCircuits,fullListIndicesToMeasure, offset

def partitionedVQE(geometry,
                   basis='STO-3G',
                   multiplicity = 1,
                   charge = 0,
                   description = '',
                   numElectrons = None,
                   quantumChemPackage = 'pyscf',
                   runFCI=0,
                   boolJWorBK=0,
                   resourceEstimator=None,
                   objectiveFunction=partitionedVQEObjectiveFunction,
                   hfStateIndex = None,
                   hfStateOplist = None):

    oplist, uccOperator, energies = openFermionUCC(geometry,
                                                           basis,
                                                           multiplicity,
                                                           charge,
                                                           description,
                                                           numElectrons,
                                                           runFCI=runFCI,
                                                           quantumChemPackage=quantumChemPackage)

    numQubits = len(oplist[0][1])
    qubits = [cirq.LineQubit(i) for i in range(numQubits)]



    pod = parameterisedOptdictFromOpenfermionUCC(uccOperator,numQubits,boolJWorBK)
    ansatzOplist, rawParameters = pod.toOplist()

    prepCirquit = cirq.Circuit()
    ansatzCircuit = vqeOplistCircuits(qubits,oplist,numElectrons,ansatzOplist,True,False,False)
    prepCirquit.append(ansatzCircuit)

    parameterNames = list(rawParameters.keys())
    initialParameters = list(rawParameters.values())
    gammas,hamiltonianCircuits,involvedIndices,offset = partitionedVQEOplistCircuits(qubits,oplist,numElectrons,ansatzOplist,onlyHamiltonian=True)

    result = scipy.optimize.minimize(objectiveFunction,
                                     x0=initialParameters,
                                     args=(parameterNames,prepCirquit,hamiltonianCircuits,qubits,involvedIndices,gammas,resourceEstimator,offset),
                                     method='L-BFGS-B')
    return result

def partitionedVQEExpectation(oplist,
                              geometry,
                   basis='STO-3G',
                   multiplicity = 1,
                   charge = 0,
                   description = '',
                   numElectrons = None,
                   quantumChemPackage = 'pyscf',
                   runFCI=0,
                   boolJWorBK=0,
                   resourceEstimator=None,
                   objectiveFunction=partitionedVQEObjectiveFunction,
                   hfStateIndex = None,
                   hfStateOplist = None):

    vqeResult = partitionedVQE(geometry,
                    basis,
                    multiplicity,
                    charge,
                    description,
                    numElectrons,
                    quantumChemPackage,
                    runFCI,
                    boolJWorBK,
                    resourceEstimator)

    _, uccOperator, energies = openFermionUCC(geometry,
                                                   basis,
                                                   multiplicity,
                                                   charge,
                                                   description,
                                                   numElectrons,
                                                   runFCI=runFCI,
                                                   quantumChemPackage=quantumChemPackage)

    numQubits = len(oplist[0][1])
    qubits = [cirq.LineQubit(i) for i in range(numQubits)]


    pod = parameterisedOptdictFromOpenfermionUCC(uccOperator,numQubits,boolJWorBK)
    ansatzOplist, rawParameters = pod.toOplist()

    prepCirquit = cirq.Circuit()
    ansatzCircuit = vqeOplistCircuits(qubits,oplist,numElectrons,ansatzOplist,True,False,False)
    prepCirquit.append(ansatzCircuit)

    parameterNames = list(rawParameters.keys())

    gammas,hamiltonianCircuits,involvedIndices,offset = partitionedVQEOplistCircuits(qubits,oplist,numElectrons,ansatzOplist,onlyHamiltonian=True)

    optimisedParameters = vqeResult.x
    #hamiltonianCircuits = vqeOplistCircuits(qubits,oplist,numElectrons,ansatzOplist,onlyHamiltonian=True)
    result = objectiveFunction(optimisedParameters,parameterNames,prepCirquit,hamiltonianCircuits,qubits,involvedIndices,gammas,resourceEstimator,offset)

    return result


def partitionedTaperedVQE(geometry,
                          basis='STO-3G',
                          multiplicity = 1,
                          charge = 0,
                          description = '',
                          numElectrons = None,
                          quantumChemPackage = 'pyscf',
                          runFCI=0,
                          boolJWorBK=0,
                          resourceEstimator=None,
                          objectiveFunction=partitionedVQEObjectiveFunction,
                          hfStateIndex = None,
                          hfStateOplist = None):

    oplist, uccOperator, energies = openFermionUCC(geometry,
                                                           basis,
                                                           multiplicity,
                                                           charge,
                                                           description,
                                                           numElectrons,
                                                           runFCI=runFCI,
                                                           quantumChemPackage=quantumChemPackage)

    kernelBasis = tapering.oplistParityKernel(oplist)
    taperedOplist = tapering.taperOplist(oplist,hfStateIndex)
    origNumQubits = len(oplist[0][1])
    numQubits = len(taperedOplist[0][1])
    qubits = [cirq.LineQubit(i) for i in range(numQubits)]

    pod = parameterisedOptdictFromOpenfermionUCC(uccOperator,origNumQubits,boolJWorBK)
    ansatzOplist, rawParameters = pod.toOplist()

    taperedAnsatzOplist = tapering.taperOplist(ansatzOplist,hfStateIndex,kernelBasis)

    prepCirquit = cirq.Circuit()
    #taperedHFCirquit(qubits,taperedhfOplist)
    prepCirquit.append(taperedHFCirquit(qubits,hfStateOplist))
    ansatzCircuit = vqeOplistCircuits(qubits,taperedOplist,numElectrons,taperedAnsatzOplist,True,False,True)
    prepCirquit.append(ansatzCircuit)

    parameterNames = list(rawParameters.keys())
    initialParameters = list(rawParameters.values())
    gammas,hamiltonianCircuits,involvedIndices,offset = partitionedVQEOplistCircuits(qubits,taperedOplist,numElectrons,ansatzOplist,onlyHamiltonian=True)
    result = scipy.optimize.minimize(objectiveFunction,
                                     x0=initialParameters,
                                     args=(parameterNames,prepCirquit,hamiltonianCircuits,qubits,involvedIndices,gammas,resourceEstimator,offset),
                                     method='L-BFGS-B')
    return result



def partitionedTaperedVQEExpectation(oplist,
                                     geometry,
                                     basis='STO-3G',
                                     multiplicity = 1,
                                     charge = 0,
                                     description = '',
                                     numElectrons = None,
                                     quantumChemPackage = 'pyscf',
                                     runFCI=0,
                                     boolJWorBK=0,
                                     resourceEstimator=None,
                                     objectiveFunction=partitionedVQEObjectiveFunction,
                                     hfStateIndex = None,
                                     hfStateOplist = None):

    vqeResult = partitionedTaperedVQE(geometry,
                                      basis,
                                      multiplicity,
                                      charge,
                                      description,
                                      numElectrons,
                                      quantumChemPackage,
                                      runFCI,
                                      boolJWorBK,
                                      resourceEstimator,
                                      hfStateIndex=hfStateIndex,
                                      hfStateOplist=hfStateOplist)

    _, uccOperator, energies = openFermionUCC(geometry,
                                                      basis,
                                                      multiplicity,
                                                      charge,
                                                      description,
                                                      numElectrons,
                                                      runFCI=runFCI,
                                                      quantumChemPackage=quantumChemPackage)

    kernelBasis = tapering.oplistParityKernel(oplist)
    taperedOplist = tapering.taperOplist(oplist,hfStateIndex)
    origNumQubits = len(oplist[0][1])
    numQubits = len(taperedOplist[0][1])
    qubits = [cirq.LineQubit(i) for i in range(numQubits)]



    #pod = parameterisedOptdictFromOpenfermionUCC(uccOperator,numQubits,boolJWorBK)
    #ansatzOplist, rawParameters = pod.toOplist()

    pod = parameterisedOptdictFromOpenfermionUCC(uccOperator,origNumQubits,boolJWorBK)
    ansatzOplist, rawParameters = pod.toOplist()

    taperedAnsatzOplist = tapering.taperOplist(ansatzOplist,hfStateIndex,kernelBasis)

    prepCirquit = cirq.Circuit()
    #taperedHFCirquit(qubits,taperedhfOplist)
    prepCirquit.append(taperedHFCirquit(qubits,hfStateOplist))
    ansatzCircuit = vqeOplistCircuits(qubits,taperedOplist,numElectrons,taperedAnsatzOplist,True,False,True)
    prepCirquit.append(ansatzCircuit)




    #prepCirquit = cirq.Circuit()
    #ansatzCircuit = vqeOplistCircuits(qubits,oplist,numElectrons,ansatzOplist,True,False,False)
    #prepCirquit.append(ansatzCircuit)

    parameterNames = list(rawParameters.keys())

    gammas,hamiltonianCircuits,involvedIndices,offset = partitionedVQEOplistCircuits(qubits,taperedOplist,numElectrons,ansatzOplist,onlyHamiltonian=True)

    optimisedParameters = vqeResult.x
    #hamiltonianCircuits = vqeOplistCircuits(qubits,oplist,numElectrons,ansatzOplist,onlyHamiltonian=True)
    result = objectiveFunction(optimisedParameters,parameterNames,prepCirquit,hamiltonianCircuits,qubits,involvedIndices,gammas,resourceEstimator,offset)

    return result

''' SCRATCH '''
def h2Geometry(bondlength):
    return [('H', (0., 0., 0.)), ('H', (0., 0., bondlength))]