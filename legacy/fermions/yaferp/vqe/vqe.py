#import pyquil
import numpy
from yaferp.circuits import circuit
from yaferp.interfaces import fermilibInterface
import openfermion
from yaferp.general import optDict
import copy
import scipy

def pauliStringBasisChangeToPyquil(pauliString, pyQuilCircuit,pyQuilReadoutRegister=None):
  #  nonIdentityGates = [x for x in enumerate(opterm) if x[1] != 0]
   # if nonIdentityGates:
    #    numNonIdentityGates = len(nonIdentityGates)
    #print(pauliString)
    for i,op in enumerate(reversed(pauliString)):
        if op == 1:
            pyQuilCircuit += pyquil.gates.H(i)
        elif op == 2:
            pyQuilCircuit += pyquil.gates.RX((-1. * numpy.pi)/2., i)

        if op != 0 and not pyQuilReadoutRegister is None:
            pyQuilCircuit += pyquil.gates.MEASURE(i,pyQuilReadoutRegister[i])
    return pyQuilCircuit


def jwHartreeFockPyquil(numElectrons, program):
    for i in range(numElectrons):
        program += pyquil.gates.X(i)
    return program


def ansatz(oplist, program, numElectrons):
    program = jwHartreeFockPyquil(numElectrons, program)
    ourCircuit = circuit.oplistToCircuit(oplist)
    memoryRequirement = ourCircuit.numRotations()
    theta = program.declare('theta', memory_type='REAL', memory_size=memoryRequirement)
    program, parameters = ourCircuit.pyquil(theta, program)
    return program, parameters


def prepareProgram(oplist, ansatzOplist, termIndex, numElectrons):
    theProgram = pyquil.Program()
    theProgram, parameters = ansatz(ansatzOplist, theProgram, numElectrons)

    ro = theProgram.declare('ro', memory_type='BIT', memory_size=len(oplist[0][1]))
    theProgram = pauliStringBasisChangeToPyquil(oplist[termIndex][1], theProgram, ro)

    return theProgram, parameters


def parityOfBitstring(listBits):
    return sum(listBits) % 2


def eigenvalueOfSingleTrial(listBits):
    result = 1. - (2. * parityOfBitstring(listBits))
    return result


def expectationValueFromResults(results):
    numTrials = float(len(results))
    individualResult = [eigenvalueOfSingleTrial(x) for x in results]
    result = float(sum(individualResult)) / numTrials
    return result


def weightedExpectationValue(coefficient, executable, qvm, ansatzParameters):
    theThing = qvm.run(executable, memory_map={'theta': ansatzParameters})
    # print(theThing)
    expValNoCoefficient = expectationValueFromResults(theThing)
    return coefficient * expValNoCoefficient


def oplistExpectationValue(oplist, executables, qvm, ansatzParameters):
    values = []
    for i in range(len(oplist)):
        if not set(oplist[i][1]) == set([0]):
            # print(ansatzParameters)
            thisValue = weightedExpectationValue(oplist[i][0], executables[i], qvm, ansatzParameters)
            values.append(thisValue)
        else:
            values.append(oplist[i][0])
    # print(values)
    return sum(values)


def generatePyquilAnsatzParameters(listExprs, symParameterValues):
    return [float(x.subs(symParameterValues) / 1.j) for x in listExprs]


def optermToPyquil(opterm):
    coefficient = opterm[0]
    pauliString = reversed(opterm[1])
    transform = {0:"I",
                 1:"X",
                 2:"Y",
                 3:"Z"}
    pyQuilTerms = [(transform[x],i) for i,x in enumerate(pauliString)]
    result = pyquil.paulis.PauliTerm.from_list(pyQuilTerms,coefficient=coefficient)
    return result

def oplistToListPyquil(oplist):
    return [optermToPyquil(x) for x in oplist]

def openFermionUCC(geometry,basis,multiplicity,charge,description,numQubits,numElectrons,boolJWorBK=0,precision=1e-12,filename=None):
    flm = fermilibInterface.FermiLibMolecule(geometry, basis, multiplicity, charge, description, filename=filename)
    flm.runPySCF(runCCSD=1, runFCI=0)
    oplist = flm.electronicHamiltonian(boolJWorBK,precision)
    molecularHamiltonian = flm.molecularData.get_molecular_hamiltonian()
    singleAmps = flm.molecularData.ccsd_single_amps
    doubleAmps = flm.molecularData.ccsd_double_amps
   # uccOperator1 = openfermion.utils.uccsd_generator(singleAmps, doubleAmps)
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
    energies = {'FCI':flm.energyFCI(),'NUCLEAR':flm.energyNuclear()}
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

def simulatorObjectiveFunction(individualParametersAsList,ansatzProgram,hamiltonian,symbolicParameters,parameterNamesAsList):
    ansatzParameters = generatePyquilAnsatzParameters(symbolicParameters,dict(zip(parameterNamesAsList,individualParametersAsList)))
    expectations = pyquil.api.WavefunctionSimulator().expectation(ansatzProgram,hamiltonian,memory_map={'theta':ansatzParameters})
    result = sum(expectations).real
    return result

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
    expectation = expectationValueComputationalBasis(state,involvedIndices)
    return expectation



