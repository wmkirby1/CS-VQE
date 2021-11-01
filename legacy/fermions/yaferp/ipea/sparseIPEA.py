'''
Created on 23 Mar 2015

@author: andrew
'''
ENTANGLING_GATE_ERROR=0.0074
SINGLE_QUBIT_GATE_ERROR=0.0008


import numpy
from yaferp.general import sparseTrotter, sparseFermions
from yaferp.orderings import commutationSubdivision
import copy


def prepareRegister():
    register = hadamard(zeroState())
    return register
def zeroState():
    return numpy.matrix([[1],[0]],dtype='complex128')
def hadamard(previousState):
    firstElement = (previousState[0,0] + previousState[1,0]) * 1/numpy.sqrt(2)
    secondElement = (previousState[0,0] - previousState[1,0]) * 1/numpy.sqrt(2)
    return numpy.matrix([[firstElement],[secondElement]])
def controlledU(compState,groundState,terms,ordering,t,n,order,power=1):
    #simulationTime = t/( 2 * numpy.pi)
    simulationTime = t
    unitary = sparseTrotter.trotterise(terms, ordering, simulationTime, n, order)
    fullUnitary = unitary**power
    phase = (groundState.H * fullUnitary * groundState)
    try:
        phase = phase.todense()
    except:
        pass
    phase = phase/numpy.linalg.norm(phase)
    compState[1] = compState[1] * phase
    return compState

def controlledUExact(compState,groundState,oplist,t):
    phase = sparseFermions.expectOfUnitary(oplist, t, groundState).todense()
    compState[1] = compState[1] * phase
    return compState
def oneIteration(oplist,groundState,t,n,order):
    compRegister = zeroState()
    compRegister2 = hadamard(compRegister)
    compRegister3 = controlledU(compRegister2,groundState,oplist,t,n,order)
    compRegister4 = hadamard(compRegister3)
    return compRegister4
def calculateCorrectionPhase(listBits,numBits):
    if not listBits:
        return 0
    phase = 0
    for bitIndex,bitState in enumerate(listBits):
        if bitState:
            denominator = 2**(numBits - bitIndex)
            phase = phase + 1/float(denominator)
    phase = phase * 2 * numpy.pi * -1j
    return phase
def newCalculateCorrectionPhase(listBits,numBits):
    if not listBits:
        return 0
    k = numBits - len(listBits)
    phase = 0
    for l in range(2,((numBits-k)+2)):
        denominator = 2**l
        numerator = list(reversed(listBits))[l-2]
        phase = phase + float(numerator)/float(denominator)
    phaseFactor = 2 * numpy.pi * -1j * phase
    return phaseFactor
    
    
def phaseGate(compReg,listBits,numBits):
    phase = newCalculateCorrectionPhase(listBits,numBits)
    factor = numpy.exp(phase)
    compReg[1] = compReg[1] * factor
    return compReg
def iterativePhaseEstimation(terms,ordering,groundState,tZero,numTrotter,trotterOrder,numBits):
    '''iterative phase estimation with long evolution times for repeated powers of U'''
    listBits = []
    for i in range(numBits):
        newTime = tZero * (2**(numBits-i-1))
        compRegister = zeroState()
        compRegister2 = hadamard(compRegister)
        compRegister3 = controlledU(compRegister2,groundState,terms,ordering,newTime,numTrotter,trotterOrder)
        compRegister4 = phaseGate(compRegister3,listBits,numBits)
        compRegister5 = hadamard(compRegister4)
        listBits.append(int(abs(compRegister5[1]) > abs(compRegister5[0])))
    return listBits

def iterativePhaseEstimationExact(oplist,groundState,numBits,tZero=1):
    '''traditional iterative phase estimation using exact unitary'''
    listBits = []
    for i in range(numBits):
        newTime = tZero * (2**(numBits-i-1))
        compRegister = zeroState()
        compRegister2 = hadamard(compRegister)
        compRegister3 = controlledUExact(compRegister2,groundState,oplist,newTime)
        compRegister4 = phaseGate(compRegister3,listBits,numBits)
        compRegister5 = hadamard(compRegister4)
        listBits.append(int(abs(compRegister5[1]) > abs(compRegister5[0])))
    return listBits

def iterativePhaseEstimationFullTrotter(oplist,groundState,t,n,order,numBits):
    '''traditional IPEA using trotterised unitary'''
    listBits = []
    for i in range(numBits):
        unitaryPower = (2**(numBits-i-1))
        compRegister = zeroState()
        compRegister2 = hadamard(compRegister)
        compRegister3 = controlledU(compRegister2,groundState,oplist,t,n,order,unitaryPower)
        compRegister4 = phaseGate(compRegister3,listBits,numBits)
        compRegister5 = hadamard(compRegister4)
        listBits.append(int(abs(compRegister5[1]) > abs(compRegister5[0])))
    return listBits
        
def oneIterationAnyBit(oplist,groundState,t,n,order,bit):
    compRegister = zeroState()
    compRegister2 = hadamard(compRegister)
    compRegister3 = controlledU(compRegister2,groundState,oplist,t*(2**bit),n,order)
    compRegister4 = hadamard(compRegister3)
    return compRegister4
    
def listDigitsToEnergy(listDigits,t=1):
    numDigits = len(listDigits)
    newListDigits = list(reversed(listDigits))
    string = ''.join(map(str, newListDigits))
    phasePrePoint = float(int(string,2))
    phase = phasePrePoint/float(2**numDigits)
    energy = phase * -2 * numpy.pi /t
    return energy

def ipeaError(terms,ordering,groundState,trueEigenvalue,tZero,trotterNumber,trotterOrder,numBits):
    readoutDigits = iterativePhaseEstimation(terms,ordering,groundState,tZero,trotterNumber,trotterOrder,numBits)
    ipeaEnergy = listDigitsToEnergy(readoutDigits,tZero)
    error = abs(ipeaEnergy-trueEigenvalue)
    return error

def errorOfTrotterNumber(oplist,groundState,trueEigenvalue,tZero,minTrotterNumber,maxTrotterNumber,trotterOrder,numBits,verbose=False):
    for trotterNumber in range(minTrotterNumber,maxTrotterNumber+1):
        try:
            error = ipeaError(oplist,groundState,trueEigenvalue,tZero,trotterNumber,trotterOrder,numBits)
        except:
            error = ipeaError(oplist,groundState,trueEigenvalue,tZero,trotterNumber,trotterOrder,numBits)[0]
        output = str(trotterNumber)+':'+str(error)
        print(output)
    return

def errorOfTZero(terms,ordering,groundState,trueEigenvalue,minTZero,maxTZero,intervalTZero,trotterNumber,trotterOrder,numBits):
    tZeroGap = maxTZero-minTZero
    numTests = int(tZeroGap/intervalTZero)+1
    for i in range(numTests):
        tZero = minTZero + i*intervalTZero
        try:
            error = ipeaError(terms,ordering,groundState,trueEigenvalue,tZero,trotterNumber,trotterOrder,numBits)[0]
        except:
            error = ipeaError(terms,ordering,groundState,trueEigenvalue,tZero,trotterNumber,trotterOrder,numBits)
        output = str(tZero)+':'+str(error)
        print(output)
    return
        
def errorOfBits(oplist,groundState,trueEigenvalue,tZero,trotterNumber,trotterOrder,minNumBits,maxNumBits):
    for numBits in range(minNumBits,maxNumBits+1):
        error = ipeaError(oplist,groundState,trueEigenvalue,tZero,trotterNumber,trotterOrder,numBits)[0]
        output = str(numBits)+':'+str(error)
        print(output)
    return

def experimentalErrorUnitary(hamiltonian,numTrotterSteps,order,entanglingError=ENTANGLING_GATE_ERROR,sqgError=SINGLE_QUBIT_GATE_ERROR):
    numSQGs,numCSQGs,numCNOTs = sparseFermions.countGates(hamiltonian, numTrotterSteps, order, True)
    numEntangling = numCNOTs + numCSQGs
    unitaryError = numEntangling*entanglingError + numSQGs*sqgError
    return unitaryError

def ipeaExperimentalError(hamiltonian,numTrotterSteps,order,sqgError=SINGLE_QUBIT_GATE_ERROR):
    unitaryError = experimentalErrorUnitary(hamiltonian,numTrotterSteps,order)
    ipeaError = unitaryError + 3*sqgError
    return ipeaError
 
def ipeaQubits(oplist,groundState,tZero,numTrotter,trotterOrder,numBits):
    '''iterative phase estimation with long evolution times for repeated powers of U'''
    listBits = []
    listQubits = []
    for i in range(numBits):
        newTime = tZero * (2**(numBits-i-1))
        compRegister = zeroState()
        compRegister2 = hadamard(compRegister)
        compRegister3 = controlledU(compRegister2,groundState,oplist,newTime,numTrotter,trotterOrder)
        compRegister4 = phaseGate(compRegister3,listBits,numBits)
        compRegister5 = hadamard(compRegister4)
        listBits.append(int(abs(compRegister5[1]) > abs(compRegister5[0])))
        listQubits.append(compRegister5)
    return listQubits

def ipeaQubitFidelity(qubitState,desiredBit):
    desiredStateAmplitude = qubitState[desiredBit]
    probability = desiredStateAmplitude * desiredStateAmplitude.conjugate()
    return probability

def ipeaQubitStringSuccessProbabilities(qubitString,desiredBits):
    '''calculate the success of each bit assuming each previous bit is correct'''
    probabilities = []
    for index,qubit in enumerate(qubitString):
        probability = ipeaQubitFidelity(qubit,desiredBits[index])
        probabilities.append(probability)
    return probabilities
                                        
def findGoodParameters(hamiltonian,state,trueEigenvalue,startTZero,endTZero,tZeroInterval,trotterNumber,trotterOrder,numBits):
    listOrderings = commutationSubdivision.generateOrderingLabels(len(hamiltonian))
    for ordering in listOrderings:
        thisHamiltonian = copy.deepcopy(hamiltonian)
        thisHamiltonian = commutationSubdivision.reorderOplist(hamiltonian, ordering)
        print(str(ordering))
        errorOfTZero(thisHamiltonian,state,trueEigenvalue,startTZero,endTZero,tZeroInterval,trotterNumber,trotterOrder,numBits)
    return

def doMagic(hamiltonian,state,trueEigenvalue,startTZero,endTZero,tZeroInterval,trotterOrder,numBits):
    '''temp, obviously'''
    for i in [1,2,3]:
        print('!!! '+ str(i) + ' Trotter Steps!!!')
        findGoodParameters(hamiltonian,state,trueEigenvalue,startTZero,endTZero,tZeroInterval,i,trotterOrder,numBits)
    return
        
        
        
        