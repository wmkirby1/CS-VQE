from yaferp.circuits import circuit
import copy
import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
ERRORTHRESHOLD = 1e-13
def testGatesCommute(gate1,gate2,state,debug=False):
    if debug:
        print(gate1.readable() + '\n' + gate2.readable())
    state2 = copy.deepcopy(state)
    commuteResult = gate1.checkCommute(gate2)
    commuteResultBackwards = gate2.checkCommute(gate1)
    assert commuteResult == commuteResultBackwards, "ERROR!  Gates report different commutativity depending on order.\nGATE 1: " + gate1.readable() + '\nGATE 2: ' + gate2.readable()
    state3 = copy.deepcopy(state)
    state4 = copy.deepcopy(state)
    newState3 = gate1.act(state3)
    hasGate1DoneSomething = scipy.linalg.norm((newState3-scipy.sparse.csc_matrix(state)).todense()) > ERRORTHRESHOLD
    if not hasGate1DoneSomething:
        return
    newState4 = gate2.act(state4)
    hasGate2DoneSomething = scipy.linalg.norm((newState4-scipy.sparse.csc_matrix(state)).todense()) > ERRORTHRESHOLD
    if not hasGate2DoneSomething:
        return
    newState1 = gate1.act(gate2.act(state))
    hasSequence1DoneSomething = scipy.linalg.norm((newState1-scipy.sparse.csc_matrix(state)).todense()) > ERRORTHRESHOLD
    if not hasSequence1DoneSomething:
        return
    newState2 = gate2.act(gate1.act(state2))
    hasSequence2DoneSomething = scipy.linalg.norm((newState2-scipy.sparse.csc_matrix(state2)).todense()) > ERRORTHRESHOLD
    if not hasSequence2DoneSomething:
        return
    normDifference = scipy.linalg.norm((newState1-newState2).todense())
    commuteResult2 = normDifference < ERRORTHRESHOLD
    assert commuteResult2 == commuteResult, "ERROR! checkCommute reports false commutativity \nGATE 1: " + gate1.readable() + '\nGATE 2: ' + gate2.readable() +'\nState ' + str(state) +'\nWanted commutativity ' + str(commuteResult2) + ' got ' + str(commuteResult)
    
    
     
def testCommute4QubitCircuit():
    """do gates properly commute?"""
    stateList = []
    for i in range(16):
        newState = [[0.]]*16
        newState[i] = [1.]
        stateList.append(newState)
    print(stateList)
    for state in stateList:
        for gate1Config in circuit.GATETYPES:
            for gate2Config in circuit.GATETYPES:
                if gate1Config[1]== 1:
                    gate1 = circuit.Gate(gate1Config[0], 0, 0.5)
                    if gate2Config[1] == 1:
                        testGatesCommute(gate1, circuit.Gate(gate2Config[0], 0, 0.5), state)
                        testGatesCommute(gate1, circuit.Gate(gate2Config[0], 1, 0.5), state)
                    else:
                        testGatesCommute(gate1, circuit.Gate(gate2Config[0], [0, 1]), state)
                        testGatesCommute(gate1, circuit.Gate(gate2Config[0], [1, 0]), state)
                        testGatesCommute(gate1, circuit.Gate(gate2Config[0], [1, 2]), state)
                else:
                    gate1 = circuit.Gate(gate1Config[0], [0, 1])
                    if gate2Config[1] == 1:
                        testGatesCommute(gate1, circuit.Gate(gate2Config[0], 0, 0.5), state)
                        testGatesCommute(gate1, circuit.Gate(gate2Config[0], 1, 0.5), state)
                        testGatesCommute(gate1, circuit.Gate(gate2Config[0], 2, 0.5), state)
                    else:
                        testGatesCommute(gate1, circuit.Gate(gate2Config[0], [0, 1]), state)
                        testGatesCommute(gate1, circuit.Gate(gate2Config[0], [1, 0]), state)
                        testGatesCommute(gate1, circuit.Gate(gate2Config[0], [1, 2]), state)
                        testGatesCommute(gate1, circuit.Gate(gate2Config[0], [2, 1]), state)
                        testGatesCommute(gate1, circuit.Gate(gate2Config[0], [2, 3]), state)
                        testGatesCommute(gate1, circuit.Gate(gate2Config[0], [3, 2]), state)
                    gate1b = circuit.Gate(gate1Config[0], [1, 0])
                    if gate2Config[1] == 1:
                        testGatesCommute(gate1b, circuit.Gate(gate2Config[0], 0, 0.5), state)
                        testGatesCommute(gate1b, circuit.Gate(gate2Config[0], 1, 0.5), state)
                        testGatesCommute(gate1b, circuit.Gate(gate2Config[0], 2, 0.5), state)
                    else:
                        testGatesCommute(gate1b, circuit.Gate(gate2Config[0], [0, 1]), state)
                        testGatesCommute(gate1b, circuit.Gate(gate2Config[0], [1, 0]), state)
                        testGatesCommute(gate1b, circuit.Gate(gate2Config[0], [1, 2]), state)
                        testGatesCommute(gate1b, circuit.Gate(gate2Config[0], [2, 1]), state)
                        testGatesCommute(gate1b, circuit.Gate(gate2Config[0], [2, 3]), state)
                        testGatesCommute(gate1b, circuit.Gate(gate2Config[0], [3, 2]), state)
    return

def testInteriorCircuitOneState(oplist,state):
    circ1 = circuit.oplistToCircuit(oplist)
    circ2 = circuit.oplistToInteriorCircuit(oplist)
    circ1Result = circ1.act(state)
    circ2Result = circ2.act(state)
    circDiff = circ1Result-circ2Result
    return (scipy.sparse.linalg.norm(circDiff))

def testInteriorCircuit(oplist):
    numQubits = len(oplist[0][1])
    stateList = []
    spaceSize = 2**numQubits
    for i in range(spaceSize):
        newState = [[0.]]*spaceSize
        newState[i] = [1.]
        stateList.append(newState)
    for state in stateList:
        fred=testInteriorCircuitOneState(oplist,state)
        print(fred)
    return

