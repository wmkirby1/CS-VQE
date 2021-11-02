'''
Created on 17 Dec 2014

@author: andrew

probably an inefficient strategy for all this but what the hell.
'''

'''define a general struct for gates'''
import string
import numpy
import scipy.sparse
import copy
import math
from yaferp.general import directFermions
import cirq

#import cPickle
try:
    import pyquil.gates
except:
    pass



GATETYPES = [['cnot',2],['rz',1],['h',1],['hy',1],['hyinv',1],['i',1],['cz',2]]

import pickle as cPickle

class Gate:
    def __init__(self,gateType,qubits,angle=0):
        '''SQRs are represented by an axis (x,y,z) specified in the type.
        angle is the angle of rotation as a fraction of 2 pi - so for instance the angle stored here is the theta in:
        i.e Rx(theta) = e^(-i*theta*2pi*X/2)
        '''
        self.type = gateType.lower()  #string - cnot, rx, ry, rz, h, hy, hyinv, i, cz
        self.qubits = qubits #list of zero-indexed ints, control is first
        self.angle = angle #rotations and "identity" need an angle specified - see function description
    def readable(self):
        if self.type == 'rx' or self.type == 'ry' or self.type == 'rz' or self.type == 'i': 
            thing = self.type.upper() + ' on qubit ' + str(self.qubits) + ' angle ' + str(self.angle)
        elif self.type == 'h':
            thing = 'H on qubit ' + str(self.qubits)
        elif self.type == 'hy':
            thing = 'Hy on qubit ' + str(self.qubits)
        elif self.type == 'hyinv':
            thing = 'Inverse Hy on qubit ' + str(self.qubits)
        elif self.type == 'cnot':
            thing = 'CNOT controlled on qubit ' + str(self.qubits[0])+' acting on qubit ' + str(self.qubits[1])
        elif self.type == 'cz':
            thing = 'Cz controlled on qubit ' + str(self.qubits[0])+' acting on qubit ' + str(self.qubits[1])
        print(thing)
        return thing
    def qasm(self):
        if self.type == 'cnot':
            return ' cnot ' + str(self.qubits[0]) + ',' + str(self.qubits[1])
        elif self.type == 'cz':
            return ' c-z ' + str(self.qubits[0]) + ',' + str(self.qubits[1])
        elif self.type == 'h':
            return ' h ' + str(self.qubits)
        elif self.type == 'i':
            return " def thing,0,'I(" +str(abs(self.angle)) +")' \n thing " + str(self.qubits)
        elif self.type == 'hy':
            return " def thing,0,'Y' \n thing " + str(self.qubits)
        elif self.type == 'hyinv':
            return " def thing,0,'Y^\\dagger' \n thing " + str(self.qubits)
        else:
            if self.type == 'rx':
                subscript = 'x'
            elif self.type == 'ry':
                subscript = 'y'
            elif self.type == 'rz':
                subscript = 'z'
            return " def thing,0,'R_" + subscript + "(" + str(abs(self.angle)) + ")' \n thing " + str(self.qubits)



    def getInverse(self):
        if self.type == 'h' or self.type == 'cnot':
            return self
        else:
            inverse = Gate(self.type, self.qubits,(self.angle * -1.))
            return inverse
    def act(self,state,debugOn=0):
        '''act the gate on a state vector, return the state vector.

        args:
        state -- a dense or sparse state vector to be acted upon by the gate
        debugOn -- print debug info?

        TODO: majorly refactor this whole thing
        '''
        #print(state)
        if debugOn:
            initialNorm = state.conjugate().transpose() * state
        state = scipy.sparse.csr_matrix(state,dtype=numpy.complex128)
        numQubits = int(math.log(state.shape[0],2))
        if self.type == 'i':
            newState = state * numpy.exp(self.angle  * -2j * numpy.pi)
        
        elif self.type == 'cnot':
            controlQubit = self.qubits[0]
            controlQubitDec = 2**controlQubit
            targetQubit = self.qubits[1]
            targetQubitDec = 2**targetQubit
            wtf = [x for x in range(2**numQubits) if (controlQubitDec & x != 0)]
            unchangedIndices = [x for x in range(2**numQubits) if (x not in wtf)]
            newState2 = [state[x,0] if x in unchangedIndices else state[x^targetQubitDec, 0] for x in range(2**numQubits)]
            newState = scipy.sparse.csr_matrix(newState2,dtype=numpy.complex128).transpose()
            
            
        elif self.type in ['rz','rx','ry']:
            realAngle = self.angle*2*numpy.pi
            identityCoefficient = numpy.cos(realAngle/2)
            pauliCoefficient = -1j * numpy.sin(realAngle/2)
            stateCopy = copy.deepcopy(state)
            
            firstComponent = identityCoefficient * stateCopy
            if self.type == 'rz':
                secondComponent = pauliCoefficient * directFermions.oneZOnState(state, self.qubits)
            newState = firstComponent + secondComponent
            
        elif self.type == 'cz':
            firstQubitMask = 2**self.qubits[0]
            secondQubitMask = 2**self.qubits[1]
            mask = firstQubitMask + secondQubitMask
            indicesToPhase = [x for x in range(2**numQubits) if x & mask == mask]
            for index in indicesToPhase:
                state[index,0] = state[index,0] * -1
            newState = state

        elif self.type == 'h':
            #really needs optimisation
            state2 = copy.deepcopy(state)
            xComponent = directFermions.oneXOnState(state, self.qubits)
            zComponent = directFermions.oneZOnState(state2, self.qubits)
            newState = (2.**-(1./2.)) * (xComponent + zComponent)
        
        elif self.type == 'hy':
            '''hy = 1/sqrt(2)  I + iX'''
            firstTerm = copy.deepcopy(state)
            zComponent = 1j * directFermions.oneXOnState(state, self.qubits)
            newState = (2.**-(1./2.)) * (firstTerm + zComponent)
            
        elif self.type == 'hyinv':
            '''hy = 1/sqrt(") I - iX'''
            firstTerm = copy.deepcopy(state)
            zComponent = -1j * directFermions.oneXOnState(state, self.qubits)
            newState = (2.**-(1./2.)) * (firstTerm + zComponent)
            
        else:
            print('error, no action for gate ' + self.type + ' found')
        
        if debugOn:
            newNorm = newState.conjugate().transpose() * newState
            if newNorm - initialNorm >= 0.01:
                print ('Error - gate ' + self.type + ' on qubit ' + str(self.qubits) + ' increasing norm')
                print ('norm ' + str(newNorm - initialNorm))
            if initialNorm - newNorm >= 0.01:
                print ('Error - gate ' + self.type + ' on qubit ' + str(self.qubits) + ' decreasing norm')
                print ('norm ' + str(newNorm - initialNorm))

        return newState

    def sparseMatrixForm(self):
        '''return the matrix form of a SQG, return matrix form of CNOT tensor identity for qubits in between control and target'''
        if self.type == 'i':
            term = numpy.exp(-1j * 2 * numpy.pi * self.angle)
            return scipy.sparse.csc_matrix([[term,0],[0,term]])
        elif self.type == 'h':
            return scipy.sparse.csc_matrix([[1,1],[1,-1]])
        elif self.type == 'cnot':
            '''do cnot'''
            pass
        elif self.type == 'rx':
            angle = numpy.pi * self.angle
            diagonal = numpy.cos(angle)
            offDiagonal = -1j * numpy.sin(angle)
            return scipy.sparse.csc_matrix([[diagonal,offDiagonal],[offDiagonal,diagonal]])
        elif self.type == 'ry':
            angle = numpy.pi * self.angle
            diagonal = numpy.cos(angle)
            sinTerms = numpy.sin(angle)
            return scipy.sparse.csc_matrix([[diagonal,-1 * sinTerms],[sinTerms,diagonal]])
        elif self.type == 'rz':
            iTheta = self.angle * numpy.pi* 1j
            firstPhase = numpy.exp(-1* iTheta)
            secondPhase = numpy.exp(iTheta)
            return scipy.sparse.csc_matrix([[firstPhase,0],[0,secondPhase]])
        return

    def toSparseUnitary(self,circuitDepth):
        '''int circuitDepth = total number of qubits in circuit'''
        if not self.type == 'cnot':
            activeMatrix = self.sparseMatrixForm()
            if self.qubits > 0:
                pregroup = scipy.sparse.identity(self.qubits,format='csc')
                unitary = scipy.sparse.kron(pregroup,activeMatrix,format='csc')
            else:
                unitary = activeMatrix
            if self.qubits < circuitDepth:
                postgroup = scipy.sparse.identity(circuitDepth - self.qubits,format='csc')
                unitary = scipy.sparse.kron(unitary,postgroup,format='csc')
            return unitary
    
    def checkCommute(self,otherGate):
        if self.type == 'i' or otherGate.type == 'i':
            return 1
        elif self.type in ['rx', 'ry', 'rz', 'h', 'hy','hyinv']:
            if otherGate.type == self.type:
                return 1 #always commute if same gate type
            elif otherGate.type == 'cnot' and self.type == 'rz':
                if self.qubits == otherGate.qubits[1]:
                    return 0
                else:
                    return 1
            elif otherGate.type == 'cnot' and self.type in ['hy','hyinv'] and self.qubits == otherGate.qubits[1]:
                return 1
            elif otherGate.type == 'cnot':
                if self.qubits in otherGate.qubits:
                    return 0
                else:
                    return 1
                
            elif otherGate.type == 'cz':
                if self.type == 'rz':
                    return 1
                elif self.qubits in otherGate.qubits:
                    return 0
                else:
                    return 1    
                
            elif self.qubits != otherGate.qubits:
                return 1 #always commute if SQG on different qubits
            else:
                return 0 #else is different gate on same qubit, hence probably don't commute
        elif self.type == 'cnot':
            if otherGate.type in ['rx', 'ry', 'h']:
                if otherGate.qubits in self.qubits:
                    return 0 #TODO:  again this should be 1 if the other qubit is a simple X rotation but w/e
                else:
                    return 1
            elif otherGate.type in ['hy','hyinv']:
                if not otherGate.qubits in self.qubits :
                    return 1
                elif self.qubits[1] == otherGate.qubits:
                    return 1
                else:
                    return 0
            elif otherGate.type == 'rz':
                if otherGate.qubits == self.qubits[1]:
                    return 0
                else:
                    return 1
            elif otherGate.type == 'cnot':
                if (self.qubits[0] == otherGate.qubits[1]) or (self.qubits[1] == otherGate.qubits[0]):
                    return 0 #CNOTs commute unless one's target is the other's control
                else:
                    return 1 
            elif otherGate.type == 'cz':
                if (self.qubits[1] in otherGate.qubits):
                    return 0
                else:
                    return 1
            else:
                return -1
        elif self.type == 'cz':
            if otherGate.type in ['rx','ry','h','hy','hyinv']:
                if otherGate.qubits in self.qubits:
                    return 0
                else:
                    return 1
            elif otherGate.type == 'rz' or otherGate.type == 'cz':
                return 1
            elif otherGate.type == 'cnot':
                if (otherGate.qubits[1] in self.qubits):
                    return 0
                else:
                    return 1
                
        return -1
    
    def canCancel(self,otherGate):
        if (self.type == 'hy' and otherGate.type == 'hyinv') or (self.type == 'hyinv' and otherGate.type == 'hy'):
            if self.qubits == otherGate.qubits:
                return 1
            else:
                return 0
        elif self.type == 'cz' and otherGate.type == 'cz':
            if (self.qubits == otherGate.qubits) or ((self.qubits[0] == otherGate.qubits[1]) and self.qubits[1] == otherGate.qubits[0]):
                return 1
            else:
                return 0
        elif self.type==otherGate.type and self.qubits==otherGate.qubits:
                return 1    
        else:
            return 0


#    def _getPyquillerisedAngle_(self,pyQuilMemoryHolder):
#        if isinstance(self.angle,sympy.Expr):
#            return self.angle.subs(parametersToPyquilMap)

    def pyquil(self,pyQuilMemoryHolder=None,pyQuilMemoryIndex=None):
        if self.type == 'cnot':
            return pyquil.gates.CNOT(self.qubits[0],self.qubits[1])
        elif self.type == 'rz':
            assert not pyQuilMemoryHolder is None
            assert not pyQuilMemoryIndex is None
       #     pyquilAngle = self._getPyquillerisedAngle_(parametersToPyquilMap)
            clive = pyQuilMemoryHolder[pyQuilMemoryIndex]
            return (pyquil.gates.RZ(clive, self.qubits),self.angle*2.*numpy.pi)
           # return (pyquil.gates.RZ(pyQuilMemoryHolder[pyQuilMemoryIndex],self.qubits), self.angle)
        elif self.type == 'h':
            return pyquil.gates.H(self.qubits)
        elif self.type == 'hy':
            return pyquil.gates.RX((-1. * numpy.pi)/2.,self.qubits)
        elif self.type == 'hyinv':
            return pyquil.gates.RX(numpy.pi/2.,self.qubits)
        else:
            raise ValueError("gate not supported!")


        return

    def toCirq(self,qubit,qubit2=None,set_to_true_if_you_are_panicking=False):

        CIRQ_GATE_DICT = {'h':cirq.H,
                            'hy':cirq.rx,
                            'hyinv':cirq.rx,
                            'rz':cirq.rz,
                            'cnot':cirq.CNOT}
        cirqGateMethod = CIRQ_GATE_DICT[self.type]

       # if self.type in ['hy','hyinv','rz']:
       #     if self.type == 'hy':
       #         cirqGateMethod1 =
        #    return cirqGateMethod(self.angle)(qubit)
        if self.type == 'hy':
            return cirqGateMethod((-1. * numpy.pi)/2.)(qubit)
        elif self.type == 'hyinv':
            return cirqGateMethod((1. * numpy.pi)/2.)(qubit)
        elif self.type == 'rz':
            if set_to_true_if_you_are_panicking:
                return cirqGateMethod(self.angle * 2. * numpy.pi)(qubit)
            else:
                return cirqGateMethod(self.angle * 2. * numpy.pi /-1j)(qubit)


        if type(self.qubits) != int:
            assert qubit2 is not None, "2 qubit gates need a second qubit."
            return cirqGateMethod(qubit,qubit2)

        else:
            return cirqGateMethod(qubit)


class Circuit:
    def __init__(self,listGates,numQubits=-1,useSubcircuits=False):
        self.listGates = listGates
        if numQubits==-1:
            maxQubitNumber = 0
            for gate in listGates:
                if isinstance(gate.qubits, list):
                    maxQubitNumberInGate = max(gate.qubits)
                else:
                    maxQubitNumberInGate = gate.qubits
                if maxQubitNumberInGate > maxQubitNumber:
                    maxQubitNumber = maxQubitNumberInGate
            self.numQubits = maxQubitNumber
        else:
            self.numQubits = numQubits
        if useSubcircuits:
            self.subcircuits = [] #list containing indices of subcircuits
        else:
            self.subcircuits = None
    def update(self):
        maxQubitNumber = 0
        for gate in self.listGates:
            if isinstance(gate.qubits, list):
                maxQubitNumberInGate = max(gate.qubits)
            else:
                maxQubitNumberInGate = gate.qubits
            if maxQubitNumberInGate > maxQubitNumber:
                maxQubitNumber = maxQubitNumberInGate
        self.numQubits = maxQubitNumber    
        return
    def numRotations(self):
        counter = 0
        for x in self.listGates:
            if x.type in ['rx','ry','rz']:
                counter += 1
        return counter

    def pyquil(self,pyQuilMemoryHolder,pyQuilProgram):
        pyQuilIndex = 0
        parametersExprs = []
        for gate in self.listGates:
            thing = gate.pyquil(pyQuilMemoryHolder,pyQuilIndex)
            if isinstance(thing,tuple):
                pyQuilIndex += 1
                parametersExprs.append(thing[1])
                pyQuilProgram += thing[0]
            else:
                pyQuilProgram += thing
        return pyQuilProgram,parametersExprs

    def toCirq(self,qubits,circuitToAppendTo=None,insertStrategy=cirq.InsertStrategy.EARLIEST,set_to_true_if_you_are_panicking=False):
        #qubits = list(reversed(qubits2))
        if circuitToAppendTo is None:
            cirquit = cirq.Circuit()
        else:
            cirquit = circuitToAppendTo

        for gate in self.listGates:
            qubitIndices = gate.qubits
            if type(qubitIndices) is not int:
                thisCirqGate = gate.toCirq(*[qubits[i] for i in qubitIndices],set_to_true_if_you_are_panicking=set_to_true_if_you_are_panicking)
            else:
                thisCirqGate = gate.toCirq(qubits[qubitIndices],set_to_true_if_you_are_panicking=set_to_true_if_you_are_panicking)

            cirquit.append(thisCirqGate,strategy=insertStrategy)

        return cirquit

    def addRight(self,target):
        try:
            self.listGates.extend(target.listGates)
        except:
            self.listGates.append(target)
        return self
    def addLeft(self,target):
        try:
            newList = target.listGates.extend(self.listGates)
            self.listGates = newList
        except:
            self.listGates.prepend(target)
        return self
    def getReverse(self):
        reversedGateList = reversed(self.listGates)
        return Circuit(reversedGateList)
    def getInverse(self):
        reversedGateList = reversed(self.listGates)
        inverseGateList = []
        for gate in reversedGateList:
            inverseGateList.append(gate.getInverse())
        inverse = Circuit(inverseGateList)
        return inverse
        
    def readable(self):
        for gate in self.listGates:
            gate.readable()
        return
    def act(self,state,debugOn=0):
        for gate in self.listGates:
            state = gate.act(state,debugOn)
        return state
    def expectationValue(self,state,debugOn=0):
  #      print(state)
        state = scipy.sparse.csr_matrix(state, dtype=numpy.complex128)
        originalState = copy.deepcopy(state)
        state = self.act(state,debugOn)
        firstBit = originalState.conjugate().transpose()
        #print(originalState.todense())
        #print(state.todense()
        expectation = firstBit * state
        return expectation[0,0]
    def angle(self,state,debugOn=0):
        expVal = self.expectationValue(state,debugOn)
        ang = numpy.angle(expVal)
        return ang
    
    def getQubitListGates(self):
        '''return a list of N lists where N is num. qubits.  
        each sublist contains the indices of self.listGates where the corresponding qubit is involved.
        TODO: store this shit'''
        qubitListGates = []
        for qubitIndex in range(self.numQubits+1):
            listOfGatesThatThisQubitIsInvolvedIn = [] #cba
            for (gateIndex,gate) in enumerate(self.listGates):
                if not (isinstance(gate.qubits, list)):
                    if qubitIndex==gate.qubits:
                        listOfGatesThatThisQubitIsInvolvedIn.append(gateIndex)
                else:
                    if qubitIndex in gate.qubits:
                        listOfGatesThatThisQubitIsInvolvedIn.append(gateIndex)
            qubitListGates.append(listOfGatesThatThisQubitIsInvolvedIn)
        return qubitListGates
    
    def parallelisedInternal(self):  
        '''good luck!'''
        numUnplacedQubitListGates = copy.deepcopy(self.getQubitListGates())
        depletedQubitsList = []
        self.listTimesteps = []
        #for j in [1,2,3,4,5,6]:
        while len(depletedQubitsList) != self.numQubits+1:
            timestep = numpy.full(self.numQubits+1,-2,dtype=int)
            for i in range(self.numQubits+1):
                if i in depletedQubitsList:
                    timestep[i] = -1
                    continue
                if timestep[i] != -2:
                    continue
                if numUnplacedQubitListGates[i] == []:
                    depletedQubitsList.append(i)
                    timestep[i] = -1
                    continue
                firstGateIndex = numUnplacedQubitListGates[i].pop(0)
                #print(firstGateIndex)
                if isinstance(self.listGates[firstGateIndex].qubits, int):
                    timestep[i] = firstGateIndex
                else: #next gate for this qubit is entangling gate
                    otherQubitIndex = [x for x in self.listGates[firstGateIndex].qubits if x != i][0]
                    if timestep[otherQubitIndex] != -2:
                        numUnplacedQubitListGates[i].insert(0,firstGateIndex)
                        timestep[i] = -1
                    else:
                       # print(otherQubitIndex)
                        #print(firstGateIndex)
                       # print(numUnplacedQubitListGates)
                       # print(i)
                        #print(otherQubitIndex)
                       # self.listGates[firstGateIndex].readable()
                        #print(numUnplacedQubitListGates[otherQubitIndex])
                        otherQubitNextGateIndex = numUnplacedQubitListGates[otherQubitIndex][0]
                        if otherQubitNextGateIndex >= firstGateIndex:
                            timestep[i] = firstGateIndex
                            timestep[otherQubitIndex] = firstGateIndex
                            numUnplacedQubitListGates[otherQubitIndex].pop(0)
                        elif otherQubitNextGateIndex < firstGateIndex:
                            numUnplacedQubitListGates[i].insert(0,firstGateIndex)
                            timestep[i] = -1
                #print(timestep)
            self.listTimesteps.append(timestep)
        return self.listTimesteps
        
    def timestepIndices(self,whichStep=None):
        try:
            timesteps = self.listTimesteps
        except AttributeError:
            timesteps = self.parallelisedInternal()
        
        if whichStep == None:
            return timesteps
        else:
            return timesteps(whichStep)
    
    def timestep(self,whichStep=None):
        try:
            timesteps = self.listTimesteps
        except AttributeError:
            timesteps = self.parallelisedInternal()
        
        if whichStep == None:
            listTimeCircs = []
            for listGatesInStep in timesteps:
                seenGates= {}
                fixedGates = []
                for thisGate in listGatesInStep:
                    if thisGate in seenGates: continue
                    seenGates[thisGate] = 1
                    fixedGates.append(thisGate)
                gatesInStep = [self.listGates[i] for i in fixedGates if i != -1]
                stepCircuit = Circuit(gatesInStep)
                listTimeCircs.append(stepCircuit)
            return listTimeCircs
        else:
            listGatesInStep = timesteps[whichStep]
            seenGates= {}
            fixedGates = []
            for thisGate in listGatesInStep:
                if thisGate in seenGates: continue
                seenGates[thisGate] = 1
                fixedGates.append(thisGate)
            gatesInStep = [self.listGates[i] for i in fixedGates if i != -1]
            stepCircuit = Circuit(gatesInStep)
            return stepCircuit
                
                
            
                
            
        
                
    
    
    def parallelDepth(self):
        '''count the depth of the circuit'''
     
        
    def qasm(self):
        text = ''
        for i in range(self.numQubits+1):
            text = text + " qubit " + str(i) + ",q_" + str(i) + "\n"
        text += "\n"
        numDefs = 0
        for gate in self.listGates:
            thisString = gate.qasm()
            includesThing = string.find(thisString,'thing')
            if includesThing != -1:
                newString = string.replace(thisString,'thing', 'thing'+str(numDefs))
                numDefs += 1
            else:
                newString = thisString
            newString = newString + '\n'
            text += newString
        return text
    
    def removeGate(self,gateIndex):
        self.listGates.pop(gateIndex)
        if self.subcircuits != None:
            self.removeGateUpdateSubcircuit(gateIndex)
        return self
    def involvedQubits(self):
        qubits = []
        for gate in self.listGates:
            if isinstance(gate.qubits,int):
                qubits.append(gate.qubits)
            else:
                for thisQubit in gate.qubits:
                    if thisQubit not in qubits:
                        qubits.append(thisQubit)
        return qubits
                
    def markSubcircuit(self):
        if self.subcircuits != None:
            self.subcircuits.append(self.numGates()-1)
        return
            
    def numGates(self):
        return len(self.listGates)
    
    def getSubcircuitBoundsByIndex(self,index):
        subCircuitEnd = self.subcircuits[index]
        if index == 0:
            subCircuitStart = 0
        else:
            subCircuitStart = self.subcircuits[index-1]+1
        return (subCircuitStart,subCircuitEnd)   
    
    def getSubcircuitByIndex(self,index):
        subCircuitStart,subCircuitEnd = self.getSubcircuitBoundsByIndex(index)
        subListGates = self.listGates[subCircuitStart:subCircuitEnd+1]
        return Circuit(subListGates)
    
    def removeSubcircuitDuplicates(self):
        seen = set()
        seen_add = seen.add
        return [x for x in self.subcircuits if x not in seen and not seen_add(x)]
        '''noDuplicates = []
        [noDuplicates.append(x) for x in self.subcircuits if not noDuplicates.count(x)]
        self.subcircuits = noDuplicates
        return self'''
    
    
    def removeGateUpdateSubcircuit(self,removedIndex):
       # self.removeSubcircuitDuplicates()
        '''newSubcircuits = []
        for x in self.subcircuits:
            if x<removedIndex:
                newSubcircuits.append(x)
            else:
                newSubcircuits.append(x-1)'''
        #newSubcircuits = [self.subcircuits[i] if self.subcircuits[i]<removedIndex else self.subcircuits[i]-1 if self.subcircuits[i]!=self.subcircuits[i+1] for i in range(len(self.subcircuits))]
        if removedIndex in self.subcircuits and removedIndex-1 in self.subcircuits: #if the removed gate is the entirety of a subcircuit
            self.subcircuits.remove(removedIndex)

        newSubcircuits = [x if x<removedIndex else x-1 for x in self.subcircuits]
        self.subcircuits = newSubcircuits
        '''  for x,i in enumerate(self.subcircuits):
            if 
        
        newSubcircuits = [x if x<removedIndex else x-1 for x in self.subcircuits]
        self.subcircuits = newSubcircuits
        self.removeSubcircuitDuplicates()
        lastseen = -1
        newSubcircuits = [x if x<removedIndex and x!= lastseen and not  else for x in self.subcircuits]
        
        return [x for x in self.subcircuits if x not in seen and not seen_add(x)]        
        
        ''' 
        return self
        
    def cancelDuplicates(self,showWarnings=1):
        '''NOTE:  THIS PROCEDURE DESTROYS SUBCIRCUIT STRUCTURE'''
        thisGate = self.listGates[0]
        thisGateIndex = 0
        while thisGate != None:
            numGates = self.numGates()
            if thisGateIndex+1 >= numGates:
                thisGate = None
                break
            nextGate = self.listGates[thisGateIndex+1]
            if not thisGate.canCancel(nextGate):
                doesCommuteNext = thisGate.checkCommute(nextGate)
                increment = 2
                hasCancelledAfterCommutation = False
                while doesCommuteNext:
                    if thisGateIndex+increment >= self.numGates():
                        break
                    followingGate = self.listGates[thisGateIndex+increment]
                    #if not (thisGate.type == followingGate.type and thisGate.qubits == followingGate.qubits):
                    if not thisGate.canCancel(followingGate):
                        doesCommuteNext = thisGate.checkCommute(followingGate)
                        increment += 1
                    else:
                            if thisGate.type == 'rx' or thisGate.type == 'ry' or thisGate.type == 'rz' or thisGate.type == 'i':
                                thisGate.angle = thisGate.angle + followingGate.angle
                                self.listGates[thisGateIndex] = thisGate
                                self.removeGate(thisGateIndex+increment)
                                doesCommuteNext = 0
                                hasCancelledAfterCommutation = True
                                continue
                            else:
                                self.removeGate(thisGateIndex+increment)
                                self.removeGate(thisGateIndex)
                                thisGate = self.listGates[thisGateIndex]
                                doesCommuteNext = 0
                                hasCancelledAfterCommutation = True
                                continue
                    
                if not hasCancelledAfterCommutation:
                    thisGateIndex += 1
                    thisGate = self.listGates[thisGateIndex]
                continue
            else:
                if thisGate.type == 'rx' or thisGate.type == 'ry' or thisGate.type == 'rz' or thisGate.type == 'i':
                    thisGate.angle = thisGate.angle + nextGate.angle
                    self.listGates[thisGateIndex] = thisGate
                    self.removeGate(thisGateIndex+1)
                    continue
                else:
                    self.removeGate(thisGateIndex)
                    self.removeGate(thisGateIndex)
                    if not thisGateIndex >= self.numGates():
                        thisGate = self.listGates[thisGateIndex]
                    continue
        return self
    
    def fullCancelDuplicates(self):
        '''NOTE:  THIS PROCEDURE DESTROYS SUBCIRCUIT STRUCTURE'''
        while True:
           # oldCircuit = copy.deepcopy(self)
            oldNumGates = self.numGates()
            if oldNumGates == 0:
                return self
            self.cancelDuplicates()
            if self.numGates() == oldNumGates:
                return self
        return
    
    def find(self, gateType, reverseSearch=0):
        '''find first instance of a specific type of gate, return its index
        return -1 on fail'''
        if reverseSearch == 1:
            for index,gate in reversed(list(enumerate(self.listGates))):
                if gate.type == gateType.lower():
                    return index
        else:
            for index,gate in enumerate(self.listGates):
                if gate.type == gateType.lower():
                    return index
        return -1
    def flipHYDirection(self):
        for gate in self.listGates:
            if gate.type == 'hy':
                gate.type = 'hyinv'
            elif gate.type == 'hyinv':
                gate.type = 'hy'
        return self        
    
    def toInteriorBasisChange(self):
        '''puts basis changes inside of the circuit.  
        NOTE:  this assumes basis changes are currently at front/end
        also returns signed int giving change in gate count'''
        originalNumGates = self.numGates()
        listBasisChangedQubits = []
        
    def splitSubcircuit(self):
        '''splits a subcircuit even further, into inital basis changes, initial CNOTS, middle rotation, final CNOTs, final basis changes'''
        numGates = self.numGates()
        initialRotationCircuit = Circuit([])
        initialCNOTCircuit = Circuit([])
        middleGate = Circuit([])
        finalCNOTCircuit = Circuit([])
        finalRotationCircuit = Circuit([])
        if numGates == 0:
            return (initialRotationCircuit,initialCNOTCircuit,middleGate,finalCNOTCircuit,finalRotationCircuit)
            
        for index,gate in enumerate(self.listGates):
            if gate.type not in ['h','hy','hyinv']:
                currentIndex = index
                if currentIndex == numGates:
                    return (initialRotationCircuit,initialCNOTCircuit,middleGate,finalCNOTCircuit,finalRotationCircuit)
                break
            else:
                initialRotationCircuit.addRight(gate)
                
        for index,gate in enumerate(self.listGates):
            if index < currentIndex:
                continue
            elif gate.type != 'cnot':
                currentIndex = index
                if currentIndex == numGates:
                    return (initialRotationCircuit,initialCNOTCircuit,middleGate,finalCNOTCircuit,finalRotationCircuit)
                break
            else:
                initialCNOTCircuit.addRight(gate)
                
        middleGate = copy.deepcopy(self.listGates[currentIndex])
        currentIndex += 1
        
        for index,gate in enumerate(self.listGates):
            if index < currentIndex:
                continue
            elif gate.type != 'cnot':
                currentIndex = index
                if currentIndex == numGates:
                    return (initialRotationCircuit,initialCNOTCircuit,middleGate,finalCNOTCircuit,finalRotationCircuit)
                break
            else:
                finalCNOTCircuit.addRight(gate)
                
        for index,gate in enumerate(self.listGates):
            if index < currentIndex:
                continue
            elif gate.type not in ['h','hy','hyinv']:
                currentIndex = index
                if currentIndex == numGates:
                    return (initialRotationCircuit,initialCNOTCircuit,middleGate,finalCNOTCircuit,finalRotationCircuit)
                break
            else:
                finalRotationCircuit.addRight(gate)
                
        return (initialRotationCircuit,initialCNOTCircuit,middleGate,finalCNOTCircuit,finalRotationCircuit)
    
    def swapHY(self):
        newcircuit = Circuit([])
        for gate in self.listGates:
            if gate.type == 'hy':
                gate.type = 'hyinv'
            newcircuit.addRight(gate)
        return newcircuit
    
    def dump(self,filepath,overwrite=1):
        if overwrite:
            mode = 'wb'
        else:
            mode = 'ab'
        with open(filepath,mode):
            cPickle.dump(self,filepath,-1)
        return
        
    
    
    def subcircuitToInterior(self):
        (initialRotationCircuit,initialCNOTCircuit,middleGate,finalCNOTCircuit,finalRotationCircuit) = self.splitSubcircuit()
        circuit = Circuit([])
        middleQubit = middleGate.qubits
        
        initialCNOTQubits = initialCNOTCircuit.involvedQubits()
        initialRotatedQubits = [x for x in initialRotationCircuit.involvedQubits() if x!= middleQubit]
        #initialRotatedQubitsNoMiddle = [x for x in initialRotatedQubits if x != middleQubit]
        initialZQubits = [x for x in initialCNOTQubits if not x in initialRotatedQubits and not x == middleQubit]
        if isinstance(initialZQubits,int):
            initialZQubits = [initialZQubits]
        if initialZQubits:
            initialLastZQubit = initialZQubits[-1]
        if initialRotatedQubits:
            initialLastRotatedQubit = initialRotatedQubits[-1]
        if len(initialRotationCircuit.listGates) == 0:
            initialTargetRotation = 'rz'
        else:
            for gate in initialRotationCircuit.listGates:
                if gate.qubits == middleQubit:
                    initialTargetRotation = gate.type
                    break
                initialTargetRotation = 'rz'
                
        
        
        
        
        
    #    '''if initialRotatedQubits:
   #         for gate in initialRotationCircuit.listGates:
     #           if gate.qubits == middleQubit:
      #              initialTargetRotation = gate.type
       #             break
        #    initialTargetRotation = 'rz'
       # else:
        #    if initialRotationCircuit.numGates() != 0:
                
                
         #   initialTargetRotation = 'rz'''
            
        finalCNOTQubits = finalCNOTCircuit.involvedQubits()
        finalRotatedQubits = [x for x in finalRotationCircuit.involvedQubits() if x!= middleQubit]
        finalZQubits = [x for x in finalCNOTQubits if not x in finalRotatedQubits and not x == middleQubit]
        if isinstance(finalZQubits,int):
            initialZQubits = [finalZQubits]
        if finalZQubits:
            finalFirstZQubit = finalZQubits[0] #-1?
        if finalRotatedQubits:
            finalFirstRotatedQubit = finalRotatedQubits[0]
        if finalRotationCircuit.numGates() == 0:
            finalTargetRotation = 'rz'
        else:
            for gate in finalRotationCircuit.listGates:
                if gate.qubits == middleQubit:
                    finalTargetRotation = gate.type
                    break
                finalTargetRotation = 'rz'
            
            
            
        if initialZQubits:    
            circuit.addRight(cnotChainGates(initialZQubits))
            if initialTargetRotation == 'h':
                circuit.addRight(Gate('cz',[initialLastZQubit,middleQubit]))
            else:
                circuit.addRight(Gate('cnot',[initialLastZQubit,middleQubit]))
        if initialRotatedQubits:
            circuit.addRight(initialRotationCircuit)
            circuit.addRight(cnotChainGates(initialRotatedQubits))
           # if initialTargetRotation == 'rz':
            circuit.addRight(Gate('cnot',[initialLastRotatedQubit,middleQubit]))
        circuit.addRight(middleGate)
        if finalRotatedQubits:
            #if finalTargetRotation == 'rz':
            circuit.addRight(Gate('cnot',[finalFirstRotatedQubit,middleQubit]))
            circuit.addRight(cnotChainGates(finalRotatedQubits,1))
            circuit.addRight(finalRotationCircuit)
        if finalZQubits:
            if finalTargetRotation == 'h':
                circuit.addRight(Gate('cz',[finalFirstZQubit,middleQubit]))
            else:
                circuit.addRight(Gate('cnot',[finalFirstZQubit,middleQubit]))
            circuit.addRight(cnotChainGates(finalZQubits,1))
        
        return circuit #finally
    
    def circuitToInterior(self):
        newCircuit = Circuit([])
        numSubcircuits = len(self.subcircuits)
        newSubcircuits = []
        for i in range(numSubcircuits):
            thisSubcircuit = self.getSubcircuitByIndex(i)
            newSubcircuit = thisSubcircuit.subcircuitToInterior()
            newCircuit.addRight(newSubcircuit)
            newSubcircuits.append(newCircuit.numGates()-1)
        newCircuit.update()
        return newCircuit

            
    
def changeBasisGates(listBasisChanges):
    '''return a circuit which implements basis changes, assuming qubits are currently in standard computational (ie pauli Z) basis)
    listBasisChanges:  list of tuples, each tuple is (index of qubit, desired basis 1=x, 2=y, 3=z).
                        nb list should be in qubit index ascending order
                        nb as ops are self-inverse this will also return qubits to computational basis'''
    
    
    
    circuit = Circuit([])
    '''  if len(listBasisChanges) == 1:
        qubitIndex = listBasisChanges[0][0]
        whichBasis = listBasisChanges[0][1]
        if whichBasis == 1:
            gate = Gate('h',qubitIndex)
            circuit.addRight(gate)
        elif whichBasis == 2:
            gate = Gate('rx',qubitIndex,-0.25)
            circuit.addRight(gate)
        return circuit
    '''
    for qubitIndex, whichBasis in listBasisChanges:
        if whichBasis == 1:
            gate = Gate('h',qubitIndex)
            circuit.addRight(gate)
        elif whichBasis == 2:
            #gate = Gate('rx',qubitIndex,-0.25)
            gate = Gate('hyinv',qubitIndex)
            circuit.addRight(gate)
    return circuit

def cnotChainGates(indices,doReverse=0):
    '''take a list of qubit indices, return a circuit applying sequential CNOTS between each pair
    i.e. [1,2,3] -> CNOT(1,2) CNOT(2,3)'''
    circuit = Circuit([])
    if len(indices) < 2:
        return circuit
    if not doReverse:
        for i in range(len(indices)-1):
            gate = Gate('cnot',[indices[i],indices[i+1]])
            circuit.addRight(gate)
    else:
        for i in range(len(indices)-1):
            gate = Gate('cnot',[indices[i+1],indices[i]])
            circuit.addRight(gate)
            
        
    return circuit

def cnotChainGatesAncilla(indices,ancillaIndex):
    '''take a list of qubit indices, return a circuit applying sequential CNOTS between each pair
    i.e. [1,2,3] -> CNOT(1,2) CNOT(2,3)'''
    
    circuit = Circuit([])
    if len(indices) < 2:
        return circuit
    for i in range(len(indices)):
       # print([indices[i],ancillaIndex])
        gate = Gate('cnot',[indices[i],ancillaIndex])
        circuit.addRight(gate)
        
    return circuit    
    
    
def pauliStringToCircuit(op):
    '''take op term as [coefficient, [list of paulis]], return circuit'''
    coefficient = op[0]
    pauliString = op[1]
    numQubits = len(pauliString)
    circuit = Circuit([])
    try:
        if list(pauliString) == [0]*numQubits: #identity is a special case, apply "global" phase
            identity = Gate('i',0,coefficient/(2 * numpy.pi))
            circuit.addRight(identity)
            return circuit
        else:
            #get (qubit,pauli) pairs for identity paulis
            nonIdentityPaulis = [(index,value) for (index,value) in enumerate(reversed(pauliString)) if value!= 0]
            sortedNonIdentityPaulis = sorted(nonIdentityPaulis,key=lambda thing:thing[0]) #sort paulis in increasing qubit order
            
            basisChangeCircuit = changeBasisGates(sortedNonIdentityPaulis)
            involvedQubitIndices = [thing[0] for thing in sortedNonIdentityPaulis]
            cnotsCircuit = cnotChainGates(involvedQubitIndices)
            leftCircuit = basisChangeCircuit.addRight(cnotsCircuit)
            rightCircuit = leftCircuit.getInverse()
            rightCircuit.flipHYDirection()
            lastPauliIndex = sortedNonIdentityPaulis[-1][0]
            middleGate = Gate('rz',lastPauliIndex,(coefficient/numpy.pi))
            leftCircuit.addRight(middleGate)
            finalCircuit = leftCircuit.addRight(rightCircuit)
    except:
        print(op)
        
     #   for thisIndex, thisPauli in enumerate(reversed(pauliString)):
       #     if thisPauli == 1:
      #          gate = Gate('rx',1/2)
    return finalCircuit

def oplistToCircuit(oplist):
    circuit = Circuit([])
    for op in oplist:
        termCircuit = pauliStringToCircuit(op)
        circuit.addRight(termCircuit)
        circuit.markSubcircuit()
    circuit.update()
    return circuit

def pauliStringToInteriorCircuit(op):
    '''take op term as [coefficient, [list of paulis]], return circuit'''
    coefficient = op[0]
    pauliString = op[1]
    numQubits = len(pauliString)
    leftCircuit = Circuit([])
    
    if list(pauliString) == [0]*numQubits: #identity is a special case, apply "global" phase
        identity = Gate('i',0,coefficient/(2 * numpy.pi))
        leftCircuit.addRight(identity)
        return leftCircuit
    else:
        nonIdentityPaulis = [(index,value) for (index,value) in enumerate(reversed(pauliString)) if value!= 0]
        sortedNonIdentityPaulis = sorted(nonIdentityPaulis,key=lambda thing:thing[0]) #sort paulis in increasing qubit order
        lastPauliIndex,lastPauliType = sortedNonIdentityPaulis[-1]
        #get list of X, Y and Z terms
        xQubits = [qubit for (qubit, pauli) in sortedNonIdentityPaulis if pauli== 1]
        yQubits = [qubit for (qubit, pauli) in sortedNonIdentityPaulis if pauli== 2]
        zQubits = [qubit for (qubit, pauli) in sortedNonIdentityPaulis if pauli== 3]
        xAndYQubits = [qubit for (qubit, pauli) in sortedNonIdentityPaulis if pauli in [1,2]]
        #build the circuit components
        zCNOTCircuit = cnotChainGates(zQubits)
        leftCircuit.addRight(zCNOTCircuit)
        if lastPauliType != 3 and zQubits != [] :
            controlledZ = Gate('cz',[max(zQubits),lastPauliIndex])
            leftCircuit.addRight(controlledZ)
            
        basisChangeCircuit = changeBasisGates(sortedNonIdentityPaulis)
        leftCircuit.addRight(basisChangeCircuit)
        xyCNOTCircuit = cnotChainGates(xAndYQubits)
        leftCircuit.addRight(xyCNOTCircuit)
        if lastPauliType == 3 and xAndYQubits != [] :
            lastCNOT = Gate('cnot', [max(xAndYQubits), lastPauliIndex])
            leftCircuit.addRight(lastCNOT)
            
        rightCircuit = leftCircuit.getInverse()
        rightCircuit.flipHYDirection()
        middleGate = Gate('rz',lastPauliIndex,(coefficient/numpy.pi))
        leftCircuit.addRight(middleGate)
        finalCircuit = leftCircuit.addRight(rightCircuit)
    
    return finalCircuit

def OBSOLETE_pauliStringToAncillaCircuit(op):
    '''take op term as [coefficient, [list of paulis]], return circuit'''
    coefficient = op[0]
    pauliString = op[1]
    numQubits = len(pauliString)
    ancillaIndex = numQubits
    leftCircuit = Circuit([])
    
    if pauliString == [0]*numQubits: #identity is a special case, apply "global" phase
        identity = Gate('i',0,coefficient/(2 * numpy.pi))
        leftCircuit.addRight(identity)
        return leftCircuit
    else:
        nonIdentityPaulis = [(index,value) for (index,value) in enumerate(reversed(pauliString)) if value!= 0]
        sortedNonIdentityPaulis = sorted(nonIdentityPaulis,key=lambda thing:thing[0]) #sort paulis in increasing qubit order
        lastPauliIndex,lastPauliType = sortedNonIdentityPaulis[-1]
        #get list of X, Y and Z terms
        xQubits = [qubit for (qubit, pauli) in sortedNonIdentityPaulis if pauli== 1]
        yQubits = [qubit for (qubit, pauli) in sortedNonIdentityPaulis if pauli== 2]
        zQubits = [qubit for (qubit, pauli) in sortedNonIdentityPaulis if pauli== 3]
        xAndYQubits = [qubit for (qubit, pauli) in sortedNonIdentityPaulis if pauli in [1,2]]
        #build the circuit components
        if lastPauliType == 3 and zQubits != []:
            zQubitsReduced = [qubit for qubit in zQubits if qubit != lastPauliIndex]
            if zQubitsReduced != []:
                zCNOTCircuit = cnotChainGatesAncilla(zQubitsReduced,ancillaIndex)
                leftCircuit.addRight(zCNOTCircuit)
                lastCNOT = Gate('cnot',[ancillaIndex,lastPauliIndex])
                leftCircuit.addRight(lastCNOT)
        elif lastPauliType != 3 and zQubits != []:
            zCNOTCircuit = cnotChainGatesAncilla(zQubits,ancillaIndex)
            leftCircuit.addRight(zCNOTCircuit)
            controlledZ = Gate('cz',[max(zQubits),lastPauliIndex])
            leftCircuit.addRight(controlledZ)
            
        basisChangeCircuit = changeBasisGates(sortedNonIdentityPaulis)
        leftCircuit.addRight(basisChangeCircuit)
        xyCNOTCircuit = cnotChainGates(xAndYQubits)
        leftCircuit.addRight(xyCNOTCircuit)
        if lastPauliType == 3 and xAndYQubits != [] :
            lastCNOT = Gate('cnot', [max(xAndYQubits), lastPauliIndex])
            leftCircuit.addRight(lastCNOT)
            
        rightCircuit = leftCircuit.getInverse()
        middleGate = Gate('rz',lastPauliIndex,(coefficient/numpy.pi))
        leftCircuit.addRight(middleGate)
        finalCircuit = leftCircuit.addRight(rightCircuit)
    
    return leftCircuit
def oplistToInteriorCircuit(oplist):
    circuit = Circuit([])
    for op in oplist:
        termCircuit = pauliStringToInteriorCircuit(op)
        circuit.addRight(termCircuit)
        circuit.markSubcircuit()
    circuit.update()
    return circuit
        
def OBSOLETE_oplistToAncillaCircuit(oplist):
    circuit = Circuit([])
    for op in oplist:
        termCircuit = pauliStringToAncillaCircuit(op)
        circuit.addRight(termCircuit)
        circuit.markSubcircuit()
    circuit.update()
    return circuit


def cnotChainAncilla(indices,ancillaIndex):
    '''take a list of qubit indices, return a circuit applying sequential CNOTS between each pair
    i.e. [1,2,3] -> CNOT(1,2) CNOT(2,3)'''
    circuit = Circuit([])
    if len(indices) < 2:
        return circuit
    for i in range(len(indices)-1):
        gate = Gate('cnot',[indices[i],ancillaIndex])
        circuit.addRight(gate)
            
        
    return circuit

def pauliStringToAncillaCircuit(op,ancillaIndex=-1):
    '''take op term as [coefficient, [list of paulis]], return circuit'''
    coefficient = op[0]
    pauliString = op[1]
    numQubits = len(pauliString)
    circuit = Circuit([])
    if ancillaIndex == -1:
        ancillaIndex = numQubits
    if list(pauliString) == [0]*numQubits: #identity is a special case, apply "global" phase
        identity = Gate('i',0,coefficient/(2 * numpy.pi))
        circuit.addRight(identity)
        return circuit
    else:
        #get (qubit,pauli) pairs for identity paulis
        nonIdentityPaulis = [(index,value) for (index,value) in enumerate(reversed(pauliString)) if value!= 0]
        sortedNonIdentityPaulis = sorted(nonIdentityPaulis,key=lambda thing:thing[0]) #sort paulis in increasing qubit order
        lastPauliIndex = sortedNonIdentityPaulis[-1][0]
        leftCircuit = changeBasisGates(sortedNonIdentityPaulis)
        involvedQubitIndices = [thing[0] for thing in sortedNonIdentityPaulis]
        leftCircuit.addRight(cnotChainAncilla(involvedQubitIndices,ancillaIndex))
        if len(involvedQubitIndices) > 1:
            leftCircuit.addRight(Gate('cnot',[ancillaIndex,lastPauliIndex]))
        rightCircuit = leftCircuit.getInverse()
        rightCircuit.flipHYDirection()
        middleGate = Gate('rz',lastPauliIndex,(coefficient/numpy.pi))
        leftCircuit.addRight(middleGate)
        finalCircuit = leftCircuit.addRight(rightCircuit)
        
        
     #   for thisIndex, thisPauli in enumerate(reversed(pauliString)):
       #     if thisPauli == 1:
      #          gate = Gate('rx',1/2)
    return finalCircuit
        
def oplistToAncillaCircuit(oplist,ancillaInd=-1):
    circuit = Circuit([])
    for op in oplist:
        termCircuit = pauliStringToAncillaCircuit(op,ancillaInd)
        circuit.addRight(termCircuit)
        circuit.markSubcircuit()
    circuit.update()
    return circuit



        
        
        
        