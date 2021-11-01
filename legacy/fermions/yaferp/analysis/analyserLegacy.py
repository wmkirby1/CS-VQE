CIRCUITS_DIR = '/home/andrew/workspace/BKData/circuit/reduced/1e-14/'
ENTANGLING_DIR = '/home/andrew/workspace/BKData/entanglingCount/reduced/1e-14/'
CIRCUIT_TYPES = ['normal',
                 'optimised',
                 'interior',
                 'interiorOptimised',
                 'ancilla',
                 'ancillaOptimised']

import cPickle
from yaferp.analysis import analyser
import os
import datetime


def saveDict(thing,dictPath):
    with open(dictPath,'wb') as f:
        cPickle.dump(thing,f,cPickle.HIGHEST_PROTOCOL)

def storeInDict(dictPath,theKey,theValue,rewrite=0):
    thing = loadDict(dictPath)
    if thing == None:
        thing = {}
    if(theKey in thing) and rewrite==0:
        return thing
    else:
        thing[theKey] = theValue
        saveDict(thing,dictPath)
    return thing

def loadDict(dictPath):
    if not os.path.isfile(dictPath):
        return None
    else:
        with open(dictPath,'rb') as f:
            it = cPickle.load(f)
    return it

def getCircGateList(circuit):
    return circuit.listGates

def countEntanglingGates(circuit):
    '''returns (numSQGs, numEntanglingGates))'''
    gates = getCircGateList(circuit)
    numSQGs = 0
    numEntanglingGates = 0
    numError = 0
    for gate in gates:
        if isinstance(gate.qubits,int) or len(gate.qubits) == 1 :
            numSQGs += 1
        elif len(gate.qubits) == 2:
            numEntanglingGates +=1
        else :
            numError +=1
    if numSQGs + numEntanglingGates != circuit.numGates():
        return -1
    return (numSQGs,numEntanglingGates)

def getEntanglingGates(molecule, circuitType='normal', boolJWorBK=0,ordering = ''):
    if boolJWorBK:
        circuitPath = CIRCUITS_DIR + '/' + str(ordering) + '/' + circuitType + '/BK/' + molecule + '.circ'
    else:
        circuitPath = CIRCUITS_DIR + '/' + str(ordering) + '/' + circuitType + '/JW/' + molecule + '.circ'
    with open(circuitPath, 'rb') as f:
        circ = cPickle.load(f)
    return (countEntanglingGates(circ))

def getAllEntanglingGates(molecule,listCircuitTypes = CIRCUIT_TYPES,recalculate=0,verbose=0,ordering=None):
    if ordering == None:
        ordering = ''
    resultsPath = ENTANGLING_DIR + str(ordering) + '/' + molecule + '.ent'
    circDict = loadDict(resultsPath)
    if circDict == None:
        circDict = {}
    for circuitType in listCircuitTypes:
        jwKey = 'JW ' + circuitType
        bkKey = 'BK ' + circuitType
        if circDict == {} or recalculate or jwKey not in circDict:
            thisJWSQGs, thisJWEntanglings = getEntanglingGates(molecule,circuitType,0,ordering)
            circDict[jwKey] = thisJWEntanglings
            saveDict(circDict,resultsPath)
        
        if circDict == {} or recalculate or bkKey not in circDict:
            
            thisBKSQGs, thisBKEntanglings = getEntanglingGates(molecule,circuitType,1,ordering)
            circDict[bkKey] = thisBKEntanglings
            saveDict(circDict,resultsPath)
            
        if circDict == {} or recalculate or 'Orbital Number' not in circDict:
            numOrbitals = analyser.loadOrbitalNumber(molecule, 0)
            circDict['Orbital Number'] = numOrbitals
            saveDict(circDict,resultsPath)
            
            
    return circDict

def getAllEntanglingGatesManyMolecules(listMolecules,listCircuitTypes=CIRCUIT_TYPES,recalculate=0,verbose=0,ordering2=None):
    result = {}
    for index,molecule in enumerate(listMolecules):
        thisDict = getAllEntanglingGates(molecule,listCircuitTypes,ordering=ordering2)
        result[molecule] = thisDict
        if verbose:
            print(datetime.datetime.now())
            print (molecule +' ' +str(index) +' of ' + str(len(listMolecules)) + ' done.' )
    return result
        

def getManyEntanglingGates(listMolecules, circuitType='normal'):
    for molecule in listMolecules:
        numOrbitals = analyser.loadOrbitalNumber(molecule, 0)
        JWSQGs, JWEntanglings = getEntanglingGates(molecule, circuitType, 0)
        BKSQGs, BKEntanglings = getEntanglingGates(molecule, circuitType, 1)
        print(molecule + ',' + str(numOrbitals) + ',' + str(JWSQGs) + ',' + str(JWEntanglings) + ',' + str(BKSQGs) + ',' + str(BKEntanglings))
    return

def getEvenMoreEntanglingGates(listMolecules, listCircuitTypes = CIRCUIT_TYPES):
    for circuitType in listCircuitTypes:
        print(circuitType)
        getManyEntanglingGates(listMolecules,circuitType)
    return
        
        

