'''
Created on 22 Sep 2016
failsafe mode for non-deleting circuits
@author: andrew
'''
DATA_DIR = './BKData/'
TESTS_DIR = DATA_DIR + 'tests/'
CIRCUIT_TEST_DIR = TESTS_DIR + 'circuit/'
CIRCUIT_ANGLE_TEST_DIR = CIRCUIT_TEST_DIR + 'angles/'
CIRCUITS_DIR = DATA_DIR + 'circuit/'
EIGENVECTORS_DIR = DATA_DIR + 'eigenvectors/'
DEFAULT_CUTOFF = 1e-12
import cPickle
import scipy.sparse
import os.path
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


def loadCircuit(filename,boolJWorBK,cutoff=DEFAULT_CUTOFF,circuitType='normal',ordering='magnitude'):
    if boolJWorBK:
        circuitPath = CIRCUITS_DIR + str(cutoff) + '/' + str(ordering) + '/' + str(circuitType) + '/BK/' + filename + '.circ'
    else:
        circuitPath = CIRCUITS_DIR + str(cutoff) + '/' + str(ordering) + '/' + str(circuitType) + '/JW/' + filename + '.circ'

    with open(circuitPath,'rb') as f:
        circuit = cPickle.load(f)
    return circuit

def loadEigenvector(filename,boolJWorBK,cutoff=DEFAULT_CUTOFF):
    if boolJWorBK:
        eigvecPath = EIGENVECTORS_DIR + '/' + str(cutoff) + '/BK/' + filename + '.evec'
    else:
        eigvecPath = EIGENVECTORS_DIR + '/' + str(cutoff) + '/JW/' + filename + '.evec'
    with open(eigvecPath,'rb') as f:
        eigenvec = cPickle.load(f)
    return eigenvec

def calculateCircuitAngle(filename,boolJWorBK,cutoff=DEFAULT_CUTOFF,circuitType='normal',ordering='magnitude'):
    circ = loadCircuit(filename,boolJWorBK,cutoff,circuitType,ordering)
    eigenvec = loadEigenvector(filename,boolJWorBK,cutoff)
    if circuitType in ['ancilla','ancillaOptimised']:
        testVec = scipy.sparse.kron([[1.],[0.]],eigenvec)
    else:
        testVec = eigenvec
    ang = circ.angle(testVec)
    return ang
    
    

def generateCircuitAngle(filename,boolJWorBK,cutoff=DEFAULT_CUTOFF,circuitType='normal',ordering='magnitude',overwrite=0):
    if cutoff != None:
        if boolJWorBK:
            outputPath = CIRCUIT_ANGLE_TEST_DIR + str(cutoff) + '/' + str(ordering) + '/BK/' + filename + '.angs'
        else:
            outputPath = CIRCUIT_ANGLE_TEST_DIR + str(cutoff) + '/' + str(ordering) + '/JW/' + filename + '.angs'
    ang = calculateCircuitAngle(filename,boolJWorBK,cutoff,circuitType,ordering)
    storeInDict(outputPath,circuitType,ang,overwrite)
    return ang

def generateManyCircuitAngles(filename,boolJWorBK,cutoff=DEFAULT_CUTOFF,listCircuitTypes='all',ordering='magnitude',overwrite=0):

    ALL_CIRCUIT_TYPES=['normal',
                       'optimised',
                       'interior',
                       'interiorOptimised',
                       'ancilla',
                       'ancillaOptimised']
    
    if listCircuitTypes == 'all':
        listCircuitTypes = ALL_CIRCUIT_TYPES
    angles = {}
    if cutoff != None:
        if boolJWorBK:
            outputPath = CIRCUIT_ANGLE_TEST_DIR + str(cutoff) + '/BK/' + filename + '.angs'
        else:
            outputPath = CIRCUIT_ANGLE_TEST_DIR + str(cutoff) +'/JW/' + filename + '.angs'
            
    for circuitType in listCircuitTypes:
        currentTestsDict = loadDict(outputPath)
        if currentTestsDict == None:
            currentTestsDict = {}
        if (not overwrite) and (not circuitType in currentTestsDict):
            thisAngle = generateCircuitAngle(filename,boolJWorBK,cutoff,circuitType,ordering,overwrite)
            angles[circuitType] = thisAngle
        
    return angles


