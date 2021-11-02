'''
Created on 22 Sep 2016

@author: andrew
'''
DATA_DIR = '/work/at1913/BKData/'
TESTS_DIR = DATA_DIR + 'tests/'
CIRCUIT_TEST_DIR = TESTS_DIR + 'circuit/'
CIRCUIT_ANGLE_TEST_DIR = CIRCUIT_TEST_DIR + 'angles/'
DEFAULT_CUTOFF = 1e-12
from yaferp.analysis import analyser
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

def calculateCircuitAngle(filename,boolJWorBK,cutoff=DEFAULT_CUTOFF,circuitType='normal',overwrite=0):
    '''!!!this WILL FAIL if eigenvectors have not been previously generated!!! (also if cutoff ==None'''
    circ = analyser.generateCircuit(filename, boolJWorBK, cutoff, circuitType, overwrite)
    eigvec = analyser.readEigenvector(filename, boolJWorBK, cutoff)
    if circuitType in ['ancilla','ancillaOptimised']:
        testVec = scipy.sparse.kron([[1.],[0.]],eigvec)
    else:
        testVec = eigvec
    ang = circ.angle(testVec)
    return ang


def generateCircuitAngle(filename,boolJWorBK,cutoff=DEFAULT_CUTOFF,circuitType='normal',overwrite=0):
    if cutoff != None:
        if boolJWorBK:
            outputPath = CIRCUIT_ANGLE_TEST_DIR + '/reduced/' + str(cutoff) + '/BK/' + filename + '.angs'
        else:
            outputPath = CIRCUIT_ANGLE_TEST_DIR + '/reduced/' + str(cutoff) +'/JW/' + filename + '.angs'
    ang = calculateCircuitAngle(filename,boolJWorBK,cutoff,circuitType,overwrite)
    storeInDict(outputPath,circuitType,ang,overwrite)
    return ang

def generateManyCircuitAngles(filename,boolJWorBK,cutoff=DEFAULT_CUTOFF,listCircuitTypes='all',overwrite=0):

    ALL_CIRCUIT_TYPES=['normal',
                       'optimised',
                       'interior',
                       'interiorOptimised',
                       'ancilla',
                       'ancillaOptimised']
    
    if listCircuitTypes == 'all':
        listCircuitTypes = ALL_CIRCUIT_TYPES
    angles = {}
    
    for circuitType in listCircuitTypes:
        thisAngle = generateCircuitAngle(filename,boolJWorBK,cutoff,circuitType,overwrite)
        angles[circuitType] = thisAngle
        
    return angles


