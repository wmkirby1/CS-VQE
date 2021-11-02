'''
Created on 24 Oct 2017

@author: andrew
'''
from yaferp.general import PATHS
import pickle as cPickle
import os
from yaferp.circuits import circuit
import datetime
from yaferp.analysis import analyser


def loadDict(dictPath):
    if not os.path.isfile(dictPath):
        return None
    else:
        with open(dictPath,'rb') as f:
            it = cPickle.load(f)
    return it


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

def loadOplist(fileName,boolJWorBK,cutoff=1e-14):
    if boolJWorBK:
        oplistPath = PATHS.OPLIST_DIR + str(cutoff) + '/BK/' + fileName + '.ham'
    else:
        oplistPath = PATHS.OPLIST_DIR + str(cutoff) + '/JW/' + fileName + '.ham'
    with open(oplistPath,'r') as oplistFile:
        oplist = cPickle.load(oplistFile)
    return oplist

def oplistToCircuit(oplist,circuitType='normal',rawCircuit=None,rawCircuitType=None,ancillaIndex=-1):
    if circuitType=='normal':
        circ = circuit.oplistToCircuit(oplist)
    elif circuitType=='optimised':
        if rawCircuit==None:
            circ = circuit.oplistToCircuit(oplist)
            circ = circ.fullCancelDuplicates()
        else:
            circ = rawCircuit.fullCancelDuplicates()
    elif circuitType=='interior':
        if rawCircuit==None:
            #circ = circuit.oplistToCircuit(oplist)
            #circ = circ.circuitToIntpython list to tupleerior()
            circ = circuit.oplistToInteriorCircuit(oplist)
        else:
            circ = rawCircuit.circuitToInterior()
    elif circuitType=='interiorOptimised':
        if rawCircuit==None:
            circ = oplistToCircuit(oplist,circuitType='interior')
            circ = circ.fullCancelDuplicates()
        elif rawCircuitType=='interior':
            circ= rawCircuit.fullCancelDuplicates()
        elif rawCircuitType =='normal':
            circ = rawCircuit.circuitToInterior()
            circ = rawCircuit.fullCancelDuplicates()
    elif circuitType == 'ancilla':
        circ = circuit.oplistToAncillaCircuit(oplist, ancillaIndex)
    elif circuitType == 'ancillaOptimised':
        if rawCircuit==None:
            circ = circuit.oplistToAncillaCircuit(oplist, ancillaIndex)
            circ = circ.fullCancelDuplicates()
        else:
            circ = rawCircuit.fullCancelDuplicates()
    return circ

def generateCircuit(fileName,boolJWorBK,cutoff=1e-12,circuitType='normal',overwrite=0, ordering=None):
    if ordering == None:
        ordering = ''
    if cutoff != -1:
        if boolJWorBK:
            outputPath = PATHS.CIRCUITS_DIR + '/' + str(cutoff) + '/' + str(ordering) + '/' + str(circuitType) + '/BK/' + fileName + '.circ'
        else:
            outputPath = PATHS.CIRCUITS_DIR + '/' + str(cutoff) + '/' + str(ordering) + '/' + str(circuitType) + '/JW/' + fileName + '.circ'
        
        if overwrite or not os.path.isfile(outputPath):
            try:
                os.remove(outputPath)
            except:
                pass
            oplist = loadOplist(fileName,boolJWorBK,cutoff)
            circ = oplistToCircuit(oplist,circuitType)
            #print(circ.readable())
            with open(outputPath,'wb') as f:
                cPickle.dump(circ,f,cPickle.HIGHEST_PROTOCOL)
        else:
            with open(outputPath,'rb') as f:
                circ = cPickle.load(f)
    return circ

def generateManyCircuit(filename,boolJWorBK,cutoff=1e-14,listCircuitTypes='all',overwrite=0):
    ALL_CIRCUIT_TYPES=['normal',
                       'optimised',
                       'interior',
                       'interiorOptimised',
                       'ancilla',
                       'ancillaOptimised']
    
    if listCircuitTypes == 'all':
        listCircuitTypes = ALL_CIRCUIT_TYPES
    circuits = {}
    
    for circuitType in listCircuitTypes:
        thisCircuit = generateCircuit(filename,boolJWorBK,cutoff,circuitType,overwrite)
        circuits[circuitType] = thisCircuit
        
    return circuits

def generateGateCount(filename,boolJWorBK,cutoff,listCircuitTypes='all',overwrite=0):

    ALL_CIRCUIT_TYPES=['normal',
                       'optimised',
                       'interior',
                       'interiorOptimised',
                       'ancilla',
                       'ancillaOptimised']
    
    if cutoff != None:
        if boolJWorBK:
            outputPath = PATHS.GATECOUNTS_DIR + '/' + str(cutoff) + '/BK/' + filename + '.gco'
        else:
            outputPath = PATHS.GATECOUNTS_DIR + '/' + str(cutoff) + '/JW/' + filename + '.gco'
    
    
    if listCircuitTypes == 'all':
        listCircuitTypes = ALL_CIRCUIT_TYPES
    gatecounts = {}
    
    for circuitType in listCircuitTypes:
        thisCircuit = generateCircuit(filename,boolJWorBK,cutoff,circuitType,overwrite)
        thisGateCount = thisCircuit.numGates()
        storeInDict(outputPath,circuitType,thisGateCount,1)
        gatecounts[circuitType]=thisGateCount
        
    return gatecounts


def generateAllOrderings(listMolecules,orderingFn,cutoff,overwrite,verbose):
    for index,molecule in enumerate(listMolecules):
        generateOrdering(molecule,0,orderingFn,cutoff,overwrite)
        if verbose:
            print(datetime.datetime.now())
            print (molecule +' JW ' +str(index*2) +' of ' + str(len(listMolecules)*2) + ' done.' )
    
        generateOrdering(molecule,1,orderingFn,cutoff,overwrite)
        if verbose:
            print(datetime.datetime.now())
            print (molecule +' BK ' +str(index*2 +1) +' of ' + str(len(listMolecules)*2) + ' done.' )
    return

def generateOrdering(filename,boolJWorBK,orderingFn,cutoff=1e-12,overwrite=0):
    if boolJWorBK:
        outputPath = PATHS.OPLIST_DIR + '/' + str(cutoff) + '/' + orderingFn.__name__ + '/BK/' + filename + '.oplist'
    else:
        outputPath = PATHS.OPLIST_DIR + '/' + str(cutoff) + '/' + orderingFn.__name__ + '/JW/' + filename + '.oplist'
        
    if overwrite or not os.path.isfile(outputPath):
        fred = loadOplist(filename,boolJWorBK,cutoff)
        fredOrdered = orderingFn(fred)
        with open(outputPath,'wb') as f:
            cPickle.dump(fredOrdered,f, protocol=cPickle.HIGHEST_PROTOCOL)
    return fredOrdered

def getMoleculesInDirectory(directory):
    '''get the files in an integral directory - remove the last four characters (.int)'''
    listFiles = os.listdir(directory)
    listFiles.sort()
    listNames = [file.split('.')[0] for file in listFiles]
    return listNames

def generateOrbitalNumberKey(directory):
    result = {}
    listMols = analyser.getMoleculesInDirectory(directory)
    for index,mol in enumerate(listMols):
        print('{} of {}'.format(index,len(listMols)))
        oplistPath = '{}{}.oplist'.format(directory,mol)
        with open(oplistPath,'rb') as f:
            thing = cPickle.load(f)
        numQubits = len(thing[0][1])
        result[mol] = numQubits
    #print result
    return result

def loadCoplistAsOplist(filepath):
    oplist = []
    with open(filepath,'rb') as f:
        f.readline() #junk first line
        for line in f:
            thisLine = line.strip().split(' ')
            coefficient = complex(float(thisLine[0]),float(thisLine[1]))
            pauliString = [int(x) for x in thisLine[2:]]
            thisTerm = [coefficient,pauliString]
            oplist.append(thisTerm)

    return oplist


def checkpointNameFromRawName(originalName,boolJWorBK):
    strJWorBK = ['JW','BK'][boolJWorBK]
    checkpointName = '{}_magnitude_weakErrorOp_{}.chk'.format(originalName,strJWorBK)
    return checkpointName


COPLIST_DIR = '/home/andrew/CLUSTER_BACKUPS/coplists_16sept/'
HAMILTONIAN_DIR = '/home/andrew/data/BKData/hamiltonian/oplist/1e-12/errorOperator/'
def coplistToOplist(filename,boolJWorBK,ordering,precision=1e-12):
    strJWorBK = ['JW','BK'][boolJWorBK]
    coplistFilename = COPLIST_DIR + checkpointNameFromRawName(filename,boolJWorBK)
    magnitudeFilename = analyser.OPLIST_DIR + '/1e-12/magnitude/{}/{}.oplist'.format(strJWorBK, filename)
    outputName = '{}{}/{}.oplist'.format(HAMILTONIAN_DIR,strJWorBK,filename)
    clive = loadCoplistAsOplist(coplistFilename)
    lenCoplist = len(clive)
    with open(magnitudeFilename,'rb') as f:
        magnitudeOplist = cPickle.load(f)
    if len(magnitudeOplist) == lenCoplist:
        with open(outputName,'wb') as f:
            cPickle.dump(clive,f,cPickle.HIGHEST_PROTOCOL)
    else:
        print('ERROR {}  {}'.format(filename, strJWorBK))
    return
