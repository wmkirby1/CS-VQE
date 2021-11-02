'''
Created on 24 Oct 2017

@author: andrew
'''
import pickle as cPickle
import os
from yaferp.circuits import circuit
import datetime
from yaferp.analysis import analyser
from yaferp.general import sparseFermions

DATA_DIR = '/home/atrant01/group/BKData/'
INTEGRALS_DIR = DATA_DIR + 'integrals/'
OPLIST_DIR = DATA_DIR + 'hamiltonian/oplist/'
OPLIST_JW_DIR = DATA_DIR + 'hamiltonian/oplist/JW/'
OPLIST_BK_DIR = DATA_DIR + 'hamiltonian/oplist/BK/'
REDUCED_OPLIST_DIR = DATA_DIR + 'hamiltonian/reducedOplist/'
ENERGY_DIR = DATA_DIR + 'energies/'
ENERGY_JW_DIR = DATA_DIR + 'energies/JW/'
ENERGY_BK_DIR = DATA_DIR + 'energies/BK/'
REDUCED_ENERGY_DIR = DATA_DIR + 'energies/reduced/'
EIGENVECS_DIR = DATA_DIR + 'eigenvectors/'
EIGENVECS_JW_DIR = DATA_DIR + 'eigenvectors/JW/'
EIGENVECS_BK_DIR = DATA_DIR + 'eigenvectors/BK/'
REDUCED_EIGENVECS_DIR = DATA_DIR + 'eigenvectors/reduced/'
GATES_JW_DIR = DATA_DIR + 'gates/JW/'
GATES_BK_DIR = DATA_DIR + 'gates/BK/'
CIRCUITS_DIR = DATA_DIR + 'circuit/'
GATECOUNT_JW_DIR = DATA_DIR + 'gatecount/JW/'
GATECOUNT_BK_DIR = DATA_DIR + 'gatecount/BK/'
GATECOUNTS_DIR = DATA_DIR + '/gatecount/'
SPARSE_DIR = DATA_DIR + '/hamiltonian/sparse/'


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

def loadOplist(fileName,boolJWorBK,cutoff=1e-14,ordering=None):
    if ordering == None:
        ordering = ''
    if boolJWorBK:
        oplistPath = OPLIST_DIR + str(cutoff) +'/' + ordering +  '/BK/' + fileName + '.oplist'
    else:
        oplistPath = OPLIST_DIR + str(cutoff) +'/' + ordering + '/JW/' + fileName + '.oplist'
    with open(oplistPath,'rb') as oplistFile:
        oplist = cPickle.load(oplistFile)
    return oplist


def getUnoptimisedCircuit(filename,boolJWorBK,cutoff=1e-13,circuitType='optimised',ordering=''):
    if circuitType == 'optimised':
        originalCircuitType = 'normal'
    elif circuitType == 'ancillaOptimised':
        originalCircuitType = 'ancilla'
    elif circuitType == 'interiorOptimised':
        originalCircuitType = 'interior'
    else:
        return None,None

    if boolJWorBK:
        originalCircuitPath = '{}/{}/{}/{}/BK/{}.circ'.format(CIRCUITS_DIR,str(cutoff),ordering,originalCircuitType,filename)
    else:
        originalCircuitPath = '{}/{}/{}/{}/JW/{}.circ'.format(CIRCUITS_DIR,str(cutoff),ordering,originalCircuitType,filename)

    try:
        with open(originalCircuitPath,'rb') as f:
            originalCircuit = cPickle.load(f)
    except IOError:
        return None,None
    return originalCircuit,originalCircuitType


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
            outputPath = CIRCUITS_DIR + '/' + str(cutoff) + '/' + str(ordering) + '/' + str(circuitType)+'/BK/' + fileName + '.circ'
        else:
            outputPath = CIRCUITS_DIR + '/' + str(cutoff) + '/' + str(ordering) + '/' + str(circuitType)+'/JW/' + fileName + '.circ'
        
        if overwrite or not os.path.isfile(outputPath):
            try:
                os.remove(outputPath)
            except:
                pass
            originalCircuit,originalCircuitType = getUnoptimisedCircuit(fileName,boolJWorBK,cutoff,circuitType,ordering)
            if originalCircuit == None or originalCircuitType == None:
                oplist = loadOplist(fileName,boolJWorBK,cutoff,ordering)
                circ = oplistToCircuit(oplist,circuitType)
            else:
                circ = oplistToCircuit(None,circuitType,originalCircuit,originalCircuitType)
            #print(circ.readable())
            with open(outputPath,'wb') as f:
                cPickle.dump(circ,f,cPickle.HIGHEST_PROTOCOL)
        else:
            with open(outputPath,'rb') as f:
                circ = cPickle.load(f)
    return circ

def generateManyCircuit(filename,boolJWorBK,cutoff=1e-12,listCircuitTypes='all',overwrite=0,ordering=None,verbose=True):
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
        thisCircuit = generateCircuit(filename,boolJWorBK,cutoff,circuitType,overwrite,ordering)
        circuits[circuitType] = thisCircuit
        print('done' + circuitType)
        
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
            outputPath = GATECOUNTS_DIR + '/' + str(cutoff) + '/BK/' + filename + '.gco'
        else:
            outputPath = GATECOUNTS_DIR + '/' + str(cutoff) +'/JW/' + filename + '.gco'
    
    
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
        outputPath = OPLIST_DIR + '/' + str(cutoff) + '/' + orderingFn.__name__ + '/BK/' + filename + '.oplist'
    else:
        outputPath = OPLIST_DIR + '/' + str(cutoff) + '/' + orderingFn.__name__ + '/JW/' + filename + '.oplist'
        
    if overwrite or not os.path.isfile(outputPath):
        fred = loadOplist(filename,boolJWorBK,cutoff)
        fredOrdered = orderingFn(fred)
        analyser.writeOplistToFile(fredOrdered, outputPath, 0)
    return fredOrdered

def getMoleculesInDirectory(directory):
    '''get the files in an integral directory - remove the last four characters (.int)'''
    listFiles = os.listdir(directory)
    listFiles.sort()
    listNames = [file.split('.')[0] for file in listFiles]
    return listNames

def eigensystem(filename, boolJWorBK, cutoff=1e-12,appendIt=1,overwriteEnergy=1):
    oplist = loadOplist(filename,boolJWorBK,cutoff,ordering='magnitude')
    energyPath = ENERGY_DIR + str(cutoff) + '/' +  filename + '.egs'
    if boolJWorBK:
        eigenvectorPath = EIGENVECS_DIR + str(cutoff) + '/BK/' + filename + '.evec'
    else:
        eigenvectorPath = EIGENVECS_DIR + str(cutoff) + '/JW/' + filename + '.evec'
        
    (ourEnergyR,ourVector) = sparseFermions.getTrueEigensystem(oplist)
    ourEnergy = ourEnergyR[0]
    with open(eigenvectorPath, 'wb') as f:
        cPickle.dump(ourVector,f,cPickle.HIGHEST_PROTOCOL)
    if boolJWorBK:
        storeInDict(energyPath,'QUBIT BK ',ourEnergy,overwriteEnergy)
    else:
        storeInDict(energyPath,'QUBIT JW', ourEnergy,overwriteEnergy)
    return

def targettedEigensystem(filename,boolJWorBK,cutoff=1e-12,overwrite=1):
    oplist = loadOplist(filename,boolJWorBK,cutoff,ordering='magnitude')
    energyPath = ENERGY_DIR + str(cutoff) + '/' + filename + '.egs'
    if boolJWorBK:
        eigenvectorPath = EIGENVECS_DIR + str(cutoff) + '/BK/'+ filename + '.evec'
    else:
        eigenvectorPath = EIGENVECS_DIR + str(cutoff) + '/JW/' + filename + '.evec'
    energyDict = loadDict(energyPath)
    fciEnergy = energyDict['FCI']

    (ourEnergyR,ourVector) = sparseFermions.findEigensystemNearTarget(oplist, fciEnergy, 1)
    ourEnergy = ourEnergyR[0]
    with open(eigenvectorPath, 'wb') as f:
        cPickle.dump(ourVector,f,cPickle.HIGHEST_PROTOCOL)
    if boolJWorBK:
        storeInDict(energyPath,'QUBIT BK ',ourEnergy,overwrite)
    else:
        storeInDict(energyPath,'QUBIT JW', ourEnergy,overwrite)
    storeInDict(energyPath,'SIGMA ENERGIES?', 1,overwrite)
    return




def analyseHamiltonianEnergyAccuracy(fileName, boolJWorBK, recalculate=1, storeEigenvectors=1, skipRB=1):
    inputPath = INTEGRALS_DIR + fileName + '.int'
    if boolJWorBK:
        outputPath = ENERGY_BK_DIR + fileName + '.egs'
    else:
        outputPath = ENERGY_JW_DIR + fileName + '.egs'
            
    if recalculate or not os.path.isfile(outputPath):
        try:
            os.remove(outputPath)
        except:
            pass
        energies = {}
        referenceEnergy = readintegrals.importRBEnergies(inputPath)
        energies["REFERENCE ENERGY"] = referenceEnergy
        #TODO: WRITE CASE WHERE OPLIST IS NOT PRESENT!
        try:
            oplist = loadOplist(fileName,boolJWorBK)
        except:
            if boolJWorBK:
                strJWorBK = 'Bravyi-Kitaev'
            else:
                strJWorBK = 'Jordan-Wigner'
                print('Cannot read oplist for' + fileName + ' ' + strJWorBK + '.  File probably not present.')
                return
        
        
        (ourEnergyR,ourVector) = sparseFermions.getTrueEigensystem(oplist)
        ourEnergy = ourEnergyR[0]
        if storeEigenvectors:
            if boolJWorBK:
                vecFilepath = EIGENVECS_BK_DIR + fileName + '.evec'
            else:
                vecFilepath = EIGENVECS_JW_DIR + fileName + '.evec'
            with open(vecFilepath, 'wb') as f:
                cPickle.dump(ourVector,f,cPickle.HIGHEST_PROTOCOL)
            
        energies["CALCULATED ENERGY"] = ourEnergy
        refDiscrepancy = preciseError(referenceEnergy,ourEnergy)
        energies["REFERENCE CALCULATED DISCREPANCY"] = refDiscrepancy

        if not boolJWorBK and not skipRB:
            workingDir = os.getcwd()
            os.chdir(RB_CODE_DIR)
            rbMolecule, rbBasis = string.split(fileName,'-',1)
            rbEnergy = operators.SparseDiagonalize(rbMolecule,rbBasis,'hamiltonian',returnEnergy=True)
            energies["RB ENERGY"] = rbEnergy
            rbRefDiscrepancy = preciseError(referenceEnergy,rbEnergy)
            energies["REFERENCE RB DISCREPANCY"] = rbRefDiscrepancy
            calcRBDiscrepancy = preciseError(ourEnergy,rbEnergy)
            energies["CALCULATED RB DISCREPANCY"] = calcRBDiscrepancy
            
            os.chdir(workingDir)
            
        for label in energies:
            writeEnergy(outputPath,energies[label],label)  
    return

