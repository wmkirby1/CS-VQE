import os
import os.path
from yaferp.integrals import readintegrals
import decimal
from yaferp.general import sparseFermions
import cPickle
import copy
from yaferp.circuits import circuit
import numpy

DATA_DIR = '/work/at1913/BKData/'
INTEGRALS_DIR = DATA_DIR + 'integrals/'
OPLIST_JW_DIR = DATA_DIR + 'hamiltonian/oplist/JW/'
OPLIST_BK_DIR = DATA_DIR + 'hamiltonian/oplist/BK/'
REDUCED_OPLIST_DIR = DATA_DIR + 'hamiltonian/reducedOplist/'
ENERGY_JW_DIR = DATA_DIR + 'energies/JW/'
ENERGY_BK_DIR = DATA_DIR + 'energies/BK/'
REDUCED_ENERGY_DIR = DATA_DIR + 'energies/reduced/'
EIGENVECS_JW_DIR = DATA_DIR + 'eigenvectors/JW/'
EIGENVECS_BK_DIR = DATA_DIR + 'eigenvectors/BK/'
REDUCED_EIGENVECS_DIR = DATA_DIR + 'eigenvectors/reduced/'
GATES_JW_DIR = DATA_DIR + 'gates/JW/'
GATES_BK_DIR = DATA_DIR + 'gates/BK/'
GATECOUNT_JW_DIR = DATA_DIR + 'gatecount/JW/'
GATECOUNT_BK_DIR = DATA_DIR + 'gatecount/BK/'
REDUCED_GATECOUNTS_DIR = DATA_DIR + 'gatecount/reduced/'
REDUCED_ETC_DIR = DATA_DIR + 'gatecount/etc/'
GATECOUNTS_DIR = DATA_DIR + 'gatecount/'
#CIRCUIT_JW_DIR
RB_CODE_DIR = '/home/andrew/workspace/RB_Fermions/'
CIRCUITS_DIR = DATA_DIR + 'circuit/'
SPARSEHAM_DIR = DATA_DIR + 'hamiltonian/sparse/'
#CIRCUIT_JW_DIR

def preciseError(energy1, energy2):
    energy1Decimal = decimal.Decimal(str(energy1))
    energy2Decimal = decimal.Decimal(str(energy2))
    energyDecimal = energy1Decimal - energy2Decimal
    return energyDecimal

def loadReducedOplist(fileName,boolJWorBK,cutoff=1e-14):
    if boolJWorBK:
        oplistPath = REDUCED_OPLIST_DIR + str(cutoff) + '/BK/' + fileName + '.oplist'
    else:
        oplistPath = REDUCED_OPLIST_DIR + str(cutoff) + '/JW/' + fileName + '.oplist'
    oplist = []
    with open(oplistPath,'r') as oplistFile:
        nextLine = oplistFile.readline()
        while nextLine:
            processedOp = eval(nextLine)
            oplist.append(processedOp)
            nextLine = oplistFile.readline()
    return oplist

def writeEnergy(filePath,energy,label,append=1):
    if append:
        openType = 'a'
    else:
        openType = 'w'
    with open(filePath,openType) as outputFile:
        outputFile.write(str(label)+'\n')
        outputFile.write(str(energy)+'\n')
    return

def saveCSC(filePath, matrix):
	numpy.savez(filePath,data=matrix.data,indices=matrix.indices,indptr=matrix.indptr,shape=matrix.shape)

def loadCSC(filePath):
	f = numpy.load(filePath)
	return scipy.sparse.csc_matrix((f['data'],f['indices'],f['indptr']),shape=f['shape'])


def reducedGenerateHamiltonian(fileName,boolJWorBK,recalculate=1,cutoff=1e-14):
	inputPath = INTEGRALS_DIR + fileName + '.int'
	if boolJWorBK:
		outputPath = SPARSEHAM_DIR + str(cutoff) + '/BK/' + fileName + '.csc'
	else:
		outputPath = SPARSEHAM_DIR + str(cutoff) + '/JW/' + fileName + '.csc'
	if recalculate or not os.path.isfile(outputPath):
		try:
			os.remove(outputPath)
		except:
			pass
		try:
			oplist = loadReducedOplist(fileName,boolJWorBK,cutoff)
		except:
			if boolJWorBK:
				strJWorBK = 'Bravyi-Kitaev'
			else:
				strJWorBK = 'Jordan-Wigner'
				print('Cannot read oplist for' + fileName + ' ' + strJWorBK + '.  File probably not present.')
				return
	hamMatrix = sparseFermions.commutingOplistToMatrix(oplist)
	saveCSC(outputPath, hamMatrix)
	return hamMatrix

def reducedHamiltonianAccuracy(fileName, boolJWorBK, recalculate=1, storeEigenvectors=1, cutoff=1e-14):
    inputPath = INTEGRALS_DIR + fileName + '.int'
    if boolJWorBK:
        outputPath = REDUCED_ENERGY_DIR + str(cutoff) + '/BK/' + fileName + '.egs'
    else:
        outputPath = REDUCED_ENERGY_DIR + str(cutoff) + '/JW/' + fileName + '.egs'
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
            oplist = loadReducedOplist(fileName,boolJWorBK,cutoff)
        except:
            if boolJWorBK:
                strJWorBK = 'Bravyi-Kitaev'
            else:
                strJWorBK = 'Jordan-Wigner'
                print('Cannot read oplist for' + fileName + ' ' + strJWorBK + '.  File probably not present.')
                return
		#hamMatrix = sparseFermions2.commutingOplistToMatrix(oplist)
		(ourEnergyR,ourVector) = sparseFermions.getTrueEigensystem(oplist)
        ourEnergy = ourEnergyR[0]
        if storeEigenvectors:
            if boolJWorBK:
                vecFilepath = REDUCED_EIGENVECS_DIR + str(cutoff) + '/BK/' + fileName + '.evec'
            else:
                vecFilepath = REDUCED_EIGENVECS_DIR + str(cutoff) + '/JW/' + fileName + '.evec'
            with open(vecFilepath, 'wb') as f:
                cPickle.dump(ourVector,f,cPickle.HIGHEST_PROTOCOL)
            
        energies["CALCULATED ENERGY"] = ourEnergy
        refDiscrepancy = preciseError(referenceEnergy,ourEnergy)
        energies["REFERENCE CALCULATED DISCREPANCY"] = refDiscrepancy
            
        for label in energies:
            writeEnergy(outputPath,energies[label],label)  
    return (ourEnergy, ourVector)

def countGatesWIP(filename,boolJWorBK,cutoff=1e-14):
    if boolJWorBK:
        outputPathCounts = REDUCED_GATECOUNTS_DIR + str(cutoff) + '/BK/' + filename + '.gco'
        outputPathValues = REDUCED_ETC_DIR + str(cutoff) + '/BK/' + filename + '.exva'
    else:
        outputPathCounts = REDUCED_GATECOUNTS_DIR + str(cutoff) + '/JW/' + filename + '.gco'
        outputPathValues = REDUCED_ETC_DIR + str(cutoff) + '/JW/' + filename + '.exva'
    
    gateCounts = {}
    expectValues = {}
    import scipy.sparse
    hamiltonian = loadReducedOplist(filename,boolJWorBK,cutoff)
    gateCounts['Negligibility threshold'] = cutoff
    
    eigval,eigvec = reducedHamiltonianAccuracy(filename,boolJWorBK,1,1,cutoff)
    eigvec2 = copy.deepcopy(eigvec)
    eigvec3 = copy.deepcopy(eigvec)
    eigvec4 = copy.deepcopy(eigvec)
    eigvec5 = scipy.sparse.kron(eigvec4,[[1.],[0.]])
    
    circ = circuit.oplistToCircuit(hamiltonian)
    circInterior = copy.deepcopy(circ)
    circInterior = circInterior.circuitToInterior()
    circAncilla = circuit.oplistToAncillaCircuit(hamiltonian, -1)
    expectValues['Initial'] = circ.expectationValue(eigvec)
    preOptimisedNumGates2 = circ.numGates()
    
    circ = circ.fullCancelDuplicates()
    numGates2 = circ.numGates()
    expectValues['Optimised'] = circ.expectationValue(eigvec2)
    
    
    
    numGatesInterior = circInterior.numGates()
    circInterior = circInterior.fullCancelDuplicates()
    numGatesOptInterior = circInterior.numGates()
    expectValues["Interior Opt"] = circInterior.expectationValue(eigvec3)
    
    gateCounts['Reduced ancilla gate count'] = circAncilla.numGates()
    circAncilla = circAncilla.fullCancelDuplicates()
    expectValues["Ancilla Opt"] = circAncilla.expectationValue(eigvec5)
    gateCounts['Reduced optimised ancilla gate count'] = circAncilla.numGates()
   # circuitReduced = circuitReduced.circuitToInterior()
   # circuitReduced = circuitReduced.fullCancelDuplicates()
   # fullOptGates = circuitReduced.numGates()
    gateCounts['Reduced standard gate count'] = preOptimisedNumGates2
    gateCounts['Reduced optimised gate count'] = numGates2
    gateCounts['Reduced interior gate count'] = numGatesInterior
    gateCounts['Reduced optimised interior gate count'] = numGatesOptInterior
    
    #gateCounts['Reduced full optimisation'] = fullOptGates
    for label in gateCounts:
        writeEnergy(outputPathCounts,gateCounts[label],label)
         
    for label in expectValues:
        writeEnergy(outputPathValues,expectValues[label],label)
    return gateCounts,expectValues

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
            circ = circuit.oplistToCircuit(oplist)
            circ = circ.circuitToInterior()
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

def generateCircuit(fileName,boolJWorBK,cutoff=1e-14,circuitType='normal',overwrite=0):
    if cutoff != -1:
        if boolJWorBK:
            outputPath = CIRCUITS_DIR + '/reduced/' + str(cutoff) + '/' + str(circuitType)+'/BK/' + fileName + '.circ'
        else:
            outputPath = CIRCUITS_DIR + '/reduced/' + str(cutoff) + '/' + str(circuitType)+'/JW/' + fileName + '.circ'


        if overwrite or not os.path.isfile(outputPath):
            try:
                os.remove(outputPath)
            except:
                pass
            oplist = loadReducedOplist(fileName,boolJWorBK,cutoff)
            circ = oplistToCircuit(oplist,circuitType)
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


def generateGateCount(filename,boolJWorBK,cutoff,listCircuitTypes='all',overwrite=0):
    
    ALL_CIRCUIT_TYPES=['normal',
                       'optimised',
                       'interior',
                       'interiorOptimised',
                       'ancilla',
                       'ancillaOptimised']
    
    if cutoff != None:
        if boolJWorBK:
            outputPath = GATECOUNTS_DIR + '/reduced/' + str(cutoff) + '/BK/' + filename + '.gco'
        else:
            outputPath = GATECOUNTS_DIR + '/reduced/' + str(cutoff) +'/JW/' + filename + '.gco'
    
    
    if listCircuitTypes == 'all':
        listCircuitTypes = ALL_CIRCUIT_TYPES
    gatecounts = {}
    
    for circuitType in listCircuitTypes:
        thisCircuit = generateCircuit(filename,boolJWorBK,cutoff,circuitType,overwrite)
        thisGateCount = thisCircuit.numGates()
        storeInDict(outputPath,circuitType,thisGateCount,1)
        gatecounts[circuitType]=thisGateCount
        
    return gatecounts
        
        


