'''
analyser.py

This contains rough scripts for working purposes, typically enabling batch computation.
Be warned that it may contain hard coded file locations and other things that would make software engineers cry.
Created on 17 Dec 2014

@author: andrew
'''
import os
import os.path
from yaferp.integrals import readintegrals
import string
import decimal
import sys
import pickle as cPickle
#import cPickle
from yaferp.general import sparseFermions, fermions, directFermions
from yaferp.orderings import directOrdering
import copy
from yaferp.circuits import circuit
import csv
import datetime

DATA_DIR = '/home/andrew/data/BKData/'
INTEGRALS_DIR = DATA_DIR + 'integrals/'
OPLIST_DIR = DATA_DIR + 'hamiltonian/oplist/'
OPLIST_JW_DIR = DATA_DIR + 'hamiltonian/oplist/JW/'
OPLIST_BK_DIR = DATA_DIR + 'hamiltonian/oplist/BK/'
REDUCED_OPLIST_DIR = DATA_DIR + 'hamiltonian/reducedOplist/'
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
DEFAULT_CUTOFF=1e-14
#CIRCUIT_JW_DIR
RB_CODE_DIR = '/home/andrew/workspace/RB_Fermions/'
DEFAULT_OPLIST_TOLERANCE=1e-14


sys.path.append(RB_CODE_DIR)
#import operators

def getFilePaths(directory):
    '''return a list of the path of all files in directory'''
    listFiles = os.listdir(directory)
    listPaths = [directory + filename for filename in listFiles]
    listPaths.sort()
    return listPaths

def oplistFromIntsFile(filePath,boolJWorBK):
    integrals = readintegrals.importRBIntegrals(filePath, True)
    numOrbitals = len(integrals[0])
    oplist = fermions.electronicHamiltonian(numOrbitals, boolJWorBK, integrals[0], integrals[1])
    return oplist

def writeOplistToFile(oplist,filePath,append):
    if append:
        openType = 'a'
    else:
        openType = 'w'
        
    with open(filePath,openType) as file:
        if not isinstance(oplist[0],list): #if only one term
            file.write(str(oplist))
        else:
            for term in oplist:
                file.write(str(term)+'\n')
    return
        
def integralFileToOplistFile(fileName,boolJWorBK,recalculate=0):
    import os.path
    inputPath = INTEGRALS_DIR + fileName + '.int'
    if boolJWorBK:
        outputPath = OPLIST_BK_DIR + fileName + '.oplist'
    else:
        outputPath = OPLIST_JW_DIR + fileName + '.oplist'
    if recalculate or not os.path.isfile(outputPath):
        oplist = oplistFromIntsFile(inputPath,boolJWorBK)
        writeOplistToFile(oplist,outputPath)    
    return

def getMoleculesInDirectory(directory):
    '''get the files in an integral directory - remove the last four characters (.int)'''
    listFiles = os.listdir(directory)
    listFiles.sort()
    listNames = [file.split('.')[0] for file in listFiles]
    return listNames

def orderMoleculesByOplistSize(listMolecules,boolJWorBK=0,reverse=0):
    for i in range(len(listMolecules)):
        if boolJWorBK:
            oplistPath = OPLIST_BK_DIR + listMolecules[i] + '.oplist'
        else:
            oplistPath = OPLIST_JW_DIR + listMolecules[i] + '.oplist'
        listMolecules[i] = (listMolecules[i],os.path.getsize(oplistPath))
    listMolecules.sort(key=lambda filename: filename[1], reverse=reverse)
    for i in range(len(listMolecules)):
        listMolecules[i] = listMolecules[i][0]

    return listMolecules
    
    
    
def integralDirectoryToOplists(integralDirectory=INTEGRALS_DIR,recalculate=0):
    listNames = getMoleculesInDirectory(integralDirectory)
    numFiles = len(listNames)
    from datetime import datetime
    for fileNum, name in enumerate(listNames):
        integralFileToOplistFile(name,0,recalculate)
        print(str(datetime.now())+'  Oplist ' + str(fileNum*2+1) + ' of ' + str(numFiles*2) + ' (' + str(name) + ' JW) '+ 'formed.')
        integralFileToOplistFile(name,1,recalculate)
        print(str(datetime.now())+'  Oplist ' + str(fileNum*2+2) + ' of ' + str(numFiles*2) + ' (' + str(name) + ' BK) '+ 'formed.')
    print("Done!")
    return

def writeEnergy(filePath,energy,label,append=1):
    if append:
        openType = 'a'
    else:
        openType = 'w'
    with open(filePath,openType) as outputFile:
        outputFile.write(str(label)+'\n')
        outputFile.write(str(energy)+'\n')
    return

def reduceHamiltonian(fileName,boolJWorBK, cutoff=1e-16,recalculate=0):
    
    if boolJWorBK:
        outputPath = REDUCED_OPLIST_DIR + str(cutoff) + '/BK/' + fileName + '.oplist'
    else:
        outputPath = REDUCED_OPLIST_DIR + str(cutoff) + '/JW/' + fileName + '.oplist'
   
    if not os.path.exists(os.path.dirname(outputPath)):
        try:
            os.makedirs(os.path.dirname(outputPath))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    
    if not recalculate and os.path.isfile(outputPath):
        return
    try:
        os.remove(outputPath)
    except:
        pass
    oplist = loadOplist(fileName,boolJWorBK)
    oplist2 = fermions.oplistRemoveNegligibles(oplist, cutoff)
    writeOplistToFile(oplist2,outputPath,0)
    return oplist2

def directoryReduceHamiltonian(directoryPath,boolJWorBK,cutoff=1e-16,recalculate=0):
    listMolecules = getMoleculesInDirectory(directoryPath)
    for molecule in listMolecules:
        reduceHamiltonian(molecule,boolJWorBK,cutoff,recalculate)
    return
    
    
    
def preciseError(energy1, energy2):
    energy1Decimal = decimal.Decimal(str(energy1))
    energy2Decimal = decimal.Decimal(str(energy2))
    energyDecimal = energy1Decimal - energy2Decimal
    return energyDecimal


def loadOplist(fileName, boolJWorBK, cutoff=None, ordering=None):
    if ordering == None:
        ordering = ''

    if cutoff == None:
        if boolJWorBK:
            oplistPath = OPLIST_DIR + ordering + '/BK/' + fileName + '.oplist'
        else:
            oplistPath = OPLIST_DIR + ordering + '/JW/' + fileName + '.oplist'

    else:
        if boolJWorBK:
            oplistPath = OPLIST_DIR + str(cutoff) + '/' + ordering + '/BK/' + fileName + '.oplist'
        else:
            oplistPath = OPLIST_DIR + str(cutoff) + '/' + ordering + '/JW/' + fileName + '.oplist'

    # '''else: oplistPath = OPLIST_DIR + '/reduced/' + str(cutoff) + '/''''
    '''oplist = []
    with open(oplistPath,'r') as oplistFile:
        nextLine = oplistFile.readline()
        while nextLine:
            processedOp = eval(nextLine)
            oplist.append(processedOp)
            nextLine = oplistFile.readline()
    '''
    with open(oplistPath, 'rb') as f:
        oplist = cPickle.load(f)
    return oplist
        


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

def energyAccuracyDirectory(integralDirectory=INTEGRALS_DIR,recalculate=0):
    listNames = getMoleculesInDirectory(integralDirectory)
    numFiles = len(listNames)
    listErrorStrings = []
    from datetime import datetime
    for fileNum, name in enumerate(listNames):
        try:
            analyseHamiltonianEnergyAccuracy(name,0,recalculate)
            print(str(datetime.now())+'  Calculation ' + str(fileNum*2+1) + ' of ' + str(numFiles*2) + ' (' + str(name) + ' JW) '+ 'done.')
        except:
            print(str(datetime.now())+'  Calculation ' + str(fileNum*2+1) + ' of ' + str(numFiles*2) + ' (' + str(name) + ' JW) '+ 'ERROR.')
            listErrorStrings.append(str(name) + ' JW')
        try:
            analyseHamiltonianEnergyAccuracy(name,1,recalculate)
            print(str(datetime.now())+'  Calculation ' + str(fileNum*2+2) + ' of ' + str(numFiles*2) + ' (' + str(name) + ' BK) '+ 'done.')
        except:
            print(str(datetime.now())+'  Calculation ' + str(fileNum*2+2) + ' of ' + str(numFiles*2) + ' (' + str(name) + ' BK) '+ 'ERROR.')
            listErrorStrings.append(str(name) + ' BK')
            
    if listErrorStrings:
        print('ERRORS occurred in files:')
        for string in listErrorStrings:
            print(string)
    
    print("Done!")
    return

def readEnergy(fileName, boolJWorBK, energyType = "CALCULATED ENERGY\n"):
    if boolJWorBK:
        energyPath = ENERGY_BK_DIR + fileName + '.egs'
    else:
        energyPath = ENERGY_JW_DIR + fileName + '.egs'
    with open(energyPath, 'rb') as energyFile:
        rawEnergyFile = energyFile.readlines()
    for index, line in enumerate(rawEnergyFile):
        if line == energyType:
            return float(rawEnergyFile[index+1])
    return -1

def readEigenvector(filename,boolJWorBK,cutoff=None):
    if cutoff==None:
        if boolJWorBK:
            vecFilepath = EIGENVECS_BK_DIR + filename + '.evec'
        else:
            vecFilepath = EIGENVECS_JW_DIR + filename + '.evec'

    else:
        if boolJWorBK:
            vecFilepath = EIGENVECS_DIR + '/reduced/' + str(cutoff) + '/BK/' + filename + '.evec'
        else:
            vecFilepath = EIGENVECS_DIR + '/reduced/' + str(cutoff) + '/JW/' + filename + '.evec'
             
    with open(vecFilepath, 'rb') as f:
        vector = cPickle.load(f)
    return vector

def EVERYTHINGISONFIRE():
    directoryWithFiles ='/home/andrew/workspace/BKData/hamiltonian/oplist/JW/'
    listNames = getMoleculesInDirectory(directoryWithFiles)
    for name in listNames:
        oplist1 = loadOplist(name,0)[0][1]
        numqubits = len(oplist1)
        print(name, ':',str(numqubits))
    return


def countGates(filename,boolJWorBK,precision=0.0001,t=0,order=1,maxIterations=20):
    hamiltonian = loadOplist(filename,boolJWorBK)
    trueEigenvalue = readEnergy(filename,boolJWorBK)
    trueEigenvector = readEigenvector(filename,boolJWorBK)
    stuff = {}
    stuff["SQ Gates"], stuff["CNOT Gates"] = sparseFermions.quickTrotterGatesToPrecision(hamiltonian, trueEigenvector, trueEigenvalue, precision, t, order, maxIterations)
    return stuff


def countGates2(filename,boolJWorBK):
    hamiltonian = loadOplist(filename,boolJWorBK)
    stuff = {}
    stuff["SQ Gates"], stuff["CNOT Gates"] = sparseFermions.countGatesOneTrotterStep(hamiltonian)
    return stuff


def processGates(fileName,boolJWorBK,recalculate=0,precision=0.0001,t=0,order=1,maxIterations=20):
    if boolJWorBK:
        outputPath = GATES_BK_DIR + fileName + '.gat'
    else:
        outputPath = GATES_JW_DIR + fileName + '.gat'   
    if recalculate or not os.path.isfile(outputPath):
        try:
            os.remove(outputPath)
        except:
            pass
        dictToPrint = countGates(fileName,boolJWorBK,precision,t,order,maxIterations)
        for label in dictToPrint:
            writeEnergy(outputPath,dictToPrint[label],label)  
    return

def processGates2(fileName,boolJWorBK,recalculate=0):
    if boolJWorBK:
        outputPath = '/home/andrew/workspace/BKData/onestepgates/BK/' + fileName + '.gat'
    else:
        outputPath = '/home/andrew/workspace/BKData/onestepgates/JW/' + fileName + '.gat'   
    if recalculate or not os.path.isfile(outputPath):
        try:
            os.remove(outputPath)
        except:
            pass
        dictToPrint = countGates2(fileName,boolJWorBK)
        for label in dictToPrint:
            writeEnergy(outputPath,dictToPrint[label],label)  
    return


def gatesDirectory(directoryWithFiles='/home/andrew/workspace/BKData/eigenvectors/JW/',recalculate=0):
    listNames = getMoleculesInDirectory(directoryWithFiles)
    numFiles = len(listNames)
    listErrorStrings = []
    from datetime import datetime
    for fileNum, name in enumerate(listNames):
        try:
            processGates(name,0,recalculate)
            print(str(datetime.now())+'  Calculation ' + str(fileNum*2+1) + ' of ' + str(numFiles*2) + ' (' + str(name) + ' JW) '+ 'done.')
        except:
            print(str(datetime.now())+'  Calculation ' + str(fileNum*2+1) + ' of ' + str(numFiles*2) + ' (' + str(name) + ' JW) '+ 'ERROR.')
            listErrorStrings.append(str(name) + ' JW')
        try:    
            processGates(name,1,recalculate)
            print(str(datetime.now())+'  Calculation ' + str(fileNum*2+2) + ' of ' + str(numFiles*2) + ' (' + str(name) + ' BK) '+ 'done.')
        except:
            print(str(datetime.now())+'  Calculation ' + str(fileNum*2+2) + ' of ' + str(numFiles*2) + ' (' + str(name) + ' BK) '+ 'ERROR.')
            listErrorStrings.append(str(name) + ' BK')
            
    if listErrorStrings:
        print('ERRORS occurred in files:')
        for string in listErrorStrings:
            print(string)
    
    print("Done!")
    return

def oneStepGatesDirectory(directoryWithFiles=OPLIST_JW_DIR,recalculate=0):
    listNames = getMoleculesInDirectory(directoryWithFiles)
    numFiles = len(listNames)
    listErrorStrings = []
    from datetime import datetime
    for fileNum, name in enumerate(listNames):
        try:
            processGates2(name,0,recalculate)
            print(str(datetime.now())+'  Calculation ' + str(fileNum*2+1) + ' of ' + str(numFiles*2) + ' (' + str(name) + ' JW) '+ 'done.')
        except:
            print(str(datetime.now())+'  Calculation ' + str(fileNum*2+1) + ' of ' + str(numFiles*2) + ' (' + str(name) + ' JW) '+ 'ERROR.')
            listErrorStrings.append(str(name) + ' JW')
        try:    
            processGates2(name,1,recalculate)
            print(str(datetime.now())+'  Calculation ' + str(fileNum*2+2) + ' of ' + str(numFiles*2) + ' (' + str(name) + ' BK) '+ 'done.')
        except:
            print(str(datetime.now())+'  Calculation ' + str(fileNum*2+2) + ' of ' + str(numFiles*2) + ' (' + str(name) + ' BK) '+ 'ERROR.')
            listErrorStrings.append(str(name) + ' BK')
            
    if listErrorStrings:
        print('ERRORS occurred in files:')
        for string in listErrorStrings:
            print(string)
    
    print("Done!")
    return

def doMethane(integrals):
    vectorPathJW = '/home/andrew/workspace/MResProject/JW_FINAL.evec'
    vectorPathBK = '/home/andrew/workspace/MResProject/BK_FINAL.evec'
    import cPickle
    hamJW = fermions.electronicHamiltonian(18, 0, integrals[0], integrals[1])
    vecJW = sparseFermions.getTrueEigensystem(hamJW)[1]
    with open(vectorPathJW,'wb') as f:
        cPickle.dump(vecJW,f,cPickle.HIGHEST_PROTOCOL)
    del vecJW
    print('JW vec stored')
    hamBK = fermions.electronicHamiltonian(18, 1, integrals[0], integrals[1])
    vecBK = sparseFermions.getTrueEigensystem(hamBK)[1]
    with open(vectorPathBK, 'wb') as f:
        cPickle.dump(vecBK,f,cPickle.HIGHEST_PROTOCOL)
    print('BK vec stored')
    with open(vectorPathJW,'rb') as f:
        vecJW = cPickle.load(f)
    import numpy
    
    for i in range(1,6):
        thing = sparseFermions.trotteriseNoMatrix(hamJW, 0.02, i, 1, vecJW)
        ans = numpy.angle((vecJW.H * thing).todense()) * 50
        print('JW 1st order ' + str(i) + ' TS: ' + str(ans))
        
    for i in range(1,6):
        thing = sparseFermions.trotteriseNoMatrix(hamBK, 0.02, i, 1, vecBK)
        ans = numpy.angle((vecBK.H * thing).todense()) * 50
        print('BK 1st order ' + str(i) + ' TS: ' + str(ans))
        
    for i in range(1,6):
        thing = sparseFermions.trotteriseNoMatrix(hamJW, 0.02, i, 2, vecJW)
        ans = numpy.angle((vecJW.H * thing).todense()) * 50
        print('JW 2nd order ' + str(i) + ' TS: ' + str(ans))
        
    for i in range(1,6):
        thing = sparseFermions.trotteriseNoMatrix(hamBK, 0.02, i, 2, vecBK)
        ans = numpy.angle((vecBK.H * thing).todense()) * 50
        print('BK 2nd order ' + str(i) + ' TS: ' + str(ans))
        
    for i in range(1,3):
        thing = sparseFermions.trotteriseNoMatrix(hamJW, 0.02, i, 4, vecJW)
        ans = numpy.angle((vecJW.H * thing).todense()) * 50
        print('JW 4th order ' + str(i) + ' TS: ' + str(ans))
        
    for i in range(1,3):
        thing = sparseFermions.trotteriseNoMatrix(hamBK, 0.02, i, 4, vecBK)
        ans = numpy.angle((vecBK.H * thing).todense()) * 50
        print('BK 4th order ' + str(i) + ' TS: ' + str(ans))
        
    return

def doMethane2(ham):   
    vectorPathJW = '/home/andrew/workspace/MResProject/BK_FINAL.evec'
    import cPickle
    with open(vectorPathJW,'rb') as f:
        vecBK = cPickle.load(f)
    import numpy
    with open('/home/andrew/workspace/MResProject/bkHam.oplist','rb') as f:
        hamBK = cPickle.load(f)
    for i in range(1,5):
        thing = sparseFermions.trotteriseNoMatrix(hamBK, 0.02, i, 1, vecBK[1])
        ans = numpy.angle((vecBK[1].H * thing).todense()) * 50
        print('JW 1st order ' + str(i) + ' TS: ' + str(ans))

    return 

def compareOrderings(filename,boolJWorBK,timeLimit,cutoff=1e-13):
    hamiltonian = loadOplist(filename,boolJWorBK,cutoff=1e-14)
    eigenvalue = readEnergy(filename,boolJWorBK)
    eigenvector = readEigenvector(filename,boolJWorBK,cutoff=1e-14)
    stuff = directOrdering.compareOrderingSchemesSample(hamiltonian, 1, timeLimit, eigenvalue, eigenvector, skipErrors=1)
    return stuff   

def processOrderings(filename,boolJWorBK,timeLimit,recalculate=0):
    if boolJWorBK:
        outputPath = '/home/andrew/workspace/BKData/orderings/BK/' + filename + '.ord'
    else:
        outputPath = '/home/andrew/workspace/BKData/orderings/JW/' + filename + '.ord'   
    if recalculate or not os.path.isfile(outputPath):
        try:
            os.remove(outputPath)
        except:
            pass
        dictErrors = compareOrderings(filename,boolJWorBK,timeLimit)
        dictWithoutTrials = {key: value for key, value in dictErrors.items() 
             if key != 'Trials'}
        for label in dictWithoutTrials:
            writeEnergy(outputPath,dictWithoutTrials[label],label)
        try:
            outputPath2 = outputPath + 't'
            trials = dictErrors['Trials']
            for label in trials:
                writeEnergy(outputPath2,trials[label],label)
        except:
            pass
    
    return

def orderingsDirectory(boolJWorBK,timeLimit,directoryWithFiles='/home/andrew/workspace/BKData/test/',recalculate=0):
    listNames = getMoleculesInDirectory(directoryWithFiles)
    listNames = ['H2-631G-CMO']
    numFiles = len(listNames)
    timeLimitOne = float(timeLimit)/float(numFiles)
    listErrorStrings = []
    from datetime import datetime
    for fileNum, name in enumerate(listNames):
        try:
            processOrderings(name,boolJWorBK,timeLimitOne,recalculate)
            print(str(datetime.now())+'  Calculation ' + str(fileNum+1) + ' of ' + str(numFiles) + 'done.')
        except:
            print(str(datetime.now())+'  Calculation ' + str(fileNum+1) + ' of ' + str(numFiles) + 'ERROR.')
            listErrorStrings.append(str(name) + ' JW')
            
    if listErrorStrings:
        print('ERRORS occurred in files:')
        for string in listErrorStrings:
            print(string)

def insertCoefficients(oplist,listCoeffs):
    newOplist = []
    for (index,term) in enumerate(oplist):
        newterm = copy.deepcopy(term)
        newterm[0] = listCoeffs[index]
        newOplist.append(newterm)
    return newOplist

def doPanicWednesdayNightThing(oplistNoCoeffs,listNewCoeffs):
    listResults = []
    for newCoeffs in listNewCoeffs:
        newOplist = insertCoefficients(oplistNoCoeffs,newCoeffs)
        bestOrdering = directOrdering.findGoodOrderings(newOplist)
        listResults.append(bestOrdering)
    for thing in listResults:
        print(thing)
    print("Done!")
    return
def doWednesdayTimeThing(oplistNoCoeffs, listNewCoeffs):
    listTimes = []
    listSteps = []
    for newCoeffs in listNewCoeffs:
        newOplist = insertCoefficients(oplistNoCoeffs,newCoeffs)
        bestOrdering = directOrdering.findBestOrdering(newOplist)
        newNewOplist = directOrdering.reorderOplist(newOplist, bestOrdering)
        timesDict = directFermions.scanTimeOneStep(newNewOplist, 0, 0, 0)
        timesKeys = list(timesDict.keys())
        timesVals = list(timesDict.values())
        #bestTime = timesKeys[timesVals.index(min(timesVals))]
        realBestTime = 0
        for index,val in enumerate(timesVals):
            if val < 0.0016:
                realBestTime = timesKeys[index]
        bestSteps = sparseFermions.trotterStepsUntilPrecision(newNewOplist, 0.0017, realBestTime, 1, 0, 20)
        listTimes.append(realBestTime)
        listSteps.append(bestSteps)
    for i in range(len(listTimes)):
        print(str(listTimes[i]) +',' + str(listSteps[i]))
    return
ELECTRONNUM_DIRECTORY = '/home/andrew/workspace/BKData/electronnumber/'
MAXNUCCHARGE_DIRECTORY = '/home/andrew/workspace/BKData/maxNuclearCharge/'
def readNumElectrons(filename):
    electronNumPath = ELECTRONNUM_DIRECTORY + filename + '.enumber'
    with open(electronNumPath,'r') as oplistFile:
        nextLine = oplistFile.readline()
        electronNum = int(nextLine)
    return electronNum

def readMaxNuclearCharge(filename):
    electronNumPath = MAXNUCCHARGE_DIRECTORY + filename + '.z'
    with open(electronNumPath,'r') as oplistFile:
        nextLine = oplistFile.readline()
        electronNum = int(nextLine)
    return electronNum
    
def compareOrderingsNew(filename,boolJWorBK, verbose=0):
    eigenvalue = readEnergy(filename,boolJWorBK)
    eigenvector = readEigenvector(filename,boolJWorBK)
    electronNumber = readNumElectrons(filename)
    oplist = loadOplist(filename,boolJWorBK)
    oplist = fermions.oplistRemoveNegligibles(oplist, 1e-12)
    if verbose:
        print('step0')
    hfState = directOrdering.numElectronsToHFState(electronNumber)
    if boolJWorBK:
        thing = getBKHFState(hfState,len(oplist[0][1]))
        hfState = thing
    overlap= checkHFState(hfState,eigenvector,len(oplist[0][1]))
    if verbose:
        print('step1')
    diagonalHamiltonian = directOrdering.separateOplist(oplist)[0]
    #hfEigval,hfState2 = directFermions.getTrueEigensystem(diagonalHamiltonian)
    hfEigval = directOrdering.hartreeFockEnergyEfficient(diagonalHamiltonian, hfState)
    if verbose:
        print('step2')
    importances= directOrdering.calcOplistImportances(oplist, hfState, hfEigval)
    if verbose:
        print('step3')
    results = directOrdering.compareOrderingSchemesLarge(oplist, 1, eigenvector, eigenvalue, importances, verbose)
    return results, overlap



def processOrderingsNew(filename,boolJWorBK,recalculate=0, threshold = 0.95):
    if boolJWorBK:
        outputPath = '/home/andrew/workspace/BKData/orderings3/BK/' + filename + '.ord'
    else:
        outputPath = '/home/andrew/workspace/BKData/orderings3/JW/' + filename + '.ord'   
    if recalculate or not os.path.isfile(outputPath):
        try:
            os.remove(outputPath)
        except:
            pass
        dictErrors,overlap = compareOrderingsNew(filename,boolJWorBK)
        dictErrors["HFoverlap"] = overlap
        for label in dictErrors:
            writeEnergy(outputPath,dictErrors[label],label)
    if overlap < threshold:
        return -1
    else:
        return 0
    return

def checkHFState(hfState,trueEigenvector,numQubits,threshold=0.95):
    hfStateVector = directOrdering.numberStateToVector(hfState, numQubits)
    overlap = abs(directFermions.overlap(hfStateVector, trueEigenvector))
    return overlap

def getBKHFState(jwHFState,numQubits):
    occupationNumber = directOrdering.numberStateToNumber(jwHFState)
    bkBasis = fermions.bkbasis(fermions.ket_basis_build(numQubits))
    thisKet = bkBasis[occupationNumber]
    bitlist = thisKet.bitlist
    activeQubits = [numQubits-x-1 for x in range(len(bitlist)) if bitlist[x] == 1]
    return activeQubits

def electronNumberDirectory():
    directoryWithFiles = '/home/andrew/workspace/BKData/hamiltonian/oplist/JW/'
    listNames = getMoleculesInDirectory(directoryWithFiles)
    for thing in listNames:
        outputPath = '/home/andrew/workspace/BKData/electronnumber/' + thing + '.enumber'
        print(thing)
        num = int(raw_input('input num electrons'))
        with open(outputPath,'wb') as outputFile:
            outputFile.write(str(num))
    return

def orderingsDirectoryNew(boolJWorBK,directoryWithFiles='/home/andrew/workspace/BKData/test2/',recalculate=0):
    listNames = getMoleculesInDirectory(directoryWithFiles)
    numFiles = len(listNames)
    listErrorStrings = []
    from datetime import datetime
    for fileNum, name in enumerate(listNames):
        try:
            hfError = processOrderingsNew(name,boolJWorBK,recalculate)
            if hfError == -1:
                print(str(datetime.now())+'  Calculation ' + str(fileNum+1) + ' of ' + str(numFiles) + '(' + str(name) + ')' + 'done.  WARNING:  CRAPPY HF STATE.')
            else:
                print(str(datetime.now())+'  Calculation ' + str(fileNum+1) + ' of ' + str(numFiles) + '(' + str(name) + ')' + 'done.')
        except:
            print(str(datetime.now())+'  Calculation ' + str(fileNum+1) + ' of ' + str(numFiles) + '(' + str(name) + ')' + 'ERROR.')
            listErrorStrings.append(str(name) + ' JW')    
    if listErrorStrings:
        print('ERRORS occurred in files:')
        for string in listErrorStrings:
            print(string)
    return
def directoryPrintNumTerms():
    directoryWithFiles = '/home/andrew/workspace/BKData/orderings3/JW/'
    listNames = getMoleculesInDirectory(directoryWithFiles)
    for thing in listNames:
        op = loadOplist(thing,0)
        op = fermions.oplistRemoveNegligibles(op, 1e-11)
        print(thing + ' JW ' + str(len(op)))
    for thing in listNames:
        op = loadOplist(thing,1)
        op = fermions.oplistRemoveNegligibles(op, 1e-11)
        print(thing + ' BK ' + str(len(op)))
              
    return




def compareImportances(filename,boolJWorBK,recalculate=0):
    if boolJWorBK:
        outputPath = '/home/andrew/workspace/BKData/importances/BK/' + fileName + '.imp'
    else:
        outputPath = '/home/andrew/workspace/BKData/importances/JW/' + fileName + '.imp'   
    if recalculate or not os.path.isfile(outputPath):
        try:
            os.remove(outputPath)
        except:
            pass
        dictToPrint = countGates2(fileName,boolJWorBK)
        for label in dictToPrint:
            writeEnergy(outputPath,dictToPrint[label],label)  
    return

def countGatesOptimized(filename,boolJWorBK,termThreshold=1e-16):
    gateCounts = {}
    hamiltonian = loadOplist(filename,boolJWorBK)
    ''' circuitNormal = circuit.oplistToCircuit(hamiltonian)
    preOptimisedNumGates = circuitNormal.numGates()
    gateCounts['Standard gate count'] = preOptimisedNumGates
    circuitNormal = circuitNormal.fullCancelDuplicates()
    numGates = circuitNormal.numGates()
    gateCounts['Optimised gate count'] = numGates'''
    gateCounts['Negligibility threshold'] = termThreshold
    hamiltonianReduced = fermions.oplistRemoveNegligibles(hamiltonian, termThreshold)
    circuitReduced = circuit.oplistToCircuit(hamiltonianReduced)
    circuitReducedCopy = copy.deepcopy(circuitReduced)
    preOptimisedNumGates2 = circuitReduced.numGates()
    circuitReduced = circuitReduced.fullCancelDuplicates()
    numGates2 = circuitReduced.numGates()
    circuitInterior = circuitReducedCopy.circuitToInterior()
    numGatesInterior = circuitInterior.numGates()
    circuitInterior = circuitInterior.fullCancelDuplicates()
    numGatesOptInterior = circuitInterior.numGates()
    circuitAncilla = circuit.oplistToAncillaCircuit(hamiltonianReduced, -1)
    gateCounts['Reduced ancilla gate count'] = circuitAncilla.numGates()
    circuitAncilla.fullCancelDuplicates()
    gateCounts['Reduced optimised ancilla gate count'] = circuitAncilla.numGates()
   # circuitReduced = circuitReduced.circuitToInterior()
   # circuitReduced = circuitReduced.fullCancelDuplicates()
   # fullOptGates = circuitReduced.numGates()
    gateCounts['Reduced standard gate count'] = preOptimisedNumGates2
    gateCounts['Reduced optimised gate count'] = numGates2
    gateCounts['Reduced interior gate count'] = numGatesInterior
    gateCounts['Reduced optimised interior gate count'] = numGatesOptInterior
    #gateCounts['Reduced full optimisation'] = fullOptGates
    return gateCounts

def processGateCounts(filename,boolJWorBK,recalculate=1):
    if boolJWorBK:
        outputPath = GATECOUNT_BK_DIR + filename + '.gat'
    else:
        outputPath = GATECOUNT_JW_DIR + filename + '.gat'   
    if recalculate or not os.path.isfile(outputPath):
        try:
            os.remove(outputPath)
        except:
            pass
        dictToPrint = countGatesOptimized(filename,boolJWorBK,termThreshold=1e-16)
        for label in dictToPrint:
            writeEnergy(outputPath,dictToPrint[label],label)  
    return

def gateCountListMolecules(boolJWorBK,listNames,recalculate2=0):
    numFiles = len(listNames)
    listErrorStrings = []
    from datetime import datetime
    for fileNum, name in enumerate(listNames):
        try:
            processGateCounts(name,boolJWorBK,recalculate2)
            print(str(datetime.now())+'  Calculation ' + str(fileNum+1) + ' of ' + str(numFiles) + '(' + str(name) + ')' + 'done.')
        except:
            print(str(datetime.now())+'  Calculation ' + str(fileNum+1) + ' of ' + str(numFiles) + '(' + str(name) + ')' + 'ERROR.')
            listErrorStrings.append(str(name))    
    if listErrorStrings:
        print('ERRORS occurred in files:')
        for string in listErrorStrings:
            print(string)
    return








def gateCountDirectory(boolJWorBK,directoryWithFiles='/home/andrew/workspace/BKData2/test2/',recalculate2=0):
    listNames = getMoleculesInDirectory(directoryWithFiles)
    numFiles = len(listNames)
    listErrorStrings = []
    from datetime import datetime
    for fileNum, name in enumerate(listNames):
        try:
            processGateCounts(name,boolJWorBK,recalculate2)
            print(str(datetime.now())+'  Calculation ' + str(fileNum+1) + ' of ' + str(numFiles) + '(' + str(name) + ')' + 'done.')
        except:
            print(str(datetime.now())+'  Calculation ' + str(fileNum+1) + ' of ' + str(numFiles) + '(' + str(name) + ')' + 'ERROR.')
            listErrorStrings.append(str(name))    
    if listErrorStrings:
        print('ERRORS occurred in files:')
        for string in listErrorStrings:
            print(string)
    return

def loadGateCount(fileName, boolJWorBK, cutoff=DEFAULT_CUTOFF):
    dictStuff = {}
    if cutoff == None:
        if boolJWorBK:
            gatecountPath = GATECOUNTS_DIR + 'BK/' + fileName + '.gco'
        else:
            gatecountPath = GATECOUNTS_DIR + 'JW/' + fileName + '.gco'
    else:
        if boolJWorBK:
            gatecountPath = GATECOUNTS_DIR + 'reduced/' + str(cutoff) + '/BK/' + fileName + '.gco'
        else:
            gatecountPath = GATECOUNTS_DIR + 'reduced/' + str(cutoff) + '/JW/' + fileName + '.gco'
    print(gatecountPath)
    with open(gatecountPath,'r') as gatecountFile:
        nextLine = gatecountFile.readline()
        while nextLine:
            if boolJWorBK:
                label = 'BK '+ nextLine[0:-1]
            else:
                label = 'JW ' + nextLine[0:-1]
            nextLine = gatecountFile.readline()
            thing = float(nextLine[0:-1])
            dictStuff[label] = thing
            nextLine = gatecountFile.readline()
    return dictStuff
import pickle
def loadGateCount2(fileName,boolJWorBK,cutoff=DEFAULT_CUTOFF,directory=None,ordering=None):
    if ordering == None:
        ordering = ''
    if directory == None:
        if cutoff == None:
            if boolJWorBK:
                gatecountPath = GATECOUNTS_DIR + 'BK/' + fileName + '.gco'
            else:
                gatecountPath = GATECOUNTS_DIR + 'JW/' + fileName + '.gco'
        else:
            if boolJWorBK:
                gatecountPath = GATECOUNTS_DIR + 'reduced/' + str(cutoff) +'/' + str(ordering) + '/BK/' + fileName + '.gco'
            else:
                gatecountPath = GATECOUNTS_DIR + 'reduced/' + str(cutoff) + '/' + str(ordering) + '/JW/' + fileName + '.gco'
    
    else:
        if boolJWorBK:
            gatecountPath = directory + 'BK/' + fileName + '.gco'
        else:
            gatecountPath = directory + 'JW/' + fileName + '.gco'
    
    with open(gatecountPath, 'r') as f:
        thing = pickle.load(f)
    
    ''' for i in range(len(thing)):
        if boolJWorBK: 
            thing.keys()[i] = 'BK ' + thing.keys()[i]
        else:
            thing.keys()[i] = 'JW ' + thing.keys()[i]'''
    
    fred = copy.deepcopy(thing.keys())
    for key in fred:
        if boolJWorBK:
            thing["BK " + key]=thing.pop(key)
        else:
            thing["JW " + key]=thing.pop(key)
            
    return thing
    
def loadOrbitalNumber(filename,boolJWorBK):
    hamiltonian = loadOplist(filename,boolJWorBK)
    orbitalNumber = len(hamiltonian[0][1])
    return orbitalNumber

def loadAllGateCountData(filename,mode=1,cutoff=DEFAULT_CUTOFF,ordering=None):
    if mode:
        jwDict = loadGateCount2(filename,0,cutoff,None,ordering)
        bkDict = loadGateCount2(filename,1,cutoff,None,ordering)
    else:
        jwDict = loadGateCount(filename,0)
        bkDict = loadGateCount(filename,1)
    gateCountsDict = dict(jwDict.items() + bkDict.items())
    gateCountsDict['Orbital Number'] = loadOrbitalNumber(filename,0)
    gateCountsDict['Electron Number'] = readNumElectrons(filename)
    gateCountsDict['Max Nuclear Charge'] = readMaxNuclearCharge(filename)
    gateCountsDict['Filename'] = filename
    return gateCountsDict

def doMultipleGateCounts(listFilenames,mode=1,cutoff=DEFAULT_CUTOFF,ordering=None):
    '''if len(listFilenames) == 0:
        return -1
    elif len(listFilenames) == 1:
        return loadAllGateCountData(listFilenames)
    else:'''
    results = {}
    for filename in listFilenames:
        results[filename] = loadAllGateCountData(filename,mode,cutoff,ordering)
    return results



def gateCountHeaders(gateCountsDict):
    firstKey = gateCountsDict.keys()[0]
    firstDict = gateCountsDict[firstKey]
    listHeaders = firstDict.keys()
    return listHeaders

def gateCountCSVOneRow(oneDict,headers):
    listValues = []
    for header in headers:
        listValues.append(oneDict[header])
    return listValues

def gateCountCSV(listFilenames,outputFileName,mode=0):
    fullDict = doMultipleGateCounts(listFilenames,mode)
    headers = gateCountHeaders(fullDict)
    with open(outputFileName,'wb') as outputFile:
        csvFile = csv.writer(outputFile)
        csvFile.writerow(headers)
        print(headers)
        for oneDict in fullDict:
            thing = gateCountCSVOneRow(fullDict[oneDict],headers)
            print(thing)
            csvFile.writerow(thing)
    return


def june14thing(origOplist,indices):
    newOplist = origOplist[indices[0]:indices[1]]
    print(newOplist)
    eigval, eigvec = directFermions.getTrueEigensystem(newOplist)
    circ = circuit.oplistToCircuit(newOplist)
    circ2 = copy.deepcopy(circ)
    circ2 = circ2.circuitToInterior()
    eigvec2 = copy.deepcopy(eigvec)
    eigvec3 = copy.deepcopy(eigvec)
    eigvec4 = copy.deepcopy(eigvec)
    print('original')
    print(circ.expectationValue(eigvec))
    print('interior')
    print(circ2.expectationValue(eigvec2))
    print('opt')
    circ.fullCancelDuplicates()
    circ3 = copy.deepcopy(circ)
    print(circ.expectationValue(eigvec3))
    print('opt int')
   # print(circ.readable())
    circ3 = circ3.circuitToInterior()
    print(circ3.expectationValue(eigvec4))
   # print(circ.readable())
    return newOplist

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
        (ourEnergyR,ourVector) = sparseFermions.getTrueEigensystem(oplist)
        ourEnergy = ourEnergyR[0]
        if storeEigenvectors:
            if boolJWorBK:
                vecFilepath = REDUCED_EIGENVECS_DIR + str(cutoff) + '/BK/' + fileName + '.evecs'
            else:
                vecFilepath = REDUCED_EIGENVECS_DIR + str(cutoff) + '/JW/' + fileName + '.evecs'
            with open(vecFilepath, 'wb') as f:
                cPickle.dump(ourVector,f,cPickle.HIGHEST_PROTOCOL)
            
        energies["CALCULATED ENERGY"] = ourEnergy
        refDiscrepancy = preciseError(referenceEnergy,ourEnergy)
        energies["REFERENCE CALCULATED DISCREPANCY"] = refDiscrepancy
            
        for label in energies:
            writeEnergy(outputPath,energies[label],label)  
    return ourEnergy,ourVector

def countGatesWIP(filename,boolJWorBK,cutoff=1e-14):
    gateCounts = {}
    expectValues = {}
    
    hamiltonian = loadReducedOplist(filename,boolJWorBK,cutoff)
    gateCounts['Negligibility threshold'] = cutoff
    import scipy.sparse
    eigval,eigvec = reducedHamiltonianAccuracy(filename,boolJWorBK,1,1,cutoff)
    eigvec2 = copy.deepcopy(eigvec)
    eigvec3 = copy.deepcopy(eigvec)
    eigvec4 = copy.deepcopy(eigvec)
    eigvec5 = scipy.sparse.kron([[1.],[0.]],eigvec4)
    
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
    gateCounts['Reduced standard gate count'] = preOptimisedNumGates2
    gateCounts['Reduced optimised gate count'] = numGates2
    gateCounts['Reduced interior gate count'] = numGatesInterior
    gateCounts['Reduced optimised interior gate count'] = numGatesOptInterior
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
            #circ = circuit.oplistToCircuit(oplist)
            #circ = circ.circuitToInterior()
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

def generateCircuit(fileName,boolJWorBK,cutoff=1e-14,circuitType='normal',overwrite=0, ordering=None):
    if ordering == None:
        ordering = ''
    if cutoff != -1:
        if boolJWorBK:
            outputPath = CIRCUITS_DIR + '/reduced/' + str(cutoff) + '/' + str(ordering) + '/' + str(circuitType)+'/BK/' + fileName + '.circ'
        else:
            outputPath = CIRCUITS_DIR + '/reduced/' + str(cutoff) + '/' + str(ordering) + '/' + str(circuitType)+'/JW/' + fileName + '.circ'
        
        if overwrite or not os.path.isfile(outputPath):
            try:
                os.remove(outputPath)
            except:
                pass
            oplist = loadReducedOplist(fileName,boolJWorBK,cutoff)
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


def generateGateCount(filename,boolJWorBK,cutoff,listCircuitTypes='all',overwrite=0,ordering=None):
    
    ALL_CIRCUIT_TYPES=['normal',
                       'optimised',
                       'interior',
                       'interiorOptimised',
                       'ancilla',
                       'ancillaOptimised']
    
    if ordering == None:
        ordering = ''
    if cutoff != None:
        if boolJWorBK:
            outputPath = GATECOUNTS_DIR + '/reduced/' + str(cutoff) + '/' + str(ordering) +'/BK/' + filename + '.gco'
        else:
            outputPath = GATECOUNTS_DIR + '/reduced/' + str(cutoff) +'/' + str(ordering) + '/JW/' + filename + '.gco'
    
    
    if listCircuitTypes == 'all':
        listCircuitTypes = ALL_CIRCUIT_TYPES
    gatecounts = {}
    
    for circuitType in listCircuitTypes:
        thisCircuit = generateCircuit(filename,boolJWorBK,cutoff,circuitType,overwrite,ordering)
        thisGateCount = thisCircuit.numGates()
        storeInDict(outputPath,circuitType,thisGateCount,1)
        gatecounts[circuitType]=thisGateCount
        
    return gatecounts

def generateManyGateCounts(listMolecules,cutoff,listCircuitTypes='all',overwrite=0,ordering=None):
    for index,molecule in enumerate(listMolecules):
        generateGateCount(molecule,0,cutoff,listCircuitTypes,overwrite,ordering)
        print(datetime.datetime.now())
        print (molecule +' ' + str((2*index)+1) + 'of' + str(len(listMolecules)*2) )
        generateGateCount(molecule,1,cutoff,listCircuitTypes,overwrite,ordering)
        print(datetime.datetime.now())
        print (molecule +' ' + str((2*index)+2) + 'of' + str(len(listMolecules)*2) )
    
    
    return
        
    
    
    
    

ORDERING_BK_DIR = '/home/andrew/workspace/BKData/orderings/BK/'
ORDERING_JW_DIR = '/home/andrew/workspace/BKData/orderings/JW/'
def loadOrderingData(fileName, boolJWorBK):
    dictStuff = {}
    if boolJWorBK:
        gatecountPath = ORDERING_BK_DIR + fileName + '.ord'
    else:
        gatecountPath = ORDERING_JW_DIR + fileName + '.ord'
    print(gatecountPath)
    with open(gatecountPath,'r') as gatecountFile:
        nextLine = gatecountFile.readline()
        while nextLine:
            if boolJWorBK:
                label = 'BK '+ nextLine[0:-1]
            else:
                label = 'JW ' + nextLine[0:-1]
            nextLine = gatecountFile.readline()
            thing = float(nextLine[0:-1])
            dictStuff[label] = thing
            nextLine = gatecountFile.readline()
    return dictStuff

def loadAllOrderingData(filename,mode=0):
    jwDict = loadOrderingData(filename,0)
    bkDict = loadOrderingData(filename,1)
    gateCountsDict = dict(jwDict.items() + bkDict.items())
    gateCountsDict['Orbital Number'] = loadOrbitalNumber(filename,0)
    gateCountsDict['Filename'] = filename
    return gateCountsDict

def doMultipleOrderingCollection(listFilenames,mode=0):
    '''if len(listFilenames) == 0:
        return -1
    elif len(listFilenames) == 1:
        return loadAllGateCountData(listFilenames)
    else:'''
    results = {}
    for filename in listFilenames:
        results[filename] = loadAllOrderingData(filename,mode)
    return results     
        
def orderingCSV(listFilenames,outputFileName,mode=0):
    fullDict = doMultipleOrderingCollection(listFilenames,mode)
    headers = gateCountHeaders(fullDict)
    with open(outputFileName,'wb') as outputFile:
        csvFile = csv.writer(outputFile)
        csvFile.writerow(headers)
        print(headers)
        for oneDict in fullDict:
            thing = gateCountCSVOneRow(fullDict[oneDict],headers)
            print(thing)
            csvFile.writerow(thing)
    return


def maxNucChargeDirectory():
    nucChargeGuide = {'Al':13,
        'Ar':18,
        'B':5,
        'Be':4,
        'BeH2':4,
        'C':6,
        'CH2':6,
        'Cl':17,
        'F':9,
        'H2':1,
        'H2O':8,
        'H':1,
        'He':2,
        'HeH+':2,
        'HF':9,
        'Li':3,
        'LiH':3,
        'Mg':12,
        'Na':11,
        'N':7,
        'Ne':10,
        'O':8,
        'P':15,
        'S':16,
        'Si':14
        }
    directoryWithFiles = '/home/andrew/workspace/BKData/hamiltonian/oplist/JW/'
    listNames = getMoleculesInDirectory(directoryWithFiles)
    for thing in listNames:
        molName = thing.split('-')[0]
        outputPath = '/home/andrew/workspace/BKData/maxNuclearCharge/' + thing + '.z'
        #print(thing)
        num = nucChargeGuide[molName]
        print(num)
        with open(outputPath,'wb') as outputFile:
            outputFile.write(str(num))
    return

def hydrogenOrderingsGatelengths():
    hydrogenJWHam = [[(0.171768686677+0j), [0, 0, 0, 3]],
 [(0.171768686678+0j), [0, 0, 3, 0]],
 [(-0.217367885286+0j), [0, 3, 0, 0]],
 [(-0.04551506155635+0j), [1, 1, 2, 2]],
 [(0.04551506155635+0j), [1, 2, 2, 1]],
 [(0.04551506155635+0j), [2, 1, 1, 2]],
 [(-0.04551506155635+0j), [2, 2, 1, 1]],
 [(-0.217367885286+0j), [3, 0, 0, 0]]]
    hydrogenBKHam = [[(0.171768686677+0j), [0, 0, 0, 3]],
 [(0.171768686678+0j), [0, 0, 3, 3]],
 [(0.04551506155635+0j), [0, 1, 3, 1]],
 [(0.04551506155635+0j), [0, 2, 3, 2]],
 [(-0.217367885286+0j), [0, 3, 0, 0]],
 [(0.04551506155635+0j), [3, 1, 3, 1]],
 [(0.04551506155635+0j), [3, 2, 3, 2]],
 [(-0.217367885286+0j), [3, 3, 3, 0]]]
    numQubits = len(hydrogenJWHam[0][1])
    permutationLabels = directOrdering.generateOrderingLabels(len(hydrogenJWHam))
    errorDict = {}
    for index, permutation in enumerate(permutationLabels):
        jwOptLen = []
        bkOptLen = []
        jwIntLen = []
        bkIntLen = []
        jwAncLen = []
        bkAncLen = []
        JWHam = copy.deepcopy(hydrogenJWHam)
        JWHam = directOrdering.reorderOplist(JWHam, permutation)
        BKHam = copy.deepcopy(hydrogenBKHam)
        BKHam = directOrdering.reorderOplist(BKHam, permutation)
        thisCircLengths = {}
        jwnormcirc = circuit.oplistToCircuit(JWHam).fullCancelDuplicates()
        jwintcirc = circuit.oplistToInteriorCircuit(JWHam).fullCancelDuplicates()
        jwanccirc = circuit.oplistToAncillaCircuit(JWHam).fullCancelDuplicates()
        bknormcirc = circuit.oplistToCircuit(BKHam).fullCancelDuplicates()
        bkintcirc = circuit.oplistToInteriorCircuit(BKHam).fullCancelDuplicates()
        bkanccirc = circuit.oplistToAncillaCircuit(BKHam).fullCancelDuplicates()
        thisCircLengths['JW optimised'] = len(jwnormcirc.listGates)
        jwOptLen.append(len(jwnormcirc.listGates))
        bkOptLen.append(len(bknormcirc.listGates))
        jwIntLen.append(len(jwintcirc.listGates))
        bkIntLen.append(len(bkintcirc.listGates))
        jwAncLen.append(len(jwanccirc.listGates))
        bkAncLen.append(len(bkanccirc.listGates))
        thisCircLengths['JW interiorOptimised'] = len(jwintcirc.listGates)
        thisCircLengths['JW ancillaOptimised'] = len(jwanccirc.listGates)
        thisCircLengths['BK optimised'] = len(bknormcirc.listGates)
        thisCircLengths['BK interiorOptimised'] = len(bkintcirc.listGates)
        thisCircLengths['BK ancillaOptimised'] = len(bkanccirc.listGates)
        # numSteps = sparseFermions2.trotterStepsUntilPrecision(newOplist)
        errorDict[tuple(permutation)] = thisCircLengths
        print ('done ' + str(index) + ' of ' + str(len(permutationLabels)))
    return errorDict


def circuitLengthDirectory(directory):
    listMols = getMoleculesInDirectory(directory)
    for molecule in listMols:
        with open(directory + molecule + '.circ', 'rb') as f:
            thisCirc = cPickle.load(f)
            print(str(thisCirc.numQubits) + ',' + str(thisCirc.numGates()))

