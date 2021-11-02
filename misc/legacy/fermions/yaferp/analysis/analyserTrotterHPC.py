'''
Created on 16 Nov 2017

@author: andrew
'''

import cPickle
from yaferp.general import directFermions

DATA_DIR = './BKData/'
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
TROTTERERROR_DIR = DATA_DIR + '/trotterErrors/'

def loadOplist(fileName,boolJWorBK,cutoff=1e-14,ordering=None):
    if ordering == None:
        ordering = ''
    if boolJWorBK:
        oplistPath = OPLIST_DIR + str(cutoff) +'/' + ordering +  '/BK/' + fileName + '.oplist'
    else:
        oplistPath = OPLIST_DIR + str(cutoff) +'/' + ordering + '/JW/' + fileName + '.oplist'
    with open(oplistPath,'r') as oplistFile:
        oplist = cPickle.load(oplistFile)
    return oplist

def loadFCIEnergy(filename,cutoff):
    energyPath = ENERGY_DIR + str(cutoff) + '/' + filename + '.egs'
    with open(energyPath,'rb') as f:
        energiesDict = cPickle.load(f)
    return energiesDict['FCI']

def getTrotterTime(filename):
    with open(DATA_DIR + 'trottertimes.dat','rb') as f:
        times = cPickle.load(f)
    return times[filename]

def getEigenvector(filename,boolJWorBK,cutoff=1e-12):
    if boolJWorBK:
        eigenvectorPath = EIGENVECS_DIR + str(cutoff) + '/BK/' + filename + '.evec'
    else:
        eigenvectorPath = EIGENVECS_DIR + str(cutoff) + '/JW/' + filename + '.evec'
    
    with open(eigenvectorPath,'rb') as f:
        eigenvector = cPickle.load(f)
    return eigenvector
    

def trotterError(filename,boolJWorBK,cutoff,ordering,trotterOrder,trotterSteps):
    eigenvector = getEigenvector(filename,boolJWorBK,cutoff)
    fciEnergy = loadFCIEnergy(filename,cutoff)
    oplist = loadOplist(filename,boolJWorBK,cutoff,ordering)
    totalTime = getTrotterTime(filename)
    error = directFermions.trotterError(oplist, trotterSteps, totalTime, trotterOrder, eigenvalue=fciEnergy, eigenvec=eigenvector)
    return error

def saveTrotterError(filename,boolJWorBK,cutoff,ordering,trotterOrder,trotterSteps):
    if boolJWorBK:
        outputFile = TROTTERERROR_DIR + str(cutoff) + '/' + ordering + '/' + str(trotterOrder) + '/BK/' + filename + '_' + str(trotterSteps) + '.terr'
    else:
        outputFile = TROTTERERROR_DIR + str(cutoff) + '/' + ordering + '/' + str(trotterOrder) + '/JW/' + filename + '_' + str(trotterSteps) + '.terr'
    
    err = trotterError(filename,boolJWorBK,cutoff,ordering,trotterOrder,trotterSteps)
    with open(outputFile, 'wb') as f:
        f.write(str(trotterSteps) + ',' + str(err) + '\n')
    return


    