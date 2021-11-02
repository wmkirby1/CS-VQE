'''
Created on 16 Nov 2017

@author: andrew
'''

import cPickle
from yaferp.general import PATHS, directFermions
from yaferp.analysis import analyser


def loadOplist(fileName,boolJWorBK,cutoff=1e-14,ordering=None):
    if ordering == None:
        ordering = ''
    if boolJWorBK:
        oplistPath = PATHS.OPLIST_DIR + str(cutoff) + '/' + ordering + '/BK/' + fileName + '.oplist'
    else:
        oplistPath = PATHS.OPLIST_DIR + str(cutoff) + '/' + ordering + '/JW/' + fileName + '.oplist'
    with open(oplistPath,'r') as oplistFile:
        oplist = cPickle.load(oplistFile)
    return oplist

def loadFCIEnergy(filename,cutoff):
    energyPath = PATHS.ENERGY_DIR + str(cutoff) + '/' + filename + '.egs'
    with open(energyPath,'rb') as f:
        energiesDict = cPickle.load(f)
    return energiesDict['FCI']

def trotterTotalTime(filename,cutoff):
    fci = loadFCIEnergy(filename,cutoff)
    time = directFermions.calculateTime(None, fci)
    return time

def genAllTrotterTimes():
    listMols = analyser.getMoleculesInDirectory('/home/andrew/workspace/BKDataNEW/energies/1e-12')
    times = {}
    for molecule in listMols:
        times[molecule] = trotterTotalTime(molecule,1e-12)
    with open('/home/andrew/workspace/BKDataNEW/trottertimes.dat', 'wb') as f:
        cPickle.dump(times,f,protocol=cPickle.HIGHEST_PROTOCOL)
    return

def getTrotterTime(filename):
    with open(PATHS.DATA_DIR + 'trottertimes.dat', 'rb') as f:
        times = cPickle.load(f)
    return times[filename]

def getEigenvector(filename,boolJWorBK,cutoff=1e-12):
    if boolJWorBK:
        eigenvectorPath = PATHS.EIGENVECS_DIR + str(cutoff) + '/BK/' + filename + '.evec'
    else:
        eigenvectorPath = PATHS.EIGENVECS_DIR + str(cutoff) + '/JW/' + filename + '.evec'
        
    eigenvector = cPickle.load(eigenvectorPath)
    return eigenvector
    

def trotterError(filename,boolJWorBK,cutoff,ordering,trotterOrder,trotterSteps):
    eigenvector = getEigenvector(filename,boolJWorBK,cutoff)
    fciEnergy = loadFCIEnergy(filename,cutoff)
    oplist = loadOplist(filename,boolJWorBK,cutoff,ordering)
    totalTime = getTrotterTime(filename)
    error = directFermions.trotterError(oplist, trotterSteps, totalTime, trotterOrder, eigenvalue=fciEnergy, eigenvec=eigenvector)
    return error

    