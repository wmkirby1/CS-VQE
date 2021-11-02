'''
Created on 5 Jun 2017

@author: andrew
'''

from yaferp.orderings import directOrdering
from yaferp.analysis import analyser
import cPickle
import os
import datetime
DATA_DIR = '/home/andrew/workspace/BKData/'
INTEGRALS_DIR = DATA_DIR + 'integrals/'
OPLIST_DIR = DATA_DIR + 'hamiltonian/oplist/'
OPLIST_JW_DIR = DATA_DIR + 'hamiltonian/oplist/JW/'
OPLIST_BK_DIR = DATA_DIR + 'hamiltonian/oplist/BK/'
REDUCED_OPLIST_DIR = DATA_DIR + 'hamiltonian/reducedOplist/'

def generateTopLexiographic(filename,boolJWorBK,cutoff=1e-14,overwrite=0):
    if boolJWorBK:
        outputPath = REDUCED_OPLIST_DIR + '/' + str(cutoff) + '/topLexiographic/BK/' + filename + '.oplist'
    else:
        outputPath = REDUCED_OPLIST_DIR + '/' + str(cutoff) + '/topLexiographic/JW/' + filename + '.oplist'
        
    if overwrite or not os.path.isfile(outputPath):
        fred = analyser.loadReducedOplist(filename, boolJWorBK, cutoff)
        fredOrdered = directOrdering.lexiographic(fred)
        analyser.writeOplistToFile(fredOrdered, outputPath, 0)
    return fredOrdered

def generateTopMagnitude(filename,boolJWorBK,cutoff=1e-14,overwrite=0):
    if boolJWorBK:
        outputPath = REDUCED_OPLIST_DIR + '/' + str(cutoff) + '/topMagnitude/BK/' + filename + '.oplist'
    else:
        outputPath = REDUCED_OPLIST_DIR + '/' + str(cutoff) + '/topMagnitude/JW/' + filename + '.oplist'
        
    if overwrite or not os.path.isfile(outputPath):
        fred = analyser.loadReducedOplist(filename, boolJWorBK, cutoff)
        fredOrdered = directOrdering.magnitude(fred)
        analyser.writeOplistToFile(fredOrdered, outputPath, 0)
    return fredOrdered

def generateAllTopLexiographics(listMolecules,cutoff=1e-14,overwrite=0,verbose=False):
   # numSys = len(listMolecules)*2
    for index,molecule in enumerate(listMolecules):
        generateTopLexiographic(molecule,0,cutoff,overwrite)
        if verbose:
            print(datetime.datetime.now())
            print (molecule +' JW ' +str(index*2) +' of ' + str(len(listMolecules)*2) + ' done.' )
        
        generateTopLexiographic(molecule,1,cutoff,overwrite)
        if verbose:
            print(datetime.datetime.now())
            print (molecule +' BK ' +str(index*2 +1) +' of ' + str(len(listMolecules)*2) + ' done.' )
    return
def generateAllTopMagnitudes(listMolecules,cutoff=1e-14,overwrite=0,verbose=False):
   # numSys = len(listMolecules)*2
    for index,molecule in enumerate(listMolecules):
        generateTopMagnitude(molecule,0,cutoff,overwrite)
        if verbose:
            print(datetime.datetime.now())
            print (molecule +' JW ' +str(index*2) +' of ' + str(len(listMolecules)*2) + ' done.' )
    
        generateTopMagnitude(molecule,1,cutoff,overwrite)
        if verbose:
            print(datetime.datetime.now())
            print (molecule +' BK ' +str(index*2 +1) +' of ' + str(len(listMolecules)*2) + ' done.' )
    return

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
def generateOrdering(filename,boolJWorBK,orderingFn,cutoff=1e-14,overwrite=0):
    if boolJWorBK:
        outputPath = REDUCED_OPLIST_DIR + '/' + str(cutoff) + '/' + orderingFn.__name__ + '/BK/' + filename + '.oplist'
    else:
        outputPath = REDUCED_OPLIST_DIR + '/' + str(cutoff) + '/' + orderingFn.__name__ + '/JW/' + filename + '.oplist'
        
    if overwrite or not os.path.isfile(outputPath):
        fred = analyser.loadReducedOplist(filename, boolJWorBK, cutoff)
        fredOrdered = orderingFn(fred)
        analyser.writeOplistToFile(fredOrdered, outputPath, 0)
    return fredOrdered


def dividedOplist(filename, boolJWorBK, dual=0, cutoff=1e-12, ordering='magnitude', verbose=False):
    initialOplist = analyser.loadOplist(filename, boolJWorBK, cutoff, ordering)
    output = directOrdering.greedyColourOplist(initialOplist, dual)
    return output


def bulkDividedOplist(listFilenames, boolJWorBK, dual=0, cutoff=1e-12, ordering='magnitude', verbose=True):
    listOplists = []
    thing = []
    for filename in listFilenames:
        fred = analyser.loadOplist(filename, boolJWorBK, cutoff, ordering)
        numQubits = len(fred[0][1])
        thing.append([filename, numQubits])
    newThing = sorted(thing, key=lambda x: x[1])
    for index, filenameThing in enumerate(newThing):
        filename = filenameThing[0]
        startTime = datetime.datetime.now()
        thisSortedOplist = dividedOplist(filename, boolJWorBK, dual, cutoff, ordering)
        listOplists.append(thisSortedOplist)
        endTime = datetime.datetime.now()
        timeTaken = endTime - startTime
        if verbose:
            print('{} ({} of {}) done.  Time taken: {}'.format(filename, index + 1, len(newThing), timeTaken))

    return listOplists


def bulkDepleteGroups(listFilenames, boolJWorBK, cutoff=1e-12, verbose=True):
    groups = readInStoredGroups(boolJWorBK, cutoff)
    mappings = ['JW', 'BK']
    done = 0
    for molecule in groups:
        startTime = datetime.datetime.now()
        thisGroups = groups[molecule]
        thisSortedOplist = directOrdering.depleteGroups(thisGroups)
        with open('/home/andrew/workspace/BKDataNEW/hamiltonian/oplist/1e-12/depleteGroups/{}/{}.ham'.format(
                mappings[boolJWorBK], molecule), 'wb') as f:
            cPickle.dump(thisSortedOplist, f)
        endTime = datetime.datetime.now()
        timeTaken = endTime - startTime
        done += 1
        if verbose:
            print('{} ({} of {}) done.  Time taken: {}'.format(molecule, done, len(groups), timeTaken))
    return


def bulkEqualiseGroups(listFilenames, boolJWorBK, cutoff=1e-12, verbose=True):
    groups = readInStoredGroups(boolJWorBK, cutoff)
    mappings = ['JW', 'BK']
    done = 0
    for molecule in groups:
        startTime = datetime.datetime.now()
        thisGroups = groups[molecule]
        thisSortedOplist = directOrdering.equaliseGroups(thisGroups)
        with open('/home/andrew/workspace/BKDataNEW/hamiltonian/oplist/1e-12/equaliseGroups/{}/{}.ham'.format(
                mappings[boolJWorBK], molecule), 'wb') as f:
            cPickle.dump(thisSortedOplist, f)
        endTime = datetime.datetime.now()
        timeTaken = endTime - startTime
        done += 1
        if verbose:
            print('{} ({} of {}) done.  Time taken: {}'.format(molecule, done, len(groups), timeTaken))
    return


def readInStoredGroups(boolJWorBK, cutoff, ordering='magnitude'):
    JWDAT = '/home/andrew/scr/dual_orderings/JW.dat'
    BKDAT = '/home/andrew/scr/dual_orderings/BK.dat'
    LISTMOLS = analyser.getMoleculesInDirectory(
        '/home/andrew/workspace/BKDataNEW/hamiltonian/cOplist/1e-12/magnitude/BK')
    if boolJWorBK:
        datFile = BKDAT
    else:
        datFile = JWDAT
    thing = []
    for mol in LISTMOLS:
        fred = analyser.loadOplist(mol, boolJWorBK, cutoff, ordering)
        numQubits = len(fred[0][1])
        thing.append([mol, numQubits])
    newThing = sorted(thing, key=lambda x: x[1])

    data = {}
    with open(datFile, 'rb') as f:
        results = cPickle.load(f)

    for i in range(len(newThing)):
        data[newThing[i][0]] = results[i]
        test1 = []
        for x in data[newThing[i][0]]:
            test1 = test1 + x
        test2 = analyser.loadOplist(newThing[i][0], boolJWorBK, cutoff, ordering)
        if len(test1) != len(test2):
            print('failed')

    return data
