'''
readintegrals
Being a module containing functions enabling the reading in of fermionic integrals.
Note that at some point this should be reviewed - I doubt passing around 4D arrays by value is sustainable...
Also NB most of these functions are woefully ungeneral, a better approach would be to read everything in
using # as a control character rather than scanning for each block, but life is short.

Created on 14 Jul 2014
@author: Andrew Tranter (a.tranter13@imperial.ac.uk)
'''

import numpy
import scipy
import cmath
import string
import copy
import decimal

TEST_FILE_PATH = '/home/andrew/workspace/MResProject/PLcode/trunk/HCQLIB/fermions/test.dat'

def readOneElectronIntegrals(filename):
    '''Read in 1 electron integrals from filename.  Returns raw imported data - a list of strings
    corresponding to each line in the section.'''
    dataFile = open(filename,'r')
    '''search the file for the #BEGIN OEI tag'''
    nextLine = dataFile.readline()
    try:
        while nextLine != '#BEGIN OEI\n':
            if nextLine == '':
                raise EOFError
            nextLine = dataFile.readline()    
    except EOFError:
        print('One electron integral section not found - missing #BEGIN OEI tag.')
    '''now read in the data as it stands.'''
    nextLine = dataFile.readline()
    data = []
    while nextLine != '#END OEI\n':
        data.append(nextLine)
        nextLine = dataFile.readline()
    dataFile.close()
    return data


       
def processOneElectronIntegrals(rawData):
    '''Convert raw imported data into a 2D tensor storing the integrals.'''
    lastLine = rawData[-1]  #first, a dirty hack to get the number of orbitals
    lastLineSplit = lastLine.split('\t')
    numOrbitals = int(lastLineSplit[0].strip())+1 #don't forget oribtal 0
 #   numOrbitals = int(lastLine[1])+1 # don't forget orbital 0
    integrals = numpy.empty([numOrbitals,numOrbitals]) #create empty tensor of correct dimensions
    integrals[:] = numpy.NaN
    '''note we make all entries NaN to avoid confusion.
    we will first import data which is in the raw file, remaining NaNs will allow us to
    determine where data was not present, requiring further effort
      if there remain NaNs on return then something has gone wrong
    - using 0 or something wouldn't have allowed us this luxury.'''
    
    '''we now build a new array with each line still corresponding to a line in the data file,
    however with whitespace removed and numbers represented as numbers not strings.
    each line within this new list has 3 entries; two indices and an integral.'''
    processedData = []
    for line in rawData:
        newLine = line.split('\t')
        newLine = [x.strip() for x in newLine]
        newLine[0] = int(newLine[0])
        newLine[1] = int(newLine[1])
        newLine[2] = float(newLine[2])
        processedData.append(newLine)
        
    '''finally we loop through all the data in the file putting the elements provided into 
    their respective place in integrals'''
    
    for line in processedData:
        i = line[0]
        j = line[1]
        thisIntegral = line[2]
        integrals[i][j]=thisIntegral
    return integrals

def addSpinOneElectronIntegrals(spatialIntegrals):
    '''take an n*n array of spatial integrals, return 2n*2n array of spatial integrals in spin-orbit basis'''
    numSpatialIntegrals = len(spatialIntegrals)
    spinIntegrals = numpy.empty([numSpatialIntegrals*2,numSpatialIntegrals*2])
    for i in range(numSpatialIntegrals*2):
        for j in range(numSpatialIntegrals*2):
            spinIntegrals[i][j] = spatialIntegrals[(i-i%2)/2][(j-j%2)/2] * (i%2 == j%2)
    return spinIntegrals        
    
def OBSOLETEaddSpinOneElectronIntegrals(spatialIntegrals):
    '''take a n*n array of spatial integrals, return 2n*2n array of spin-orbit integrals
    even indices:  up spin, odd indices:  down spin.'''
    numSpatialIntegrals = len(spatialIntegrals)
    spinIntegrals = numpy.empty([numSpatialIntegrals*2,numSpatialIntegrals*2])
    for i in range(len(spinIntegrals)):
        for j in range(len(spinIntegrals)):
            spatialI = (i - i%2)/2
            spatialJ = (j - j%2)/2
            spinIntegrals[i][j] = (spatialIntegrals[spatialI,spatialJ] * (i%2 == j%2))
    return spinIntegrals
    
def readTwoElectronIntegrals(filename):
    '''Read in 1 electron integrals from filename.  Returns raw imported data - a list of strings
    corresponding to each line in the section.'''
    dataFile = open(filename,'r')
    '''search the file for the #BEGIN TEI tag'''
    nextLine = dataFile.readline()
    try:
        while nextLine != '#BEGIN TEI\n':
            if nextLine == '':
                raise EOFError
            nextLine = dataFile.readline()    
    except EOFError:
        print('One electron integral section not found - missing #BEGIN TEI tag.')
    '''now read in the data as it stands.'''
    nextLine = dataFile.readline()
    data = []
    while nextLine != '#END TEI\n':
        data.append(nextLine)
        nextLine = dataFile.readline()
    dataFile.close()
    return data
   
def processTwoElectronIntegrals(rawData):
    '''Convert raw imported data into a 2D tensor storing the integrals.'''
    lastLine = rawData[-1]  #first, a dirty hack to get the number of orbitals
    lastLineSplit = lastLine.split('\t')
    numOrbitals = int(lastLineSplit[0].strip())+1 #don't forget oribtal 0
 #   numOrbitals = int(lastLine[1])+1 # don't forget orbital 0
    integrals = numpy.empty([numOrbitals,numOrbitals,numOrbitals,numOrbitals]) #create empty tensor of correct dimensions
    integrals[:] = numpy.NaN
    '''note we make all entries NaN to avoid confusion.
    we will first import data which is in the raw file, remaining NaNs will allow us to
    determine where data was not present, requiring further effort
      if there remain NaNs on return then something has gone wrong
    - using 0 or something wouldn't have allowed us this luxury.'''
    
    '''we now build a new array with each line still corresponding to a line in the data file,
    however with whitespace removed and numbers represented as numbers not strings.
    each line within this new list has 3 entries; two indices and an integral.'''
    processedData = []
    for line in rawData:
        newLine = line.split('\t')
        newLine = [x.strip() for x in newLine]
        newLine[0] = int(newLine[0])
        newLine[1] = int(newLine[1])
        newLine[2] = int(newLine[2])
        newLine[3] = int(newLine[3])
        #newLine[4] = float(newLine[4])
        newLine[4] = decimal.Decimal(newLine[4])
        processedData.append(newLine)
        
    '''finally we loop through all the data in the file putting the elements provided into 
    their respective place in integrals'''
    
    for line in processedData:
        i = line[0]
        j = line[1]
        k = line[2]
        l = line[3]
        thisIntegral = line[4]
        integrals[i][j][k][l]=thisIntegral
    
    return integrals
    
def fixOEIPermutationalSymmetry(rawIntegrals):
    for i in reversed(range(len(rawIntegrals))):
        for j in reversed(range(len(rawIntegrals))):
            if numpy.isnan(rawIntegrals[i][j]):
                if not numpy.isnan(rawIntegrals[j][i]):
                    rawIntegrals[i][j]=rawIntegrals[j][i]
                else:
                    rawIntegrals[i][j] = 0.
                    
    return rawIntegrals
                    
def fixTEIPermutationalSymmetry(rawIntegralsOrig,inPhysicist=False):
    '''account for permutational symmetry in 2-electron spatial integrals tensor.
    this is possibly the worst function ever written.  i apologise sincerely if you have to use it.'''
    steve = 'nope'
    rawIntegrals = copy.deepcopy(rawIntegralsOrig)
    if not inPhysicist:
        for i in reversed(range(len(rawIntegrals))):
            for j in reversed(range(len(rawIntegrals[i]))):
                for k in reversed(range(len(rawIntegrals[i][j]))):
                    for l in reversed(range(len(rawIntegrals[i][j][k]))):
                        if numpy.isnan(rawIntegrals[i][j][k][l]):
                            if not numpy.isnan(rawIntegrals[i][j][l][k]):
                                rawIntegrals[i][j][k][l] = rawIntegrals[i][j][l][k]
                            elif not numpy.isnan(rawIntegrals[j][i][l][k]):
                                rawIntegrals[i][j][k][l] = rawIntegrals[j][i][l][k]
                            elif not numpy.isnan(rawIntegrals[k][l][i][j]):
                                rawIntegrals[i][j][k][l] = rawIntegrals[k][l][i][j]
                            elif not numpy.isnan(rawIntegrals[l][k][j][i]):
                                rawIntegrals[i][j][k][l] = rawIntegrals[l][k][j][i]
                            elif not numpy.isnan(rawIntegrals[j][i][k][l]):
                                rawIntegrals[i][j][k][l] = rawIntegrals[j][i][k][l]
                            elif not numpy.isnan(rawIntegrals[l][k][i][j]):
                                rawIntegrals[i][j][k][l] = rawIntegrals[l][k][i][j]
                            elif not numpy.isnan(rawIntegrals[k][l][j][i]):
                                rawIntegrals[i][j][k][l] = rawIntegrals[k][l][j][i]
                            elif (i >= j) and (k >= l) and((( (i * (i+1))/2) +j) >= (( (k * (j+1))/2) + l)):  #even i don't know at this point.  (no, but seriously, HOPEFULLY this line essentially says 'if this is a unique permutation and still isn't present, the integral is below the notability threshold so set to 0.')
                                rawIntegrals[i][j][k][l] = 0.
                                
    elif inPhysicist:
        for i in reversed(range(len(rawIntegrals))):
            for j in reversed(range(len(rawIntegrals[i]))):
                for k in reversed(range(len(rawIntegrals[i][j]))):
                    for l in reversed(range(len(rawIntegrals[i][j][k]))):
                        if numpy.isnan(rawIntegrals[i][j][k][l]):
                            if not numpy.isnan(rawIntegrals[j][i][l][k]):
                                rawIntegrals[i][j][k][l] = rawIntegrals[j][i][l][k]
                            elif not numpy.isnan(rawIntegrals[k][l][i][j]):
                                rawIntegrals[i][j][k][l] = rawIntegrals[k][l][i][j]
                            elif not numpy.isnan(rawIntegrals[l][k][j][i]):
                                rawIntegrals[i][j][k][l] = rawIntegrals[l][k][j][i]
                            elif not numpy.isnan(rawIntegrals[k][j][i][l]):
                                rawIntegrals[i][j][k][l] = rawIntegrals[k][j][i][l]
                            elif not numpy.isnan(rawIntegrals[l][i][j][k]):
                                rawIntegrals[i][j][k][l] = rawIntegrals[l][i][j][k]
                            elif not numpy.isnan(rawIntegrals[i][l][k][j]):
                                rawIntegrals[i][j][k][l] = rawIntegrals[i][l][k][j]
                            elif not numpy.isnan(rawIntegrals[j][k][l][i]):
                                rawIntegrals[i][j][k][l] = rawIntegrals[j][k][l][i]
                            #elif (i >= j) and (k >= l) and((( (i * (i+1))/2) +j) >= (( (k * (j+1))/2) + l)):  #even i don't know at this point.  (no, but seriously, HOPEFULLY this line essentially says 'if this is a unique permutation and still isn't present, the integral is below the notability threshold so set to 0.')
                            else: 
                                rawIntegrals[i][j][k][l] = 0.    
    return rawIntegrals

def convertChemistToPhysicist(chemistIntegrals):
    '''convert from chemists' notation two electron tensor to physicists' notation.
    in theory.
    in practice i have no idea whether this works or why it works.
    i *do* know that i'm 99% certain it won't work with complex orbitals, though.'''
    numOrbitals = len(chemistIntegrals)
    physicistIntegrals = numpy.empty([numOrbitals,numOrbitals,numOrbitals,numOrbitals])
    for i in range(numOrbitals):
        for j in range(numOrbitals):
            for k in range(numOrbitals):
                for l in range(numOrbitals):
                    physicistIntegrals[i][j][k][l]=chemistIntegrals[i][k][l][j]
    return physicistIntegrals
    
def addSpinTwoElectronIntegrals(spatialIntegrals,boolInPhysicists=False):
    '''take n^4 array of spatial integrals, return (2n)^4 array of spatial integrals
    in a spin-orbit basis'''
    numSpatialIntegrals = len(spatialIntegrals)
    spinIntegrals = numpy.empty([numSpatialIntegrals*2,numSpatialIntegrals*2,numSpatialIntegrals*2,numSpatialIntegrals*2])
    for i in range(numSpatialIntegrals*2):
        for j in range(numSpatialIntegrals*2):
            for k in range(numSpatialIntegrals*2):
                for l in range(numSpatialIntegrals*2):
                    if boolInPhysicists:
                        spinIntegrals[i][j][k][l] = spatialIntegrals[(i-i%2)/2][(j-j%2)/2][(k-k%2)/2][(l-l%2)/2] * (j%2 == k%2) * (i%2 == l%2)
                    else:
                        spinIntegrals[i][j][k][l] = spatialIntegrals[(i-i%2)/2][(j-j%2)/2][(k-k%2)/2][(l-l%2)/2] * (j%2 == i%2) * (k%2 == l%2)
    return spinIntegrals
    


def OBSOLETEaddSpinTwoElectronIntegrals(spatialIntegrals):
    '''take a n^4 array of spatial integrals, return (2n)^4 array of spin-orbit integrals
    even indices:  up spin, odd indices:  down spin.'''
    numSpatialIntegrals = len(spatialIntegrals)
    spinIntegrals = numpy.empty([numSpatialIntegrals*2,numSpatialIntegrals*2,numSpatialIntegrals*2,numSpatialIntegrals*2])
    for i in numSpatialIntegrals*2:
        for j in numSpatialIntegrals*2:
            for k in numSpatialIntegrals*2:
                for l in numSpatialIntegrals*2:
                    spatialI = (i - i%2)/2
                    spatialJ = (j - j%2)/2
                    spatialK = (k - k%2)/2
                    spatialL = (l - l%2)/2
                    '''not entirely certain how this works, it's from crawdad'''
                    value1 = spatialIntegrals[spatialI][spatialK][spatialJ][spatialL] * (i%2 == k%2) * (j%2 == l%2)
                    value2 = spatialIntegrals[spatialI][spatialL][spatialJ][spatialK] * (i%2 == l%2) * (j%2 == k%2)
                    spinIntegrals[i][j][k][l] = value1 + value2
    return spinIntegrals

def importOneElectronIntegrals(filename=TEST_FILE_PATH,noSpin=False):
        rawData = readOneElectronIntegrals(filename)
        processedData = processOneElectronIntegrals(rawData)
        processedData = fixOEIPermutationalSymmetry(processedData)
        if not noSpin:
            processedData = addSpinOneElectronIntegrals(processedData)
        return processedData

def realPhysicistToCurrent(ints):
    newInts = copy.deepcopy(ints)
    thing = range(len(ints))
    for i in thing:
        for j in thing:
            for k in thing:
                for l in thing:
                    newInts[i][j][k][l] = ints[i][j][l][k]
    return newInts
                    
        
    
    
def importTwoElectronIntegrals(filename=TEST_FILE_PATH,noSpin=False,inPhysicist=False,inRealPhysicist=False):
    '''import two electron spatial orbitals from filename, return a 4d tensor array'''
    rawData = readTwoElectronIntegrals(filename)
    processedData = processTwoElectronIntegrals(rawData)
    processedData = fixTEIPermutationalSymmetry(processedData,inPhysicist)
    if inRealPhysicist:
        processedData = realPhysicistToCurrent(processedData)
    if not noSpin:
        processedData = addSpinTwoElectronIntegrals(processedData,inPhysicist)
    if not inPhysicist:
        processedData = convertChemistToPhysicist(processedData)
    return processedData
    
def importIntegrals(filename=TEST_FILE_PATH,noSpin=False,inPhysicist=False):
    '''read in integrals from data file.  return list, entry 0 is 1E integrals, entry 1 is 2E ints'''
    oneEIntegrals = importOneElectronIntegrals(filename,noSpin)
    twoEIntegrals = importTwoElectronIntegrals(filename,noSpin,inPhysicist)
    return [oneEIntegrals,twoEIntegrals]

def importJMNew(filename='/home/andrew/workspace/MResProject/methane.ARGH'):    
    oneEIntegrals = importOneElectronIntegrals(filename,True)
    twoEIntegrals = importTwoElectronIntegrals(filename,False,True,True)
    return [oneEIntegrals,twoEIntegrals]
def readRBInts(filePath):
    dataFile = open(filePath,'r')
    nextLine = dataFile.readline()
    rawData = []
    while nextLine != '':
        rawData.append(nextLine)
        nextLine = dataFile.readline()
    return rawData
    
def splitRBInts(rawData):
    rawOEI = []
    rawTEI = []
    for line in rawData:
        if len(string.split(line)) == 3 and string.split(line)[0].isdigit():
            rawOEI.append(line)
        elif len(string.split(line)) == 5 and string.split(line)[0].isdigit():
            rawTEI.append(line)
    return (rawOEI,rawTEI)

def rawListToDict(rawData,flipKet=False):
    dict = {}
    for line in rawData:
        splitLine = string.split(line)
        coefficient = float(splitLine[-1])
        indicesStrings = splitLine[0:(len((splitLine))-1)]
        indices = map(int,indicesStrings)
        if len(indices) == 4 and flipKet:
            newIndices = [indices[0],indices[1],indices[3],indices[2]]
            indices = newIndices
        dict[tuple(indices)] = coefficient
    return dict

def findNumOrbitals(dictIntegrals):
    keys = list(dictIntegrals.keys())
    highestIndex = 0
    for key in keys:
        for index in key:
            if index > highestIndex:
                highestIndex = index
    return highestIndex

def formNANTensor(dimensions):
    tensor = numpy.empty(list(dimensions))
    tensor[:] = numpy.NaN
    return tensor

def negateTermsInDict(dictTerms):
    for key in dictTerms:
        dictTerms[key] = -1 * dictTerms[key]
    return dictTerms

def fillTensor(tensor,dictTerms):
    for indices in dictTerms:
        tensor[indices] = dictTerms[indices]
    return tensor

def importRBIntegrals_OBSOLETE(filePath, negateTEI = True):
    
    rawData = readRBInts(filePath)
    rawOEIs, rawTEIs = splitRBInts(rawData)
    dictOEI = rawListToDict(rawOEIs)
    dictTEI = rawListToDict(rawTEIs)
    if negateTEI:
        dictTEI = negateTermsInDict(dictTEI)
        
    numOrbitals = max(findNumOrbitals(dictOEI)+1,findNumOrbitals(dictTEI)+1)
    tensorOEI = formNANTensor([numOrbitals,numOrbitals])
    tensorTEI = formNANTensor([numOrbitals,numOrbitals,numOrbitals,numOrbitals])
    
    tensorOEI = fillTensor(tensorOEI,dictOEI)
    tensorTEI = fillTensor(tensorTEI,dictTEI)
    finalTensorOEI = fixOEIPermutationalSymmetry(tensorOEI)
    finalTensorTEI = fixTEIPermutationalSymmetry(tensorTEI,True)
    return(finalTensorOEI,finalTensorTEI)
        
def importRBEnergies(filePath):
    import decimal
    rawData = readRBInts(filePath)
    (nuclearEnergy,totalEnergy) = findRBEnergies(rawData)
    nuclearEnergyDecimal = decimal.Decimal(str(nuclearEnergy))
    totalEnergyDecimal = decimal.Decimal(str(totalEnergy))
    electronicEnergyDecimal = totalEnergyDecimal - nuclearEnergyDecimal
    return electronicEnergyDecimal
    
    
def findRBEnergies(rawData):
    for (index,line) in enumerate(rawData):
        if line == "NUCLEAR REPULSION ENERGY\n":
            nuclearEnergy = rawData[index+1]
        elif line == "FCI TOTAL ENERGY\n":
            totalEnergy = rawData[index+1]
    return(nuclearEnergy,totalEnergy)

def importRBIntegrals(filePath):
    rawData = readRBInts(filePath)
    rawOEIs, rawTEIs = splitRBInts(rawData)
    
    dictOEI = rawListToDict(rawOEIs)
    dictTEI = rawListToDict(rawTEIs,flipKet=True)
    numOrbitals = max(findNumOrbitals(dictOEI)+1,findNumOrbitals(dictTEI)+1)
    
    tensorOEI = numpy.zeros([numOrbitals,numOrbitals])
    tensorTEI = numpy.zeros([numOrbitals,numOrbitals,numOrbitals,numOrbitals])
    tensorOEI = fillTensor(tensorOEI,dictOEI)
    tensorTEI = fillTensor(tensorTEI,dictTEI)
    return(tensorOEI,tensorTEI)
