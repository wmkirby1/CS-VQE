'''
Created on 28 Oct 2014

@author: andrew
'''

def codeToInt(code):
    '''takes a string of form '[integer]'+'[a, b or x]'.
    integer corresponds to which spatial orbital is being referenced.
    a : alpha spin, b: beta spin, x: both occupied.
    returns an integer corresponding to the index of 
    the Fock state where only the relevant orbital(s) are occupied'''
    
    whichSpin = code[-1]
    whichSpatial = int(code[:-2])-1
    if whichSpin == 'A' or whichSpin == 'a':
        index = 1 << 2*whichSpatial
    elif whichSpin == 'B' or whichSpin == 'b':
        index = 1 << (2*whichSpatial + 1)
    elif whichSpin == 'X' or whichSpin == 'x':
        index1 = 1 << (2*whichSpatial)
        index2 = 1 << (2*whichSpatial + 1)
        index = index1 | index2
    
    return index

def listCodesToInt(listCodes):
    '''take a list of codes i.e. '1A 2X 3B', return the integer corresponding
    to the index of the relevant Fock state'''
    import numpy    
    listIndices = [codeToInt(code) for code in listCodes]
    overallIndex = numpy.bitwise_or.reduce(listIndices)
    return overallIndex

def readState(numOrbitals, filePath='/home/andrew/scratch/testListDets.csv'):
    '''read in a state, store in vector.  nb numOrbitals is spin orbitals'''
    import csv
    import numpy
    import scipy
    import scipy.sparse
    with open(filePath,'rb') as fileStream:
        rawFile = csv.reader(fileStream)
        rawData = []
        for row in rawFile:
            rawData.append(row)
    
    state = numpy.zeros((1, 2**numOrbitals))
    for determinant in rawData:
       # test1 = numpy.count_nonzero(state)
        coefficient = float(determinant[0])
        listBasisStates = determinant[1:]
        listBasisStates = [state2 for state2 in listBasisStates if state2] #remove empty strings
        index = listCodesToInt(listBasisStates)
        if not state[0,index]:
            state[0,index] = float(coefficient)
        else:
            state[0,index] = state[0,index]+coefficient
       # state[0,index] = float(coefficient)
       # test2 = numpy.count_nonzero(state)
    normState = numpy.linalg.norm(state)
    normalisedState = state/normState
    normalisedState = scipy.sparse.coo_matrix(normalisedState).T
    return normalisedState
