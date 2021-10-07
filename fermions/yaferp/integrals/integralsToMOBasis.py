'''
Created on 11 Nov 2014

@author: andrew
'''
def transformOEIs(numOrbitals,oeiTensor,moList):
    '''nb CHECK moList is a list of lists, SECOND index refers to MO index'''
    import numpy
    oeiMO = numpy.empty((numOrbitals,numOrbitals))
    oeiMO[:] = numpy.NAN
    
    for i in range(numOrbitals):
        for j in range(numOrbitals):
            thisIntegral = 0
            for m in range(numOrbitals):
                for n in range(numOrbitals):
                    thisIntegral = thisIntegral + moList[i][m] * moList[j][n] * oeiTensor[m][n]          
            oeiMO[i][j] = thisIntegral
            
    return oeiMO