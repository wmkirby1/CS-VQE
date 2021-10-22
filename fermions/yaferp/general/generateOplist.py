'''
Created on 17 Oct 2014

@author: andrew
'''

def addHermitianConjugates(dictTerms):
    '''take a dictionary of terms w/ keys designating indices, add in terms corresponding to H conjugates of terms'''
    import numpy
    newDictTerms = {}
    for indices in dictTerms:
        coefficient = dictTerms[indices]
        newCoefficient = numpy.conj(coefficient)
        newDictTerms[indices] = coefficient
        conjugateIndices = indices[::-1] #reverse terms
        conjugateIndices = tuple([index*-1 for index in conjugateIndices])
        if not conjugateIndices in newDictTerms:
            newDictTerms[conjugateIndices] = newCoefficient
        
    return newDictTerms
        


def indicesToOplist(indices, numOrbitals, boolJordanOrBravyi, coefficient=1.):
    '''take a list of indices for fermionic creation/annihilation ops as an array, return an oplist representing term'''
    from yaferp.general.fermions import twoprod, fourprod
    
    boolsCreationOrAnnihilation = []
    absoluteIndices = []
    for index in indices:
        boolsCreationOrAnnihilation.append(int(index > 0))
        absoluteIndices.append(abs(index) - 1)

    if len(indices) == 2:
        oplist = twoprod(absoluteIndices[0],absoluteIndices[1],numOrbitals,boolsCreationOrAnnihilation[0],boolsCreationOrAnnihilation[1],boolJordanOrBravyi,coefficient)
        
    elif len(indices) == 4:
        oplist = fourprod(absoluteIndices[0],absoluteIndices[1],absoluteIndices[2],absoluteIndices[3],numOrbitals,boolsCreationOrAnnihilation[0],boolsCreationOrAnnihilation[1],boolsCreationOrAnnihilation[2],boolsCreationOrAnnihilation[3],boolJordanOrBravyi,coefficient)
    
    return oplist 

def fermionicHamiltonianToQubit(dictTerms,numOrbitals,boolJordanOrBravyi):
    '''take a dictionary w/ terms as keys, coefficients as values.  note terms are zero indexed.'''
    from yaferp.general.fermions import simplify
    oplist = []
    for indices in dictTerms:
        indicesList = list(indices)
        coefficient = dictTerms[indices]
        newQubitTerm = indicesToOplist(indicesList,numOrbitals,boolJordanOrBravyi,coefficient)
        oplist = oplist + newQubitTerm
    simplify(oplist)
    return oplist


def indicesToHermitianSums(indices,numOrbitals,boolJordanOrBravyi):
    from yaferp.general.fermions import oplist_sum
    from yaferp.general.fermions import coefficient
    conjugateIndices = indices[::-1] #reverse indices
    conjugateIndices[:] = [-1*x for x in conjugateIndices]
    
    firstOplistList = []
    firstOplistList.append(indicesToOplist(indices,numOrbitals,boolJordanOrBravyi)) #first the original term
    firstOplistList.append(indicesToOplist(conjugateIndices,numOrbitals,boolJordanOrBravyi)) #now the conjugate
    firstOplist = oplist_sum(firstOplistList)
    
    secondOplistList = []
    secondOplistList.append(indicesToOplist(indices,numOrbitals,boolJordanOrBravyi))
    secondOplistList.append(indicesToOplist(conjugateIndices,numOrbitals,boolJordanOrBravyi))
    secondOplistList[1] = coefficient(-1,secondOplistList[1])
    secondOplist = oplist_sum(secondOplistList)
    secondOplist = list(coefficient(1.j, secondOplist))
    return [firstOplist,secondOplist]
    
def listIndicesToOplists(listIndices,numOrbitals,boolJordanOrBravyi):
    listOplists = []
    for indices in listIndices:
        if len(indices) == 2:
            isHermitian = int(indices[0] + indices[1] == 0 )
        if len(indices) == 4:
            isHermitian = int(((indices[0]+indices[3]) == 0) and ((indices[1]+indices[2])==0))
            
        if isHermitian:
            oplist = indicesToOplist(indices,numOrbitals,boolJordanOrBravyi)
        else:
            oplist = indicesToHermitianSums(indices,numOrbitals,boolJordanOrBravyi)
        
        listOplists.append(oplist) 
    return listOplists

def printListIndicesToOplists(listIndices,numOrbitals,boolJordanOrBravyi,filepath='/home/andrew/LiH.dat'):
    fileStream = open(filepath,'w')
    
    for indices in listIndices:
        if len(indices) == 2:
            isHermitian = int(indices[0] + indices[1] == 0 )
        if len(indices) == 4:
            isHermitian = int(((indices[0]+indices[3]) == 0) and ((indices[1]+indices[2])==0))
            
        if isHermitian:
            oplist = indicesToOplist(indices,numOrbitals,boolJordanOrBravyi)
            lineToWrite = str(indices) + '\t' + str(oplist) + '\n'
            fileStream.write(lineToWrite)
        else:
            conjugateIndices = indices[::-1] #reverse indices
            conjugateIndices[:] = [-1*x for x in conjugateIndices]
            oplists = indicesToHermitianSums(indices,numOrbitals,boolJordanOrBravyi)
            lineToWrite = str(indices) + ' + ' + str(conjugateIndices) + '\t' + str(oplists[0]) + '\n'
            fileStream.write(lineToWrite)
            lineToWrite = 'i(' + str(indices) + ' - ' + str(conjugateIndices) + ')' + '\t' + str(oplists[1]) + '\n'
            fileStream.write(lineToWrite)
            
            
    fileStream.close()
    return
            
    
