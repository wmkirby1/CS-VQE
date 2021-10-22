''' sparseFermions - Sparse matrix methods for simulation of fermionic quantum simulation
Created on 5 Aug 2014

@author: Andrew Tranter (a.tranter13@imperial.ac.uk)
'''
import math
import scipy
import numpy
import scipy.sparse
import scipy.sparse.linalg
from datetime import datetime
import copy
from fermions.yaferp.general import sparseLinalg, fermions, oneXOnState, oneZOnState


def kronProductListSparse(listMatrices):
    product = 1
    for thisMatrix in listMatrices:
        product = scipy.sparse.kron(product,thisMatrix, 'coo')
    result = scipy.sparse.csc_matrix(product)
    return result
        

_PAULIS_AS_SPARSE_ = [scipy.sparse.coo_matrix(numpy.array([[1.0,0.],     [0.,1.0]])),
                      scipy.sparse.coo_matrix(numpy.array([[0.0,1.0],       [1.0,0.0]])),
                      scipy.sparse.coo_matrix(numpy.array([[0.,0.-1.j],   [0.+1.j,0.]])),
                      scipy.sparse.coo_matrix(numpy.array([[1.,0.],       [0.,-1.]]))]


def symPauliToSparseMatrix(whichPauli):


    '''if whichPauli == 1:
        matrix = numpy.array([[0.0,1.0]
                              ,[1.0,0.0]])
    elif whichPauli == 2:
        matrix = numpy.array([[0.,0.-1.j],[0.+1.j,0.]]) 
    elif whichPauli == 3:
        matrix = numpy.array([[1.,0.],[0.,-1.]])
    else:
        matrix = numpy.array([[1.,0.],[0.,1.]])
    sparseMatrix = scipy.sparse.csc_matrix(matrix)
    '''
    return (_PAULIS_AS_SPARSE_[whichPauli])

def pauliStringToListSparseMatrices(pauliString):
    '''take a list indicating a string of pauli matrices, return those matrices as a list.'''
    listMatrices = []
    for thisPauli in pauliString:
        thisMatrix = symPauliToSparseMatrix(thisPauli)
        listMatrices.append(thisMatrix)
    return listMatrices

def exponentiatePauliString(inputString):
    '''take a pauli string in form [coefficient, [string]].
    exponentiate and return a one or two element list of the same form
    which when summed are the exponential of the input string.
    NOTE A FACTOR OF i IN THE EXPONENT IS IMPLIED YES THIS IS INDEED GETTING INCREDIBLY ROUGH
    TODO: CHECK THE EQUATION!!!!!!!! '''

    coefficient = inputString[0]
    pauliString = inputString[1]
    numQubits = len(pauliString)
    sinCoefficient = numpy.sin(coefficient)
    cosCoefficient = numpy.cos(coefficient)
    firstTerm = [sinCoefficient * 1.0j, pauliString]
    secondTerm = [cosCoefficient, [0]*numQubits]
    output = [firstTerm,secondTerm]
    output = fermions.simplify(output)
    return output


def exponentiateCommutingOplist(oplist, outputFile=''):
    '''take an oplist, exponentiate each term.  multiply exponentials together.
    returns answer as a matrix.'''
    if isinstance(oplist[0],list): #if oplist has > 1 element
        overallMatrix = commutingOplistToMatrix(exponentiatePauliString(oplist[0]))
        if len(oplist) > 1:
            for i in range(1,len(oplist)):
                newMatrix = commutingOplistToMatrix(exponentiatePauliString(oplist[i]))
                overallMatrix = overallMatrix * newMatrix
    else:
        overallMatrix = commutingOplistToMatrix(exponentiatePauliString(oplist))
    return overallMatrix


def commutingOplistUnitary(oplist,t):
    '''find the unitary evolution operator for a sum of commuting hamiltonians at time t
    (u = e^(-iHt) )'''
    oplistCopy = list(oplist)
    oplistNewCoefficients = []
    if isinstance(oplistCopy[0],list): #if oplist has > 1 element
        for term in oplistCopy:
            coefficient = term[0] * -1. * t
            oplistNewCoefficients.append([coefficient,term[1]])
        return exponentiateCommutingOplist(oplistNewCoefficients)
    else:
        coefficient = oplistCopy[0] * -1. * t
        newOplist = [coefficient,oplist[1]]
        return exponentiateCommutingOplist(newOplist)

def subsetsToPropagators(oplist,t):
    '''find the unitary evolution operator for a sum of commuting hamiltonians at time t
    (u = e^(-iHt) ).  matrix-free version'''
    oplistCopy = list(oplist)
    oplistNewCoefficients = []
    listOplists = []
    for term in oplistCopy:
        coefficient = term[0] * -1. * t
        oplistNewCoefficients.append([coefficient,term[1]])
        
    return exponentiateCommutingOplist(oplistNewCoefficients)


def directPropagatorFromOplist(oplist,t):
    '''find the propagator for an oplist corresponding to a sum of commuting hamiltonians at time t
    returns a list of oplists, each must be effected serially (each oplist is 2 elements from exponentiation)'''
    oplistCopy = list(oplist)
    if not isinstance(oplistCopy[0],list):
        coefficient = oplistCopy[0]* -1. * t
        oplistCopy[0] = coefficient
        return (exponentiatePauliString(oplistCopy))
    else:
        listOplists = []
        for term in oplistCopy:
            coefficient = term[0] * -1. * t
            newTerm = [coefficient, term[1]]
            listOplists.append(exponentiatePauliString(newTerm))
            return listOplists


#from memory_profiler import profile
#@profile(precision=4)
#import objgraph
#@profile(precision=4)
def commutingOplistToMatrix(oplist, outputFile = '/run/media/andrew/b2305290-df58-489c-a39a-d25f8b52c664/data/data.dat', boolPrintDebugInfo=0):
    '''take in a oplist where all elements commute.  output one H matrix.
    nb note oplist is list of [(coefficient), (pauli string)] entries.  see standard established above
    TODO:  do a better outputFile thing.  (make it store to a temp file which is deleted on output).'''
    import sys
    if isinstance(oplist[0],list):
        numQubits = len(oplist[0][1])
        listMatrixTerms = []
    #if boolPrintDebugInfo:
     #   print('Creating overallMatrix memory map.')
       # sys.stdout.flush()
   # overallMatrix = numpy.memmap(outputFile, dtype='float32', mode='w+', shape=(2**numQubits,2**numQubits))
   # overallMatrix[:] = 0.+0.j
    #if boolPrintDebugInfo:
     #   print('overallMatrix memory map created.  Calculating individual matrix terms.')
      #  sys.stdout.flush()
    
        for index, term in enumerate(oplist):
           # print(str(term))
            #print(objgraph.show_most_common_types())
            if boolPrintDebugInfo == 1:
                print(str(datetime.now()))
                print('Calculating full dimensional treatment of term', index)
                sys.stdout.flush()
            
            coefficient = term[0]
            pauliString = term[1]
            listPauliMatrices = pauliStringToListSparseMatrices(pauliString)

            if boolPrintDebugInfo:
                print(str(datetime.now()))
                print('Pauli matrices created, creating tensor product represntation.')
                sys.stdout.flush()

            tensoredTerm = kronProductListSparse(listPauliMatrices)
            tensoredTerm = coefficient * tensoredTerm
        
            if index == 0:
                overallMatrix = tensoredTerm
            else:
                overallMatrix = overallMatrix + tensoredTerm
    else:
        coefficient = oplist[0]
        pauliString = oplist[1]
        listPauliMatrices = pauliStringToListSparseMatrices(pauliString)
        tensoredTerm = kronProductListSparse(listPauliMatrices)
        tensoredTerm = coefficient * tensoredTerm
        overallMatrix = tensoredTerm
         
    #if boolPrintDebugInfo == 1:
     #   print(str(datetime.now()))
      #  print('Full dimensional calculation of Pauli strings completed.  Summing.')
       # sys.stdout.flush()
    
 #   overallMatrix = listMatrixTerms[0]
  #  for i in range(1,len(listMatrixTerms)):
   #     overallMatrix = overallMatrix + listMatrixTerms[i]
    
    #overallMatrix = numpy.asmatrix(overallMatrix) #convert to numpy matrix type for compatibility
    
    
    
    
    return overallMatrix


'''below lies old trotter code for adaptation.  all code not mine except where noted. -AT
TODO:  clean up, liberally.'''
 
import numpy as np
import scipy.linalg
from cmath import pi
#import matplotlib.pyplot as plt
#All terms for H. from Jake Seeley, to trotterize his work
# Bravyi-Kitaev Terms




def tsFormulaTermOrdering(listHamiltonians,order):
    '''order terms in listHamiltonians such that they can subsequently be exponentiated
    and multiplied to form a trotter-suzuki approximation of the unitary evolution operator corresponding
    to their sum. x is an internal parameter used when recursively doing stuff.  
    NB only supports order 2^(positive integer).  i also haven't tested this for order > 4.'''
    if order == 1:
        return listHamiltonians #TS order 1 involves no reordering
    elif order == 2:
        newListHamiltonians = list(listHamiltonians)
        newListHamiltonians.extend(list(reversed(newListHamiltonians))) #AB -> ABBA
        return newListHamiltonians
    else: #the fun part.  strap in...
        k = order/2
        z = (4. - 4.**(1./(2.*k - 1.)))**(-1.)
        middleTermMultiplier = 1. - 4.*z
        newListHamiltonians = []
        
        hamiltoniansTimesZ = [h*z for h in listHamiltonians]
        hamiltoniansMiddle = [h*middleTermMultiplier for h in listHamiltonians]
        
        outsideTerm = list(tsFormulaTermOrdering(hamiltoniansTimesZ,k))
        middleTerm = list(tsFormulaTermOrdering(hamiltoniansMiddle,k))
        
        newListHamiltonians.extend(outsideTerm)
        newListHamiltonians.extend(outsideTerm)
        newListHamiltonians.extend(middleTerm)
        newListHamiltonians.extend(outsideTerm)
        newListHamiltonians.extend(outsideTerm)
        
        return newListHamiltonians
        
def tsFormulaOrder(numTerms,order,currentList=None):
    '''return a list indicating the ordering of terms in a given trotter-suzuki approximation.
    numTerms:  number of terms to be trotterized
    order:  order of the approximation
    currentList:  internal recursion parameter
    returns list of [index, factor] entries where index is the index of the 
    hamiltonian in that exponential and factor is the factor in the exponential
    NB only supports order 2^(integer).  untested for order >4'''
    if order == 1:
        listIndicesFactors=[]
        for i in range(numTerms):
            newTerm = [i,1]
            listIndicesFactors.append(newTerm)
        return listIndicesFactors
    
    elif order == 2:
        if currentList == None:
            currentList = []
            for i in range(numTerms):
                newTerm = [i,1]
                currentList.append(newTerm)
        copyCurrentList = list(currentList)
        for term in copyCurrentList:
            term[1] = term[1]/2.  #halve the time length for symmetrization    
        copyCurrentList.extend(list(reversed(copyCurrentList))) #ab -> abba
        return copyCurrentList
    
    else:
        k = order/2
        z = (4. - 4.**(1./(2.*k - 1.)))**(-1.)
        middleTermMultiplier = 1. - 4.*z
        indicesCoefficients = []
        
        if currentList == None:
            currentList = []
            for i in range(numTerms):
                newTerm = [i,1]
                currentList.append(newTerm)        
        copyCurrentList = list(currentList)
        listForMiddle = []
        listTimesZ = []
        
        for indexAndCoefficient in copyCurrentList:
            newTermMiddle = [indexAndCoefficient[0],indexAndCoefficient[1]*middleTermMultiplier]
            newTermEdges = [indexAndCoefficient[0],indexAndCoefficient[1]*z]
            listForMiddle.append(newTermMiddle)
            listTimesZ.append(newTermEdges)
        
        outsideTerm = list(tsFormulaOrder(numTerms,k,listTimesZ))
        middleTerm = list(tsFormulaOrder(numTerms,k,listForMiddle))
        
        indicesCoefficients.extend(outsideTerm)
        indicesCoefficients.extend(outsideTerm)
        indicesCoefficients.extend(middleTerm)
        indicesCoefficients.extend(outsideTerm)
        indicesCoefficients.extend(outsideTerm)
        return indicesCoefficients





# t/2*t/2 is the same as t

#Functions that given an input of a list of terms will order them to take product according to Trotter approx
#def ord2_generator(termlist):
 #   """given an input of a list of terms will order them for taking product of
  #  2nd order Trotter approx. ex: For input 'ab' output is 'abba'"""
   # copy=termlist[:]
    #copy.reverse()
    #for x in copy:
     #   termlist.append(x)
    #return termlist

#def ord4_generator(termlist):
#    """given an input of a list of terms will order them for taking product of 4th order Trotter approx"""
 #   p=(1.0/(4.-4.**(1./3.)))
  #  p3=1-(4.0)*p
   # copy=termlist[:]
    #copy.reverse()
    #for x in copy:
     #   termlist.append(x) #form is abba
   # copy1=termlist[:]
   # copyp=[elem*p for elem in copy1]
   # copyp*=2 #abbaabba
   # copyp_end=copyp[:]
   # copyp3=[elem*p3 for elem in copy1]
   # for y in copyp3:
   #     copyp.append(y)
   # for z in copyp_end:
   #     copyp.append(z)
   # return copyp

#def orderTermsForTrotter(termlist, order):
#    '''wrapper function - order the hamiltonian terms for trotterization,
#    using suzuki-trotter approximation of order given in parameter.'''
#    if order == 2:
#        return ord2_generator(termlist)
#    elif order == 4:
#        return ord4_generator(termlist)
#    elif order == 1:
#        return termlist


'''rewrote code to get the overall unitary from a list of Hamiltonian terms.
also note this uses sparse matrices. - AT'''

def hamiltonianToUnitary(hamiltonian,t):
    '''take a hamiltonian as a sparse (theoretically, also dense) matrix,
    return its corresponding unitary evolution operator at time t
    TODO:  this will return dense matrices as an array.  this needs to be fixed.
    it may also throw up problems with formatting sparse matrices which must be looked into.'''
    exponent = -1 * hamiltonian * 1.0j * t
    unitary = scipy.sparse.linalg.expm(exponent)
    return unitary

def listHamiltoniansToUnitaries(listHamiltonians,t):
    '''take a list of hamiltonians as sparse matrices, exponentiate each and return their
    respective unitary evolution operator at time t.
    TODO:  get the type of sparse matrix, shape of matrix, create identity matrix of shape and type,
    '''
    unitariesList = []
    for hamiltonian in listHamiltonians:
        unitary = hamiltonianToUnitary(hamiltonian, t)
        unitariesList.append(unitary)
  #  overallUnitary = unitariesList[0]
 #   for i in range(1,len(unitariesList)):
   #     overallUnitary = overallUnitary*unitariesList[i]
    return unitariesList
 
 
 
 
 
 
#def NewExp(term,t):
 #   """Exponentiating a matrix for time propogator e^(-iHt)"""
  #  return scipy.linalg.matfuncs.expm(term*-1.0j*t)
#propagator is -iHt
#output is an array not matrix

#def list_exp(hamil_terms, t):
 ##   """Exponentiates each matrix term in a list"""
   # termlist = []
    #for n in range(0,len(hamil_terms)):
     #   termlist.append(np.mat(NewExp(hamil_terms[n],t)))
   # return termlist

#def trot_term_mult(termlist):
 #   """Takes a list of matrices and returns their matrix product"""
  #  n = len(termlist)
   # if n == 1:
    #    return termlist[0]
    #else:
     #   output = termlist[0]*termlist[1]
      #  for j in range(2,n):
       #     output = output*termlist[j]
        #return output

#Combining previous steps to get actual trotter approximations
#Puts terms in correct order, exponentiates, adds t/n, multiplies all terms together

def OBSOLETE_trotterise(termList,t,n, order):
    '''calculate the unitary evolution operator corresponding to a sum
    of hamiltonian operators given as list termList, using a 
    suzuki-trotter expansion.  order is the order of the expansion to be used.
    n is the number of time steps.
    TODO:  write the function
    '''
    
    orderedTermList = tsFormulaTermOrdering(termList,order)
    if order != 1:
        propagationTime = float(t)/(2.*float(n)) #factor of 2 for order > 1 due to symmetrization
    else:
        propagationTime = float(t)/float(n)
    listUnitaries = listHamiltoniansToUnitaries(orderedTermList,propagationTime)
    
    overallUnitary = listUnitaries[0]
    if len(listUnitaries) > 1:
        for unitary in listUnitaries[1:]:
            overallUnitary = overallUnitary * unitary
    overallUnitary = overallUnitary ** n
    return overallUnitary






def orderForTrotter(listCommutingOplists,order):
    
    numSets = len(listCommutingOplists)
    indicesMultipliers = tsFormulaOrder(numSets,order)
    oplistsOrdered = []
    for indexMultiplier in indicesMultipliers:
        index = indexMultiplier[0]
        multiplier = indexMultiplier[1]
        newOplist = list(listCommutingOplists[index])
        newNewOplist = []
        for i in range(len(newOplist)):
            newNewOp = list(newOplist[i])
            newNewOp[0] = newNewOp[0] * multiplier
            newNewOplist.append(newNewOp)
        oplistsOrdered.append(newNewOplist)
    return oplistsOrdered
    
def oneXOnState(state,xIndex, useC=1):
    '''apply a pauli x to qubit xIndex of a given register, with the register
    state given as a (sparse) vector.
    RETURNS A MATRIX IN COO SPARSE FORMAT!'''
    if not useC:
        oldState = scipy.sparse.coo_matrix(state) #replace with tocoo if optimisation needed
        newIndices = []
        for rowIndex in oldState.col:
            newRowIndex = rowIndex ^ (1<<xIndex) #flip the xIndexth bit of rowIndex
            newIndices.append(newRowIndex)
        newState = scipy.sparse.coo_matrix((oldState.data,([0]*oldState.nnz,newIndices)),shape=oldState.shape)
        return newState
    
    if useC:
        oldState = scipy.sparse.coo_matrix(state)
        indices = oldState.row
        import oneXOnState
        oldState.row = oneXOnState.oneXOnState(indices,xIndex)  
        return oldState

def oneZOnState(state,zIndex, useC=1):
    '''useC : do we use a separate Cython handler for the main loop?  should be substantially faster but cause architectural chaos.
    non-C algorithm doesn't currently work (needs to be rewritten to be in-place'''
    if not useC:
        oldState = scipy.sparse.coo_matrix(state)
  
   # if not method:
        newData = []
        for datum,index in zip(oldState.data,oldState.col):
            if index & (1<<zIndex):
                newData.append(-1 * datum)
            else:
                newData.append(datum)    
   # if method:
    #    newData = [oldState.data[i] * (1 - ((oldState.col[i] & (1<<zIndex))!=0)*2) for i in range(len(oldState.data))] #wait what    
        newState = scipy.sparse.coo_matrix((newData,(oldState.row,oldState.col)),shape=oldState.shape)
        oldState = newState
    if useC:
        if state.dtype != 'complex128':
            state = state.astype(numpy.complex128)
        oldState = scipy.sparse.coo_matrix(state)
        data = oldState.data
        indices = oldState.row
        import oneZOnState
        oldState.data = oneZOnState.oneZOnState(data,indices,zIndex)
        
        #newData = cythonFunction(double complex data[n], long indices[n], long zIndex)
        
        
        #newState = scipy.sparse.coo_matrix((newData,(oldState.row,indices)),shape=oldState.shape)
        
    return oldState

def oneYOnState(state,yIndex, boolIncludeI = 1):
    '''NB set boolIncludeI to 0 to NOT multiply the state by i.
    calculate the number of i's for a given pauli string separately for greater efficiency'''
    newState = oneZOnState(state,yIndex)
    newState = oneXOnState(newState,yIndex)
    if boolIncludeI:
        newState = newState * 1j
    return newState
    
def applyPauliStringToState(string,state):
    '''take a pauli string in format [coeff, [string]], apply each pauli, return new vector'''
    coefficient = string[0]
    pauliString = string[1]
    newState = state
    numYs = 0 #count the number of y's applied, add a phase at end for efficiency
    for whichQubit,pauli in enumerate(reversed(pauliString)):
        if pauli == 1:
            newState = oneXOnState(newState,whichQubit)
        elif pauli == 2:
            newState = oneYOnState(newState,whichQubit, 0)
            numYs = numYs + 1
        elif pauli == 3:
            newState = oneZOnState(newState,whichQubit)
    if numYs > 0:
        newState = (1j** numYs) * newState
    newState = newState * coefficient    
    return newState

def applyOplistToState(oplist,state):
    '''apply an oplist to a state, summing the new states created by each string'''
    if not isinstance(oplist,list): #if oplist has only one string
        newState = applyPauliStringToState(oplist,state)
    else:
        newState = applyPauliStringToState(oplist[0],state)
        for string in oplist[1:]:
            newStatePart = applyPauliStringToState(string,state)
            newState = newState + newStatePart
    return newState

def applyListOplistsToState(listOplists,state):
    '''take a list of oplists, apply all to state, add states'''
    if not isinstance(listOplists[0],list): #if there is only one oplist with only one string
        newState = applyOplistToState(listOplists,state)
        return newState
    else:
        if not isinstance(listOplists[0][0],list): #if only one oplist
            newState = applyOplistToState(listOplists,state)
            return newState
        else:
            newState = state
            for oplist in reversed(listOplists):
                newState = applyOplistToState(oplist,newState)
            return newState

def trotteriseNoMatrix(listCommutingOplists,t,n, order, groundState):
    '''calculate the unitary evolution operator corresponding to a sum
    of hamiltonian operators given as list termList, using a 
    suzuki-trotter expansion.  
    
    NB This is the one you want for direct methods - it requires specification of the ground state (i.e. from a
    classical FCI calculation), but is MUCH faster and less memory intensive than explicitly forming a unitary.
    
    order: order of the expansion to be used.
    n: number of time steps
    '''
    import time
    import scipy.sparse
    
    startTime = time.clock()
    
    numSets = len(listCommutingOplists)
    indicesMultipliers = tsFormulaOrder(numSets,order)
    oplistsOrdered = []
    for indexMultiplier in indicesMultipliers:
        index = indexMultiplier[0]
        multiplier = indexMultiplier[1]
        newOplist = list(listCommutingOplists[index])
        if isinstance(newOplist[0],list):
            newNewOplist = []
            for i in range(len(newOplist)):
                newNewOp = list(newOplist[i])
                newNewOp[0] = newNewOp[0] * multiplier
                newNewOplist.append(newNewOp)
            oplistsOrdered.append(newNewOplist)
           # newOplist[i][0] = newOplist[i][0]*multiplier
        else:
           newNewOp = list(newOplist)
           newNewOp[0] = newNewOp[0]*multiplier
           oplistsOrdered.append(newNewOp)
           
           
    '''we now have a list of oplists which must be individually unitarised to give the TS solution'''
    import copy
    groundState2 = copy.deepcopy(groundState)
    stateVector = scipy.sparse.coo_matrix(groundState2)
    for i in range(n):
        for oplist in reversed(oplistsOrdered): #right to left
            listOplists = directPropagatorFromOplist(oplist,float(t)/float(n))
            stateVector = applyListOplistsToState(listOplists,stateVector)
        #print('done' + str(i) +'overall trotter step '+ str(time.clock()-startTime))
    endTime = time.clock()
   # print ('Execution time')
   # print (endTime-startTime)
    return stateVector    

    
def trotterise(listCommutingOplists,t,n, order,verbose=False):
    '''calculate the unitary evolution operator corresponding to a sum
    of hamiltonian operators given as list termList, using a 
    suzuki-trotter expansion.  order is the order of the expansion to be used.
    n is the number of time steps.
    TODO:  write the function
    '''
    import time
    startTime = time.clock()
    
    numSets = len(listCommutingOplists)
    indicesMultipliers = tsFormulaOrder(numSets,order)
    oplistsOrdered = []
    for indexMultiplier in indicesMultipliers:
        index = indexMultiplier[0]
        multiplier = indexMultiplier[1]
        newOplist = list(listCommutingOplists[index])
        if isinstance(newOplist[0],list):
            newNewOplist = []
            for i in range(len(newOplist)):
                newNewOp = list(newOplist[i])
                newNewOp[0] = newNewOp[0] * multiplier
                newNewOplist.append(newNewOp)
            oplistsOrdered.append(newNewOplist)
           # newOplist[i][0] = newOplist[i][0]*multiplier
        else:
           newNewOp = list(newOplist)
           newNewOp[0] = newNewOp[0]*multiplier
           oplistsOrdered.append(newNewOp)
    '''we now have a list of oplists which must be individually unitarised to give the TS solution'''
    
    if len(oplistsOrdered) > 1:
        overallUnitary = commutingOplistUnitary(oplistsOrdered[0],float(t)/float(n))
        for i in range(1,len(oplistsOrdered)):
            unitary = commutingOplistUnitary(oplistsOrdered[i], float(t)/float(n))
            overallUnitary = overallUnitary * unitary
    else:
        overallUnitary = commutingOplistUnitary(oplistsOrdered,float(t)/float(n))
    overallUnitary = overallUnitary ** n
    endTime = time.clock()
    if verbose:
        print ('Execution time')
        print (endTime-startTime)
    return overallUnitary

def trotteriseSerial(listCommutingOplists,t,n, order, groundState):
    '''calculate the unitary evolution operator corresponding to a sum
    of hamiltonian operators given as list termList, using a 
    suzuki-trotter expansion.  order is the order of the expansion to be used.
    n is the number of time steps.
    TODO:  write the function
    '''
    import time
    startTime = time.clock()
    
    numSets = len(listCommutingOplists)
    indicesMultipliers = tsFormulaOrder(numSets,order)
    oplistsOrdered = []
    for indexMultiplier in indicesMultipliers:
        index = indexMultiplier[0]
        multiplier = indexMultiplier[1]
        newOplist = list(listCommutingOplists[index])
        newNewOplist = []
        for i in range(len(newOplist)):
            newNewOp = list(newOplist[i])
            newNewOp[0] = newNewOp[0] * multiplier
            newNewOplist.append(newNewOp)
           # newOplist[i][0] = newOplist[i][0]*multiplier
        oplistsOrdered.append(newNewOplist)
    '''we now have a list of oplists which must be individually unitarised to give the TS solution'''
    stateVector = groundState
    for i in range(n):
        for oplist in reversed(oplistsOrdered): #right to left
            unitary = commutingOplistUnitary(oplist,float(t)/float(n))
            stateVector = unitary * stateVector
        print('done' + str(i) +'overall trotter step '+ str(time.clock()-startTime))
            
    
    
    endTime = time.clock()
    print ('Execution time')
    print (endTime-startTime)
    return stateVector

def trotteriseWriteToFiles(listCommutingOplists,t,n, order):
    '''calculate the unitary evolution operator corresponding to a sum
    of hamiltonian operators given as list termList, using a 
    suzuki-trotter expansion.  order is the order of the expansion to be used.
    n is the number of time steps.  nb instead of multiplying all time steps together
    and powering by the number of time steps here we simply save every time step 
    to disk.
    TODO:  write the function
    '''
    
    WORKINGDIR = '/home/andrew/scratch/timesteptests/'
    import time
    import pickle
    startTime = time.clock()
    
    numSets = len(listCommutingOplists)
    indicesMultipliers = tsFormulaOrder(numSets,order)
    oplistsOrdered = []
    for indexMultiplier in indicesMultipliers:
        index = indexMultiplier[0]
        multiplier = indexMultiplier[1]
        newOplist = list(listCommutingOplists[index])
        newNewOplist = []
        for i in range(len(newOplist)):
            newNewOp = list(newOplist[i])
            newNewOp[0] = newNewOp[0] * multiplier
            newNewOplist.append(newNewOp)
           # newOplist[i][0] = newOplist[i][0]*multiplier
        oplistsOrdered.append(newNewOplist)
    '''we now have a list of oplists which must be individually unitarised to give the TS solution'''
    
    if len(oplistsOrdered) > 1:
        overallUnitary = commutingOplistUnitary(oplistsOrdered[0],float(t)/float(n))
        output = open((WORKINGDIR+'0'+'.dat'), 'wb')
        pickle.dump(overallUnitary,output)
        for i in range(1,len(oplistsOrdered)):
            unitary = commutingOplistUnitary(oplistsOrdered[i], float(t)/float(n))
            output = open((WORKINGDIR+str(i)+'.dat'), 'wb')
            pickle.dump(unitary,output)
            #overallUnitary = overallUnitary * unitary
    else:
        overallUnitary = commutingOplistUnitary(oplistsOrdered,float(t)/float(n))
   # overallUnitary = overallUnitary ** n
    
    endTime = time.clock()
    print ('Execution time')
    print (endTime-startTime)
    return overallUnitary

def trotteriseMultiply(listCommutingOplists,t,n, order):
    '''calculate the unitary evolution operator corresponding to a sum
    of hamiltonian operators given as list termList, using a 
    suzuki-trotter expansion.  order is the order of the expansion to be used.
    n is the number of time steps.  nb here we just multiply
    TODO:  write the function
    '''
    import time
    import pickle
    startTime = time.clock()
    with open((WORKINGDIR+'0'+'.dat'),'rb') as f:
        overallUnitary = pickle.load(f)
    for i in range(1,NUMDATA):
        with open((WORKINGDIR+str(i)+'.dat'),'rb') as f:
            unitary = pickle.load(f)
        overallUnitary = overallUnitary * unitary
        output = open((WORKINGDIR+str(i)+'.dat'), 'wb')
    overallUnitary = overallUnitary ** NUMSTEPS
    with open((WORKINGDIR+'overallUnitary.dat'),'wb') as f:
        pickle.dump(overallUnitary,f)
    endTime = time.clock()
    print ('Execution time')
    print (endTime-startTime)
    return overallUnitary

        
def evolvedToInitialFidelity(initialState,unitary):
    '''takes an initial state vector and a unitary evolution operator at a certain time,
    calculates the fidelity between the evolved state and the initial state.'''
    evolvedState = unitary * initialState
    overlap = initialState.H * evolvedState
    fidelityAsMatrix = numpy.conj(overlap) * overlap
    fidelity = fidelityAsMatrix.item(0) #gross hack to remove singleton dimensions
    return fidelity

def estimatedPhase(initialState,unitary):
    '''takes an initial state vector and a unitary evolution operator.
    calculates the phase which will be determined through phase estimation.
    is equivalent to calculating the overlap between the evolved state and initial state.'''
    evolvedState = unitary * initialState
    overlap = initialState.H * evolvedState
   # overlap = overlapAsMatrix.item(0) #gross hack to remove singleton dimenstions
    return overlap.todense()

def angleZeroInterval(complexNumber):
    '''take a complex number, return angle in 0 - 2pi interval'''
    angle = numpy.angle(complexNumber)
    if angle < 0:
        return angle + 2*numpy.pi
    else:
        return angle

def hamiltonianEigenvalueFromUnitary(state,unitary,t):
    overlap = estimatedPhase(state,unitary)
    return -1 * (angleZeroInterval(overlap)/t)

def hamiltonianEigenvalueFromState(evolvedState,initialState,t):
    overlap = initialState.H * evolvedState
    overlapDense = overlap.todense()
    return -1 * (angleZeroInterval(overlapDense)/t)

#def Trotter_Ord1(termlist, t, n):
 #   b=list_exp(termlist, t/float(n))
  #  c=trot_term_mult(b)**n
   # return c

#def Trotter_Ord2(termlist, t, n):
 #   a=ord2_generator(termlist)
  #  b=list_exp(a, t/float(n)/2.0)
   # c=trot_term_mult(b)**n
    #return c

#def Trotter_Ord4(termlist, t, n):
 #   a=ord4_generator(termlist)
  #  b=list_exp(a, t/float(n)/2.0)
   # c=trot_term_mult(b)**n
    #return c
#make code where order is an input?


#def fidelity(ground, evolved):
 #   """fidelity determined by norm of overlap of ground state and approximated ground state"""
  #  return np.linalg.norm(ground.H*evolved*ground)**2


#def eigen_rotation(ground, evolved):
 #   """Overlap between actual approx, used to give angle that hamilitonian rotates to get eigenvalue approx"""
  #  return ground.H*evolved*ground
#better name would be eigenphase

#Functions to generate list of y values for plotting
#This is where terms input are, need to change depending on B-K or J-W

'''these functions are for generating plots.  they have been altered from their original forms to use 
our sparse matrix code, but will be rewritten in due course.
TODO:  see above'''


def n_plotter(t,max_n):
    """produces plot of fidelity with constant time, changing number of time steps, JW or BK chosen in plotter functions"""
    plt.plot(range(1,max_n),plotter1(t,max_n),'r--',range(1,max_n),plotter2(t,max_n),'b--', range(1,max_n), plotter4(t,max_n),'g--')
    plt.axis([0,max_n,0.0,1.1])
    plt.ylabel('Fidelity')
    plt.xlabel('Number of Time Steps')
    plt.legend(('First Order','Second Order','Fourth Order'),loc=4)
    plt.show()


def dt_plotter(t, max_n):
    """plot of fidelity, constant t, varying number of steps.  JW or BK chosen in plotter functions"""
    xlist=[t/n for n in range(1,max_n)]
    plt.plot(xlist,plotter1(t,max_n),'r--', xlist,plotter2(t,max_n), 'b--', xlist,plotter4(t,max_n),'g--')
    plt.axis([0,5,0.0,1.1])
    plt.ylabel('Fidelity')
    plt.xlabel('dt')
    plt.legend(('First Order','Second Order','Fourth Order'),loc=4)
    plt.show()


def estimatedPhases(t, maxNumSteps, order, listHamiltonians, initialState):
    '''estimate the phase given by phase estimation for a given list of hamiltonians,
    employing a trotter suzuki approximation.  returns a list of phases where the 
    number of time steps used in the approximation increases from 1 to maxNumSteps.
    
    t: propagation time, typically set to unity for phase estimation of eigenvalues
    maxNumSteps: the maximum number of time steps to consider
    order:  the order of the TS approximation (currently supports 1,2,4)
    listHamiltonians: a list of Hamiltonian terms
    initialState: the initial state of the simulation, typically the ground state of the overall Hamiltonian'''
    
    '''listPhases = []
    for i in range(1,maxNumSteps+1): # +1 as python range gives array-like indices
        unitary = trotterise(listHamiltonians,t,i,order)
        phase = estimatedPhase(initialState,unitary)
        listPhases.append(phase)
    return listPhases
    '''
    unitaryForNumSteps = lambda numSteps: trotterise(listHamiltonians,t,numSteps,order)
    phaseForNumSteps = lambda numSteps: estimatedPhase(initialState,unitaryForNumSteps(numSteps))
   # phaseForNumSteps = lambda numSteps: estimatedPhase(initialState,trotterise([tt.hz,tt.hxy],t,numSteps,order))
    listPhases = list(map(phaseForNumSteps, range(1,maxNumSteps)))
  #  listPhases = map(lambda numSteps: estimatedPhase(g,trotterise([tt.hz,tt.hxy],t,numSteps,order)),range(1,maxNumSteps))
    return listPhases
    
def rot_adder(i):
    """angle function range is '-pi to pi', changing it to '0 to 2pi'"""
    if np.angle(i)<0:
        return np.angle(i)+2*pi
    else:
        return np.angle(i)

#e^-iHt=e^i(angle) so angle=(Eg)t, solving for Eg

def doStuff():   
    import pickle
    from yaferp.interfaces import readStates
    argh = []
    for i in range(1,11):
        with(open('/home/andrew/workspace/MResProject/data/fullJW10its.dat','rb')) as f:
            ham = pickle.load(f)
        state = readStates.readState(18)
        bob = trotteriseNoMatrix(ham,0.01,i,1,state)
        argh.append(bob)
    return argh
        
#if __name__ == "__main__":
 #   main()
'''
def getTrueEigenvalue(oplist):
   ''' '''take a whole hamiltonian as an oplist, get the real eigenvalue by diagonalisation''''''
    hamMatrix = commutingOplistToMatrix(oplist)
    return scipy.sparse.linalg.eigsh(hamMatrix)[0][0]

def getTrueEigenvector(oplist):
    hamMatrix = commutingOplistToMatrix(oplist)
    vector = scipy.sparse.linalg.eigsh(hamMatrix)[1][:,0]
    vectorSparse = scipy.sparse.coo_matrix(vector).T
    return vectorSparse
'''
#from memory_profiler import profile
#@profile(precision=4)
def getTrueEigensystem(oplist,numEigenvalues=1):
    hamMatrix = sparseLinalg.oplistToSparse(oplist)
    eigenvalue,rawVector = scipy.sparse.linalg.eigsh(hamMatrix, numEigenvalues, which='SA')
    eigenvector = scipy.sparse.coo_matrix(rawVector[:, 0]).T
    return eigenvalue, eigenvector

def findEigensystemNearTarget(oplist,targetEigenvalue,numEigenvalues=1):
    hamMatrix = commutingOplistToMatrix(oplist)
    eigenvalue,rawVector = scipy.sparse.linalg.eigsh(hamMatrix,numEigenvalues,sigma=targetEigenvalue,which='LM')
    eigenvector = scipy.sparse.coo_matrix(rawVector[:,0]).T
    return eigenvalue,eigenvector

def calculateTime(ham,eigenvalue=False):
    if not eigenvalue:
        eigenvalue = getTrueEigensystem(ham)[0]
    if not math.floor(abs(eigenvalue/(2*math.pi))):
        t = 1
    else:
        t = 1/((2 * math.pi)*math.floor(abs(eigenvalue/(2 * math.pi))))
    return t
    
def trotterStepsUntilPrecision(ham,precision=0.0001,t=0,order=1,boolGroupTerms=False,maxN=20):
    from general import fermions
    n = 1
    try:
       # trueEigenvalue = getTrueEigenvalue(ham)
       # groundState = getTrueEigenvector(ham)
        trueEigenvalue, groundState = getTrueEigensystem(ham)
    except:
        print("Couldn't calculate eigensystem of hamiltonian")
    floatPrecision = float(precision)
    if not t:
        t = calculateTime(ham,trueEigenvalue)
    if boolGroupTerms:
        ham = fermions.groupCommutingTerms(ham)
    while n <= maxN:
        unitary = trotterise(ham,t,n,order)
        trotterEigenvalue = hamiltonianEigenvalueFromUnitary(groundState,unitary,t)
        difference = numpy.linalg.norm(numpy.linalg.norm(trueEigenvalue)-numpy.linalg.norm(trotterEigenvalue))
        if difference < floatPrecision:
            return n
        n+=1
        
    return 0

def fastStepsUntilPrecision(hamiltonian,trueEigenstate=[],trueEigenvalue=[],precision=0.0001,t=0,order=1,maxN=20):
    if trueEigenstate == [] or trueEigenvalue == []:
        try:
            trueEigenstate, trueEigenvalue = getTrueEigensystem(hamiltonian)
        except:
            print("Couldn't calculate eigensystem of hamiltonian")
    floatPrecision = float(precision)
    if not t: 
        t = calculateTime(hamiltonian,trueEigenvalue)
    n = 1
    while n <= maxN:
        testEigenstate = copy.deepcopy(trueEigenstate)
        thisHamiltonian = copy.deepcopy(hamiltonian)
        finalState = trotteriseNoMatrix(thisHamiltonian,t,n,order, testEigenstate)
        eigenvalue = hamiltonianEigenvalueFromState(finalState,trueEigenstate,t)
        difference = numpy.linalg.norm(numpy.linalg.norm(trueEigenvalue) - numpy.linalg.norm(eigenvalue))
        if difference < floatPrecision:
            return n
        n+=1
    return 0
        
    
    
def countGatesOneTrotterStep(hamiltonian):
    numCNOTs = 0
    numSQGs = 0
    numNonIdentity = 0
    numXandY = 0
    
    for term in hamiltonian:
        numNonIdentity = 0
        numXandY = 0
        pauliString = term[1]
        for pauli in pauliString:
            if int(pauli) == 1 or int(pauli) == 2:
                numXandY += 1
                numNonIdentity += 1
            elif int(pauli) == 3:
                numNonIdentity += 1
                
        if numNonIdentity:
            numCNOTs += 2 * (numNonIdentity - 1)
            numSQGs += 1 + 2*numXandY
        
    return (numSQGs, numCNOTs)

def countGates(hamiltonian,numTrotterSteps,order,countCSQGs=0):
    '''only supports order 1,2,4!!!!
    note that this is slightly off from the real gate count as it doesn't combine middle terms.
    i.e. this calculates ABBA for 2 groups in 2nd order.  however this shouldn't make a huge difference
    apart from very small hamiltonians.
    TODO:  see above'''
    oneStepSQGs, oneStepCNOTs = countGatesOneTrotterStep(hamiltonian)
    
    if order == 1:
        orderMultiplier = 1
    elif order == 2:
        orderMultiplier = 2
    elif order == 4:
        orderMultiplier = 10
        
    numSQGs = oneStepSQGs * orderMultiplier * numTrotterSteps
    numCNOTs = oneStepCNOTs * orderMultiplier * numTrotterSteps
    if countCSQGs:
        return(correctGatesCountCSQG((numSQGs,numCNOTs),hamiltonian,numTrotterSteps,order))
    else:
        return(numSQGs,numCNOTs) 

def correctGatesCountCSQG(gateCount,hamiltonian,numTrotterSteps,order):
    numSQGs = gateCount[0]
    numCNOTs = gateCount[1]
    if order == 1:
        multiplier = numTrotterSteps
    elif order == 2:
        multiplier = 2*numTrotterSteps
    elif order == 4:
        multiplier = 10*numTrotterSteps
    numCSQGs = len(hamiltonian)*multiplier
    numNotCSQGs = numSQGs - numCSQGs
    return(numNotCSQGs,numCSQGs,numCNOTs)
     
       
def trotterGatesToPrecision(ham,precision,t,order,boolgroupTerms=False,maxIterations=20):
    '''returns (numSQGs, numCNOTs).  (0,0) on failure to achieve desired precision.'''
    numSteps = trotterStepsUntilPrecision(ham,precision,t,order,boolgroupTerms,maxIterations)
    gates = countGates(ham,numSteps,order)
    return gates

def quickTrotterGatesToPrecision(hamiltonian,trueEigenstate=[],trueEigenvector=[],precision=0.0001,t=0,order=1,maxIterations=20):
    numSteps = fastStepsUntilPrecision(hamiltonian,trueEigenstate,trueEigenvector,precision,t,order,maxIterations)
    gates = countGates(hamiltonian,numSteps,order)
    return gates
        
        
def oneStepTrotterError(ham,t=0,order=1,boolGroupTerms=False,maxN=20):    
    from general import fermions
    try:
       # trueEigenvalue = getTrueEigenvalue(ham)
       # groundState = getTrueEigenvector(ham)
        trueEigenvalue, groundState = getTrueEigensystem(ham)
    except:
        print("Couldn't calculate eigensystem of hamiltonian")
    if not t:
        t = calculateTime(ham,trueEigenvalue)
    if boolGroupTerms:
        ham = fermions.groupCommutingTerms(ham)
    unitary = trotterise(ham,t,1,order)
    trotterEigenvalue = hamiltonianEigenvalueFromUnitary(groundState,unitary,t)
    difference = numpy.linalg.norm(numpy.linalg.norm(trueEigenvalue)-numpy.linalg.norm(trotterEigenvalue))
    return difference

def errorByTrotterNumber(ham,t=0,order=1,listTrotterNumbers=[1],boolGroupTerms=False):    
    from general import fermions
    try:
       # trueEigenvalue = getTrueEigenvalue(ham)
       # groundState = getTrueEigenvector(ham)
        trueEigenvalue, groundState = getTrueEigensystem(ham)
    except:
        print("Couldn't calculate eigensystem of hamiltonian")
    if not t:
        t = calculateTime(ham,trueEigenvalue)
    if boolGroupTerms:
        ham = fermions.groupCommutingTerms(ham)
    errors = []
    for thisN in listTrotterNumbers:
        unitary = trotterise(ham,t,thisN,order)
        trotterEigenvalue = hamiltonianEigenvalueFromUnitary(groundState,unitary,t)
        #difference = numpy.linalg.norm(trueEigenvalue)-numpy.linalg.norm(trotterEigenvalue)
        difference = trueEigenvalue - trotterEigenvalue
        errors.append(difference)
    return errors
def oplistExpectation(oplist,vector):
    opMatrix= sparseLinalg.oplistToSparse(oplist)
    expectation = vector.H * opMatrix * vector
    return expectation


def expectationPauliString(operator,vector):
    opMatrix = commutingOplistToMatrix(operator)
    expectation = vector.H * opMatrix * vector
    return expectation

def trotterEigenvalue(oplist,t,n,order):
    try:
       # trueEigenvalue = getTrueEigenvalue(ham)
       # groundState = getTrueEigenvector(ham)
        trueEigenvalue, groundState = getTrueEigensystem(oplist)
    except:
        print("Couldn't calculate eigensystem of hamiltonian")
    unitary=trotterise(oplist,t,n,order)
    trotterEigenvalue = hamiltonianEigenvalueFromUnitary(groundState,unitary,t)
    return trotterEigenvalue

def exactUnitary(oplist,t):
    hamMatrix = commutingOplistToMatrix(oplist)
    exponent = -1j * t * hamMatrix
    thing = scipy.sparse.linalg.expm(exponent)
    return thing

def expectOfUnitary(oplist,t,groundState,timesToApply=1):
    unitary = exactUnitary(oplist,t)
    repeatedUnitary = unitary**timesToApply
    expectation = groundState.H * repeatedUnitary * groundState
    return expectation

def expectOfTrotter(oplist,t,n,order,groundState,timesToApply=1):
    unitary = trotterise(oplist,t,n,order)
    repeatedUnitary = unitary**timesToApply
    expectation = groundState.H*repeatedUnitary*groundState
    return expectation

def expectAsPhase(oplist,t,groundState,timesToApply=1):
    expectation = expectOfUnitary(oplist,t,groundState,timesToApply).todense()
    angle = angleZeroInterval(expectation)
    phase = (angle * -1) /(2*numpy.pi)
    return phase

def trotterExpectAsPhase(oplist,t,n,order,groundState,timesToApply=1):
    expectation = expectOfTrotter(oplist,t,n,order,groundState,timesToApply).todense()
    angle = angleZeroInterval(expectation)
    phase = (angle*-1)/(2*numpy.pi)
    return phase

def getLongPhases(oplist,t,groundState,maxBits):
    thing = []
    for i in range(maxBits):
        time = t* (2**i)
        phase = expectAsPhase(oplist,time,groundState)
        thing.append(phase)
    return thing

def getLongTrotterPhases(oplist,t,n,order,groundState,maxBits):
    thing = []
    for i in range(maxBits):
        time = t*(2**i)
        phase = trotterExpectAsPhase(oplist,time,n,order,groundState,1)
        newPhase = phase
        thing.append(newPhase)
    return thing

def getRepeatedPhases(oplist,t,groundState,maxBits):
    thing = []
    for i in range(maxBits):
        phase = expectAsPhase(oplist,t,groundState,2**i)
        thing.append(phase)
    return thing

def getRepeatedTrotterPhases(oplist,t,n,order,groundState,maxBits):
    thing = []
    for i in range(maxBits):
        phase = trotterExpectAsPhase(oplist,t,n,order,groundState,2**i)
        thing.append(phase)
    return thing


