''' sparseFermions - Sparse matrix methods for simulation of fermionic quantum simulation
Created on 5 Aug 2014

@author: Andrew Tranter (a.tranter13@imperial.ac.uk)
'''

import scipy
import numpy
import scipy.sparse
import scipy.sparse.linalg
from datetime import datetime

def kronProductListSparse(listMatrices):
    product = 1
    for thisMatrix in listMatrices:
        product = scipy.sparse.kron(product,thisMatrix, 'csc')
    return product
        

def symPauliToSparseMatrix(whichPauli):
    if whichPauli == 1:
        matrix = numpy.array([[0.0,1.0],[1.0,0.0]])
    elif whichPauli == 2:
        matrix = numpy.array([[0.,0.-1.j],[0.+1.j,0.]]) 
    elif whichPauli == 3:
        matrix = numpy.array([[1.,0.],[0.,-1.]])
    else:
        matrix = numpy.array([[1.,0.],[0.,1.]])
    sparseMatrix = scipy.sparse.csc_matrix(matrix)
    return sparseMatrix

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
    from yaferp.general import fermions

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
    overallMatrix = commutingOplistToMatrix(exponentiatePauliString(oplist[0]))
    if len(oplist) > 1:
        for i in range(1,len(oplist)):
            newMatrix = commutingOplistToMatrix(exponentiatePauliString(oplist[i]))
            overallMatrix = overallMatrix * newMatrix
    return overallMatrix

def commutingOplistUnitary(oplist,t):
    '''find the unitary evolution operator for a sum of commuting hamiltonians at time t
    (u = e^(-iHt) )'''
    oplistCopy = list(oplist)
    oplistNewCoefficients = []
    for term in oplistCopy:
        coefficient = term[0] * -1. * t
        oplistNewCoefficients.append([coefficient,term[1]])
    return exponentiateCommutingOplist(oplistNewCoefficients)

def commutingOplistToMatrix(oplist, outputFile = '/run/media/andrew/b2305290-df58-489c-a39a-d25f8b52c664/data/data.dat', boolPrintDebugInfo=0):
    '''take in a oplist where all elements commute.  output one H matrix.
    nb note oplist is list of [(coefficient), (pauli string)] entries.  see standard established above
    TODO:  do a better outputFile thing.  (make it store to a temp file which is deleted on output).'''
    import sys
    
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
        
         
    #if boolPrintDebugInfo == 1:
     #   print(str(datetime.now()))
      #  print('Full dimensional calculation of Pauli strings completed.  Summing.')
       # sys.stdout.flush()
    
 #   overallMatrix = listMatrixTerms[0]
  #  for i in range(1,len(listMatrixTerms)):
   #     overallMatrix = overallMatrix + listMatrixTerms[i]
    
    #overallMatrix = numpy.asmatrix(overallMatrix) #convert to numpy matrix type for compatibility
    return overallMatrix


def main():
    import pickle
    from yaferp.integrals import readintegrals
    from yaferp.general import fermions
    one = readintegrals.importIntegrals()
    two = fermions.electronicHamiltonian(18, 0, one[0], one[1])
    three = fermions.groupCommutingTerms(two)
    with open('/home/andrew/scratch/timesteptests/data.dat','wb') as f:
        pickle.dump(three,f) 
'''    
    with open('/home/andrew/scratch/data.dat', 'rb') as f:
        steve = pickle.load(f)
    try:
        steve2 = reversed(steve)
        for index, commutingGroup in enumerate(steve2):
            if index == 305:
                hamiltonian = commutingOplistToMatrix(commutingGroup)
                outputpath = '/home/andrew/scratch/' + str(index) + '.dat'
                with open(outputpath, 'wb') as f:
                    pickle.dump(hamiltonian, f)
            
            output = open(('/home/andrew/scratch/'+str(index)+'.dat'), 'wb')
            pickle.dump(hamiltonian,output)
            output.close()
            
        
    except MemoryError:
        print('oh, balls.')
'''        
if __name__ == "__main__":
    main()


'''below lies old trotter code for adaptation.  all code not mine except where noted. -AT
TODO:  clean up, liberally.'''
 
import numpy as np
import scipy.linalg
from cmath import pi
import matplotlib.pyplot as plt
#import trotterms as tt

#All terms for H. from Jake Seeley, to trotterize his work
# Bravyi-Kitaev Terms

#Bravy-Kitaev eigenvector and value (g for ground state)
g=np.linalg.eig((tt.hz+tt.hxy))[1][:,1]
# Lowest Eigenvalue is -1.851
eigval=np.linalg.eig((tt.hz+tt.hxy))[0][1]

#Jordan-Wigner eigen 
JW_g=np.linalg.eig((tt.JW_hz+tt.JW_hxy))[1][:,3]
JW_eigval=np.linalg.eig((tt.JW_hz+tt.JW_hxy))[0][3]


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

def trotterise(listCommutingOplists,t,n, order):
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
    
    if len(oplistsOrdered) > 1:
        overallUnitary = commutingOplistUnitary(oplistsOrdered[0],float(t)/float(n))
        for i in range(1,len(oplistsOrdered)):
            unitary = commutingOplistUnitary(oplistsOrdered[i], float(t)/float(n))
            overallUnitary = overallUnitary * unitary
    else:
        overallUnitary = commutingOplistUnitary(oplistsOrdered,float(t)/float(n))
    overallUnitary = overallUnitary ** n
    
    endTime = time.clock()
    print ('Execution time')
    print (endTime-startTime)
    return overallUnitary

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
    overlapAsMatrix = initialState.H * evolvedState
    overlap = overlapAsMatrix.item(0) #gross hack to remove singleton dimenstions
    return overlap
    
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

def plotter1(t,z):
    listFidelities = map(lambda n: evolvedToInitialFidelity(JW_g,trotterise([scipy.sparse.csc_matrix(tt.JW_hz),scipy.sparse.csc_matrix(tt.JW_hxy)],t,n,1)),range(1,z))
    return listFidelities
def plotter2(t,z):
    listFidelities = map(lambda n: evolvedToInitialFidelity(JW_g,trotterise([scipy.sparse.csc_matrix(tt.JW_hz),scipy.sparse.csc_matrix(tt.JW_hxy)],t,n,2)),range(1,z))
    return listFidelities
def plotter4(t,z):
    listFidelities = map(lambda n: evolvedToInitialFidelity(JW_g,trotterise([scipy.sparse.csc_matrix(tt.JW_hz),scipy.sparse.csc_matrix(tt.JW_hxy)],t,n,4)),range(1,z))
    return listFidelities

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

#list of rotations used for plotting eigenvalue approx
#BK terms
#def eigen1(t,z):
 #   return map(lambda n: estimatedPhase(g,trotterise([tt.hz,tt.hxy],t,n,1)),range(1,z))
def eigen2(t,z):
    return map(lambda n: estimatedPhase(g,trotterise([tt.hz,tt.hxy],t,n,2)),range(1,z))
#def eigen4(t,z):
 #   return map(lambda n: estimatedPhase(g,trotterise([scipy.sparse.csc_matrix(tt.hz),scipy.sparse.csc_matrix(tt.hxy)],t,n,4)),range(1,z))

def estimatedPhases(t, maxNumSteps, order, listHamiltonians = [tt.hz,tt.hxy], initialState = g):
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
    
#JW terms
def eigen1JW(t,z):
    return map(lambda n: eigen_rotation(JW_g,Trotter_Ord1([tt.JW_hz,tt.JW_hxy],t,n)),range(1,z))
def eigen2JW(t,z):
    return map(lambda n: eigen_rotation(JW_g,Trotter_Ord2([tt.JW_hz,tt.JW_hxy],t,n)),range(1,z))
def eigen4JW(t,z):
    return map(lambda n: eigen_rotation(JW_g,Trotter_Ord4([tt.JW_hz,tt.JW_hxy],t,n)),range(1,z))

def rot_adder(i):
    """angle function range is '-pi to pi', changing it to '0 to 2pi'"""
    if np.angle(i)<0:
        return np.angle(i)+2*pi
    else:
        return np.angle(i)

#e^-iHt=e^i(angle) so angle=(Eg)t, solving for Eg
def eigenvalue(t,max_n):
    """plotting how close the approximation gets to the actual eigenvalue with increasing n"""
    a = []
    b = []
    c = []
    
    '''
    for i in estimatedPhases(t,max_n+1,1):
        def f(i=i): return ((rot_adder(i))/t)
        a.append(f)
    for i in estimatedPhases(t,max_n+1,2):
        def f(i=i): return ((rot_adder(i))/t)
        b.append(f)                                            
    for i in estimatedPhases(t,max_n+1,4):
        def f(i=i): return ((rot_adder(i))/t)
        c.append(f)'''
    '''
    a=[(lambda i=i: (rot_adder(i))/t) for i in estimatedPhases(t, max_n, 1)]
    b=[(lambda i=i: (rot_adder(i))/t) for i in estimatedPhases(t, max_n, 2)]
    c=eval('[(lambda i=i: (rot_adder(i))/t) for i in estimatedPhases(t, max_n, 4)]')'''
    for i in range(len(estimatedPhases(t,max_n,1))):
        ARGH1 = estimatedPhases(t,max_n,1)[i]
        ARGH2 = estimatedPhases(t,max_n,2)[i]
        ARGH3 = estimatedPhases(t,max_n,4)[i]
        
        WHY1 = rot_adder(ARGH1)/t
        WHY2 = rot_adder(ARGH2)/t
        WHY3 = rot_adder(ARGH3)/t
        
        a.append(WHY1)
        b.append(WHY2)
        c.append(WHY3)
        
        '''a.append(estimatedPhases(t,max_n,1))[i]
        b.append(estimatedPhases(t,max_n,2))[i]
        c.append(estimatedPhases(t,max_n,4))[i]'''
                   
    plt.plot(range(1,max_n),a,'r-o',range(1,max_n),b,'b-o', range(1,max_n), c,'g-o')
    plt.axis([0,max_n,-eigval-0.001,-eigval+0.001])
    plt.axhline(y=-eigval-0.0001, color='purple')
    plt.ylabel('Eigenvalue')
    plt.xlabel('Number of Time Steps')
    plt.legend(('First Order','Second Order','Fourth Order'),loc=4)
    plt.ticklabel_format(useOffset=False)
    plt.show()

    #eigenvalues are technically the same but should be changed depending on method to split terms
def dt_eigenvalue(t,max_n):
    """eigenvalue with dt on x axis (constant t, changing n)"""
    xlist=[t/n for n in range(1,max_n)]
    a=[(rot_adder(i)[0,0])/t*(-1) for i in eigen1(t,max_n)]
    b=[(rot_adder(i)[0,0])/t*(-1) for i in eigen2(t,max_n)]
    c=[(rot_adder(i)[0,0])/t*(-1) for i in eigen4(t,max_n)]
    plt.plot(xlist,a,'r-s', xlist,b, 'b-o', xlist,c,'g-d')
    plt.axis([0,1,eigval-0.001,eigval+0.001])
    plt.ylabel('Eigenvalue')
    plt.xlabel('dt')
    plt.legend(('First Order','Second Order','Fourth Order'),loc=4)
    plt.axhline(y=eigval-0.0001, color='purple')
    plt.axhline(y=eigval+0.0001, color='purple')
    plt.ticklabel_format(useOffset=False)
    plt.show()
 
 
 
 