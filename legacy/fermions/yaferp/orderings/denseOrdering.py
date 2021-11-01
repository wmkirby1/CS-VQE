'''
Created on 18 Dec 2014

@author: andrew
'''
import numpy
import random
import itertools
from yaferp.general import sparseFermions, fermions
import copy
#import pygraph
#import pygraph.classes
#import pygraph.classes.graph
import datetime

def generateAdjacencyMatrix(oplist,dual=0):
    '''take an oplist, generate the adjacency matrix for that oplist.  edges represent noncommutativity.
    set dual for edges to represent commutativity
    '''
    numTerms = len(oplist)
    zeroArray = numpy.zeros((numTerms,numTerms),dtype=int)
    adjacencyMatrix = numpy.matrix(zeroArray)
    for (index1, term1) in enumerate(oplist):
        for (index2,term2) in enumerate(oplist):
            if not dual and not fermions.checkCommute(term1[1], term2[1]):
                adjacencyMatrix[index1,index2] = 1
            elif dual and fermions.checkCommute(term1[1], term2[1]):
                adjacencyMatrix[index1,index2] = 1
    '''  for (index1,term1) in enumerate(oplist):
        for (index2,term2) in enumerate(oplist[(index1+1):]):
            realIndex2 = index2 + index1
            if not fermions.symcommutes(term1,term2):
                adjacencyMatrix[index1][realIndex2] = 1
                adjacencyMatrix[realIndex2][index1] = 1'''
    
    return adjacencyMatrix

def generateCommutatorGraph(oplist):
    numTerms = len(oplist)
    graph = pygraph.classes.graph.graph()
    graph.add_nodes(tuple(range(numTerms)))
    for (index1,term1) in enumerate(oplist):
        for (index2,term2) in enumerate(oplist):
            if fermions.symcommutes(term1[1], term2[1]):
                if not graph.has_edge(tuple([index1,index2])):
                    graph.add_edge(tuple([index1,index2]),wt=0)
            else:
                if not graph.has_edge(tuple([index1,index2])):
                    graph.add_edge(tuple([index1,index2]),wt=1)
    return graph
        
        
    return graph




 
def generateOrderingLabels(numTerms):
    '''generate a list of all the possible orderings of terms (using ints to label the terms).
    please don't put a big number in here.'''
    return list(itertools.permutations(range(numTerms)))

def reorderOplist(oplist,order):
    newOplist = []
    for newOrderIndex in order:
        newOplist.append(oplist[newOrderIndex])
    return newOplist
            
def randomTerm(numQubits):
    coefficient = random.random()
    pauliString = []
    for i in range(numQubits):
        pauliString.append(random.randint(0,3))    
    return [coefficient,pauliString]

def randomOplist(numQubits,numTerms):
    oplist = []
    for i in range(numTerms):
        oplist.append(randomTerm(numQubits))
    return oplist
        
def trotterStepsForAllPermutations(hamiltonianOplist):
    numQubits = len(hamiltonianOplist[0][1])
    permutationLabels = generateOrderingLabels(numQubits)
    errorDict = {}
    for permutation in permutationLabels:
        newOplist = copy.deepcopy(hamiltonianOplist)
        newOplist = reorderOplist(newOplist,permutation)
        numSteps = sparseFermions.trotterStepsUntilPrecision(newOplist)
        errorDict[tuple(permutation)] = numSteps
    return errorDict

def trotterErrorForAllPermutations(hamiltonianOplist,listPermutations=None, verbose=0):
    numTerms = len(hamiltonianOplist)
    if listPermutations == None:
        listPermutations = generateOrderingLabels(numTerms)
    errorDict = {}
    numPerms = len(listPermutations)
    permsCounter = 1
    for permutation in listPermutations:
        newOplist = copy.deepcopy(hamiltonianOplist)
        newOplist = reorderOplist(newOplist,permutation)
        error = sparseFermions.oneStepTrotterError(newOplist)
        errorDict[tuple(permutation)] = error
        if verbose:
            print(str(datetime.datetime.now()) + ' Permutation ' + str(permsCounter) + ' of ' + str(numPerms) + ' completed.')
            permsCounter += 1
    return errorDict
def trotterErrorForAllPermutationsPregroup(hamiltonianOplist,pregroupOplist,listPermutations=None):
    numTerms = len(hamiltonianOplist)
    if listPermutations == None:
        listPermutations = generateOrderingLabels(numTerms)
    errorDict = {}
    for permutation in listPermutations:
        newOplist = copy.deepcopy(hamiltonianOplist)
        newOplist = reorderOplist(newOplist,permutation)
        fullOplist = pregroupOplist + newOplist
        error = sparseFermions.oneStepTrotterError(fullOplist)
        errorDict[tuple(permutation)] = error
    return errorDict    
    
    
    
    
    
def spliceNewTermOrdering(ordering):
    newTermIndex = max(ordering) + 1
    possibleOrderings = []
    for i in range(newTermIndex+1):
        thisOrdering = ordering[0:i] + (newTermIndex,) + ordering[i:]
        possibleOrderings.append(thisOrdering)
    return possibleOrderings


def pickBestSplice(hamiltonianOplist,currentOrdering,newTerm):
    newOrderings = spliceNewTermOrdering(currentOrdering)
    hamiltonianOplist.append(list(newTerm))
    errorDict = trotterErrorForAllPermutations(hamiltonianOplist,newOrderings)
    bestOrder = min(errorDict,key=errorDict.get)
    #newHamiltonian = reorderOplist(hamiltonianOplist,bestOrder)
    return bestOrder

def greedySortOplist(hamiltonianOplist):
    oplistCopy = copy.deepcopy(hamiltonianOplist) #better safe than sorry.
    orderedOplist = []
    '''try and minimize the impact of the first three choices by picking the best permutation of these.
    TODO:  refactor the crap out of this whole chain, it's awful.'''
    firstIndex = random.randint(0,len(oplistCopy)-1)
    orderedOplist.append(oplistCopy.pop(firstIndex))
    secondIndex = random.randint(0,len(oplistCopy)-1)
    orderedOplist.append(oplistCopy.pop(secondIndex))
    thirdIndex = random.randint(0,len(oplistCopy)-1)
    orderedOplist.append(oplistCopy.pop(thirdIndex))
    errorDict = trotterErrorForAllPermutations(orderedOplist)
    currentOrdering = min(errorDict,key=errorDict.get)
    
    while oplistCopy:
        if isinstance(oplistCopy[0],list):
            randomIndex = random.randint(0,len(oplistCopy)-1)
            newTerm = oplistCopy.pop(randomIndex)
        else:
            newTerm = copy.deepcopy(oplistCopy)
            oplistCopy = []
            
        currentOrdering = pickBestSplice(orderedOplist,currentOrdering,newTerm)
    return currentOrdering
    
def bestPermutationError(hamiltonianOplist):
    dictAllErrors = trotterErrorForAllPermutations(hamiltonianOplist)
    smallestError = min(dictAllErrors.values())
    return smallestError

def testGreedySort(hamiltonianOplist):
    smallestError = bestPermutationError(hamiltonianOplist)
    greedyOrdering = greedySortOplist(hamiltonianOplist)
    greedyHam = reorderOplist(hamiltonianOplist,greedyOrdering)
    greedyError = sparseFermions.oneStepTrotterError(greedyHam)
    return smallestError-greedyError
    
def randomHamErrorForAllPermutations(numQubits,numTerms,calcAdjacencyMatrix = 1):
    ham = reliableRandomOplist(numQubits,numTerms)
    if calcAdjacencyMatrix:
        adjacencyMatrix = generateAdjacencyMatrix(ham,1)
        return (trotterErrorForAllPermutations(ham), ham, adjacencyMatrix)
    else:
        return (trotterErrorForAllPermutations(ham),ham)

def averageOrderingError(orderings):
    errors = orderings.values()
    return numpy.mean(errors)

def minimumOrderingError(orderings):
    errors = orderings.values()
    minError = min(errors)
    return minError

def maximumOrderingError(orderings):
    errors = orderings.values()
    maxError = max(errors)
    return maxError
    
def getLexiographicOrdering(oplist):
    orderedOplist = copy.deepcopy(oplist)
    orderedOplist.sort(key=lambda item:item[1])
    ordering = []
    for orderedTerm in orderedOplist:
        for index,originalTerm in enumerate(oplist):
            if orderedTerm == originalTerm:
                ordering.append(index)
                break
    return tuple(ordering)

def getMagnitudeOrdering(oplist):
    orderedOplist = copy.deepcopy(oplist)
    orderedOplist.sort(key=lambda item:abs(item[0]))
    orderedOplist.reverse()
    ordering = []
    for orderedTerm in orderedOplist:
        for index,originalTerm in enumerate(oplist):
            if orderedTerm == originalTerm:
                ordering.append(index)
                break
    return tuple(ordering)

def compareOrderingSchemes(oplist):
    errors = {}
    orderingErrors = trotterErrorForAllPermutations(oplist)
    errors["Minimum"] = minimumOrderingError(orderingErrors)
    errors["Maximum"] = maximumOrderingError(orderingErrors)
    errors["Average"] = averageOrderingError(orderingErrors)
    errors["Lexiographic"] = orderingErrors[getLexiographicOrdering(oplist)]
    errors["Magnitude"] = orderingErrors[getMagnitudeOrdering(oplist)]
    errors["Commutator"] = orderingErrors[getCommutatorOrdering(oplist)]
    return errors
def readoutSetOfOrderingSchemes(listOrderings):
    listOrderingNames = listOrderings[0].keys()
    for ordering in listOrderingNames:
        print(str(ordering)+','),
    print('')
    for thisDict in listOrderings:
        for ordering in listOrderingNames:
            print(str(thisDict[ordering])+','),
        print('')
    return
        
        
        
def compareOrderingSchemesLarge(oplist,trotterOrder):
    errors = {}
    lexiographicOrdering = getLexiographicOrdering(oplist)
    magnitudeOrdering = getMagnitudeOrdering(oplist)
    commutatorOrdering = getCommutatorOrdering(oplist)
    lexiographicOplist = reorderOplist(oplist,lexiographicOrdering)
    magnitudeOplist = reorderOplist(oplist,magnitudeOrdering)
    commutatorOplist = reorderOplist(oplist,commutatorOrdering)
    lexiographicError = sparseFermions.oneStepTrotterError(lexiographicOplist, order=trotterOrder)
    magnitudeError = sparseFermions.oneStepTrotterError(magnitudeOplist, order=trotterOrder)
    commutatorError = sparseFermions.oneStepTrotterError(commutatorOplist, order=trotterOrder)
    errors["Lexiographic"] = lexiographicError
    errors["Magnitude"] = magnitudeError
    errors["Commutator"] = commutatorError
    return errors
    
    
    

def commutatorOrderingFindNextTerm(oplist,adjacencyMatrix,ordering=()):
    numTerms = len(oplist)
    if isinstance(ordering,int):
        ordering = [ordering]
    unsortedIndices = [x for x in range(numTerms) if x not in ordering]
    recordNonCommuting = 0
    recordCoefficient = 0.
    currentPickIndex = -1
    for unsortedIndex in unsortedIndices:
        thisTerm = oplist[unsortedIndex]
        thisCoefficient = thisTerm[0]
        thisAdjacencyRow = adjacencyMatrix[unsortedIndex]
        numNonCommuting = 0
        #count non-commuting
        for sortedIndex in ordering:
            numNonCommuting = numNonCommuting + thisAdjacencyRow[0,sortedIndex]
        if numNonCommuting > recordNonCommuting:
            currentPickIndex = unsortedIndex
            recordNonCommuting = numNonCommuting
            recordCoefficient = thisCoefficient
        elif numNonCommuting == recordNonCommuting and abs(thisCoefficient) > abs(recordCoefficient):
            currentPickIndex = unsortedIndex
            recordCoefficient = thisCoefficient
            
    return currentPickIndex
            

def getCommutatorOrdering(oplist):
    adjacencyMatrix = generateAdjacencyMatrix(oplist,0)
    ordering = []
    for i in range(len(oplist)):
        newIndex = commutatorOrderingFindNextTerm(oplist,adjacencyMatrix,ordering)
        ordering.append(newIndex)
    return tuple(ordering)
    

def randomHamCompareOrderings(numQubits,numTerms):
    ham = reliableRandomOplist(numQubits,numTerms)
    return compareOrderingSchemes(ham)

def reliableRandomOplist(numQubits,numTerms):
    ham = randomOplist(numQubits,numTerms)
    hamiltonianWorks = False
    while not hamiltonianWorks:
        hamiltonianWorks = True
        try:
            eigenvalue = sparseFermions.getTrueEigensystem(ham)[0]
        except:
            hamiltonianWorks = False
            ham = randomOplist(numQubits,numTerms)
    return ham
    
    
def batchRandomGreedyTest(numQubits,numTerms,numTrials):
    listDiscrepancies = []
    for i in range(numTrials):
        ham = randomOplist(numQubits,numTerms)
        hamiltonianWorks = False
        while not hamiltonianWorks:
            hamiltonianWorks = True
            try:
                eigenvalue = sparseFermions.getTrueEigensystem(ham)[0]
            except:
                hamiltonianWorks = False
                ham = randomOplist(numQubits,numTerms)
        greedyResult = testGreedySort(ham)
        listDiscrepancies.append(greedyResult)
    return listDiscrepancies

def commutator(op1,op2):
    if not op1 or not op2:
        return []
    numQubits = len(op1[1])
    commutatorString = []
    isZero = True
    negated = False

    coefficient = 2j * op1[0] * op2[0]
    for i in range(numQubits):
        if op1[1][i] == op2[1][i]:
            commutatorString.append(0)
        elif op1[1][i] == 0:
            commutatorString.append(op2[1][i])
        elif op2[1][i] == 0:
            commutatorString.append(op1[1][i])
        else:
            newOp = 6 - op1[1][i] - op2[1][i]
            isZero = not isZero
            commutatorString.append(newOp)
            if ((op1[1][i] == 0 and op2[1][i] == 2) or (op1[1][i] == 1 and op2[1][i] == 0)) or (op1[1][i] == 2 and op2[1][i] == 1):
                negated = not negated
    if isZero:
        return []
    if negated:
        coefficient = -1 * coefficient
    
    return [coefficient, commutatorString]

def errorOperatorInterior(oplist,alpha,beta,gamma):
    hamAlpha = oplist[alpha]
    hamBeta = oplist[beta]
    hamGamma = oplist[gamma]
    innerCommutator = commutator(hamBeta,hamGamma)
    if not innerCommutator:
        return []
    tripleCommutator = commutator(hamAlpha,innerCommutator)
    if not tripleCommutator:
        return []
    if alpha == beta:
        tripleCommutator[0] = tripleCommutator[0] * 0.5
    return tripleCommutator

def errorOperator(oplist):
    errorOperator = []
    numTerms = len(oplist)
    for beta in range(numTerms):
        for alpha in range(beta+1):
            for gamma in range(beta):
                thisTerm = errorOperatorInterior(oplist,alpha,beta,gamma)
                errorOperator.append(thisTerm)
    reducedErrorOperator = [term for term in errorOperator if term]
    return reducedErrorOperator

def errorOperatorNorm(oplist):
    errorOp = errorOperator(oplist)
    if not isinstance(oplist[0],list):
        return oplist[0] * oplist[0].conjugate()
    norm = 0
    for term in errorOp:
        norm = norm + term[0] * term[0].conjugate()
    return norm

def errorOperatorExpectation(oplist,state):
    errorOp = errorOperator(oplist)
    opMatrix = sparseFermions.commutingOplistToMatrix(errorOp)
    expectation = state.H * opMatrix * state
    return expectation

def compareErrorsForAllPermutationsPregroup(hamiltonianOplist,pregroupOplist,listPermutations=None):
    numTerms = len(hamiltonianOplist)
    if listPermutations == None:
        listPermutations = generateOrderingLabels(numTerms)
    numPermutations = len(listPermutations)
    fullOplist = pregroupOplist + hamiltonianOplist
    eigvec = sparseFermions.getTrueEigensystem(fullOplist)[1]
    errorDict = {}
    for index,permutation in enumerate(listPermutations):
        newOplist = copy.deepcopy(hamiltonianOplist)
        newOplist = reorderOplist(newOplist,permutation)
        fullOplist = pregroupOplist + newOplist
        errorActual = sparseFermions.oneStepTrotterError(fullOplist, order=2)
        errorOprNorm = errorOperatorNorm(fullOplist)
        errorOpExpectation = errorOperatorExpectation(fullOplist,eigvec)
        errorDict[tuple(permutation)] = (errorActual,errorOprNorm,errorOpExpectation)
        print("Permutation " + str(index+1) + " of " + str(numPermutations) + " done.")
    return errorDict            
        
def compareErrorsForAllPermutationsPregroupDBG(hamiltonianOplist,pregroupOplist,listPermutations=None):
    numTerms = len(hamiltonianOplist)
    if listPermutations == None:
        listPermutations = generateOrderingLabels(numTerms)
    numPermutations = len(listPermutations)
    fullOplist = pregroupOplist + hamiltonianOplist
    eigvec = sparseFermions.getTrueEigensystem(fullOplist)[1]
    errorDict = {}
    for index,permutation in enumerate(listPermutations):
        newOplist = copy.deepcopy(hamiltonianOplist)
        newOplist = reorderOplist(newOplist,permutation)
        fullOplist = pregroupOplist + newOplist
        errorActual = sparseFermions.oneStepTrotterError(fullOplist, order=2)
        errorOprNorm = errorOperatorNorm(fullOplist)
        errorOpExpectation = errorOperatorExpectation(fullOplist,eigvec)
        errorDict[tuple(permutation)] = (errorActual,errorOprNorm,errorOpExpectation)
        print(str(errorActual),str(errorOprNorm),str(errorOpExpectation))                           


def readoutErrorsComparison(dictErrors):
    for key in dictErrors:
        thing = dictErrors[key]
        print(str(key) + ','+str(thing[0]) + ',' +str(abs(thing[1])) + ',' + str(abs(thing[2].todense()[0,0])))
    return
def compareErrorsRandomUnitary(numUnitaries,numQubits,numTerms):
    listErrors = []
    i = 0
    while i < numUnitaries:
        try:
            thisErrors = randomHamCompareOrderings(numQubits,numTerms)
        except:
            continue
        listErrors.append(thisErrors)
        print(str(datetime.datetime.now()) + "   " + str(i+1)+ " of " +str(numUnitaries) + " trials done.")
        i += 1
    return listErrors
        
def michaelHamToOplist(michaelHam):
    oplist = []
    translateDict = {}
    translateDict['Id'] = 0
    translateDict['X'] = 1
    translateDict['Y'] = 2
    translateDict['Z'] = 3
    for term in michaelHam:
        pauliString = []
        for pauli in term[1]:
            thing = translateDict[pauli]
            pauliString.append(thing)
        oplist.append([term[0],pauliString])
    return oplist

def numElectronsToHFState(numElectrons):
    return range(numElectrons)


def getHFState(exactState):
    '''input an exact ground state in canonical basis, get the HF state by finding max element'''
    pass 