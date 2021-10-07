'''
Created on 18 Dec 2014

@author: andrew
'''
import numpy
import random
import itertools
from yaferp.general import fermions, directFermions
import copy
try:
    import pygraph
    import pygraph.classes
    import pygraph.classes.graph
except:
    pass
import datetime
import math
import numpy.random
import time
import networkx
import matplotlib.pyplot as plt

def numElectronsToHFState(numElectrons):
    return list(reversed(range(numElectrons)))

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

def networkXGraph(oplist, dual=0):
    adjacencyMatrix = generateAdjacencyMatrix(oplist, dual)
    theGraph = networkx.Graph()
    for term in oplist:
        theGraph.add_node(tuple(term[1]), magnitude=term[0])

    for term1 in oplist:
        for term2 in oplist:
            if not dual and not fermions.checkCommute(term1[1], term2[1]):
                theGraph.add_edge(term1[1], term2[1])
            elif dual and fermions.checkCommute(term1[1], term2[1]):
                theGraph.add_edge(term1[1], term2[1])

    return theGraph

    # def networkXGraph(oplist,dual=0):
    #    adjacencyMatrix = generateAdjacencyMatrix(oplist,dual)
    #    graph = networkx.from_numpy_matrix(adjacencyMatrix)
    #    return graph

def greedyColourOplist(oplist, dual=0,strategy=networkx.coloring.strategy_independent_set):
    if strategy == "independent_set": #for backwards compatibility
        strategy = networkx.coloring.strategy_independent_set
    if isinstance(oplist[0][1],list):
        newOplist = [[x[0],tuple(x[1])] for x in oplist]
        oplist=newOplist

    graph = networkXGraph(oplist,dual)
    #assert strategy == 'independent_set'
    thing = networkx.algorithms.coloring.greedy_color(graph, strategy=networkx.coloring.strategy_independent_set)
    numColours = max(thing.values()) + 1
    listOplists = []
    for i in range(numColours):
        thisOplist = [term for term in oplist if thing[term[1]] == i]
        listOplists.append(thisOplist)

    return listOplists
    #
    # print(listOplists)
    # for term in oplist:
    #   thisColour = thing[term[1]]
    #   print(thisColour)
    #   print(term)
    #   print(listOplists[thisColour])
    #   print('\n\n\n')
    ##   print(listOplists)
    #   listOplists[thisColour] = listOplists[thisColour].append(term)

    return listOplists


def plotNetworkXGraph(graph):
    plt.cla()
    # plt.subplot(121)
    networkx.draw_circular(graph)  # ,pos=networkx.spectral_layout(graph))
    # plt.subplot(122)
    # nx.draw(G, pos=nx.circular_layout(G), nodecolor='r', edge_color='b')
    return


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
        numSteps = sparseFermions2.trotterStepsUntilPrecision(newOplist)
        errorDict[tuple(permutation)] = numSteps
    return errorDict



def trotterErrorForAllPermutations(hamiltonianOplist,listPermutations=None,verbose=0):
    thisEigenvalue,thisEigenvec = directFermions.getTrueEigensystem(hamiltonianOplist)
    numTerms = len(hamiltonianOplist)
    if listPermutations == None:
        listPermutations = generateOrderingLabels(numTerms)
    errorDict = {}
    numPerms = len(listPermutations)
    permsCounter = 1
    for permutation in listPermutations:
        newOplist = copy.deepcopy(hamiltonianOplist)
        newOplist = reorderOplist(newOplist,permutation)
        error = directFermions.oneStepTrotterError(newOplist, eigenvalue=thisEigenvalue, eigenvec=thisEigenvec)
        errorDict[tuple(permutation)] = error
        if verbose:
            print(str(datetime.datetime.now()) + ' Permutation ' + str(permsCounter) + ' of ' + str(numPerms) + ' completed.')
            permsCounter += 1
    return errorDict

def findBestOrdering(hamiltonianOplist):
    '''ROUGH'''
    orderings = trotterErrorForAllPermutations(hamiltonianOplist)
    vals = list(orderings.values())
    keys = list(orderings.keys())
    return(keys[vals.index(min(vals))])

def findGoodOrderings(hamiltonianOplist):
    results = []
    TOLERANCE = 0.0000001
    orderings = trotterErrorForAllPermutations(hamiltonianOplist)
    vals = list(orderings.values())
    keys = list(orderings.keys())
    lowestVal = min(vals)
    for index,thisVal in enumerate(vals):
        if abs(thisVal-lowestVal) < TOLERANCE:
            results.append(keys[index])
    return results
            
            
    
def trotterErrorForAllPermutationsPregroup(hamiltonianOplist,pregroupOplist,listPermutations=None):
    numTerms = len(hamiltonianOplist)
    if listPermutations == None:
        listPermutations = generateOrderingLabels(numTerms)
    errorDict = {}
    for permutation in listPermutations:
        newOplist = copy.deepcopy(hamiltonianOplist)
        newOplist = reorderOplist(newOplist,permutation)
        fullOplist = pregroupOplist + newOplist
        error = sparseFermions2.oneStepTrotterError(fullOplist)
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
    greedyError = sparseFermions2.oneStepTrotterError(greedyHam)
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
    orderedOplist.sort(key=lambda item: list(reversed(item[1])))
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

def compareOrderingSchemes(oplist,includeTestStates=0):
    errors = {}
    orderingErrors = trotterErrorForAllPermutations(oplist)
    if includeTestStates:
        eigenval, eigenstate = directFermions.getTrueEigensystem(oplist)
    errors["Minimum"] = minimumOrderingError(orderingErrors)
    errors["Maximum"] = maximumOrderingError(orderingErrors)
    errors["Average"] = averageOrderingError(orderingErrors)
    errors["Lexiographic"] = orderingErrors[getLexiographicOrdering(oplist)]
    errors["Magnitude"] = orderingErrors[getMagnitudeOrdering(oplist)]
    errors["Commutator"] = orderingErrors[getCommutatorOrdering(oplist)]
    errors["Counting"] = orderingErrors[getCountingOrdering(oplist)]
    if includeTestStates:
        errors["TeststateExact"] = orderingErrors[getCommutatorTestStateOrdering(oplist,eigenstate)]
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
        
def randomOrdering(numTerms):
    ordering = range(numTerms)
    random.shuffle(ordering)
    return tuple(ordering)

def hartreeFockEnergyEfficient(diagOplist,hfState):
    hfEnergy = 0
    for term in diagOplist:
        coefficient = term[0]
        pauliString = term[1]
        zIndices = [len(pauliString)-x-1 for x in range(len(pauliString)) if list(pauliString)[x] == 3]
        intersect = intersection(zIndices,list(hfState))
        isOdd = len(intersect)%2
        if isOdd:
            hfEnergy -= coefficient
        else:
            hfEnergy += coefficient
    return hfEnergy
        
def compareOrderingSchemesLarge(oplist,trotterOrder,eigenvector=None,eigenvalue=None,importances=None,verbose=0):
    if eigenvalue == None or eigenvector == None:
        eigenvalue,eigenvector = sparseFermions2.getTrueEigensystem(oplist)
    errors = {}
    lexiographicOrdering = getLexiographicOrdering(oplist)
    magnitudeOrdering = getMagnitudeOrdering(oplist)
    if verbose:
        print('done non-com orderings')
    commutatorOrdering = getCommutatorOrdering(oplist)
    countingOrdering = getCountingOrdering(oplist)
    if verbose:
        print('done comm ordering')
    commutatorPeturbativeOrdering = getCommutatorPeturbativeOrdering(oplist,importances)
    countingPeturbativeOrdering = getCountingPeturbativeOrdering(oplist,importances)
    if verbose:
        print('done ordering, calcultatin error')
    lexiographicOplist = reorderOplist(oplist,lexiographicOrdering)
    magnitudeOplist = reorderOplist(oplist,magnitudeOrdering)
    commutatorOplist = reorderOplist(oplist,commutatorOrdering)
    commutatorPeturbativeOplist = reorderOplist(oplist,commutatorPeturbativeOrdering)
    countingOplist = reorderOplist(oplist,countingOrdering)
    countingPeturbativeOplist = reorderOplist(oplist,countingPeturbativeOrdering)
    lexiographicError = directFermions.oneStepTrotterError(lexiographicOplist, order=trotterOrder)
    magnitudeError = directFermions.oneStepTrotterError(magnitudeOplist, order=trotterOrder)
    commutatorError = directFermions.oneStepTrotterError(commutatorOplist, order=trotterOrder)
    commutatorPeturbativeError = directFermions.oneStepTrotterError(commutatorPeturbativeOplist, order=trotterOrder)
    countingError = directFermions.oneStepTrotterError(countingOplist, order=trotterOrder)
    countingPeturbativeError = directFermions.oneStepTrotterError(countingPeturbativeOplist, order=trotterOrder)
    errors["Lexiographic"] = lexiographicError
    errors["Magnitude"] = magnitudeError
    errors["Commutator"] = commutatorError
    errors["CommutatorPeturbative"] = commutatorPeturbativeError
    errors["Counting"] = countingError
    errors["CountingPeturbative"] = countingPeturbativeError
    return errors
def compareOrderingSchemesSample(oplist,trotterOrder,timeLimit,evalue=None,evec=None,skipErrors=0):
    errors = {}
    if evalue == None or evec == None:
        try:
            eigval, eigvec = directFermions.getTrueEigensystem(oplist)
        except:
            print("Couldn't calculate eigensystem of hamiltonian") 
            return -1
    else:
        eigval = evalue
        eigvec = evec
    startTime = time.clock()
    lexiographicOrdering = getLexiographicOrdering(oplist)
    lexiographicOplist = reorderOplist(oplist,lexiographicOrdering)
    lexiographicError = directFermions.oneStepTrotterError(lexiographicOplist, order=trotterOrder, eigenvalue=eigval, eigenvec=eigvec)
    errors["Lexiographic"] = lexiographicError
    endTime = time.clock()
    timeForOne = endTime-startTime
    magnitudeOrdering = getMagnitudeOrdering(oplist)
    commutatorOrdering = getCommutatorOrdering(oplist)
    countingOrdering = getCountingOrdering(oplist)
    magnitudeOplist = reorderOplist(oplist,magnitudeOrdering)
    commutatorOplist = reorderOplist(oplist,commutatorOrdering)
    countingOplist = reorderOplist(oplist,countingOrdering)
    magnitudeError = directFermions.oneStepTrotterError(magnitudeOplist, order=trotterOrder, eigenvalue=eigval, eigenvec=eigvec)
    errors["Magnitude"] = magnitudeError
    commutatorError = directFermions.oneStepTrotterError(commutatorOplist, order=trotterOrder, eigenvalue=eigval, eigenvec=eigvec)
    errors["Commutator"] = commutatorError
    countingError = directFermions.oneStepTrotterError(countingOplist, order=trotterOrder, eigenvalue=eigval, eigenvec=eigvec)
    errors["Counting"] = countingError
    timeLimit = timeLimit - 3*timeForOne
    if timeLimit > 0:
        averageError,fractionalOrderingsTried,sampleErrorsDict = randomSampleOrderings(oplist,trotterOrder,timeLimit,timeForOne,eigval,eigvec,skipErrors)
        if averageError != None:
            errors["Average"] = averageError
            errors["Average Sample Size"] = fractionalOrderingsTried
            errors["Trials"] = sampleErrorsDict
    return errors
    
    
def randomSampleOrderings(oplist,trotterOrder,timeLimit,timeForOne=-1,eigval=None,eigvec=None,skipErrors=1):
    possibleOrderings = math.factorial(len(oplist))
    numTerms = len(oplist)
    if eigval == None or eigvec == None:
        try:
            eigval, eigvec = directFermions.getTrueEigensystem(oplist)
        except:
            print("Couldn't calculate eigensystem of hamiltonian") 
    errorsDict = {}
    if timeForOne == -1:
        startTime = time.clock()
        firstOrdering = randomOrdering(numTerms)
        firstOplist = reorderOplist(oplist,firstOrdering)
        firstError = directFermions.oneStepTrotterError(firstOplist, order=trotterOrder, eigenvalue=eigval, eigenvec=eigvec)
        endTime = time.clock()
        timeForOne = endTime-startTime  
        numOrderingsToTry = int(math.floor(timeLimit/timeForOne)) - 1
        errorsDict[firstOrdering] = firstError
        sumErrors = firstError
    else:
        numOrderingsToTry = int(math.floor(timeLimit/timeForOne))
        sumErrors = 0
        
    for i in range(numOrderingsToTry):
        if skipErrors:
            try:
                thisOrdering = randomOrdering(numTerms)
                thisOplist = reorderOplist(oplist,thisOrdering)
                thisError = directFermions.oneStepTrotterError(thisOplist, order=trotterOrder, eigenvalue=eigval, eigenvec=eigvec)
                errorsDict[thisOrdering] = thisError
                sumErrors += thisError
            except:
                pass
        else:
            thisOrdering = randomOrdering(numTerms)
            thisOplist = reorderOplist(oplist,thisOrdering)
            thisError = directFermions.oneStepTrotterError(thisOplist, order=trotterOrder, eigenvalue=eigval, eigenvec=eigvec)
            errorsDict[thisOrdering] = thisError
            sumErrors += thisError
            
    numTrials = len(errorsDict)
    if not numTrials:
        return (None,None,None)
    averageError = sumErrors/float(numTrials)
    return (averageError,numTrials,errorsDict)

def countingOrderingNextTermOptimised(oplist,adjacencyMatrix,ordering=(),tiebreakMetric = 'magnitude', testState = None, importances=None, listNumNonCommuting=None):
    numTerms = len(oplist)
    if isinstance(ordering,int):
        ordering = [ordering]
    unsortedIndices = [x for x in range(numTerms) if x not in ordering]
    maximumPossibleCommutativity = min([listNumNonCommuting[x] for x in unsortedIndices])
    candidateIndices =[x for x in unsortedIndices if listNumNonCommuting[x] == maximumPossibleCommutativity]
    topScore = 0.
    bestIndex = None
    for i in candidateIndices:
        if tiebreakMetric == 'magnitude' and abs(oplist[i][0]) > abs(topScore):
            bestIndex = i
            topScore = oplist[i][0]
        elif tiebreakMetric =='peturbative':
            thisImportance = importances[i]
            if abs(thisImportance) > abs(topScore):
                bestIndex = i
                topScore = thisImportance
                
    for i in unsortedIndices:
        listNumNonCommuting[i] -= adjacencyMatrix[i,bestIndex]
    return bestIndex,listNumNonCommuting

def getCountingOrdering(oplist,importances=None,verbose=0):
    if verbose:
        print('start comm order')
    adjacencyMatrix = generateAdjacencyMatrix(oplist,0)
    if verbose:
        print('adj mat done')
    listCountNonCommuting = numpy.sum(adjacencyMatrix,axis=0).T
    #print(listCountNonCommuting)
    ordering2 = []
    for i in range(len(oplist)):
        if verbose:
            print('done a thing' + str(i))
            print(str(i))
        newIndex,listCountNonCommuting = countingOrderingNextTermOptimised(oplist,adjacencyMatrix,ordering=ordering2,listNumNonCommuting=listCountNonCommuting)
        ordering2.append(newIndex)
    return tuple(ordering2)

def getCountingPeturbativeOrdering(oplist,importancez = None):
    adjacencyMatrix = generateAdjacencyMatrix(oplist,0)
    ordering2 = []
    listCountNonCommuting = numpy.sum(adjacencyMatrix,axis=0).T
    for i in range(len(oplist)):
        newIndex,listCountNonCommuting = countingOrderingNextTermOptimised(oplist,adjacencyMatrix,ordering=ordering2,tiebreakMetric = 'peturbative',importances=importancez,listNumNonCommuting=listCountNonCommuting)
        ordering2.append(newIndex)
    return tuple(ordering2)







def commutatorOrderingNextTermOptimised(oplist,adjacencyMatrix,ordering=(),tiebreakMetric = 'magnitude', testState = None, importances=None, listNumNonCommuting=None):
    numTerms = len(oplist)
    if isinstance(ordering,int):
        ordering = [ordering]
    unsortedIndices = [x for x in range(numTerms) if x not in ordering]
    maximumPossibleCommutativity = max([listNumNonCommuting[x] for x in unsortedIndices])
    candidateIndices =[x for x in unsortedIndices if listNumNonCommuting[x] == maximumPossibleCommutativity]
    topScore = 0.
    bestIndex = None
    for i in candidateIndices:
        if tiebreakMetric == 'magnitude' and abs(oplist[i][0]) > abs(topScore):
            bestIndex = i
            topScore = oplist[i][0]
        elif tiebreakMetric =='peturbative':
            thisImportance = importances[i]
            if abs(thisImportance) > abs(topScore):
                bestIndex = i
                topScore = thisImportance
                
    for i in unsortedIndices:
        listNumNonCommuting[i] += adjacencyMatrix[i,bestIndex]
    return bestIndex,listNumNonCommuting
        
    
    
#@profile
def commutatorOrderingFindNextTerm(oplist,adjacencyMatrix,ordering=(),tiebreakMetric = 'magnitude', testState = None,importances = None, listNumNonCommuting = None):
    numTerms = len(oplist)
    if isinstance(ordering,int):
        ordering = [ordering]
    unsortedIndices = [x for x in range(numTerms) if x not in ordering]
    #unsortedImportances = [importances[x] for x in unsortedIndices]
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
        elif numNonCommuting == recordNonCommuting:
            if tiebreakMetric == 'magnitude' and abs(thisCoefficient) > abs(recordCoefficient):
                currentPickIndex = unsortedIndex
                recordCoefficient = thisCoefficient
            elif tiebreakMetric == 'teststate':
                expectation = testState.H * directFermions.applyPauliStringToState(thisTerm, testState)
                if abs(expectation) > abs(recordCoefficient):
                    currentPickIndex = unsortedIndex
                    recordCoefficient = expectation
            elif tiebreakMetric =='peturbative':
                thisImportance = importances[unsortedIndex]
                if abs(thisImportance) > abs(recordCoefficient):
                    currentPickIndex = unsortedIndex
                    recordCoefficient = thisImportance
                
    return currentPickIndex

        

def getCommutatorOrdering(oplist,importances=None,verbose=0):
    if verbose:
        print('start comm order')
    adjacencyMatrix = generateAdjacencyMatrix(oplist,0)
    if verbose:
        print('adj mat done')
    listCountNonCommuting = [0]*len(oplist)
    ordering2 = []
    for i in range(len(oplist)):
        if verbose:
            print('done a thing' + str(i))
            print(str(i))
        newIndex,listCountNonCommuting = commutatorOrderingNextTermOptimised(oplist,adjacencyMatrix,ordering=ordering2,listNumNonCommuting=listCountNonCommuting)
        ordering2.append(newIndex)
    return tuple(ordering2)

def getCommutatorPeturbativeOrdering(oplist,importancez = None):
    adjacencyMatrix = generateAdjacencyMatrix(oplist,0)
    ordering2 = []
    listCountNonCommuting = [0]*len(oplist)
    for i in range(len(oplist)):
        newIndex,listCountNonCommuting = commutatorOrderingNextTermOptimised(oplist,adjacencyMatrix,ordering=ordering2,tiebreakMetric = 'peturbative',importances=importancez,listNumNonCommuting=listCountNonCommuting)
        ordering2.append(newIndex)
    return tuple(ordering2)
 
def getCommutatorTestStateOrdering(oplist,testState):
    adjacencyMatrix = generateAdjacencyMatrix(oplist,0)
    ordering = []
    for i in range(len(oplist)):
        newIndex = commutatorOrderingFindNextTerm(oplist,adjacencyMatrix,ordering,'teststate',testState)
        ordering.append(newIndex)
    return tuple(ordering)        

def randomHamCompareOrderings(numQubits,numTerms,doTestState=0):
    ham = reliableRandomOplist(numQubits,numTerms)
    return compareOrderingSchemes(ham,doTestState)

def reliableRandomOplist(numQubits,numTerms):
    ham = randomOplist(numQubits,numTerms)
    hamiltonianWorks = False
    while not hamiltonianWorks:
        hamiltonianWorks = True
        try:
            eigenvalue = directFermions.getTrueEigensystem(ham)[0]
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
                eigenvalue = sparseFermions2.getTrueEigensystem(ham)[0]
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
    opMatrix = sparseFermions2.commutingOplistToMatrix(errorOp)
    expectation = state.H * opMatrix * state
    return expectation

def compareErrorsForAllPermutationsPregroup(hamiltonianOplist,pregroupOplist,listPermutations=None):
    numTerms = len(hamiltonianOplist)
    if listPermutations == None:
        listPermutations = generateOrderingLabels(numTerms)
    numPermutations = len(listPermutations)
    fullOplist = pregroupOplist + hamiltonianOplist
    eigvec = sparseFermions2.getTrueEigensystem(fullOplist)[1]
    errorDict = {}
    for index,permutation in enumerate(listPermutations):
        newOplist = copy.deepcopy(hamiltonianOplist)
        newOplist = reorderOplist(newOplist,permutation)
        fullOplist = pregroupOplist + newOplist
        errorActual = sparseFermions2.oneStepTrotterError(fullOplist,order=2)
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
    eigvec = sparseFermions2.getTrueEigensystem(fullOplist)[1]
    errorDict = {}
    for index,permutation in enumerate(listPermutations):
        newOplist = copy.deepcopy(hamiltonianOplist)
        newOplist = reorderOplist(newOplist,permutation)
        fullOplist = pregroupOplist + newOplist
        errorActual = sparseFermions2.oneStepTrotterError(fullOplist,order=2)
        errorOprNorm = errorOperatorNorm(fullOplist)
        errorOpExpectation = errorOperatorExpectation(fullOplist,eigvec)
        errorDict[tuple(permutation)] = (errorActual,errorOprNorm,errorOpExpectation)
        print(str(errorActual),str(errorOprNorm),str(errorOpExpectation))                           


def readoutErrorsComparison(dictErrors):
    for key in dictErrors:
        thing = dictErrors[key]
        print(str(key) + ','+str(thing[0]) + ',' +str(abs(thing[1])) + ',' + str(abs(thing[2].todense()[0,0])))
    return
def compareErrorsRandomUnitary(numUnitaries,numQubits,numTerms,includeTestStates=0):
    listErrors = []
    i = 0
    while i < numUnitaries:
        try:
            thisErrors = randomHamCompareOrderings(numQubits,numTerms,includeTestStates)
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

def support(term):
    pauliString = term[1]
    numQubits = len(pauliString)
    support = []
    for index, whichPauli in enumerate(pauliString):
        if whichPauli != 0:
            support.append(numQubits-index-1)
    return support

def supportXY(term):
    pauliString = term[1]
    numQubits = len(pauliString)
    support = []
    for index,whichPauli in enumerate(pauliString):
        if whichPauli == 1 or whichPauli == 2:
            support.append(numQubits-index-1) 
    return support

def intersection(list1,list2):
    return [term for term in list1 if term in list2]

def calcOmega(hamDiag, offDiagonalTerm,hfState):
    '''note hf state is input as a list of which qubits are in the 1 state - so |5> is [2,0]'''
    offDiagonalSupport = supportXY(offDiagonalTerm)
    total=0
    for term in hamDiag:
        diagonalSupport = support(term)
        intersect1 = intersection(diagonalSupport,offDiagonalSupport)
        intersect2 = intersection(hfState,diagonalSupport)
        boolNegate1 = len(intersect1)%2
        boolNegate2 = len(intersect2)%2
        boolNegate = (boolNegate1+boolNegate2)%2
        total += (-1)**boolNegate * term[0]
    return total

def calcImportance(diagOplist,term,hfState,hfEnergy):
    if checkTermDiagonal(term):
        return abs(term[0])
    else:
        denom = hfEnergy - calcOmega(diagOplist,term,hfState)
        coeff = abs(term[0])**2
        return abs(coeff/denom)

def calcOplistImportances(oplist,hfState,hfEnergy):
    diagOplist,nonDiagOplist = separateOplist(oplist)
    importances = []
    for term in oplist:
        importance = calcImportance(diagOplist,term,hfState,hfEnergy)
        importances.append(importance)
    return importances
        
def calcTrueImportances(oplist,whichTerm):
    trueEigenval = directFermions.getTrueEigensystem(oplist)[0]
    importances = []
    for i in range(len(oplist)):
        newOplist = copy.deepcopy(oplist)
        newOplist.pop(i)
        newEigenval = directFermions.getTrueEigensystem(newOplist)[0]
        diff = abs(trueEigenval-newEigenval)
        importances.append(diff)
    return importances

def separateOplist(oplist):
    '''separate an oplist into diagonal and offdiagonal sets of terms'''
    diagonalOplist = []
    offDiagonalOplist = []
    if not isinstance(oplist[0],list):
        if checkTermDiagonal(oplist):
            return([oplist],[])
        else:
            return ([],[oplist])
    for term in oplist:
        if checkTermDiagonal(term):
            diagonalOplist.append(term)
        else:
            offDiagonalOplist.append(term)
    return (diagonalOplist,offDiagonalOplist)
        

def checkTermDiagonal(term):
    '''check to see if a term is diagonal (i.e. contains only pauli Zs and Is)
    return 0 if false, 1 if true'''
    pauliString = term[1]
    for gate in pauliString:
        if gate == 1 or gate == 2:
            return 0
    return 1

def hartreefockRandomOplist(numQubits,numTerms,threshold=0.95):
    '''an attempt to generate a random oplist w/ a close number state approx. to ground state.
    doesn't work, in retrospect for obvious reasons.'''
    ham = randomOplist(numQubits,numTerms)
    hamiltonianWorks = False
    while not hamiltonianWorks:
        hamiltonianWorks = True
        try:
            eigenvalue,eigenvector = directFermions.getTrueEigensystem(ham)[0]
            maxCoefficient = abs(max(eigenvector.todense()))
            if maxCoefficient < threshold:
                hamiltonianWorks = False
                ham = randomOplist(numQubits,numTerms)
        except:
            hamiltonianWorks = False
            ham = randomOplist(numQubits,numTerms)
    return ham
def numberStateToNumber(onQubits):
    thing = 0
    if isinstance(onQubits,tuple):
        onQubits = list(onQubits)
    if not isinstance(onQubits,list):
        thing = 2**onQubits
    else:
        for thing2 in onQubits:
            thing += 2**thing2
    return thing
def numberStateToVector(onQubits,numQubits):
    state = numpy.mat(numpy.zeros((2**numQubits,1)))
    index = 0
    if isinstance(onQubits,tuple):
        onQubits = list(onQubits)
    if not isinstance(onQubits,list):
        index = 2**onQubits
    else:
        for thing in onQubits:
            index += 2**thing
    state[index,0] = 1.
    return state

#ham=reliableRandomOplist(6,150)
#getCommutatorOrdering(ham)   

def lexiographic2(oplist):
    ''' orderedOplist = copy.deepcopy(oplist)
    orderedOplist.sort(key=lambda item:reversed(item[1]))'''
    compressedOplist = []
    for index,term in enumerate(oplist):
        decRep = 0
        for index2, i in enumerate(term[1]):
            decRep += i * (5**index2) #hackish af
        compressedOplist.append([index,decRep])
    compressedOplist.sort(key=lambda item:item[1])
    newOplist = []
    for term in compressedOplist:
        newOplist.append(oplist[term[0]])
    return newOplist

def lexiographic(oplist):
    '''WARNING:  IN-PLACE
    no idea why i didn't do one this in the first place.
    it's even commented out above the damn function above. works perfectly fine.
    '''
    oplist.sort(key=lambda item:list(reversed(item[1])))
    return oplist

def reverseLexiographic(oplist):
    oplist.sort(key=lambda item:item[1])
    return oplist
            
        
        
    '''
    newOplist = []
    for term in oplist:
        newOplist.append([term[0],list(reversed(term[1]))])
    newOplist.sort(key=lambda item:item[1])
    newNewOplist = []
    for term in newOplist:
        newNewOplist.append([term[0],list(reversed(term[1]))])
    return newNewOplist'''


def magnitude(oplist):
    oplist2 = copy.deepcopy(oplist)
    oplist2.sort(key=lambda item:(numpy.absolute(item[0])),reverse=True)
    return oplist2
'''
def lexoMagOLD(oplist):
    oplistMag = copy.deepcopy(oplist)
    oplistMag = magnitude(oplistMag)
    #print(len(oplistMag))
    oplistLexy = copy.deepcopy(oplistMag)
    oplistLexy = lexiographic(oplistLexy)
    
    magIterator = iter(oplistMag)
    lexyIterator = iter(oplistLexy)
    #print(len(oplistLexy))
    oplist2 = []
    try:
        while True:
            check=0
            while check==0 :
                nextMag = next(magIterator)
                if not (nextMag in oplist2):
                    oplist2.append(nextMag)
                    check += 1
                
            check=0
            while check==0:
                nextLex = next(lexyIterator)
                if not (nextLex in oplist2):
                    oplist2.append(nextLex)
                    check += 1
    except StopIteration:
        pass

    for x in magIterator:
        if not (x in oplist2):
            oplist2.append(x)
    for x in lexyIterator:
        if not (x in oplist2):
            oplist2.append(x)
    
    if len(oplist) != len(oplist2):
        print("oops lol")
    return oplist2
'''
def lexoMag(oplist):

    oplistMag = copy.deepcopy(oplist)
    oplistMag = magnitude(oplistMag)
    #print(len(oplistMag))
    oplistLexy = copy.deepcopy(oplistMag)
    oplistLexy = lexiographic(oplistLexy)
    
    magIterator = iter(oplistMag)
    lexyIterator = iter(oplistLexy)
    #print(len(oplistLexy))
    oplist2 = []
    seenTerms = {}
    try:
        while True:
            while True :
                nextMag = next(magIterator)
                if tuple(nextMag[1]) in seenTerms:continue
                oplist2.append(nextMag)
                seenTerms[tuple(nextMag[1])] = 1
                break
                
            while True:
                nextLex = next(lexyIterator)
                if tuple(nextLex[1]) in seenTerms:continue
                oplist2.append(nextLex)
                seenTerms[tuple(nextLex[1])] = 1
                break
                
    except StopIteration:
        pass

    for x in magIterator:
        if tuple(x[1]) in seenTerms: continue
        oplist2.append(x)
        seenTerms[tuple(x[1])] = 1
    for x in lexyIterator:
        if tuple(x[1]) in seenTerms: continue
        oplist2.append(x)
        seenTerms[tuple(x[1])] = 1
    
    if len(oplist) != len(oplist2):
        print("oops lol")
    return oplist2

def inverseLexoMag(oplist):
    '''yeah, i factored this terribly'''
    oplistMag = copy.deepcopy(oplist)
    oplistMag = list(reversed(magnitude(oplistMag)))
    #print(len(oplistMag))
    oplistLexy = copy.deepcopy(oplistMag)
    oplistLexy = lexiographic(oplistLexy)
    
    magIterator = iter(oplistMag)
    lexyIterator = iter(oplistLexy)
    #print(len(oplistLexy))
    oplist2 = []
    seenTerms = {}
    try:
        while True:
            while True :
                nextMag = next(magIterator)
                if tuple(nextMag[1]) in seenTerms:continue
                oplist2.append(nextMag)
                seenTerms[tuple(nextMag[1])] = 1
                break
                
            while True:
                nextLex = next(lexyIterator)
                if tuple(nextLex[1]) in seenTerms:continue
                oplist2.append(nextLex)
                seenTerms[tuple(nextLex[1])] = 1
                break
    except StopIteration:
        pass

    for x in magIterator:
        if tuple(x[1]) in seenTerms: continue
        oplist2.append(x)
        seenTerms[tuple(x[1])] = 1
    for x in lexyIterator:
        if tuple(x[1]) in seenTerms: continue
        oplist2.append(x)
        seenTerms[tuple(x[1])] = 1
    
    if len(oplist) != len(oplist2):
        print("oops lol")
    return oplist2
    
    '''
    while True:
        try:
            magIndex,nextMag = next(term for term in enumerate(oplistMag[magIndex:]) if not (term[1] in oplist2))
            magIndex += 1
            print(nextMag)
        except StopIteration:
            boolMagFinished = 1
            break
        oplist2.append(nextMag)
        
        try:
            lexIndex,nextLex = next(term for term in enumerate(oplistLexy[lexIndex:]) if not (term[1] in oplist2))
            lexIndex += 1
            print(nextLex)
        except StopIteration:
            boolLexFinished = 1
            break
        oplist2.append(nextLex)
 
    if not boolMagFinished:
        if not (oplistMag[magIndex:] == []):
            oplist2 += oplistMag[magIndex:]
        
    if not boolLexFinished:
        if not (oplistLexy[lexIndex:] == []):
            oplist2 += oplistLexy[lexIndex:]
    '''
        
    if len(oplist) != len(oplist2):
        print("oops lol")
    return oplist2
                
    '''     
    while True: #yolo
        if oplistMag == [] or oplistLexy == []:
            break
        oplistMagTerm = oplistMag.pop(0)
        oplistLexyTerm = oplistLexy.pop(0)
        
        while oplistMagTerm in oplist2:
            if oplistMag == []:
                oplistMagTerm = -1
                break
            oplistMagTerm = oplistMag.pop(0)
        if oplistMagTerm == -1:
            break
        oplist2.append(oplistMagTerm)
        
        
        while oplistLexyTerm in oplist2:
            if oplistLexy == []:
                oplistLexyTerm = -1
                break
            oplistLexyTerm = oplistLexy.pop(0)
        if oplistLexyTerm == -1:
            break
        oplist2.append(oplistLexyTerm)
    
    if oplistMag != []:
        for term in oplistMag:
            if term not in oplist2:
                oplist2.append(term)
    if oplistLexy != []:
        for term in oplistLexy:
            if term not in oplist2:
                oplist2.append(term)
    
    if len(oplist) != len(oplist2):
        print("oops lol")
        return
    else:
        return oplist2

def inverseLexoMag(oplist):
    oplistMag2 = magnitude(oplist)
    oplistMag = list(reversed(oplistMag2))
    oplistLexy = lexiographic(oplist)
    oplist2 = []
    while True: #yolo
        if oplistMag == [] or oplistLexy == []:
            break
        oplistMagTerm = oplistMag.pop(0)
        oplistLexyTerm = oplistLexy.pop(0)
        
        while oplistMagTerm in oplist2:
            if oplistMag == []:
                oplistMagTerm = -1
                break
            oplistMagTerm = oplistMag.pop(0)
        if oplistMagTerm == -1:
            break
        oplist2.append(oplistMagTerm)
        
        
        while oplistLexyTerm in oplist2:
            if oplistLexy == []:
                oplistLexyTerm = -1
                break
            oplistLexyTerm = oplistLexy.pop(0)
        if oplistLexyTerm == -1:
            break
        oplist2.append(oplistLexyTerm)
    
    if oplistMag != []:
        for term in oplistMag:
            if term not in oplist2:
                oplist2.append(term)
    if oplistLexy != []:
        for term in oplistLexy:
            if term not in oplist2:
                oplist2.append(term)
    
    if len(oplist) != len(oplist2):
        print(oplist2)
        return
    else:
        return oplist2'''

def randomOrd(oplist):
    unsorted = range(len(oplist))
    sortedOplist = []
    while unsorted != []:
        thisIndex = random.randint(0,len(unsorted)-1)
        bob = unsorted.pop(thisIndex)
        sortedOplist.append(oplist[bob])
    return sortedOplist

def commu(oplist):
    commutatorOrdering = getCommutatorOrdering(oplist)
    commutatorOplist = reorderOplist(oplist, commutatorOrdering)
    return commutatorOplist

def reverseCommu(oplist):
    commutatorOrdering = getCountingOrdering(oplist)
    commutatorOplist = reorderOplist(oplist, commutatorOrdering)
    return commutatorOplist
