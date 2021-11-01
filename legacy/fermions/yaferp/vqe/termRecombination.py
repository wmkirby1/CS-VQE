import numpy
from yaferp.circuits import circuit
import copy
import collections
from yaferp.general import directFermions
from yaferp.vqe import vqe
import random
try:
    import cPickle
except:
    import pickle as cPickle
#import cPickle

def pauliProduct(leftPauli,rightPauli):
    index = leftPauli*4 + rightPauli
    chart = [(1.,0), #ii
             (1.,1), #ix
             (1.,2), #iy
             (1.,3), #iz
             (1.,1), #xi
             (1.,0), #xx
             (1.j,3), #xy
             (-1.j,2), #xz
             (1.,2), #yi
             (-1.j,3),#yx
             (1.,0), #yy
             (1.j,1), #yz
             (1.,3), #zi
             (1.j,2), #zx
             (-1.j,1), #zy
             (1., 0)] #zz
    return(chart[index])



def xOperator(leftPauliString,rightPauliString):
    coefficient = 1.j
    resultPauliString = []
    for i in range(len(leftPauliString)):
        thisThing = pauliProduct(leftPauliString[i],rightPauliString[i])
        coefficient = coefficient * thisThing[0]
        resultPauliString.append(thisThing[1])
    return [coefficient,resultPauliString]

    #terms = [pauliProduct(leftPauliString[i],rightPauliString[i]) for i in range(len(leftPauliString))]
    #individualCoefficients,resultPauliString = zip(*terms)


def normaliseOplist(oplist):
    squaredCoefficients = [(abs(x[0]))**2 for x in oplist]
    gamma = ((sum(squaredCoefficients))**(1./2.))
    newOplist = [[x[0]/gamma,x[1]] for x in oplist]
    return gamma,newOplist

def fixRange(angle):
    if angle < 0.:
        return (2.*numpy.pi) + angle
    else:
        return angle

def thetasFromOplist(normalisedOplist):
    betas = [x[0] for x in normalisedOplist]
    squaredBetas = [x**2 for x in betas]

    runningTotal = squaredBetas[-1]
    squaredBetaSums = [runningTotal]
    for i in range(1,len(normalisedOplist)-1):
        runningTotal += squaredBetas[i-1]
        squaredBetaSums.append(runningTotal)

    l2Betas = [x**(1./2.) for x in squaredBetaSums]
    l2Betas[0] = betas[-1]
    thetas = [numpy.arctan(betas[i]/l2Betas[i]) for i in range(len(l2Betas))]
    if betas[-1].real < 0.:
        thetas[0] = thetas[0] + numpy.pi
    return thetas

def xOperatorsOplist(oplist):
    finalPauliString = oplist[-1][1]
    result = [xOperator(oplist[i][1],finalPauliString) for i in range(len(oplist)-1)]
    return result

def correctTheta(theta): #switch domain & divide by 2i as code does rotation of -2j theta
    if theta < 0.:
        #return -(2.*numpy.pi + theta)/2.
        return -theta/2.
    else:
        return -theta/2.

def singleTermAdjointHHCircuit(theta,xOperator):
    correctedTheta = correctTheta(theta)
    thetaX = [xOperator[0]*correctedTheta,xOperator[1]]
    return circuit.oplistToCircuit([thetaX])

def changeBasisCircuitForMeasurement(finalPauliString):
    nonIdentityPaulis = [(index,value) for (index,value) in enumerate(reversed(finalPauliString)) if value != 0]
    sortedNonIdentityPaulis = sorted(nonIdentityPaulis, key=lambda thing: thing[0])
    basisChangeCircuit = circuit.changeBasisGates(sortedNonIdentityPaulis)
    return basisChangeCircuit


def putTermWithMostIdentitiesAtEnd(oplist):
    '''in principle we want the final term to have the most similarity to other terms
    -  if the terms have linear scaling of support (JW) or better (BK) then we expect
    the best strategy to be to find the term that has the most identities in it, possibly.'''
    newOplist = copy.deepcopy(oplist)
    numIdentities = [len([x for x in y[1] if x == 0]) for y in newOplist]
    maxNumIdentites = max(numIdentities)
    identitiesIndex = numIdentities.index(maxNumIdentites)
    newOplist.append(newOplist.pop(identitiesIndex))
    return newOplist


def anticommutingSetToCircuit(oplist):
    result = circuit.Circuit([])
    if len(oplist) == 1:
        result.addRight(changeBasisCircuitForMeasurement(oplist[0][1]))
        indicesToMeasure = [i for i,x in enumerate(reversed(oplist[-1][1])) if x!= 0]
        gamma = oplist[0][0]
        return gamma,result, indicesToMeasure


    reorderedOplist = putTermWithMostIdentitiesAtEnd(oplist)

    gamma,normalisedOplist = normaliseOplist(reorderedOplist)
    thetas = thetasFromOplist(normalisedOplist)
    xOperators = xOperatorsOplist(normalisedOplist)
    for i in range(len(thetas)):
        thisCircuit = singleTermAdjointHHCircuit(thetas[i],xOperators[i])
        result.addRight(thisCircuit)

    result.addRight(changeBasisCircuitForMeasurement(normalisedOplist[-1][1]))
    indicesToMeasure = [i for i, x in enumerate(reversed(normalisedOplist[-1][1])) if x != 0]
    #print (gamma)
    #print(result)
    #print(indicesToMeasure)
    return gamma, result, indicesToMeasure

def findOptimalRecombinationCircuit(oplist,maxGates=None,boolOptimiseCircuits=True):
    queue = collections.deque([oplist])
    circuits = []
    gammas = []
    unDoableIndices = []
    listIndicesToMeasure = []

    if maxGates == None:
        gamma,circuit,indicesToMeasure = anticommutingSetToCircuit(oplist)
        if boolOptimiseCircuits:
            circuit.fullCancelDuplicates()
        return [gamma], [circuit],[],[indicesToMeasure]

    while queue:
        currentOplist = queue.pop()
        #print(currentOplist)
        #print(currentOplist)
        try:
            gamma,circuit,indicesToMeasure = anticommutingSetToCircuit(currentOplist)
        except:
            while True:
                newList = copy.deepcopy(currentOplist)
                random.shuffle(newList)
                try:
                    gamma,circuit,indicesToMeasure = anticommutingSetToCircuit(newList)
                    break
                except:
                    pass
        if boolOptimiseCircuits:
            circuit.fullCancelDuplicates()
        if circuit.numGates() <= maxGates:
            circuits.append(circuit)
            gammas.append(gamma)
            listIndicesToMeasure.append(indicesToMeasure)
        elif len(currentOplist) == 1:
            unDoableIndices.append(len(circuits))
            circuits.append(circuit)
            gammas.append(gamma)
            listIndicesToMeasure.append(indicesToMeasure)
        else:
            firstHalf = currentOplist[:len(currentOplist)//2]
            secondHalf = currentOplist[len(currentOplist)//2:]
            queue.appendleft(firstHalf)
            queue.appendleft(secondHalf)

    return gammas,circuits,unDoableIndices,listIndicesToMeasure


def removeIdentity(oldOplist):
    coefficient = 0.
    oplist = copy.deepcopy(oldOplist)
    for i,term in enumerate(oplist):
        pauliStringWithoutIdentities = [x for x in term[1] if x != 0]
        if not pauliStringWithoutIdentities:
            identityTerm = oplist.pop(i)
            coefficient = identityTerm[0]
            break
    return oplist,coefficient


def optimalCircuitsFromAnticommutingSets(oplists,maxGates=None,boolOptimiseCircuits=True):
    fullGammas = []
    fullCircuits = []
    fullUndoableIndices = []
    fullListIndicesToMeasure = []
    offset = 0.
    for i,x in enumerate(oplists):
        oplist,newOffset = removeIdentity(x)
        offset += newOffset
        if not oplist:
            continue
        origNumCircuits = len(fullCircuits)
        gammas,circuits,unDoableIndices,listIndicesToMeasure = findOptimalRecombinationCircuit(oplist,maxGates,boolOptimiseCircuits)
        fullGammas += gammas
        fullCircuits += circuits
        fullUndoableIndices += [x + origNumCircuits for x in unDoableIndices]
        fullListIndicesToMeasure += listIndicesToMeasure
    return fullGammas,fullCircuits,fullUndoableIndices,fullListIndicesToMeasure, offset



def testOptimalCircuitsFromAnticommutingSets(oplists,maxGates,boolOptimiseCircuits=True,verbosity=0):
    fullOplist = [x for y in oplists for x in y]
    eigval,eigvec = directFermions.getTrueEigensystem(fullOplist)
    gammas,circuits,undoableIndices, indicesToMeasure, offset = optimalCircuitsFromAnticommutingSets(oplists,maxGates,boolOptimiseCircuits)
    fullExpVal = offset
    for i in range(len(circuits)):
        thisState = circuits[i].act(eigvec)
        expval = vqe.expectationValueComputationalBasis(thisState, indicesToMeasure[i])
        fullExpVal += gammas[i] * expval

    if verbosity > 0:
        print('TRUE EIGENVALUE:  {}'.format(eigval))
        print('ANTICOMMUTERISED EIGENVALUE:  {}'.format(fullExpVal))
        print('ERROR:   {}'.format(abs(eigval-fullExpVal)))
    else:
        if abs(eigval-fullExpVal) > 1e-13:
            print ('is broke lol')
        else:
            print ('OK')

    return


def doTheThing(molecule,boolJWorBK,maxGates):
    strJWorBK = ['JW','BK']
    inputFilename = '/home/andrew/data/BKData/anticommutingHamiltonians/{}/{}.sets'.format(strJWorBK[boolJWorBK],molecule)
    outputFilename = '/home/andrew/data/BKData/anticommutingCircuits/{}/{}_{}.stuff'.format(strJWorBK[boolJWorBK],molecule,str(maxGates))
    with open(inputFilename,'rb') as f:
        inputSets = cPickle.load(f)

    fullOplist = [x for y in inputSets for x in y]
    gammas,circuits,undoables,ind2meas,offset = optimalCircuitsFromAnticommutingSets(inputSets,maxGates)
    results = {}
    results['gammas'] = gammas
    results['circuits'] = circuits
    results['undoables'] = undoables
    results['ind2meas'] = ind2meas
    results['offset'] = offset
    results['fullLength'] = len(fullOplist)
    with open(outputFilename,'wb') as f:
        cPickle.dump(results,f,cPickle.HIGHEST_PROTOCOL)
    return

