from yaferp.general import sparseFermions, fermions

#import commutator
#from commutator import multiplyPauli
#from commutator import multiplyPauliString
#from commutator import commutator
OPLIST_CUTOFF = 1e-12
from yaferp.orderings import directOrdering

'''
multiplyPauliLookup = {  # try working this out when you're drunk.  sorry.  i needed the speed.
    (0, 0): (1., 0),
    (0, 1): (1., 1),
    (0, 2): (1., 2),
    (0, 3): (1., 3),
    (1, 0): (1., 1),
    (1, 1): (1., 0),
    (1, 2): (1j, 3),
    (1, 3): (-1j, 2),
    (2, 0): (1., 2),
    (2, 1): (-1j, 3),
    (2, 2): (1., 0),
    (2, 3): (1j, 1),
    (3, 0): (1., 3),
    (3, 1): (1j, 2),
    (3, 2): (-1j, 1),
    (3, 3): (1., 0)
}
def multiplyPauli(pauli1,pauli2):
    return(multiplyPauliLookup[(pauli1,pauli2)])
'''
'''
multiplyPauliLookup = {  # OK take two.  if this is faster then shoutout to my mate Greg.
    0 : (1., 0),
    1: (1., 1),
    2: (1., 2),
    3: (1., 3),
    4: (1., 1),
    5: (1., 0),
    6: (1j, 3),
    7: (-1j, 2),
    8: (1., 2),
    9: (-1j, 3),
    10: (1., 0),
    11: (1j, 1),
    12: (1., 3),
    13: (1j, 2),
    14: (-1j, 1),
    15: (1., 0)
}



def multiplyPauli(pauli1,pauli2):
    return(multiplyPauliLookup[(pauli1*4)+pauli2])
'''

'''
def multiplyPauli(pauli1,pauli2):
    if pauli1 == 0:
        return (1,pauli2)
    elif pauli2 == 0:
        return(1,pauli1)
    elif pauli1 == pauli2:
        return(1,0)
    else:
        newPauli = 6 - pauli1 - pauli2
        if (pauli1 == 2 and pauli2 == 1) or (pauli1 == 3 and pauli2 == 2) or (pauli1 == 1 and pauli2 == 3):
           return (-1j,newPauli)
        else:
            return(1j,newPauli)
'''

# def multiplyPauliString(pauliString1,pauliString2):
#     newPauliString = []
#     coeffFactor = 1
#     for i in range(len(pauliString1)):
#         #newFactor,newPauli = multiplyPauli(pauliString1[i],pauliString2[i])
#         newFactor,newPauli = multiplyPauli(pauliString1[i],pauliString2[i])
#         coeffFactor = coeffFactor*newFactor
#         newPauliString.append(newPauli)
#   return (coeffFactor,newPauliString)

#
#
#
# def multiplyOpterm(opterm1,opterm2,negate=False):
#     coeffFactor,newString = multiplyPauliString(opterm1[1],opterm2[1])
#     if negate:
#         newCoeff = opterm1[0]*opterm2[0]*coeffFactor*-1.
#     else:
#         newCoeff = opterm1[0]*opterm2[0]*coeffFactor
#     return ([newCoeff,newString])
#
# def commutator(term1,term2):
#     if not term1 or not term2:
#         return []
#     try:
#         firstPart = multiplyOpterm(term1,term2)
#     except:
#         print(term1)
#         print(term2)
#     secondPart = multiplyOpterm(term2,term1,True)
#     opComm = fermions.oplist_sum([[firstPart],[secondPart]])
#     opComm = fermions.oplistRemoveNegligibles(opComm)
#     return opComm







def commutator(op1,op2):
    #print(op1)
    #print(op2)

    if not op1 or not op2:
        return []
    numQubits = len(op1[1])
    commutatorString = []
   # isZero = True
    negated = False
    nonCommutingPairs = 0

    #coefficient = 2 * op1[0] * op2[0]
    for i in range(numQubits):
        if op1[1][i] == op2[1][i]:
            commutatorString.append(0)
        elif op1[1][i] == 0:
            commutatorString.append(op2[1][i])
        elif op2[1][i] == 0:
            commutatorString.append(op1[1][i])
        else:
            newOp = 6 - op1[1][i] - op2[1][i]
            nonCommutingPairs += 1
            #isZero = not isZero
            commutatorString.append(newOp)
            if ((op1[1][i] == 3 and op2[1][i] == 2) or (op1[1][i] == 2 and op2[1][i] == 1)) or (op1[1][i] == 1 and op2[1][i] == 3):
                negated = not negated

    numCommutingPairsModulo = nonCommutingPairs % 4
    if numCommutingPairsModulo == 0 or numCommutingPairsModulo == 2:
        return []
    elif numCommutingPairsModulo == 1:
        coefficient = 2j * op1[0] * op2[0]
    elif numCommutingPairsModulo == 3:
        coefficient = -2j * op1[0] * op2[0]
 #   if isZero:
  #      return []
    if negated:
        coefficient = -1. * coefficient
    
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
       # tripleCommutator = tripleCommutator[0]
    if alpha == beta:
        tripleCommutator[0] = tripleCommutator[0] * 0.5
#  print(tripleCommutator)
    tripleCommutator[0] = tripleCommutator[0]/12.
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
    reducedErrorOperator = fermions.simplify(reducedErrorOperator)
    reducedErrorOperator = fermions.oplistRemoveNegligibles(reducedErrorOperator, OPLIST_CUTOFF)
    #reducedErrorOperator = 
    return reducedErrorOperator

def errorOperatorNorm(oplist,returnLen=False):
    #using the 1-norm for now at least.  it's a comically loose upper bound but hopefully we'll get some nice correlation
    errorOp = errorOperator(oplist)
    if not isinstance(oplist[0],list):
        return oplist[0] * oplist[0].conjugate()
    norm = 0.
    for term in errorOp:
        #norm = norm + abs(term[0] * term[0].conjugate())
        norm = norm + abs(term[0])
        #norm =  norm + term[0]
        #print(norm)
    if returnLen:
        return(norm,len(errorOp))
    return abs(norm)

def errorOperatorExpectation(oplist,state):
    errorOp = errorOperator(oplist)
    opMatrix = sparseFermions.commutingOplistToMatrix(errorOp)
    expectation = state.H * opMatrix * state
    return expectation



def errorOperatorExpectationByTerm(oplist,state):
    errorOp = errorOperator(oplist)
    expVals = []
    for term in errorOp:
        termMatrix = sparseFermions.commutingOplistToMatrix(term)
        thisVal = (state.H * termMatrix * state)[0,0]
        expVals.append([term, thisVal])
    return expVals
import copy

def errorOpPlaceTerm(sortedOplist,term):
    bestError = None
    for i in range(len(sortedOplist)+1):
        newOplist = copy.deepcopy(sortedOplist)
        newOplist.insert(i,term)
        '''try:
            thisNorm = errorOperatorNorm(newOplist)
        except:
            print(newOplist)'''
        thisNorm = errorOperatorNorm(newOplist)
        if bestError is None or thisNorm < bestError:
            bestPosition = i
            bestError = thisNorm
    return bestPosition,bestError


def errorOpFindNextTerm(sortedOplist,unsortedOplist):
    bestPositions = {}
    bestTerms = []
    bestError = None
    for i,term in enumerate(unsortedOplist):
        thisBestPosition, thisError = errorOpPlaceTerm(sortedOplist,term)
        if bestError is None or thisError < bestError:
            bestTerms = [i]
            bestError = thisError
            bestPositions = {}
            bestPositions[i] = thisBestPosition
        elif thisError == bestError:
            bestTerms.append(i)
            bestPositions[i] = thisBestPosition
    return bestTerms, bestPositions

def errorOpSortNextTerm(sortedOplist,unsortedOplist):

    bestTerms,bestPositions = errorOpFindNextTerm(sortedOplist,unsortedOplist)
    #print(bestTerms)
    #print(bestPositions)
    bestMagnitudes = [abs(unsortedOplist[x][0]) for x in bestTerms]
    theTermIndex = bestTerms[bestMagnitudes.index(max(bestMagnitudes))]
    theTerm = unsortedOplist.pop(theTermIndex)

    theNewPosition = bestPositions[theTermIndex]

    sortedOplist.insert(theNewPosition,theTerm)
    return(sortedOplist,unsortedOplist)

def errorOpOrdering(oplist):
    unsortedOplist = copy.deepcopy(oplist)
    sortedOplist = []
    while unsortedOplist:
        errorOpSortNextTerm(sortedOplist,unsortedOplist)
    return sortedOplist


def weakErrorOpOrderingNextTerm(sortedOplist,unsortedOplist):
    thisTerm = unsortedOplist.pop()
    bestPosition = errorOpPlaceTerm(sortedOplist,thisTerm)[0]
    sortedOplist.insert(bestPosition,thisTerm)
    return(sortedOplist,unsortedOplist)

def weakErrorOpOrdering(oplist):
    unsortedOplist = directOrdering.magnitude(oplist)[::-1]
    sortedOplist = []
    while unsortedOplist:
        sortedOplist,unsortedOplist = weakErrorOpOrderingNextTerm(sortedOplist,unsortedOplist)
    return sortedOplist

def reverseWeakErrorOpOrdering(oplist):
    unsortedOplist = directOrdering.magnitude(oplist)
    sortedOplist = []
    while unsortedOplist:
        sortedOplist,unsortedOplist = weakErrorOpOrderingNextTerm(sortedOplist,unsortedOplist)
    return sortedOplist

def randomWeakErrorOpOrdering(oplist):
    unsortedOplist = directOrdering.randomOrd(oplist)
    sortedOplist = []
    while unsortedOplist:
        sortedOplist,unsortedOplist = weakErrorOpOrderingNextTerm(sortedOplist,unsortedOplist)
    return sortedOplist

def whatIsTheNewErrorOperatorIfYouInsertATermIntoTheHamiltonian(oplist,insertIndex,term,oldErrorOperator):
    '''hopefully i've written up somewhere how this works by now...'''

    newOplist = oplist[0:insertIndex] + [term] +oplist[insertIndex:]
    errorOperator = copy.deepcopy(oldErrorOperator)
    for alpha in range(0,insertIndex+1):
        for gamma in range(0,insertIndex):
            thisTerm = errorOperatorInterior(newOplist, alpha, insertIndex, gamma)
            errorOperator.append(thisTerm)

    for beta in range(insertIndex+1,len(newOplist)):
        for gamma in range(0,beta):
            thisTerm = errorOperatorInterior(newOplist,insertIndex,beta,gamma)
            errorOperator.append(thisTerm)

        for alpha in [x for x in range(0,beta+1) if x != insertIndex]:
            thisTerm = errorOperatorInterior(newOplist,alpha,beta,insertIndex)
            errorOperator.append(thisTerm)

    reducedErrorOperator = [term for term in errorOperator if term]
    reducedErrorOperator = fermions.simplify(reducedErrorOperator)
    reducedErrorOperator = fermions.oplistRemoveNegligibles(reducedErrorOperator, OPLIST_CUTOFF)


    return reducedErrorOperator

def testWhatIsTheEtc(oplist,term):
    originalErrorOperator = errorOperator(oplist)
    for i in range(len(oplist)):
        newOplist = oplist[0:i] + [term] + oplist[i:]
        thisErrorOperatorSequential = whatIsTheNewErrorOperatorIfYouInsertATermIntoTheHamiltonian(oplist,i,term,originalErrorOperator)
        thisErrorOperator = errorOperator(newOplist)
        thisErrorOperatorFuckedUp = [[-1*thisErrorOperator[i][0],thisErrorOperator[i][1]] for i in range(len(thisErrorOperator))]
        testList = thisErrorOperatorSequential + thisErrorOperatorFuckedUp

        reducedTestList = fermions.simplify(testList)
        reducedTestList = fermions.oplistRemoveNegligibles(reducedTestList, 1e-12)

        if reducedTestList:
            print("error")
            print (reducedTestList)
        else:
            print("all cool")
    return
