import scipy.sparse
import numpy
import random
import multiprocessing
def innermostFactor(pauli):
    if pauli == 0:
        return [(0,0),(1,1)], [], [], []
    elif pauli == 1:
        return [(0,1),(1,0)], [], [], []
    elif pauli == 2:
        return [],[(1,0)],[],[(0,1)]
    elif pauli == 3:
        return [(0,0)],[],[(1,1)],[]


def pauliI(bitSetter,set1,setI,setNeg,setNegI):
    resSet1 = set1 + [(x[0] + bitSetter,x[1] + bitSetter) for x in set1]
    resSetI = setI + [(x[0] + bitSetter, x[1] + bitSetter) for x in setI]
    resSetNeg = setNeg + [(x[0] + bitSetter, x[1] + bitSetter) for x in setNeg]
    resSetNegI = setNegI + [(x[0] + bitSetter, x[1] + bitSetter) for x in setNegI]
    return resSet1,resSetI,resSetNeg,resSetNegI

def pauliX(bitSetter,set1,setI,setNeg,setNegI):
    resSet1 = [(x[0] + bitSetter, x[1]) for x in set1] + [(x[0],x[1] + bitSetter) for x in set1]
    resSetI = [(x[0] + bitSetter, x[1]) for x in setI] + [(x[0],x[1] + bitSetter) for x in setI]
    resSetNeg = [(x[0] + bitSetter, x[1]) for x in setNeg] + [(x[0], x[1] + bitSetter) for x in setNeg]
    resSetNegI = [(x[0] + bitSetter, x[1]) for x in setNegI] + [(x[0], x[1] + bitSetter) for x in setNegI]
    return resSet1,resSetI,resSetNeg,resSetNegI

def pauliY(bitSetter,set1,setI,setNeg,setNegI):
    resSet1 = [(x[0],x[1] + bitSetter) for x in setI] + [(x[0] + bitSetter,x[1]) for x in setNegI]
    resSetI = [(x[0], x[1] + bitSetter) for x in setNeg] + [(x[0] + bitSetter, x[1]) for x in set1]
    resSetNeg = [(x[0], x[1] + bitSetter) for x in setNegI] + [(x[0] + bitSetter, x[1]) for x in setI]
    resSetNegI = [(x[0], x[1] + bitSetter) for x in set1] + [(x[0] + bitSetter, x[1]) for x in setNeg]
    return resSet1,resSetI,resSetNeg,resSetNegI

def pauliZ(bitSetter,set1,setI,setNeg,setNegI):
    resSet1 = set1 + [(x[0] + bitSetter,x[1] + bitSetter) for x in setNeg]
    resSetI = setI + [(x[0] + bitSetter,x[1] + bitSetter) for x in setNegI]
    resSetNeg = setNeg + [(x[0] + bitSetter,x[1] + bitSetter) for x in set1]
    resSetNegI = setNegI + [(x[0] + bitSetter, x[1] + bitSetter) for x in setI]
    return resSet1,resSetI,resSetNeg,resSetNegI

def pauliStringToListsIndices(pauliString):
    thisPauli = pauliString[0]
    if len(pauliString) == 1:
        return innermostFactor(thisPauli)
    pauliStringWithoutMe = pauliString[1:]
    set1,setI,setNeg,setNegI = pauliStringToListsIndices(pauliStringWithoutMe)
    bitSetter = 2 ** len(pauliStringWithoutMe)
    if thisPauli == 0:
        return pauliI(bitSetter,set1,setI,setNeg,setNegI)
    elif thisPauli == 1:
        return pauliX(bitSetter,set1,setI,setNeg,setNegI)
    elif thisPauli == 2:
        return pauliY(bitSetter,set1,setI,setNeg,setNegI)
    else:
        return pauliZ(bitSetter,set1,setI,setNeg,setNegI)

def optermToSparseData(opterm):
    coeffsBase = [1.,1.j,-1.,-1.j]
    coeffs = [opterm[0] * x for x in coeffsBase]
    shape = (2**len(opterm[1]),2**len(opterm[1]))
    set1,setI,setNeg,setNegI = pauliStringToListsIndices(opterm[1])
    lengths = [len(set1),len(setI),len(setNeg),len(setNegI)]
    data = [coeffs[0]] * lengths[0] + [coeffs[1]] * lengths[1] + [coeffs[2]] * lengths[2] + [coeffs[3]] * lengths[3]
    #print(set1)
    #print(setI)
    #print(setNegI)
    #print(setNegI)
    if set1:
        xs1,ys1 = zip(*set1)
    else:
        xs1,ys1 = (),()
    if setI:
        xs2,ys2 = zip(*setI)
    else:
        xs2,ys2 = (),()
    if setNeg:
        xs3,ys3 = zip(*setNeg)
    else:
        xs3,ys3 = (),()
    if setNegI:
        xs4,ys4 = zip(*setNegI)
    else:
        xs4,ys4 = (),()
    #print (xs1)
    #print(xs2)
    xs = xs1 + xs2 + xs3 + xs4
    ys = ys1 + ys2 + ys3 + ys4
    return data,xs,ys


def oplistToSparseData(oplist2):
    oplist = oplist2
    #manager = multiprocessing.Manager()
    #oplist=manager.list(oplist2)
    #pool = multiprocessing.Pool(multiprocessing.cpu_count())
    #fullDats = pool.map(optermToSparseData,oplist)
    #pool.close()
    #pool.join()
    fullDats = [optermToSparseData(term) for term in oplist]
    dats = (thing[0] for thing in fullDats)
    xs = (thing[1] for thing in fullDats)
    ys = (thing[2] for thing in fullDats)
    data = list(item for sublist in dats for item in sublist)
    x = list(item for sublist in xs for item in sublist)
    y = list(item for sublist in ys for item in sublist)
    return data,x,y


def oplistToSparseData_multithreaded(oplist2):
    #oplist = oplist2
    manager = multiprocessing.Manager()
    oplist=manager.list(oplist2)
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    fullDats = pool.map(optermToSparseData,oplist)
    pool.close()
    pool.join()
    #fullDats = [optermToSparseData(term) for term in oplist]
    dats = (thing[0] for thing in fullDats)
    xs = (thing[1] for thing in fullDats)
    ys = (thing[2] for thing in fullDats)
    data = list(item for sublist in dats for item in sublist)
    x = list(item for sublist in xs for item in sublist)
    y = list(item for sublist in ys for item in sublist)
    return data,x,y

def oplistToSparse(oplist):
    shape = (2**len(oplist[0][1]), 2**len(oplist[0][1]))
    data,x,y = oplistToSparseData(oplist)
    result1 = scipy.sparse.coo_matrix((data,(x,y)),shape=shape,dtype=numpy.complex128)
    result2 = scipy.sparse.csc_matrix(result1)
    return result2

def OLD_oplistToSparse2(oplist):
    shape = (2**len(oplist[0][1]), 2**len(oplist[0][1]))
    result = scipy.sparse.csc_matrix(shape,dtype=numpy.complex128)
    for term in oplist:
        data,x,y = optermToSparseData(term)
        thisTerm = scipy.sparse.csc_matrix((data,(x,y)),dtype=numpy.complex128,shape=shape)
        result += thisTerm
    return result



def randomOplist(maxNumQubits=8,maxTerms=1000):
    numTerms = random.randint(2,maxTerms)
    numQubits = random.randint(2,maxNumQubits)
    oplist = []
    for i in range(numTerms):
        coefficient =  random.random()
        pauliString = [random.randint(0,3) for i in range(numQubits)]
        term = [coefficient,pauliString]
        oplist.append(term)
    return oplist


def tester(n=100,maxnumQubits=8,maxTerms=100):
    from yaferp.general import sparseFermions
    for i in range(n):
        oplist = randomOplist(maxnumQubits,maxTerms)
        mat1 = sparseFermions.commutingOplistToMatrix(oplist)
        mat2 = oplistToSparse(oplist)
        isOK = abs(mat1-mat2) < 1e-13
        if False in isOK.todense():
            return False
    return True


