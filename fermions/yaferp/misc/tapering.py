import numpy
import scipy
import sympy
import scipy.linalg
import copy
import utils.qonversion_tools as qonvert
import fermions.yaferp.general.fermions as fermions
import fermions.yaferp.general.sparseFermions as sparseFermions

def pauliStringGVectors(pauliString):
    gX = numpy.zeros((len(pauliString)),dtype=numpy.bool)
    gZ = numpy.zeros((len(pauliString)),dtype=numpy.bool)
    result = numpy.zeros((len(pauliString)*2,1))
    for i in range(len(pauliString)):
        if pauliString[i] in [1,2]:
            gX[i] = 1
        if pauliString[i] in [2,3]:
            gZ[i] = 1
    return gX, gZ
def gVectorToPauliString(gVector):
    numPaulis = int(len(gVector)/2)
    result = []
    for i in range(numPaulis):
        if gVector[i]:
            if gVector[i+numPaulis]:
                result.append(2)
            else:
                result.append(1)
        else:
            if gVector[i+numPaulis]:
                result.append(3)
            else:
                result.append(0)
    return result

def gMatrixToPauliStrings(gMatrix):
    result = []
    for i in range(gMatrix.shape[0]):
        thing = gVectorToPauliString(gMatrix[i,:])
        result.append(thing)
    return result
def singleSymGenSQG(term,otherTerms):
    nonzeroIndices = [x for x,y in enumerate(term) if y != 0]
    otherNonZeroIndices = [x for thing in otherTerms for x,y in enumerate(thing) if y!= 0]
    candidates = [x for x in nonzeroIndices if x not in otherNonZeroIndices]
    assert candidates
    result = [0] * len(term)
    result[candidates[0]] = 1
    return result
def symGenOplists(terms):
    results = []
    for i in range(len(terms)):
        thisTerm = terms[i]
        otherTerms = terms[:i] + terms[i+1:]
        thisSQG = singleSymGenSQG(thisTerm,otherTerms)
        thisResult = [[2.**(-0.5),thisSQG],[2.**(-0.5),thisTerm]]
        results.append(thisResult)
    return results

def listOplistsProduct(listOplists):
    if len(listOplists) == 0:
        return listOplists
    if len(listOplists) == 1:
        return listOplists[0]
    result = copy.deepcopy(listOplists[0]) #slinging copies everywhere because i'm not sure how safe this is
    for nextTerm in listOplists[1:]:
        nextTermCopy = copy.deepcopy(nextTerm)
        result = copy.deepcopy(fermions.oplist_prod(result,nextTerm))
    return result

def nullspaceBasisToUOplist(vecs):
    thing = symGenOplists(vecs)
    return listOplistsProduct(thing)




def oplistGMatrices(oplist):
    gX = numpy.zeros((len(oplist[0][1]),len(oplist)),dtype=numpy.bool)
    gZ = numpy.zeros((len(oplist[0][1]),len(oplist)),dtype=numpy.bool)
    for i,term in enumerate(oplist):
        x, z = pauliStringGVectors(term[1])
        gX[:,i] = x
        gZ[:,i] = z
    return gX,gZ


def oplistParityCheck(oplist):
    e = numpy.zeros((len(oplist),len(oplist[0][1])*2),dtype=numpy.uint)
    gX,gZ = oplistGMatrices(oplist)
    eX = gZ.T
    eZ = gX.T
    e[:,0:len(oplist[0][1])] = eX
    e[:,len(oplist[0][1]):] = eZ
    #e[:,len(oplist[0][1])+1:2*len(oplist[0][1])] = eZ
    return e

def pivotCols(m):
    result = []
    currentLeadingRow = -1
    for colI in range(m.shape[1]):
        if 1 in m[currentLeadingRow+1:,colI]:
            currentLeadingRow += 1
            result.append(colI)
    return result

def nonPivotCols(m):
    pivots = pivotCols(m)
    result = [x for x in list(range(m.shape[1])) if not x in pivots]
    return result
def reBinary(m):
    shape = m.shape
    row=0
    col=0
    currentLeadingRow = -1
    #currentLeadingRow = 0
    for col in range(shape[1]):
        #for col in range(0,i):
        if 1 in m[currentLeadingRow+1:,col]: #if pivot
            currentLeadingRow += 1
            lowerIndicesToKill = numpy.nonzero(m[currentLeadingRow+1:,col])[0] + currentLeadingRow + 1
            #lowerIndicesToKill = [x + currentLeadingRow + 1 for x in numpy.nonzero(m[currentLeadingRow+1:,col])]
            #print(lowerIndicesToKill)
            if m[currentLeadingRow,col] == 0: # make sure we have 1 in right position
                #print(lowerIndicesToKill)
                #print(m[lowerIndicesToKill[0],:])
                #print(m[currentLeadingRow,:])
                #print(m[currentLeadingRow,:])
                #print('go!')
                m[currentLeadingRow,:] = (m[lowerIndicesToKill[0],:] + m[currentLeadingRow,:]) % 2
            for index in lowerIndicesToKill:
                m[index,:] = (m[currentLeadingRow,:] + m[index,:]) % 2

            upperIndicesToKill = numpy.nonzero(m[:currentLeadingRow,col])
            #print(upperIndicesToKill)
            for index in upperIndicesToKill:
                m[index,:] = (m[currentLeadingRow,:] + m[index,:]) % 2

        else:
            pass
    # print(m)

    mask = numpy.all(m == 0, axis=1)
    #print(type(mask))
    m = m[~mask]
    return m


def oplistParityCheckRRE(oplist):
    e = oplistParityCheck(oplist)
    eE = reBinary(e)
    #eE, independentIndices  = sympy.Matrix(e).rref(iszerofunc=lambda x: x % 2 ==0)
    #eE, independentIndices = sympy.Matrix(e).rref()
    #print(numpy.array(eE))
    #print(independentIndices)
    #print(eE.nullspace(iszerofunc=lambda x: x % 2==0))
    #zeroeVector = numpy.zeros(len(oplist[0][1])*2,dtype=numpy.int)
    #bigIndex = max(independentIndices)+1
    #clive = numpy.array(eE,dtype=numpy.int)
    #result = clive[numpy.any(clive != 0, axis=1)]
    #result = numpy.matrix(eE).astype(numpy.int)
    result = eE
    return result


'''
def oplistParityKernel(oplist):
    x = oplistParityCheckRRE(oplist)
    xT = x.T
    augmented = numpy.concatenate((xT,numpy.identity(xT.shape[0],dtype=numpy.uint64)),axis=1)

    thing = reBinary(augmented)[:,x.shape[0]:]
    thing2 = thing.dot(numpy.ones((x.shape[1],1)))
    assert (not 0. in thing2) and (not 0 in thing2)
    nullRows = numpy.argwhere(thing2 > 1)[:,0]
    basis = thing[nullRows,:]
    return basis
'''

def oplistParityKernel(oplist):
    x = oplistParityCheckRRE(oplist)
    pivots = pivotCols(x)
    nonPivots = nonPivotCols(x)
    basis = numpy.zeros((len(nonPivots),x.shape[1]))
    for i,nonPivot in enumerate(nonPivots):

        #basis[i,:] = x[:,nonPivot].T
        for otherNonPivot in nonPivots:
            if otherNonPivot == nonPivot:
                basis[i,nonPivot] = 1
            else:
                basis[i,otherNonPivot] = 0
        for k,pivot in enumerate(pivots):
            basis[i,pivot] = x[k,nonPivot]
    return basis



PAULI_STRINGS_LOOKUP = {'I':0,
                        'X':1,
                        'Y':2,
                        'Z':3}

def oplistHC(oplist): #probably code to do this more robustly somewhere else but i've lost track
    result = []
    for term in oplist:
        result.append(copy.deepcopy([term[0].conjugate(),term[1]]))
    return result

def parametrisedOplistRemoveNegligibles(oplist):
    '''TODO: this desperately needs to go somewhere else.  also it'll probably break from FP errors at some point'''
    newOplist = []
    for term in oplist:
        coeff = term[0]
        if sympy.simplify(coeff) != 0:
            newOplist.append(term)
    return newOplist

def transformedHamiltonian(oplist,kernelBasis=None):
    if kernelBasis is None:
        kernelBasis = oplistParityKernel(oplist)
    generators = gMatrixToPauliStrings(kernelBasis)
    unitary = nullspaceBasisToUOplist(generators)
    unitaryC = oplistHC(unitary)
    listTerms = []
    for term in oplist:
        #result = fermions.oplist_prod(fermions.oplist_prod(unitaryC,[term]),unitary)
        result = fermions.oplist_prod(fermions.oplist_prod(unitary,[term]),unitaryC)
        if isinstance(result[0][0], sympy.Basic): #clunky af hack to deal with parametrised data (ie ansatzes)
            result = parametrisedOplistRemoveNegligibles(result)
        else:
            result = fermions.oplistRemoveNegligibles(result)
        listTerms += result
    return listTerms

def stabEigvals(oplist,hfStateIndex,kernelBasis=None,explicitEigvals=False):

    if kernelBasis is None:
        kernelBasis = oplistParityKernel(oplist)
    generators = gMatrixToPauliStrings(kernelBasis)
    if explicitEigvals:
        result = stabEigvalsExplicit(oplist,generators)
        return result
        #return stabEigvalsExplicit(oplist,generators)
    eigvals = []
    for generator in generators:
        assert not any(x in generator for x in [1,2]) #panic if we have X or Ys
        thisEigval = 1.
        for i,thisPauli in enumerate(generator):
            if thisPauli == 3 and hfStateIndex >= 2**i: #if Z in this qubit is present in gen, and bit is set in HF state
                thisEigval *= -1.
        eigvals.append(thisEigval)
    return eigvals

def stabEigvalsExplicit(oplist,thing):
    eigvals = []
    hfGroundState = hfDegenerateGroundState(oplist)[0]
    for generator in thing:
        thisOplist = [[1.,generator]]
        assert not any(x in generator for x in [1,2])
        thisExpVal = sparseFermions.oplistExpectation(thisOplist,hfGroundState).todense()[0,0]
        eigvals.append(thisExpVal)
    return eigvals

def hfDegenerateGroundState(oplist):
    oplistDiag = [x for x in oplist if 2 not in x[1] and 1 not in x[1]] 
    mat = sparseFermions.commutingOplistToMatrix(oplistDiag)
    matDiag = mat.diagonal()
    hfEnergy = matDiag.min()
    indices = numpy.nonzero(abs(matDiag - hfEnergy) < 1e-8)[0]
    normalization = 1./(len(indices)**(1/2))
    fullSpace = mat.shape[0]
    matDat = [normalization]*len(indices)
    hfState = scipy.sparse.csc_matrix((matDat,(indices,[0]*len(indices))),shape=(fullSpace,1),dtype=numpy.complex128)
    #print(indices)
    return hfState, indices

def replaceOperators(transformedOplist,xOperatorIndices,eigenvalues):
    newOplist = []
    eigvalDict = {}
    for i,thing in enumerate(xOperatorIndices):
        eigvalDict[thing] = eigenvalues[i]
    for term in transformedOplist:
        coefficient = term[0]
        pauliString = []
        for i,pauli in enumerate(term[1]):
            if i in eigvalDict and pauli == 1:
                pauliString.append(0)
                coefficient *= eigvalDict[i]
            else:
                pauliString.append(pauli)
        newOplist.append([coefficient,pauliString])
    return newOplist

def getXOperatorIndices(xOperators):
    result = []
    for thing in xOperators:
        assert sum(thing) == 1
        assert not any(x in thing for x in [2,3])
        result.append(thing.index(1))
    return result

def reducedHamiltonian(oplist,anzlist,hfStateIndex,kernelBasis=None,explicitEigvals=False):
    if kernelBasis is None:
        kernelBasis = oplistParityKernel(oplist)
    generators = gMatrixToPauliStrings(kernelBasis)
    thing = symGenOplists(generators)
    xOperators = [x[0][1] for x in thing]

    xOperatorIndices = getXOperatorIndices(xOperators)
    transformed_ham = transformedHamiltonian(oplist,kernelBasis)
    transformed_anz = transformedHamiltonian(anzlist,kernelBasis)

    eigvals = stabEigvals(oplist,hfStateIndex,kernelBasis,explicitEigvals)
    reducedHam = replaceOperators(transformed_ham,xOperatorIndices,eigvals)
    reducedAnz = replaceOperators(transformed_anz,xOperatorIndices,eigvals)

    if isinstance(reducedHam[0][0], sympy.Basic): #clunky af hack to deal with parametrised data (ie ansatzes)
        result_ham = parametrisedOplistRemoveNegligibles(fermions.simplify(reducedHam))
        result_anz = parametrisedOplistRemoveNegligibles(fermions.simplify(reducedAnz))
    else:
        result_ham = fermions.oplistRemoveNegligibles(fermions.simplify(reducedHam),1e-12)
        result_anz = fermions.oplistRemoveNegligibles(fermions.simplify(reducedAnz),1e-12)
    return result_ham, result_anz


def taperableQubits(reducedOplist):
    result = []
    for i in range(len(reducedOplist[0][1])):
        thisQubitPaulis = [x[1][i] for x in reducedOplist]
        if sum(thisQubitPaulis) == 0:
            result.append(i)
    return result

def reducedToTaperedOplist(reducedOplist, taper_oplist):
    taperableIndices = taperableQubits(taper_oplist)
    result = []
    for term in reducedOplist:
        coeff = term[0]
        pauliString = [x for i,x in enumerate(term[1]) if i not in taperableIndices]
        result.append([coeff,pauliString])
    return result



def taperOplist(oplist,anzlist,hfStateIndex,kernel=None,explicitEigvals=False):
    reduced_ham, reduced_anz = reducedHamiltonian(oplist,anzlist,hfStateIndex,kernel,explicitEigvals)
    tapered_ham = reducedToTaperedOplist(reduced_ham, reduced_ham)
    tapered_anz = reducedToTaperedOplist(reduced_anz, reduced_ham)
    return tapered_ham, tapered_anz


'''
def taperableQubits(transformedOplist):
    result = []
    for i in range(len(transformedOplist[0][1])):
        factors = [x[1][i] for x in transformedOplist]
        if not (2 in factors or 3 in factors):
            result.append(i)
    return result

'''

def taperTransformedOplist(transformedOplist):
    taperableIndices = taperableQubits(transformedOplist)
    indicesToInclude = [x for x in range(len(transformedOplist[0][1])) if not x in taperableIndices]
    pass
    #for
    return

def taperingPauliStringToOurs(taperingPauliString):
    pauliString = []
    #for char in list(reversed(taperingPauliString)):
    #for char in list(reversed(taperingPauliString)):
    for char in taperingPauliString:
        pauliString.append(PAULI_STRINGS_LOOKUP[char])
    return pauliString

def readTaperingHamiltonian(filepath):
    oplist = []
    with open(filepath,'rb') as f:
        raw = [x.strip().decode('UTF-8') for x in f.readlines()]
    for i in range(0,len(raw),2):
        coeffReal = float(raw[i+1])
        pauliStringOrig = raw[i]
        pauliString = taperingPauliStringToOurs(pauliStringOrig)
        coeff = complex(coeffReal,0.)
        opterm = [coeff,pauliString]
        oplist.append(opterm)

    return(oplist)


def taper_dict(ham, anz):
    ham_list = qonvert.dict_to_list_index(ham)
    anz_list = qonvert.dict_to_list_index(anz)

    hf_index = hfDegenerateGroundState(ham_list)[1][-1]
    tap_ham, tap_anz = taperOplist(ham_list, anz_list, hf_index)

    new_ham = qonvert.index_list_to_dict(tap_ham)
    new_anz = qonvert.index_list_to_dict(tap_anz)

    num_qubits = len(list(new_ham.keys())[0])

    return new_ham, new_anz, num_qubits


TEST_OPLIST = [[(-0.8026507290573593+0j), (0, 0, 0, 0)],
                [1.,[0,0,0,3]],
               [1.,[0,0,3,0]],
               [1.,[0,3,0,0]],
               [1.,[3,0,0,0]],
               [1.,[0,0,3,3]],
               [1.,[0,3,0,3]],
               [1.,[3,0,0,3]],
               [1.,[0,3,3,0]],
               [1.,[3,0,3,0]],
               [1.,[3,3,0,0]],
               [1.,[1,1,2,2]],
               [1.,[1,2,2,1]],
               [1.,[2,1,1,2]],
               [1.,[2,2,1,1]]]

TEST_OPLIST_2 = [[(-0.8026507290573593+0j), (0, 0, 0, 0)],
                 [(-0.23665448233398706+0j), (3, 0, 0, 0)],
                 [(-0.236654482333987+0j), (0, 3, 0, 0)],
                 [(0.1757540240298699+0j), (0, 0, 3, 0)],
                 [(0.17575402402986984+0j), (0, 0, 0, 3)],
                 [(0.1757017919455566+0j), (3, 3, 0, 0)],
                 [(0.17001384213276516+0j), (0, 0, 3, 3)],
                 [(0.1671428215197555+0j), (0, 3, 3, 0)],
                 [(0.1671428215197555+0j), (3, 0, 0, 3)],
                 [(0.12222518427388038+0j), (3, 0, 3, 0)],
                 [(0.12222518427388038+0j), (0, 3, 0, 3)],
                 [(-0.044917637245875136+0j), (2, 2, 1, 1)],
                 [(-0.044917637245875136+0j), (1, 1, 2, 2)],
                 [(0.044917637245875136+0j), (1, 2, 2, 1)],
                 [(0.044917637245875136+0j), (2, 1, 1, 2)]]

