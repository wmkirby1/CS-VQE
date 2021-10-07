from yaferp.analysis import analyser
from yaferp.misc import tapering
from yaferp.misc import constants
from yaferp.general import fermions
import pickle
DATA_DIR = '/home/andrew/data/BKData/'
TAPERED_DIR = DATA_DIR + 'hamiltonian/tapered/'


'''
FLOW:
READ NUMBER OF ELECTRONS FROM FILE -> GET HF STATE
-> generate tapered ham, store

'''

def stateIndexToBK(index):
    jwKet = fermions.ket(format(index,'b'))
    bkKet = fermions.newBK(jwKet)
    result = int(''.join([str(x) for x in bkKet.bitlist]),2)
    return result

def numElectronsToStateIndex(numElectrons,boolJWorBK,numSOO):
    if numSOO != 0:

        assert ((numElectrons-numSOO)%2) == 0
        numDOO = (numElectrons - numSOO)//2
        jwStateIndexDOOsStr = ''.join(['11']*numDOO)
        jwStateIndexSOOsStr = ''.join(['10']*numSOO)
        jwStateIndex = int(jwStateIndexSOOsStr + jwStateIndexDOOsStr,2)
    else:
        jwStateIndex = int(''.join(['1']*numElectrons),2)
    if boolJWorBK:
        return stateIndexToBK(jwStateIndex)
    else:
        return jwStateIndex

multiplicityToNumSOO = {'singlet':0,
                        'doublet':1,
                        'triplet':2,
                        'quartet':3,
                        'quintet':4}

def filenameToNumElectrons(filename):  #this is the hackiest nonsense ever
    numElectrons = 0
    splitFilename = filename.split('_')
    numSOO = multiplicityToNumSOO[splitFilename[2]]

    molname = splitFilename[0]
    atomsAndNums = molname.split('-')
    atoms = [''.join(c for c in x if not c.isnumeric()) for x in atomsAndNums]
    #atoms = [x[:-1] for x in atomsAndNums]
    nums = [''.join(c for c in x if c.isnumeric()) for x in atomsAndNums]
    atomicNumbers = [constants.atomicNumbers[x] for x in atoms]
    for i in range(len(atoms)):
        numElectrons += atomicNumbers[i] * int(nums[i])

    possibleCharge = splitFilename[3]
    possibleChargeLastCharacter = possibleCharge[-1]
    if possibleChargeLastCharacter == '+':
        numElectrons -= int(possibleCharge[:-1])
    elif possibleChargeLastCharacter =='-':
        numElectrons += int(possibleCharge[:-1])

    return numElectrons,numSOO


def generateTaperedHamiltonian(filename, boolJWorBK, cutoff=1e-12, ordering='magnitude'):
    numElectrons, numSOO= filenameToNumElectrons(filename)
    hfStateIndex = numElectronsToStateIndex(numElectrons,boolJWorBK,numSOO)
    outputPath = '{}{}/{}/{}/{}.oplist'.format(TAPERED_DIR,str(cutoff),ordering,['JW','BK'][boolJWorBK],filename)
    thisOplist = analyser.loadOplist(filename,boolJWorBK,cutoff=1e-12,ordering=ordering)
    taperedOplist = tapering.taperOplist(thisOplist,hfStateIndex)
    with open(outputPath,'wb') as f:
        pickle.dump(taperedOplist,f)
    return
from yaferp.general import sparseFermions
import numpy
import scipy.sparse
def hfDegenerateGroundState(oplist):
    oplistDiag = [x for x in oplist if 2 not in x[1] and 1 not in x[1]]
    mat = sparseFermions.commutingOplistToMatrix(oplistDiag)
    matDiag = mat.diagonal()
    hfEnergy = matDiag.min()
    indices = numpy.nonzero(abs(matDiag - hfEnergy) < 1e-8)[0]
    #hfState = scipy.sparse.coo_matrix((mat.shape[0],1))
    normalization = 1./(len(indices)**(1/2))
    fullSpace = mat.shape[0]
    matDat = [normalization]*len(indices)
    hfState = scipy.sparse.csc_matrix((matDat,(indices,[0]*len(indices))),shape=(fullSpace,1),dtype=numpy.complex128)
    #for i in indices:
     #   hfState[i,0] = normalization
    return hfState
