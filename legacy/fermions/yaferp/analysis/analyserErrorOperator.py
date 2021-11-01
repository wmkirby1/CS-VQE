'''
Created on 30 May 2017

@author: andrew
'''
DATA_DIR = '/home/andrew/workspace/BKData/'
INTEGRALS_DIR = DATA_DIR + 'integrals/'
OPLIST_DIR = DATA_DIR + 'hamiltonian/oplist/'
OPLIST_JW_DIR = DATA_DIR + 'hamiltonian/oplist/JW/'
OPLIST_BK_DIR = DATA_DIR + 'hamiltonian/oplist/BK/'
REDUCED_OPLIST_DIR = DATA_DIR + 'hamiltonian/reducedOplist/'
ENERGY_JW_DIR = DATA_DIR + 'energies/JW/'
ENERGY_BK_DIR = DATA_DIR + 'energies/BK/'
REDUCED_ENERGY_DIR = DATA_DIR + 'energies/reduced/'
EIGENVECS_DIR = DATA_DIR + 'eigenvectors/'
EIGENVECS_JW_DIR = DATA_DIR + 'eigenvectors/JW/'
EIGENVECS_BK_DIR = DATA_DIR + 'eigenvectors/BK/'
REDUCED_EIGENVECS_DIR = DATA_DIR + 'eigenvectors/reduced/'
GATES_JW_DIR = DATA_DIR + 'gates/JW/'
GATES_BK_DIR = DATA_DIR + 'gates/BK/'
CIRCUITS_DIR = DATA_DIR + 'circuit/'
GATECOUNT_JW_DIR = DATA_DIR + 'gatecount/JW/'
GATECOUNT_BK_DIR = DATA_DIR + 'gatecount/BK/'
GATECOUNTS_DIR = DATA_DIR + '/gatecount/'
ERROR_OPERATOR_DIR = DATA_DIR + '/errorOperator/'
DEFAULT_CUTOFF=1e-14
#CIRCUIT_JW_DIR
RB_CODE_DIR = '/home/andrew/workspace/RB_Fermions/'
DEFAULT_OPLIST_TOLERANCE=1e-14
TESTS_DIR = DATA_DIR + 'tests/'
ERROR_OPERATOR_TESTS = TESTS_DIR + 'errop/'

from yaferp.orderings import errorOperator
from yaferp.analysis import analyser
from yaferp.general import sparseFermions
import scipy
import copy
import scipy.sparse
import numpy
import cPickle
import os

def generateErrorOperator(filename, boolJWorBK,cutoff=DEFAULT_CUTOFF,recalculate=0):
    if boolJWorBK:
        outputPath = ERROR_OPERATOR_DIR + '/reduced/' + str(cutoff) + '/BK/' + filename + '.erop'
    else:
        outputPath = ERROR_OPERATOR_DIR + '/reduced/' + str(cutoff) +'/JW/' + filename + '.erop'
        
    if recalculate or not os.path.isfile(outputPath):
        try:
            os.remove(outputPath)
        except:
            pass
        oplist = analyser.loadReducedOplist(filename, boolJWorBK, cutoff)
        errorOp = errorOperator.errorOperator(oplist)
        with open(outputPath,'wb') as f:
            cPickle.dump(errorOp,f,cPickle.HIGHEST_PROTOCOL)
    else:
        with open(outputPath,'rb') as f:
            errorOp = cPickle.load(f)
    return errorOp
        
            


def estimateError (filename,boolJWorBK,cutoff=DEFAULT_CUTOFF,order=2,recalculate=0):
    '''needs eigenvectors calced'''
    
    if boolJWorBK:
        outputPath = ERROR_OPERATOR_TESTS + '/reduced/' + str(cutoff) + '/BK/' + filename + '.erop'
    else:
        outputPath = ERROR_OPERATOR_TESTS + '/reduced/' + str(cutoff) +'/JW/' + filename + '.erop'
    
    if recalculate or not os.path.isfile(outputPath):
        try:
            os.remove(outputPath)
        except:
            pass
        errorOp = generateErrorOperator(filename,boolJWorBK,cutoff,recalculate)
        eigvec = analyser.readEigenvector(filename, boolJWorBK, cutoff)
        state = scipy.sparse.csr_matrix(eigvec, dtype=numpy.complex128)
        errorOpMat = sparseFermions.commutingOplistToMatrix(errorOp)
        
        originalState = copy.deepcopy(state)
        newState = errorOpMat * state
        firstBit = originalState.conjugate().transpose()
        expectation = (firstBit * newState)
        with open(outputPath,'wb') as f:
            cPickle.dump(expectation,f,cPickle.HIGHEST_PROTOCOL)
    else:
        with open(outputPath,'rb') as f:
            errorOp = cPickle.load(f)
    
    
    return expectation
  
    
