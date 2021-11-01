'''
Created on 28 Jun 2017

@author: andrew
'''

DATA_DIR = '/home/andrew/workspace/BKData/'
ERROR_DATA_DIR= DATA_DIR + 'errors/'
ONE_STEP_ERROR = ERROR_DATA_DIR + 'oneStep/'
NUM_STEPS_TEST = ERROR_DATA_DIR + 'numSteps/'
DEFAULT_CUTOFF = 1e-14
from yaferp.analysis import analyser

'''
def oneStepErrorCalculate(filename,boolJWorBK,cutoff=DEFAULT_CUTOFF,ordering='topMagnitude'):
    oplist = analyser.loadOplist(filename,boolJWorBK,cutoff,ordering)
    eigenvalue2 = analyser.readEnergy(filename,boolJWorBK)
    eigenvector = analyser.readEigenvector(filename,boolJWorBK,cutoff)
    error = directFermions.oneStepTrotterError(oplist,eigenvalue=eigenvalue2,eigenvec=eigenvector)
    '''
def numSteps(filename,boolJWorBK,precision=0.0017, cutoff=DEFAULT_CUTOFF,ordering='topMagnitude'):
    oplist = analyser.loadOplist(filename, boolJWorBK, cutoff, ordering)
    eigenvalue = analyser.readEnergy(filename, boolJWorBK)
    eigenvector = analyser.readEigenvector(filename, boolJWorBK, cutoff)
    
    
    
