'''
denseTrotter
Created on 21 May 2015
summary:  trotterise things as dense matrices - for VERY small hamiltonians only (no more than 12 qubits or so) (!!!!!!!!)
@author: andrew
'''
from yaferp.general import denseFermions
import scipy.linalg

def oplistToMatrices(oplist):
    '''return a list of matrix representations of the '''
    listMatrices = []
    for op in oplist:
        listMatrices.append(denseFermions.commutingOplistToMatrix(op))
    return listMatrices

def unitaryManual(hamiltonian,time):
    '''generate a unitary from explicit settings.
    hamiltonian:  sparse matrix representations of the hamiltonian
    time:  overall propagation time'''
    exponent = hamiltonian * time * -1.j
    unitary = scipy.mat(scipy.linalg.expm(exponent))
    return unitary

def trotterise(listTerms,ordering,t,trotterNumber,trotterOrder=1):
    '''trotterise a list of terms. note this DOES NOT SUPPORT trotter approximation order != 1 at the moment
    ordering:  tuple specifying the order the terms are applied - note this goes LEFT to RIGHT'''
    trueOrdering = ordering[::-1] #reverse the ordering so we apply leftmost first
    unitaryShape = listTerms[0].shape[0]
    fullUnitary = scipy.identity(unitaryShape)
    for whichTerm in trueOrdering:
        thisUnitary = unitaryManual(listTerms[whichTerm],float(t)/float(trotterNumber))
        fullUnitary = fullUnitary * thisUnitary
    fullUnitary = fullUnitary**trotterNumber
    return fullUnitary