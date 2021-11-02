'''header'''
from yaferp.general import generateOplist


def test_addHermitianConjugates():
    
    input = {}
    input[tuple([1,-2])] = 1.
    output = {}
    output[tuple([1,-2])] = 1.
    output[tuple([2,-1])] = 1.
    assert generateOplist.addHermitianConjugates(input) == output
    
    