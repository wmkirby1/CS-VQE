'''
Created on 25 Feb 2015

@author: andrew
'''
import numpy

def testNums(float1,float2):
    float3 = float1-float2
    if abs(float3) == numpy.nan:
        return True
    elif abs(float3) >= 1e-14:
        return False
    else:
        return True

def checkPermutationPhys(integrals,index):
    i = index[0]
    j = index[1]
    k = index[2]
    l = index[3]
    test = integrals[i][j][k][l]==integrals[j][i][l][k]==integrals[k][l][i][j]==integrals[l][k][j][i]==integrals[k][j][i][l] == integrals[l][i][j][k] == integrals[i][l][k][j] == integrals[j][k][l][i]
    return test

def checkAllPermutationsPhys(integrals):
    numOrbs = len(integrals)
    for i in range(numOrbs):
        for j in range(numOrbs):
            for k in range(numOrbs):
                for l in range(numOrbs):
                    works = checkPermutationPhys(integrals,(i,j,k,l))
                    if not works:
                        print(str(i)+str(j)+str(k)+str(l))
    return

def checkPermutationChem(integrals,index):
    i = index[0]
    j = index[1]
    k = index[2]
    l = index[3]
    test = integrals[i][j][k][l]==integrals[k][l][i][j]==integrals[j][i][l][k]==integrals[l][k][j][i]==integrals[j][i][k][l]==integrals[l][k][i][j]==integrals[i][j][l][k]==integrals[k][l][j][i]
    return test

def checkAllPermutationsChem(integrals):
    numOrbs = len(integrals)
    for i in range(numOrbs):
        for j in range(numOrbs):
            for k in range(numOrbs):
                for l in range(numOrbs):
                    works = checkPermutationChem(integrals,(i,j,k,l))
                    if not works:
                        print(str(i)+str(j)+str(k)+str(l))
    return

def checkPermutationRawChem(integrals,index):
    i = index[0]
    j = index[1]
    k = index[2]
    l = index[3]
    thisIntegral = integrals[i][j][k][l]
    if thisIntegral == numpy.NaN:
        return True
    if not testNums(thisIntegral,integrals[k][l][i][j]):
        return False
    if not testNums(thisIntegral,integrals[j][i][l][k]):
        return False
    if not testNums(thisIntegral,integrals[l][k][j][i]):
        return False
    if not testNums(thisIntegral,integrals[j][i][k][l]):
        return False
    if not testNums(thisIntegral,integrals[l][k][i][j]):
        return False
    if not testNums(thisIntegral,integrals[i][j][l][k]):
        return False
    if not testNums(thisIntegral,integrals[k][l][j][i]):
        return False
    return True
def checkAllPermutationsRawChem(integrals):
    numOrbs = len(integrals)
    for i in range(numOrbs):
        for j in range(numOrbs):
            for k in range(numOrbs):
                for l in range(numOrbs):
                    works = checkPermutationRawChem(integrals,(i,j,k,l))
                    if not works:
                        print(str(i)+str(j)+str(k)+str(l))
    return
