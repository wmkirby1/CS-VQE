'''
Created on 27 Oct 2017

@author: andrew
'''
from yaferp.general import fermions
from operator import mul
import sympy
import copy
from functools import reduce


def pauliprod(twoplist):
    phase = 1 + 0j
    
    if twoplist[0] == 0:
        return [phase,twoplist[1]]
    if twoplist[1] == 0:
        return [phase,twoplist[0]]
    
    if twoplist[0] == 1:
        if twoplist[1] == 1:
            return [phase,0]
        if twoplist[1] == 2:
            phase = 0 + 1j
            return [phase,3]
        if twoplist[1] == 3:
            phase = 0 - 1j
            return [phase, 2]
                    
    if twoplist[0] == 2:
        if twoplist[1] == 1:
            phase = 0 - 1j
            return [phase,3]
        if twoplist[1] == 2:
            return [phase,0]
        if twoplist[1] == 3:
            phase = 0 + 1j
            return [phase, 1]

    if twoplist[0] == 3:
        if twoplist[1] == 1:
            phase = 0 + 1j
            return [phase,2]
        if twoplist[1] == 2:
            phase = 0 - 1j
            return [phase,1]
        if twoplist[1] == 3:
            return [phase,0]
    
def op_prod(op1Phase,op2Phase,op1Paulis,op2Paulis):
    
    
    phase = op1Phase*op2Phase
    listPhasesAndStrings = [pauliprod([op1Paulis[i],op2Paulis[i]]) for i in range(len(op1Paulis))]
    phases,strings = zip(*listPhasesAndStrings)
    newPhase = phase * reduce(mul,phases)
    return(newPhase,tuple(strings))
    ''' phase = op1Phase*op2Phase
    listPhasesAndStrings = map((lambda i: pauliprod([op1Paulis[i],op2Paulis[i]])),range(len(op1Paulis)))
    newList = zip(*listPhasesAndStrings)
    newPhase = phase * reduce((lambda x,y: x*y),newList[0])
    return(newPhase,tuple(newList[1]))'''

    '''
    for i in range(len(op1Paulis)):
        [cphase,newnum] = pauliprod([op1Paulis[i],op2Paulis[i]])
        newnumlist.append(newnum)
        phase = cphase*phase'''
   # return (phase,tuple(newnumlist))  

class ferm_op:
    def __init__(self,j,n,an_cr,JW_BK):
        self.dim = n
        self.index = j
        if an_cr == 1:
            self.kind = 'Creation'
        if an_cr == 0:
            self.kind = 'Annihilation'
        if JW_BK == 0:
            self.encoding = 'Jordan-Wigner'
        if JW_BK == 1:
            self.encoding = 'Bravyi-Kitaev'
        self.oplist = fermions.oplist_create(j, n, an_cr, JW_BK)
        self.opDict = opDict(self.oplist)
        #self.algexp = fermions.cleandisplay(self.oplist)

    def __repr__(self):
                return self.kind + ' operator acting on qubit ' + str(self.index) + ' of ' + str(self.dim) + ' total qubits in the ' + self.encoding + ' basis.'

    def op(self,state):
        if self.encoding == 'Jordan-Wigner':
            if self.kind == 'Creation':
                return fermions.jw_cr_op(state, self.index)
            if self.kind == 'Annihilation':
                return fermions.jw_an_op(state, self.index)
        if self.encoding == 'Bravyi-Kitaev':
            if self.kind == 'Creation':
                return fermions.bk_cr_op(state, self.index)
            if self.kind == 'Annihilation':
                return fermions.bk_an_op(state, self.index)


class opDict(dict):
    '''
    classdocs
    '''


    def __init__(self, oplist=None,cutoff=1e-12):
        dict.__init__(self)
        if not oplist == None:
            for item in oplist:
                self[tuple(item[1])] =item[0]
        self.cutoff=cutoff
        '''
        Constructor
        '''
    
    def sum(self,other):
        #in-place
        for otherKey in other:
            if otherKey in self:
                self[otherKey] += other[otherKey]
                if abs(self[otherKey]) < self.cutoff:
                    self.pop(otherKey)
            else:
                self[otherKey] = other[otherKey]
        return self

    def product(self,other):
        #in-place
        result = opDict()
        if type(other) is opDict:
            for thisPString in self:
                for otherPString in other:
                    coeff,newString = op_prod(self[thisPString],other[otherPString],thisPString,otherPString)
                    if newString in result:
                        result[newString] += coeff
                    else:
                        result[newString] = coeff
        else:
            for key in self:
                self[key] = self[key] * other
            result = self
        return result
    
    def removeNegligibles(self,cutoff=1e-12):
        if cutoff > self.cutoff:
            self.cutoff = cutoff

        stringsToDelete = [pString for pString in list(self.keys()) if abs(self[pString]) < self.cutoff]
        for pString in stringsToDelete:
            del self[pString]
        #for pString, coeff in self.items():
        #    if abs(coeff) < self.cutoff:
        #        del self[pString]
        return self
    
    def oplist(self):
        data = self.items()
        fred = list(map((lambda x: list(reversed(x))), data))
        return fred

class parameterisedOptDict(dict):
    '''like an optdict, but stores values as a pair (parameterisation,startingvalue).
    parameterisation is a sympy expression
    TODO: make this less dogshit'''


    def __init__(self, opdict, startingValue, cutoff=1e-12,negated=False):
        dict.__init__(self)
        for term in opdict:
            if negated:
                self[term] = opdict[term] * -1. * sympy.symbols('a0',real=True)
            else:
                self[term] = opdict[term] * sympy.symbols('a0',real=True)
        self.cutoff = cutoff
        self.parameters = dict()
        self.parameters[sympy.symbols('a0',real=True)] = startingValue
        '''
        Constructor
        '''

    def nextParameterIndex(self):
        return len(self.parameters)

    def relabelParameter(self,oldSymbol,newSymbol):
        for key in self:
            oldTerm = self[key]
            newTerm = oldTerm.subs(oldSymbol,newSymbol)
            self[key] = newTerm
        self.parameters[newSymbol] = self.parameters.pop(oldSymbol)
        return

    def parametersOverlap(self,otherParameterisedOptDict):

        return set(self.parameters.keys()).intersection(set(otherParameterisedOptDict.parameters.keys()))

    def findParameterRemap(self,otherParameterisedOptDict):
        parametersToBeRenamed = self.parametersOverlap(otherParameterisedOptDict)
        theMap = {}
        testIndex = 0
        for thisParameter in parametersToBeRenamed:
            looping = True
            while looping:
                testSymbol = sympy.symbols('a{}'.format(testIndex),real=True)
                if not testSymbol in self.parameters:
                    if not testSymbol in otherParameterisedOptDict.parameters:
                        theMap[thisParameter] = testSymbol
                        looping = False
                testIndex += 1
        #print (theMap)
        return theMap

    def relabelMappedParameters(self,map):
        for thisParameter in map:
            self.relabelParameter(thisParameter,map[thisParameter])
        return self



    def sum(self, thing):
        other = copy.deepcopy(thing)
       # print(self.parameters)
        # in-place
        parametersRenameMap = self.findParameterRemap(other)
        #print(self.parameters) #gives result 0
        other.relabelMappedParameters(parametersRenameMap)
        #print(self.parameters) #gives result 1 jesus christ fuck this shit
        for otherKey in other:
            if otherKey in self:
                self[otherKey] += other[otherKey]
            else:
                self[otherKey] = other[otherKey]
       # print(self)
       # print('\n\n')
        #print(other)
        #print('\n\n')
        #print(self.parameters)
        fred = self.parameters.copy()
        #print(fred)
        #print(other.parameters)
        fred.update(other.parameters)
        self.parameters = fred
        return self

    def removeNegligibles(self, cutoff=1e-12):
        if cutoff > self.cutoff:
            self.cutoff = cutoff

        for pString, coeff in self.items():
            if abs(coeff[1]) < self.cutoff:
                del self[pString]
        return self

    def collectConstants(self):
        newStuff = {}
        for x in self:
            #print(self[x])
            newStuff[x] = sympy.separatevars(self[x])
        self.update(newStuff)
        return self



    def _separateConstants_(self):
        self.collectConstants()
        result = {}
        for x in self:
            if self[x].func == sympy.Mul:
                #potentialConstant = self[x].args[0]
                potentialConstant = [(i,y) for i,y in enumerate(self[x].args) if y.is_constant()]
                if potentialConstant:
                    argsWithoutConstant = [y for i,y in enumerate(self[x].args) if not (i,y) in potentialConstant]
                    exprWithoutConstant = sympy.Mul(*argsWithoutConstant)
                    result[x] = (sympy.Mul(*[y[1] for y in potentialConstant]),exprWithoutConstant)
        return result

    def optimise(self):
        thing = self._separateConstants_();
        uniqueDenormalisedCoefficients = list(set([x[1] for x in thing.values()]))
        sympyMagicBullshit = sympy.cse([sympy.collect_const(x) for x in uniqueDenormalisedCoefficients])
        denormalisedCoefficientsMap = {uniqueDenormalisedCoefficients[i]:sympyMagicBullshit[1][i] for i in range(len(uniqueDenormalisedCoefficients))}
        #print(denormalisedCoefficientsMap)
        newThing = {}
        for x in thing:
            newThing[x] = (thing[x][0],denormalisedCoefficientsMap[thing[x][1]])

        newCoefficientValues = {x[0]:x[1].subs(self.parameters) for x in sympyMagicBullshit[0]}
        #print(newCoefficientValues)
        #print(newThing)
        newValues = {}
        for x in newThing:
            newValues[x] = newThing[x][0] * newThing[x][1]
        #print(newValues)
        self.update(newValues)
        self.parameters.update(newCoefficientValues)
        self.killUnusedParameters()
        #print(self)
        #print('\n\n\n')
        #print(self.parameters)
       # substitutions = {x[1]:x[0]sympyMagicBullshit[0]
        #newUniqueDenormalisedCoefficients = sympyMagicBullshit[1]
        #denormalisedCoefficientMap = {uniqueDenormalisedCoefficients[i]:newUniqueDenormalisedCoefficients[i] for i in range(len(uniqueDenormalisedCoefficients))}
        #newMe = copy.deepcopy(self)
       # for x in newMe:
       #     newMe[x] = denormalisedCoefficientMap[self[x]]
        #print(newMe)

        #theMap = {}
        #for i,x in enumerate(uniqueCoefficients):
        #    theMap[x] = sympy.symbols('c{}'.format(i))
        #print (theMap)



        return self

    def sumSkippingRelabel(self,other):
        for x in other:
            if not x in self:
                self[x] = other[x]
            else:
                self[x] += other[x]
        self.parameters.update(other.parameters)
        self.killZeroTerms()
        self.killUnusedParameters()
        return self

    def killZeroTerms(self):
        killlist = []
        for x in self:
            if self[x] == 0:
                killlist.append(x)
        for x in killlist:
            self.pop(x)
        return

    def killUnusedParameters(self):
        freeSymbols = []
        for x in self:
            for y in self[x].free_symbols:
                freeSymbols.append(y)
        uniqueSymbols = set(freeSymbols)
        newParameters = {}
        for x in self.parameters:
            if x in uniqueSymbols:
                newParameters[x] = self.parameters[x]
        self.parameters = newParameters
        return

    def toOplist(self):
        theOplist = []
        for term in self:
            theOplist.append([self[term],term])
        return (theOplist,self.parameters)



def twoProd(i,j,n,option1,option2,JW_BK,coefficient=1):
    opDict1 = ferm_op(i,n,option1,JW_BK).opDict
    opDict2 = ferm_op(j,n,option2,JW_BK).opDict
    result = opDict1.product(opDict2).product(coefficient)
    return result

def fourProd(i,j,k,l,n,option1,option2,option3,option4,JW_BK,coefficient=1):
    opDict1 = ferm_op(i,n,option1,JW_BK).opDict
    opDict2 = ferm_op(j,n,option2,JW_BK).opDict
    opDict3 = ferm_op(k,n,option3,JW_BK).opDict
    opDict4 = ferm_op(l,n,option4,JW_BK).opDict
    result = opDict1.product(opDict2).product(opDict3).product(opDict4).product(coefficient)
    return result

def oneElectronHamiltonian(numOrbitals,boolJordanOrBravyi, integrals,verbose=False,cutoff=1e-12):
    '''calculate the one electron hamiltonian
    numOrbitals: integer number of orbitals
    boolJordanOrBravyi:  boolean, 0 for jordan-wigner mapping, 1 for bravyi-kitaev mapping
    integrals:  2D tensor of one electron integrals'''
    result = opDict()
    for i in range(numOrbitals):
        for j in range(numOrbitals):
            if verbose:
                print(str(i)+str(j))
            thisTwoProd = twoProd(i,j,numOrbitals,1,0,boolJordanOrBravyi,integrals[i][j])
            result.sum(thisTwoProd)
    result.removeNegligibles(cutoff)
    return result

#from memory_profiler import profile
#@profile
def twoElectronHamiltonian(numOrbitals,boolJordanOrBravyi, integrals, negateIntegrals = False,verbose=False,cutoff=1e-12):
    '''calculate the two electron hamiltonian
    numOrbitals: integer number of orbitals
    boolJordanOrBravyi:  boolean, 0 for jordan-wigner mapping, 1 for bravyi-kitaev mapping
    integrals:  4D tensor of two electron integrals'''
    result = opDict()
    #fourProds = [fourProd(i,j,k,l,numOrbitals,1,1,0,0,boolJordanOrBravyi,(1 - 2*negateIntegrals) * 0.5*integrals[i][j][k][l])
    #             for i in range(numOrbitals)
    #             for j in range(numOrbitals)
     #            for k in range(numOrbitals)
     #            for l in range(numOrbitals)]
    for i in range(numOrbitals):
        for j in range(numOrbitals):
            for k in range(numOrbitals):
                for l in range(numOrbitals):
                    if integrals[i][j][k][l]:
                        if verbose:
                            print(str(i)+str(j)+str(k)+str(l))
                        thisFourProd = fourProd(i,j,k,l,numOrbitals,1,1,0,0,boolJordanOrBravyi,(1 - 2*negateIntegrals) * 0.5*integrals[i][j][k][l])
                        result.sum(thisFourProd)
    result.removeNegligibles(cutoff)
    return result

def electronicHamiltonian(numOrbitals,boolJordanOrBravyi,oneEIntegrals,twoEIntegrals, negateTwoEIntegrals=False,verbose=False,cutoff=1e-12):
    oneEHam = oneElectronHamiltonian(numOrbitals,boolJordanOrBravyi,oneEIntegrals,verbose,cutoff)
    twoEHam = twoElectronHamiltonian(numOrbitals,boolJordanOrBravyi,twoEIntegrals,negateTwoEIntegrals,verbose,cutoff)
    result = oneEHam.sum(twoEHam)
    return result.oplist()
