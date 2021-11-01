# Automates the construction of quasi-quantized models for noncontextual Hamiltonians, as described in https://arxiv.org/abs/1904.02260 and ...
# William M. Kirby, 2020
# IMPORTANT VCS NOTE:  This is only in the SVN to make my life easier moving it between systems.  It's probably an outdated version.  - AT

import numpy as np
from openfermion.ops import QubitOperator
from openfermion.utils import commutator
from functools import reduce
import itertools


########################
### Helper functions ###
########################

# Takes a Pauli operator specified as a string (e.g. 'XZIIYXY') and returns the corresponding QubitOperator:
def qo(s):
    out = ''
    for i in range(len(s)):
        if s[i] != 'I':
            out = out + s[i] + str(i+1) + ' '
    out = out[:-1]
    return QubitOperator(out)

# Takes two Pauli operators specified as strings and determines whether they commute:

from yaferp.general import fermions
from functools import lru_cache
@lru_cache(maxsize=None) #cache the commutator results
def commute(x,y):
    #return commutator(qo(x),qo(y))==0*QubitOperator() #sorry openfermion, not today!
    pauliString1 = pauliStringStr2List(x)
    pauliString2 = pauliStringStr2List(y)
    result = fermions.checkCommute(pauliString1, pauliString2)
    return result

# Input: S, a list of Pauli operators specified as strings.
# Output: a boolean indicating whether S is contextual or not.
# Runtime: O(|S|**3).
def contextualQ(S,verbose=False):
    # Store T all elements of S that anticommute with at least one other element in S (takes O(|S|**2) time).
    T=[]
    Z=[] # complement of T
    for i in range(len(S)):
        if any(not commute(S[i],S[j]) for j in range(len(S))):
            T.append(S[i])
        else:
            Z.append(S[i])
    # Search in T for triples in which exactly one pair anticommutes; if any exist, S is contextual.
    for i in range(len(T)): # WLOG, i indexes the operator that commutes with both others.
        for j in range(len(T)):
            for k in range(j,len(T)): # Ordering of j, k does not matter.
                if i!=j and i!=k and commute(T[i],T[j]) and commute(T[i],T[k]) and not commute(T[j],T[k]):
                    return True
    if verbose:
        return False,Z,T
    else:
        return False
    
# Input: ham, a Hamiltonian specified as a dict mapping Pauli strings to coefficients.
# Output: a boolean indicating whether ham is contextual or not.
# Runtime: O(|S|**3).
def contextualQ_ham(ham,verbose=False):
    S = list(ham.keys())
    # Store T all elements of S that anticommute with at least one other element in S (takes O(|S|**2) time).
    T=[]
    Z=[] # complement of T
    for i in range(len(S)):
        if any(not commute(S[i],S[j]) for j in range(len(S))):
            T.append(S[i])
        else:
            Z.append(S[i])
    # Search in T for triples in which exactly one pair anticommutes; if any exist, S is contextual.
    for i in range(len(T)): # WLOG, i indexes the operator that commutes with both others.
        for j in range(len(T)):
            for k in range(j,len(T)): # Ordering of j, k does not matter.
                if i!=j and i!=k and commute(T[i],T[j]) and commute(T[i],T[k]) and not commute(T[j],T[k]):
                    return True
    if verbose:
        return False,Z,T
    else:
        return False

# determine the action of a Pauli operator P (a QubitOperator) on qubit i (e.g. for qo('XYZI',0), returns 'X'):
def pauli_action(P,i):
    xi=QubitOperator('X'+str(i))
    yi=QubitOperator('Y'+str(i))
    zi=QubitOperator('Z'+str(i))
    if (commutator(P,xi)==0*QubitOperator()) and (commutator(P,yi)!=0*QubitOperator()) and (commutator(P,zi)!=0*QubitOperator()):
        return 'X'
    elif (commutator(P,xi)!=0*QubitOperator()) and (commutator(P,yi)==0*QubitOperator()) and (commutator(P,zi)!=0*QubitOperator()):
        return 'Y'
    elif (commutator(P,xi)!=0*QubitOperator()) and (commutator(P,yi)!=0*QubitOperator()) and (commutator(P,zi)==0*QubitOperator()):
        return 'Z'
    else:
        return 'I'

# multiply two Pauli operators p,q, represented as strings
# output has the form [r, sgn], where r is a Pauli operator specified as a string,
# and sgn is the complex number such that p*q == sgn*r.
def pauli_mult(p,q):
    assert(len(p)==len(q))
    sgn=1
    out=''
    for i in range(len(p)):
        if p[i]=='I':
            out+=q[i]
        elif q[i]=='I':
            out+=p[i]
        elif p[i]=='X':
            if q[i]=='X':
                out+='I'
            elif q[i]=='Y':
                out+='Z'
                sgn=sgn*1j
            elif q[i]=='Z':
                out+='Y'
                sgn=sgn*-1j
        elif p[i]=='Y':
            if q[i]=='Y':
                out+='I'
            elif q[i]=='Z':
                out+='X'
                sgn=sgn*1j
            elif q[i]=='X':
                out+='Z'
                sgn=sgn*-1j
        elif p[i]=='Z':
            if q[i]=='Z':
                out+='I'
            elif q[i]=='X':
                out+='Y'
                sgn=sgn*1j
            elif q[i]=='Y':
                out+='X'
                sgn=sgn*-1j
    return [out,sgn]


#######################################
### Generate quasi-quantized models ###
#######################################

# Given a commuting set of Pauli strings, input as a dict mapping each to None,
# return a independent generating set in the same format, together with the dict mapping
# each original element to its equivalent product in the new set
def to_indep_set(G_w_in):
    G_w = G_w_in
    G_w_keys = [[str(g),1] for g in G_w.keys()]
    G_w_keys_orig = [str(g) for g in G_w.keys()]
    generators = []
    for i in range(len(G_w_keys[0][0])):
        # search for first X,Y,Z in ith position (not including operators that are already in generators)
        fx=None
        fy=None
        fz=None
        j=0
        while fx==None and j<len(G_w_keys):
            if G_w_keys[j][0][i]=='X' and not any(G_w_keys[j][0]==g[0] for g in generators):
                fx=G_w_keys[j]
            j+=1
        j=0
        while fy==None and j<len(G_w_keys):
            if G_w_keys[j][0][i]=='Y' and not any(G_w_keys[j][0]==g[0] for g in generators):
                fy=G_w_keys[j]
            j+=1
        j=0
        while fz==None and j<len(G_w_keys):
            if G_w_keys[j][0][i]=='Z' and not any(G_w_keys[j][0]==g[0] for g in generators):
                fz=G_w_keys[j]
            j+=1
        # multiply to eliminate all other nonidentity entries in ith position
        if fx!=None:
            generators.append(fx)
            for j in range(len(G_w_keys)):
                if G_w_keys[j][0][i]=='X': # if any other element of G_w has 'X' in the ith position...
                    # multiply it by fx
                    G_w[G_w_keys_orig[j]]=G_w[G_w_keys_orig[j]]+[fx]
                    sgn=G_w_keys[j][1]*fx[1]
                    G_w_keys[j]=pauli_mult(G_w_keys[j][0],fx[0])
                    G_w_keys[j][1]=G_w_keys[j][1]*sgn
        
        if fz!=None:
            generators.append(fz)
            # if any other element of G_w has 'Z' in the ith position...
            for j in range(len(G_w_keys)):
                if G_w_keys[j][0][i]=='Z': 
                    # multiply it by fz
                    G_w[G_w_keys_orig[j]]=G_w[G_w_keys_orig[j]]+[fz] # update the factor list for G_w_keys[j]
                    sgn=G_w_keys[j][1]*fz[1] # update the sign for G_w_keys[j]
                    G_w_keys[j]=pauli_mult(G_w_keys[j][0],fz[0]) # multiply G_w_keys[j] by fz...
                    G_w_keys[j][1]=G_w_keys[j][1]*sgn # ... and by the associated sign.
        
        if fx!=None and fz!=None:
            for j in range(len(G_w_keys)):
                if G_w_keys[j][0][i]=='Y': # if any other element of G_w has 'Y' in the ith position...
                    # multiply it by fx and fz
                    G_w[G_w_keys_orig[j]]=G_w[G_w_keys_orig[j]]+[fx,fz]
                    sgn=G_w_keys[j][1]*fx[1]
                    G_w_keys[j]=pauli_mult(G_w_keys[j][0],fx[0])
                    G_w_keys[j][1]=G_w_keys[j][1]*sgn
                    sgn=G_w_keys[j][1]*fz[1]
                    G_w_keys[j]=pauli_mult(G_w_keys[j][0],fz[0])
                    G_w_keys[j][1]=G_w_keys[j][1]*sgn
        # If both fx and fz are not None, then at this point we are done with this position.
        # Otherwise, there may be remaining 'Y's at this position:
        elif fy!=None:
            generators.append(fy)
            # if any other element of G_w has 'Y' in the ith position...
            for j in range(len(G_w_keys)):
                if G_w_keys[j][0][i]=='Y': 
                    # multiply it by fy
                    G_w[G_w_keys_orig[j]]=G_w[G_w_keys_orig[j]]+[fy]
                    sgn=G_w_keys[j][1]*fy[1]
                    G_w_keys[j]=pauli_mult(G_w_keys[j][0],fy[0])
                    G_w_keys[j][1]=G_w_keys[j][1]*sgn                    
    for j in range(len(G_w_keys)):
        G_w[G_w_keys_orig[j]]=G_w[G_w_keys_orig[j]]+[G_w_keys[j]]
    
    return generators,G_w


# Input: a noncontextual Hamiltonian encoded as a dict of the form e.g. {'III':0.123, 'XII':1.234, 'YII':-5.678,...}
# Output: 

def quasi_model(ham_dict):
    terms = [str(k) for k in ham_dict.keys()]
    terms_qo = [qo(t) for t in terms] # terms as a list of QubitOperators
    assert(not contextualQ(terms)) # Hamiltonian should be noncontextual
    c,Z,T = contextualQ(terms,verbose=True) # get set of universally-commuting terms, Z, and its complement, T
#     for i in range(len(Z)):
#         j = terms_qo.index(Z[i])
#         Z[i] = terms[j]
#     for i in range(len(T)):
#         j = terms_qo.index(T[i])
#         T[i] = terms[j]
    
    # Partition T into cliques:
    C=[]
    while T:
        C.append([T.pop()]) # remove the last element from T and put it in a new sublist in C
        for i in range(len(T)-1,-1,-1): # among the remaining elements in T...
            t=T[i]
            if commute(C[-1][0],t): # check if each commutes with the current clique
                C[-1].append(t) # if so, add it to the current clique...
                T.remove(t) # and remove it from T
                
    # Get full set of universally-commuting component operators:
    Gprime = [[z,1] for z in Z] # elements are stored together with their sign
    Ci1s=[]
    for Cii in C: # for each clique...
        Ci=Cii
        Ci1=Ci.pop() # pull out one element
        Ci1s.append(Ci1) # append it to a list of these
        for c in Ci: Gprime.append(pauli_mult(c,Ci1)) # add the remaining elements, multiplied by Ci1, to the commuting set
    
    # Get independent generating set for universally-commuting component operators:
    G_p = dict.fromkeys([g[0] for g in Gprime],[])
    G,G_mappings = to_indep_set(G_p)
    
    # Remove duplicates and identities from G:
    G = list(dict.fromkeys([g[0] for g in G]))
    # Remove identities from product list:
    i=len(G)-1
    while i>=0:
        if qo(G[i]) == 1.0*QubitOperator(''):
            del G[i]
        i=i-1
    
    # Rewrite the values in G_mappings as lists of the form e.g. [sgn, 'XYZ', 'XZY',...]:
    Gprime = list(dict.fromkeys([g[0] for g in Gprime]))
    for g in G_mappings.keys():
        ps = G_mappings[g]
        sgn = int(np.real(np.prod([p[1] for p in ps])))
        ps = [[p[0] for p in ps],sgn]
        # Remove identities from product list:
        i=len(ps[0])-1
        while i>=0:
            if qo(ps[0][i]) == 1.0*QubitOperator(''):
                del ps[0][i]
            i=i-1
        G_mappings[g] = ps
        
    # Assemble all the mappings from terms in the Hamiltonian to their products in R:
    all_mappings = dict.fromkeys(terms)
    for z in Z:
        mapping = G_mappings[z]
        all_mappings[z] = [mapping[0]]+[[]]+[mapping[1]]
        
    for Ci1 in Ci1s:
        all_mappings[Ci1] = [[],[Ci1],1]
    
    for i in range(len(C)):
        Ci=C[i]
        Ci1=Ci1s[i]
        for Cij in Ci:
            mult = pauli_mult(Cij,Ci1)
            mapping = G_mappings[mult[0]]
            all_mappings[Cij] = [mapping[0]]+[[Ci1]]+[mult[1]*mapping[1]]
    
    return G,Ci1s,all_mappings

# Input: a Hamiltonian ham_dict, two lists of epistemic parameters, q and r, and model, the output of a quasi-model.
# Output: the corresponding energy objective function, encoded as a list with the following form:
# [ dim of q, dim of r, list of form [coeff, indices of q's, indices of r's] ]
def energy_function_form(ham_dict,model):
    terms = [str(k) for k in ham_dict.keys()]
    q = model[0]
    r = model[1]
    out = []
    for t in terms:
        mappings = model[2][t]
        coeff = ham_dict[t]*mappings[2] # mappings[2] is the sign
        q_indices = [q.index(qi) for qi in mappings[0]]
        r_indices = [r.index(ri) for ri in mappings[1]]
        out.append([coeff, q_indices, r_indices])
    return [len(q),len(r),out]

# Given fn_form, the output of an energy_function_form, returns the corresponding function definition.
def energy_function(fn_form):
    dim_q = fn_form[0]
    return lambda *args: np.real(sum(
        [
            (t[0] if len(t[1])==0 and len(t[2])==0 else
            (t[0]*(reduce(lambda x, y: x * y, [args[i] for i in t[1]]))) if len(t[1])>0 and len(t[2])==0 else
            (t[0]*(reduce(lambda x, y: x * y, [args[dim_q+i] for i in t[2]]))) if len(t[1])==0 and len(t[2])>0 else
            (t[0]*(reduce(lambda x, y: x * y, [args[i] for i in t[1]]))*(reduce(lambda x, y: x * y, [args[dim_q+i] for i in t[2]]))))
            for t in fn_form[2]
        ]
    ))


##############################################
### Finding noncontextual sub-Hamiltonians ###
##############################################


# Returns the highest-weight noncontextual set containing candidate and some subset of remaining_terms.
def best_noncon(candidate,remaining_terms,ham_dict):
    if remaining_terms == []:
        return candidate # no more terms to add
    else:
        current_best = candidate
        for i in range(len(remaining_terms)):
            next_candidate = candidate + [remaining_terms[i]]
            if not contextualQ(next_candidate):
                #print(next_candidate,reduce(lambda x,y: x+y,[abs(ham_dict[t]) for t in next_candidate])-reduce(lambda x,y: x+y,[abs(ham_dict[t]) for t in current_best]))
                #print(current_best)
                # If next_candidate is noncontextual, try to add even more terms to it:
                next_candidate = best_noncon(next_candidate,remaining_terms[i+1:],ham_dict)
                # Check whether next_candidate has more weight than current_best:
                if reduce(lambda x,y: x+y,[abs(ham_dict[t]) for t in next_candidate]) > reduce(lambda x,y: x+y,[abs(ham_dict[t]) for t in current_best]):
                    # If it does, update current_best:
                    current_best = next_candidate
#             else:
#                 print(next_candidate,'contextual')
        return current_best


# Input: a Hamiltonian ham_dict specified as a dict mapping Pauli strings to coefficients.
# Output: an approximation of the highest-weight noncontextual subset of the terms in H.
# Assumes that the highest-weight commuting subset is the strings over {I,Z}.
# Output is exact if H is itself noncontextual.

# WARNING: runtime is exponential in the number of terms that are not strings over {I,Z}.

def toNCHamiltonian(ham_dict):
    if not contextualQ([str(t) for t in ham_dict.keys()]):
        return ham_dict
    S = [str(t) for t in ham_dict.keys()] # all terms
    Z = [] # max commuting set
    T = [] # S\Z
    for s in S:
        if all(s[i]=='I' or s[i]=='Z' for i in range(len(s))):
            Z.append(s)
        else:
            T.append(s)
    print(Z,'\n')
    print(T,'\n')
    # Add as high total-weight terms as possible from T while remaining noncontextual: depth-first search to save space.
    best_noncon_set = best_noncon(Z,T,ham_dict)
    return {t:ham_dict[t] for t in best_noncon_set}




# Input: a Hamiltonian H specified as a dict mapping Pauli strings to coefficients.
# Output: an approximation of the highest-weight noncontextual subset of the terms in H.
# Assumes that the highest-weight commuting subset is the strings over {I,Z}.
# Output is exact if H is itself noncontextual.

# Guaranteed to run in polynomial time in the number of terms.
def toNCHamiltonian_fast(ham_dict,weighting=None,strategy='greedy',step_size=1,show_progress=True):
#     if not contextualQ([str(t) for t in ham_dict.keys()]): # O(n^3)
#         return ham_dict
    if weighting==None:
        weights = {t:abs(ham_dict[t]) for t in ham_dict}
    else:
        weights = weighting
    S = [k for k, v in sorted(weights.items(), key=lambda item: item[1])] # sort in increasing order of weight: O(n^2)
    candidate = []
    remaining = [] # S\candidate
    if strategy == 'chem':
        for s in S: # O(n^3)
            if all(s[i]=='I' or s[i]=='Z' for i in range(len(s))):
                candidate.append(s)
            else:
                remaining.append(s)
    if strategy == 'greedy':
        candidate = []
        remaining = S
    Z = []
    T = []
    init_len = len(remaining)
    if step_size > 0:
        s = step_size
        while s > 0:
            complete = False
            while not complete:
                if show_progress:
                    print('Progress: ',100*(1-len(remaining)/init_len)-100*(1-len(remaining)/init_len)%1,'%',end='     \r')
                # possible additional sets of size s to be added:
                poss_add_sets = list(itertools.combinations(remaining,s))
                poss_add_sets_dict = {}
                for add_set in poss_add_sets:
                    poss_add_sets_dict[add_set] = reduce(lambda x, y: x + y,[weights[t] for t in add_set])
                # sort possible additional sets in increasing order of total weight:
                poss_add_sets = [k for k, v in sorted(poss_add_sets_dict.items(), key=lambda item: item[1])] # O(n^(2*step_size))
                # try to add a set, heaviest first:
                added = False
                while (not added) and (poss_add_sets != []):
                    next_set = list(poss_add_sets.pop())
                    candidate_temp = list(candidate)
                    Z_temp = list(Z)
                    T_temp = list(T)
                    bad = False
                    if not bad:
                        # Try adding the elements of next_set:
                        for t in next_set:
                            # If adding t will result in a set of size 3 or less, noncontextual automatically.
                            if len(candidate_temp)<3:
                                # If t commutes universally, can be added to Z:
                                if all(commute(t,c) for c in candidate_temp): # O(n)
                                    Z_temp.append(t)
                                else:
                                    T_temp.append(t)
                            else:
                                # If t commutes universally, can be added to Z:
                                if all(commute(t,c) for c in candidate_temp): # O(n)
                                    Z_temp.append(t)
                                else:
                                    # Check whether t is part of a non-transitive subset:
                                    for p in itertools.combinations(T_temp,2):
                                        if commute(t,p[0]) and commute(t,p[1]) and (not commute(p[0],p[1])):
                                            bad = True
                                        if commute(t,p[0]) and (not commute(t,p[1])) and commute(p[0],p[1]):
                                            bad = True
                                        if (not commute(t,p[0])) and commute(t,p[1]) and commute(p[0],p[1]):
                                            bad = True
                                    T_temp.append(t)
                            for u in Z_temp:
                                if not commute(u,t): # O(n)
                                    Z_temp.remove(u)
                                    T_temp.append(u)
                            if not bad:
                                candidate_temp.append(t)
                        if not bad:
                            candidate = candidate_temp
                            Z = Z_temp
                            T = T_temp
                            for t in next_set:
                                remaining.remove(t)
                            added = True
                if not added:
                    complete = True
            # we should now have added as many sets of size s as possible: repeat with sets of size s-1
            s = s-1
    #
    if show_progress:
        print('Progress: ',100,'%                            \n')
    return {t:ham_dict[t] for t in candidate}



############################
#########AT scratch#########
############################

pauliStr2Int = {'I':0,
                'X':1,
                'Y':2,
                'Z':3}
pauliInt2Str = ['I','X','Y','Z']


def pauliStringStr2List(pauliStringStr):
    return [pauliStr2Int[x] for x in pauliStringStr]

def pauliStringList2Str(pauliStringList):
    return ''.join([pauliInt2Str[x] for x in pauliStringList])

def ham_dict2Oplist(hamDict):
    oplist = [[hamDict[x],pauliStringStr2List(x)] for x in hamDict]
    return oplist

def oplist2ham_dict(oplist):
    ham_dict = {pauliStringList2Str(x[1]):x[0] for x in oplist}
    return ham_dict

import yaferp.interfaces.fermilibInterface as fermilibInterface
import yaferp.bomd.trajectories as trajectories

#def geometrySpecToNC(geometry,quantumChemPackage='pyscf')
#    flmGeometry = geometry.cartesian
#    flm = fermilibInterface.FermiLibMolecule(flmGeometry, "STO-3G", geometry.multiplicity, geometry.charge,
#                                             str(geometry), str(geometry))
#    trajectories.runQuantumChemistry(flm, quantumChemPackage)
#    oplist= flm.electronicHamiltonian(0,1e-12)
 #   hamDict = oplist2ham_dict(oplist)
 ##   ncHamDict = toNCHamiltonian_fast(hamDict,
 #                                            weighting=None,
 #                                            strategy='greedy',
 #                                            step_size=1,
 #                                            show_progress=False)
  #  return ncHamDict
#