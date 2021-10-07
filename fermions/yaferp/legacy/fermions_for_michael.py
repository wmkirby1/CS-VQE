import copyimport numpyimport timeimport numpy.randomdef oplistCoefficientNorm(oplist):    if not isinstance(oplist[0],list):        return abs(oplist[0])    else:        norm = 0        for op in oplist:            norm = norm + (op[0] ** 2)        norm = numpy.sqrt(norm)    return norm            def groupedOplistsDeviation(listOplists):    '''calculate the standard deviation of the norm of the coefficients of the terms within each group.'''    listNorms = [0]*len(listOplists)    for i in range(len(listOplists)):        listNorms[i] = oplistCoefficientNorm(listOplists[i])    sd = numpy.std(listNorms)    return sddef doesTermCommuteWithGroup(op,groupOplist):    if not isinstance(groupOplist[0],list):        return checkCommute(op,groupOplist)    for groupOp in groupOplist:        if not checkCommute(op,groupOp):            return False    return Truedef doGroupsCommute(group1,group2):    for op1 in group1:        for op2 in group2:            if not checkCommute(op1[1],op2[1]):                return False    return True    def oplistToGroups(oplist):    groups = []    for op in oplist:        groups.append([op])    return groupsdef findBestGroupMerge(group,listGroups):    '''take one group of oplists and a list of grouped oplists, find the best choice of merger    including just having the first group as a new group in the list'''    listGroups.append(group)    bestStandardDeviation = groupedOplistsDeviation(listGroups)    listGroups.pop()    lengthGroup = len(group)    currentBestGroupMergeIndex = -1    for i in range(len(listGroups)):        if doGroupsCommute(group,listGroups[i]):            listGroups[i] = listGroups[i] + group            thisStandardDeviation = groupedOplistsDeviation(listGroups)            if abs(thisStandardDeviation) < abs(bestStandardDeviation):                bestStandardDeviation = thisStandardDeviation                currentBestGroupMergeIndex = i            listGroups[i] = listGroups[i][:-lengthGroup]    return (currentBestGroupMergeIndex,bestStandardDeviation)            def greedyGroupPickMerge(listGroups):    bestStandardDeviation = groupedOplistsDeviation(listGroups)    bestGroupToMergeIndex = -1    bestRelativeMergeIndex = -1    for i in range(len(listGroups)):        thisGroup = listGroups.pop(i)        (mergeIndex,thisStandardDeviation) = findBestGroupMerge(thisGroup,listGroups)        if abs(thisStandardDeviation) < abs(bestStandardDeviation):            bestStandardDeviation = thisStandardDeviation            bestGroupToMergeIndex = i            bestRelativeMergeIndex = mergeIndex        listGroups.insert(i,thisGroup)    return(bestGroupToMergeIndex,bestRelativeMergeIndex)            def mergeRelative(listGroups, groupToMergeIndex, relativeMergeIndex):    groupToMerge = listGroups.pop(groupToMergeIndex)    listGroups[relativeMergeIndex] = listGroups[relativeMergeIndex] + groupToMerge    return listGroups                        def greedyGroupOplist(oplist):    listGroups = copy.deepcopy(oplist)    listGroups = oplistToGroups(listGroups)    boolFinished = 0    while not boolFinished:        (bestGroupToMergeIndex,bestRelativeMergeIndex) = greedyGroupPickMerge(listGroups)        if ((not (bestGroupToMergeIndex == -1)) and (not (bestRelativeMergeIndex == -1))):            listGroups = mergeRelative(listGroups,bestGroupToMergeIndex,bestRelativeMergeIndex)        else:            boolFinished = 1    return listGroupsdef groupCommutingTerms(rawHamiltonian):    '''group commuting terms in a hamiltonian, return a list of groups.    note for now this randomly selects terms to find other commuting terms,    iterates MAX_ITERATIONS times, selecting the best choice.    obviously this is not the best way of doing this.        TODO:  !!!! make this return explicitly, at the moment the interpreter can    get confused and references to different subgroups can cause the groups to be    recalculated, leading to the subgroups being different, leading to chaos.!!! '''    startTime = time.clock()    MAX_ITERATIONS = 1    storedGroupsList = []    numStoredGroups = -1    for iteration in range(MAX_ITERATIONS): #try procedure MAX_ITERATIONS times        unsortedHamiltonian = list(rawHamiltonian)        newGroupsList = []                while len(unsortedHamiltonian) > 0: #while we have unsorted terms in the Hamiltonian            randomIndex = numpy.random.random_integers(0,len(unsortedHamiltonian)-1)            firstTermInNewGroup = list(unsortedHamiltonian[randomIndex])            newGroup = [firstTermInNewGroup]            unsortedHamiltonian.pop(randomIndex)            tmpUnsortedHamiltonian = list(unsortedHamiltonian)            for term in unsortedHamiltonian:                boolAddTerm = 1                 for termInNewGroup in newGroup: #for all terms in the group                        #abort if we've already detected noncommutativity                    if not checkCommute(termInNewGroup[1],term[1]): #check commutativity                        boolAddTerm = 0 #don't add term if noncommutativity                        break #                   if boolAddTerm == 1: #having checked term against entire subset, add term if commutes with all.                if boolAddTerm:                    newGroup.append(term)                    tmpUnsortedHamiltonian.remove(term)            newGroupsList.append(newGroup)            unsortedHamiltonian = list(tmpUnsortedHamiltonian)        print(len(newGroupsList))        '''the following line is crucial - it determines the metric we are using to judge grouping quality.'''             if ((len(newGroupsList) > numStoredGroups) or numStoredGroups < 1): #check to see if this iteration "beats the current record"            storedGroupsList = list(newGroupsList)            numStoredGroups = len(storedGroupsList)        endTime = time.clock()   # print(endTime-startTime)    return storedGroupsListdef checkCommute(oplist1,oplist2):    '''takes oplists of same length, returns true if they commute otherwise false'''    commute = True    for i in range(len(oplist1)):        if oplist1[i] and oplist2[i] and oplist1[i] != oplist2[i]:            commute = not commute    return commute                        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  TESTS  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
