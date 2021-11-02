import datetime
import pickle
import os
from yaferp.bomd import GeometrySpec, IndependentAdaptiveIntegrator, trajectoriesConstants, trajectories, \
    AdaptiveIntegrator

H3_TRAJECTORIES_DATA_DIR = '/home/andrew/data/molecular_dynamics/H3/'
H2_TRAJECTORIES_DATA_DIR = '/home/andrew/data/molecular_dynamics/H2/'
def makeFilename(startingCoords,startingVels,gradientFunction,i,returnResources,returnSamples):
    vals = [str(startingCoords),str(startingVels),gradientFunction.__name__,str(i)]

    if returnResources:
        vals.append('resources')
    if returnSamples:
        vals.append('samples')
    return '_'.join(vals) + '.dat'

def makeFilenameAdaptive(startingCoords,
                         startingVels,
                         badGradientFunction,
                         goodGradientFunction,
                         i,
                         returnResources,
                         returnSamples,
                         tolerance,
                         testerFunction,
                         badEnergyFunction=None,
                         goodEnergyFunction=None,
                         branch=False,
                         independent=True):
    vals = [str(startingCoords),str(startingVels),badGradientFunction.__name__,goodGradientFunction.__name__,str(i),str(tolerance),testerFunction.__name__]

    if returnResources:
        vals.append('resources')
    if returnSamples:
        vals.append('samples')
    if branch:
        vals.append('branch')
    if independent:
        vals.append('independent')
    if badEnergyFunction is not None:
        vals.append(badEnergyFunction.__name__)
        vals.append(goodEnergyFunction.__name__)
    return '_'.join(vals) + '.dat'



def compareGradientFunctionsFile(its=200,
                                 geometryFunction=trajectories.colinearH3JacobiGeometry,
                                 startingCoords=[0.71,1.555],
                                 startingVels=[0.,0.],
                                 masses=trajectories.H3JacobiMasses,
                                 multiplicity=2,
                                 gradientFunctions=[trajectories.openFermionOplistExpectationGradient,
                                                    trajectories.openFermionGradientFunction],
                                 returnResources=False,
                                 countSampling=False):
    REPORT_FREQUENCY = datetime.timedelta(seconds=60)
    lastReport = datetime.datetime.now()
    startTime = lastReport
    results = []
    fullNumTrials = its * len(gradientFunctions)
    doneTrials = 0
    for thisGradientFunction in gradientFunctions:
        result = []
        initialGeometry = GeometrySpec.GeometrySpec(geometryFunction, startingCoords, startingVels, masses, multiplicity)
        simulator = trajectories.run(initialGeometry, thisGradientFunction, returnResources, countSampling)
        #firstFile = makeFilenameH3Jacobi(startingCoords,startingVels,thisGradientFunction,0,returnResources,countSampling)
        #firstPath = H3_TRAJECTORIES_DATA_DIR + firstFile
       # with open(firstPath,'wb') as f:
       #     pickle.dump(simulator,f)
        for i in range(its+1):
            thing = next(simulator)
            result.append(thing)
            doneTrials += 1
            if datetime.datetime.now() - lastReport > REPORT_FREQUENCY:
                fracDone = doneTrials/float(fullNumTrials)
                estTime = (datetime.datetime.now() - startTime) * ((1./fracDone) - 1.)
                print('{}  {}% DONE, EST TIME TO COMPLETE: {}'.format(datetime.datetime.now(),fracDone*100,estTime))
                lastReport = datetime.datetime.now()
            thisFile = makeFilename(startingCoords,startingVels,thisGradientFunction,i,returnResources,countSampling)
            thisPath = H3_TRAJECTORIES_DATA_DIR + thisFile
            with open(thisPath,'wb') as f:
                pickle.dump(thing,f)
        results.append(result)
    return(results)

def energyDifferencesH3(its=200,
                        geometryFunction=trajectories.colinearH3JacobiGeometry,
                        startingCoords=[0.714,2.357],
                        startingVels=[0.,-0.2],
                        masses=trajectories.H3JacobiMasses,
                        multiplicity=2,
                        badGradientFunctions=[trajectories.openFermionHartreeFock],
                        goodGradientFunctions=[trajectories.openFermionGradientFunction],
                        gradientPrecision=trajectoriesConstants.GRADIENT_PRECISION
                        ):

    for badGradientFunction in badGradientFunctions:
        for goodGradientFunction in goodGradientFunctions:
            initialGeometry = GeometrySpec.GeometrySpec(geometryFunction, startingCoords, startingVels, masses, multiplicity)
            simulator = trajectories.runAdaptive(initialGeometry, badGradientFunction, goodGradientFunction, gradientPrecision=gradientPrecision)
            i = 0
            results = []
            branches = []
            while i < its:
                thing,newStep = next(simulator)
                differences = thing.verificationValues()
                if newStep < i:
                    branches += results[newStep:]
                    results = results[:newStep]

                results.append(differences)
                i = newStep
    return results

def compareGradientFunctionsFileAdaptiveH2(its=100,
                                           geometryFunction=trajectories.h2Geometry,
                                           startingCoords=[1.0],
                                           startingVels=[0.],
                                           masses=trajectories.H2JacobiMasses,
                                           multiplicity=1,
                                           unadaptiveGradientFunctions = [trajectories.openFermionHartreeFock,
                                                                          trajectories.openFermionGradientFunction,
                                                                          trajectories.openFermionGradientVQE],
                                           badGradientFunctions=[trajectories.openFermionHartreeFock],
                                           goodGradientFunctions=[trajectories.openFermionGradientVQE],
                                           returnResources=False,
                                           countSampling=False,
                                           gradientPrecision=trajectoriesConstants.GRADIENT_PRECISION):


    REPORT_FREQUENCY = datetime.timedelta(seconds=60)
    lastReport = datetime.datetime.now()
    startTime = lastReport
    unAdaptiveResults = []
    fullNumTrials = its * len(unadaptiveGradientFunctions)
    doneTrials = 0
    for thisGradientFunction in unadaptiveGradientFunctions:
        result = []
        initialGeometry = GeometrySpec.GeometrySpec(geometryFunction, startingCoords, startingVels, masses, multiplicity)
        simulator = trajectories.run(initialGeometry, thisGradientFunction, returnResources, countSampling)
        #firstFile = makeFilenameH3Jacobi(startingCoords,startingVels,thisGradientFunction,0,returnResources,countSampling)
        #firstPath = H3_TRAJECTORIES_DATA_DIR + firstFile
        # with open(firstPath,'wb') as f:
        #     pickle.dump(simulator,f)
        for i in range(its+1):
            thing = next(simulator)
            result.append(thing)
            doneTrials += 1
            if datetime.datetime.now() - lastReport > REPORT_FREQUENCY:
                fracDone = doneTrials/float(fullNumTrials)
                estTime = (datetime.datetime.now() - startTime) * ((1./fracDone) - 1.)
                print('{}  {}% DONE, EST TIME TO COMPLETE: {}'.format(datetime.datetime.now(),fracDone*100,estTime))
                lastReport = datetime.datetime.now()
            thisFile = makeFilename(startingCoords,startingVels,thisGradientFunction,i,returnResources,countSampling)
            thisPath = H2_TRAJECTORIES_DATA_DIR + thisFile
            with open(thisPath,'wb') as f:
                pickle.dump(thing,f)
        unAdaptiveResults.append(result)

    for badGradientFunction in badGradientFunctions:
        for goodGradientFunction in goodGradientFunctions:
            initialGeometry = GeometrySpec.GeometrySpec(geometryFunction, startingCoords, startingVels, masses, multiplicity)
            simulator = trajectories.runAdaptive(initialGeometry, badGradientFunction, goodGradientFunction, returnResources, countSampling, gradientPrecision=gradientPrecision)
            i = 0
            while i < its:
                thing,newStep = next(simulator)
                if newStep < i: #if we branched
                    for k in [x for x in range(i+1) if x >= newStep]:
                        oldFile = H2_TRAJECTORIES_DATA_DIR + makeFilenameAdaptive(startingCoords,startingVels,badGradientFunction,goodGradientFunction,k,returnResources,countSampling,gradientPrecision)
                        newFile = H2_TRAJECTORIES_DATA_DIR + makeFilenameAdaptive(startingCoords,startingVels,badGradientFunction,goodGradientFunction,k,returnResources,countSampling,gradientPrecision,branch=True)
                        os.rename(oldFile,newFile)
                i = newStep
                filePath = H2_TRAJECTORIES_DATA_DIR + makeFilenameAdaptive(startingCoords,startingVels,badGradientFunction,goodGradientFunction,i,returnResources,countSampling,gradientPrecision)
                with open(filePath,'wb') as f:
                    pickle.dump(thing,f)
    return


def compareGradientFunctionsFileAdaptiveH3(its=300,
                                           geometryFunction=trajectories.colinearH3JacobiGeometry,
                                           startingCoords=[0.71,1.555],
                                           startingVels=[0.,0.],
                                           masses=trajectories.H3JacobiMasses,
                                           multiplicity=2,
                                           unadaptiveGradientFunctions = [trajectories.openFermionHartreeFock,
                                                                          trajectories.openFermionGradientFunction,
                                                                          trajectories.openFermionGradientVQE,
                                                                          trajectories.openFermionCCSD,
                                                                          trajectories.openFermionMP2],
                                           badGradientFunctions=[trajectories.openFermionHartreeFock,
                                                                 trajectories.openFermionCCSD,
                                                                 trajectories.openFermionMP2],
                                           goodGradientFunctions=[trajectories.openFermionGradientVQE,
                                                                  trajectories.openFermionGradientFunction],
                                           returnResources=True,
                                           countSampling=True,
                                           gradientPrecisions=[1e-06,1e-05,1e-04,1e-07,1e-08,1e-09,5e-08,5e-07,5e-09]):




    REPORT_FREQUENCY = datetime.timedelta(seconds=60)
    lastReport = datetime.datetime.now()
    startTime = lastReport
    unAdaptiveResults = []
    fullNumTrials = its * len(unadaptiveGradientFunctions)
    doneTrials = 0
    for gradientPrecision in gradientPrecisions:
        for thisGradientFunction in unadaptiveGradientFunctions:
            result = []
            initialGeometry = GeometrySpec.GeometrySpec(geometryFunction, startingCoords, startingVels, masses,
                                                        multiplicity)
            simulator = trajectories.run(initialGeometry, thisGradientFunction, returnResources, countSampling)
            # firstFile = makeFilenameH3Jacobi(startingCoords,startingVels,thisGradientFunction,0,returnResources,countSampling)
            # firstPath = H3_TRAJECTORIES_DATA_DIR + firstFile
            # with open(firstPath,'wb') as f:
            #     pickle.dump(simulator,f)
            for i in range(its + 1):
                thing = next(simulator)
                result.append(thing)
                doneTrials += 1
                if datetime.datetime.now() - lastReport > REPORT_FREQUENCY:
                    fracDone = doneTrials / float(fullNumTrials)
                    estTime = (datetime.datetime.now() - startTime) * ((1. / fracDone) - 1.)
                    print('{}  {}% DONE, EST TIME TO COMPLETE: {}'.format(datetime.datetime.now(), fracDone * 100,
                                                                          estTime))
                    lastReport = datetime.datetime.now()
                thisFile = makeFilename(startingCoords, startingVels, thisGradientFunction, i, returnResources,
                                        countSampling)
                thisPath = H3_TRAJECTORIES_DATA_DIR + thisFile
                with open(thisPath, 'wb') as f:
                    pickle.dump(thing, f)
            unAdaptiveResults.append(result)

    for badGradientFunction in badGradientFunctions:
        for goodGradientFunction in goodGradientFunctions:
            initialGeometry = GeometrySpec.GeometrySpec(geometryFunction, startingCoords, startingVels, masses,
                                                        multiplicity)
            simulator = trajectories.runAdaptive(initialGeometry, badGradientFunction, goodGradientFunction,
                                                 returnResources, countSampling, gradientPrecision=gradientPrecision)
            i = 0
            while i < its:
                thing, newStep = next(simulator)
                if newStep < i:  # if we branched
                    for k in [x for x in range(i + 1) if x >= newStep]:
                        oldFile = H3_TRAJECTORIES_DATA_DIR + makeFilenameAdaptive(startingCoords, startingVels,
                                                                                  badGradientFunction,
                                                                                  goodGradientFunction, k,
                                                                                  returnResources, countSampling,
                                                                                  gradientPrecision)
                        newFile = H3_TRAJECTORIES_DATA_DIR + makeFilenameAdaptive(startingCoords, startingVels,
                                                                                  badGradientFunction,
                                                                                  goodGradientFunction, k,
                                                                                  returnResources, countSampling,
                                                                                  gradientPrecision, branch=True)
                        os.rename(oldFile, newFile)
                i = newStep
                filePath = H3_TRAJECTORIES_DATA_DIR + makeFilenameAdaptive(startingCoords, startingVels,
                                                                           badGradientFunction, goodGradientFunction, i,
                                                                           returnResources, countSampling,
                                                                           gradientPrecision)
                with open(filePath, 'wb') as f:
                    pickle.dump(thing, f)
    return



def compareGradientFunctionsFileAdaptiveH2Big(its=100,
                                              geometryFunction=trajectories.h2Geometry,
                                              startingCoords=[1.0],
                                              startingVels=[0.],
                                              masses=trajectories.H2JacobiMasses,
                                              multiplicity=1,
                                              unadaptiveGradientFunctions = [trajectories.openFermionHartreeFock,
                                                                             trajectories.openFermionGradientFunction],
                                              badGradientFunctions=[trajectories.openFermionHartreeFock,
                                                                    trajectories.openFermionCCSD,
                                                                    trajectories.openFermionMP2],
                                              goodGradientFunctions=[trajectories.openFermionGradientVQE,
                                                                     trajectories.openFermionGradientFunction],
                                              returnResources=True,
                                              countSampling=True,
                                              gradientPrecisions=[0.016,1e-8],
                                              verifyFunctionsGradient=[AdaptiveIntegrator.compareGradients],
                                              verifyFunctionsEnergy=[AdaptiveIntegrator.compareEnergy],
                                              energyFunctions={trajectories.openFermionHartreeFock: trajectories.openFermionHartreeFockEnergy,
                                                               trajectories.openFermionGradientVQE: trajectories.openFermionEnergyVQE,
                                                               trajectories.openFermionGradientFunction: trajectories.openFermionPotentialFunction,
                                                               trajectories.openFermionCCSD: trajectories.openFermionCCSDEnergy,
                                                               trajectories.openFermionMP2: trajectories.openFermionMP2Energy}):






    REPORT_FREQUENCY = datetime.timedelta(seconds=60)
    lastReport = datetime.datetime.now()
    startTime = lastReport
    unAdaptiveResults = []
    fullNumTrials = its * len(unadaptiveGradientFunctions)
    doneTrials = 0
    for gradientPrecision in gradientPrecisions:
        for thisGradientFunction in unadaptiveGradientFunctions:
            result = []
            initialGeometry = GeometrySpec.GeometrySpec(geometryFunction, startingCoords, startingVels, masses,
                                                        multiplicity)
            simulator = trajectories.run(initialGeometry, thisGradientFunction, returnResources, countSampling)
            # firstFile = makeFilenameH3Jacobi(startingCoords,startingVels,thisGradientFunction,0,returnResources,countSampling)
            # firstPath = H3_TRAJECTORIES_DATA_DIR + firstFile
            # with open(firstPath,'wb') as f:
            #     pickle.dump(simulator,f)
            for i in range(its + 1):
                thing = next(simulator)
                result.append(thing)
                doneTrials += 1
                if datetime.datetime.now() - lastReport > REPORT_FREQUENCY:
                    fracDone = doneTrials / float(fullNumTrials)
                    estTime = (datetime.datetime.now() - startTime) * ((1. / fracDone) - 1.)
                    print('{}  {}% DONE, EST TIME TO COMPLETE: {}'.format(datetime.datetime.now(), fracDone * 100,
                                                                          estTime))
                    lastReport = datetime.datetime.now()
                thisFile = makeFilename(startingCoords, startingVels, thisGradientFunction, i, returnResources,
                                        countSampling)
                thisPath = H2_TRAJECTORIES_DATA_DIR + thisFile
                with open(thisPath, 'wb') as f:
                    pickle.dump(thing, f)
            unAdaptiveResults.append(result)
        for verifyFunction in verifyFunctionsGradient:
            for badGradientFunction in badGradientFunctions:
                for goodGradientFunction in goodGradientFunctions:
                    initialGeometry = GeometrySpec.GeometrySpec(geometryFunction, startingCoords, startingVels, masses,
                                                                multiplicity)
                    simulator = trajectories.runAdaptive(initialGeometry, badGradientFunction, goodGradientFunction,
                                                         returnResources, countSampling, gradientPrecision=gradientPrecision, testerFunction=verifyFunction)
                    i = 0
                    while i < its:
                        clive = next(simulator)
                        if returnResources:
                            thing = clive[0:2]
                            newStep = clive[2]
                        else:
                            thing = clive[0]
                            newStep = clive[1]
                        #thing, newStep = next(simulator)
                        if newStep < i:  # if we branched
                            for k in [x for x in range(i + 1) if x >= newStep]:
                                oldFile = H2_TRAJECTORIES_DATA_DIR + makeFilenameAdaptive(startingCoords, startingVels,
                                                                                          badGradientFunction,
                                                                                          goodGradientFunction, k,
                                                                                          returnResources, countSampling,
                                                                                          gradientPrecision,testerFunction=verifyFunction,
                                                                                          )
                                newFile = H2_TRAJECTORIES_DATA_DIR + makeFilenameAdaptive(startingCoords, startingVels,
                                                                                          badGradientFunction,
                                                                                          goodGradientFunction, k,
                                                                                          returnResources, countSampling,
                                                                                          gradientPrecision,testerFunction=verifyFunction,
                                                                                          branch=True)
                                os.rename(oldFile, newFile)
                        i = newStep
                        filePath = H2_TRAJECTORIES_DATA_DIR + makeFilenameAdaptive(startingCoords, startingVels,
                                                                                   badGradientFunction, goodGradientFunction, i,
                                                                                   returnResources, countSampling,
                                                                                   gradientPrecision,testerFunction=verifyFunction,
                                                                                   )
                        with open(filePath, 'wb') as f:
                            pickle.dump(thing, f)




        for verifyFunction in verifyFunctionsEnergy:
            for badGradientFunction in badGradientFunctions:
                for goodGradientFunction in goodGradientFunctions:
                    badEnergyFunction=energyFunctions[badGradientFunction]
                    goodEnergyFunction=energyFunctions[goodGradientFunction]
                    initialGeometry = GeometrySpec.GeometrySpec(geometryFunction, startingCoords, startingVels, masses,
                                                                multiplicity)
                    simulator = trajectories.runAdaptive(initialGeometry, badGradientFunction, goodGradientFunction,
                                                         returnResources, countSampling, gradientPrecision=gradientPrecision, testerFunction=verifyFunction,
                                                         badEnergyFunction=badEnergyFunction,
                                                         goodEnergyFunction=goodEnergyFunction)
                    i = 0
                    while i < its:
                        clive = next(simulator)
                        if returnResources:
                            thing = clive[0:2]
                            newStep = clive[2]
                        else:
                            thing = clive[0]
                            newStep = clive[1]
                        if newStep < i:  # if we branched
                            for k in [x for x in range(i + 1) if x >= newStep]:
                                oldFile = H2_TRAJECTORIES_DATA_DIR + makeFilenameAdaptive(startingCoords, startingVels,
                                                                                          badGradientFunction,
                                                                                          goodGradientFunction, k,
                                                                                          returnResources, countSampling,
                                                                                          gradientPrecision,
                                                                                          testerFunction=verifyFunction,
                                                                                          badEnergyFunction=badEnergyFunction,
                                                                                          goodEnergyFunction=goodEnergyFunction)
                                newFile = H2_TRAJECTORIES_DATA_DIR + makeFilenameAdaptive(startingCoords, startingVels,
                                                                                          badGradientFunction,
                                                                                          goodGradientFunction, k,
                                                                                          returnResources, countSampling,
                                                                                          gradientPrecision,
                                                                                          testerFunction=verifyFunction,
                                                                                          badEnergyFunction=badEnergyFunction,
                                                                                          goodEnergyFunction=goodEnergyFunction,
                                                                                          branch=True)
                                os.rename(oldFile, newFile)
                        i = newStep
                        filePath = H2_TRAJECTORIES_DATA_DIR + makeFilenameAdaptive(startingCoords, startingVels,
                                                                                   badGradientFunction, goodGradientFunction, i,
                                                                                   returnResources, countSampling,
                                                                                   gradientPrecision,
                                                                                   testerFunction=verifyFunction,
                                                                                   badEnergyFunction=badEnergyFunction,
                                                                                   goodEnergyFunction=goodEnergyFunction)
                        with open(filePath, 'wb') as f:
                            pickle.dump(thing, f)
    return

def compareGradientFunctionsFileAdaptiveH3Big(its=300,
                                              geometryFunction=trajectories.colinearH3JacobiGeometry,
                                              startingCoords=[0.71,1.555],
                                              startingVels=[0.,0.],
                                              masses=trajectories.H3JacobiMasses,
                                              multiplicity=2,
                                              unadaptiveGradientFunctions = [trajectories.openFermionHartreeFock,
                                                                             trajectories.openFermionGradientFunction,
                                                                             trajectories.openFermionGradientVQE,
                                                                             trajectories.openFermionCCSD,
                                                                             trajectories.openFermionMP2],
                                              badGradientFunctions=[trajectories.openFermionHartreeFock,
                                                                    trajectories.openFermionCCSD,
                                                                    trajectories.openFermionMP2],
                                              goodGradientFunctions=[trajectories.openFermionGradientVQE,
                                                                     trajectories.openFermionGradientFunction],
                                              returnResources=True,
                                              countSampling=True,
                                              gradientPrecisions=[1e-08,1e-09,1e-07,1e-06,1e-05,1e-04,5e-08,5e-07,5e-09,0.016,0.001,0.01],
                                              verifyFunctionsGradient=[AdaptiveIntegrator.compareGradients],
                                              verifyFunctionsEnergy=[AdaptiveIntegrator.compareEnergy],
                                              energyFunctions={trajectories.openFermionHartreeFock: trajectories.openFermionHartreeFockEnergy,
                                                               trajectories.openFermionGradientVQE: trajectories.openFermionEnergyVQE,
                                                               trajectories.openFermionGradientFunction: trajectories.openFermionPotentialFunction,
                                                               trajectories.openFermionCCSD: trajectories.openFermionCCSDEnergy,
                                                               trajectories.openFermionMP2: trajectories.openFermionMP2Energy},
                                              quantumChemPackage='pyscf'):






    REPORT_FREQUENCY = datetime.timedelta(seconds=60)
    lastReport = datetime.datetime.now()
    startTime = lastReport
    unAdaptiveResults = []
    fullNumTrials = its * len(unadaptiveGradientFunctions)
    doneTrials = 0
    for gradientPrecision in [0]:
        for thisGradientFunction in unadaptiveGradientFunctions:
            result = []
            initialGeometry = GeometrySpec.GeometrySpec(geometryFunction, startingCoords, startingVels, masses,
                                                        multiplicity)
            simulator = trajectories.run(initialGeometry, thisGradientFunction, returnResources, countSampling)
            # firstFile = makeFilenameH3Jacobi(startingCoords,startingVels,thisGradientFunction,0,returnResources,countSampling)
            # firstPath = H3_TRAJECTORIES_DATA_DIR + firstFile
            # with open(firstPath,'wb') as f:
            #     pickle.dump(simulator,f)
            for i in range(its + 1):
                thing = next(simulator)
                result.append(thing)
                doneTrials += 1
                if datetime.datetime.now() - lastReport > REPORT_FREQUENCY:
                    fracDone = doneTrials / float(fullNumTrials)
                    estTime = (datetime.datetime.now() - startTime) * ((1. / fracDone) - 1.)
                    print('{}  {}% DONE, EST TIME TO COMPLETE: {}'.format(datetime.datetime.now(), fracDone * 100,
                                                                          estTime))
                    lastReport = datetime.datetime.now()
                thisFile = makeFilename(startingCoords, startingVels, thisGradientFunction, i, returnResources,
                                        countSampling)
                thisPath = H3_TRAJECTORIES_DATA_DIR + thisFile
                with open(thisPath, 'wb') as f:
                    pickle.dump(thing, f)
            unAdaptiveResults.append(result)
    for gradientPrecision in gradientPrecisions:
        for verifyFunction in verifyFunctionsGradient:
            for badGradientFunction in badGradientFunctions:
                for goodGradientFunction in goodGradientFunctions:
                    initialGeometry = GeometrySpec.GeometrySpec(geometryFunction, startingCoords, startingVels, masses,
                                                                multiplicity)
                    simulator = trajectories.runAdaptive(initialGeometry, badGradientFunction, goodGradientFunction,
                                                         returnResources, countSampling, gradientPrecision=gradientPrecision, testerFunction=verifyFunction)
                    i = 0
                    while i < its:
                        clive = next(simulator)
                        if returnResources:
                            thing = clive[0:2]
                            newStep = clive[2]
                        else:
                            thing = clive[0]
                            newStep = clive[1]
                        #thing, newStep = next(simulator)
                        if newStep < i:  # if we branched
                            for k in [x for x in range(i + 1) if x >= newStep]:
                                oldFile = H3_TRAJECTORIES_DATA_DIR + makeFilenameAdaptive(startingCoords, startingVels,
                                                                                          badGradientFunction,
                                                                                          goodGradientFunction, k,
                                                                                          returnResources, countSampling,
                                                                                          gradientPrecision,testerFunction=verifyFunction,
                                                                                          )
                                newFile = H3_TRAJECTORIES_DATA_DIR + makeFilenameAdaptive(startingCoords, startingVels,
                                                                                          badGradientFunction,
                                                                                          goodGradientFunction, k,
                                                                                          returnResources, countSampling,
                                                                                          gradientPrecision,testerFunction=verifyFunction,
                                                                                          branch=True)
                                os.rename(oldFile, newFile)
                        i = newStep
                        filePath = H3_TRAJECTORIES_DATA_DIR + makeFilenameAdaptive(startingCoords, startingVels,
                                                                                   badGradientFunction, goodGradientFunction, i,
                                                                                   returnResources, countSampling,
                                                                                   gradientPrecision,testerFunction=verifyFunction,
                                                                                   )
                        with open(filePath, 'wb') as f:
                            pickle.dump(thing, f)




        for verifyFunction in verifyFunctionsEnergy:
            for badGradientFunction in badGradientFunctions:
                for goodGradientFunction in goodGradientFunctions:
                    badEnergyFunction=energyFunctions[badGradientFunction]
                    goodEnergyFunction=energyFunctions[goodGradientFunction]
                    initialGeometry = GeometrySpec.GeometrySpec(geometryFunction, startingCoords, startingVels, masses,
                                                                multiplicity)
                    simulator = trajectories.runAdaptive(initialGeometry, badGradientFunction, goodGradientFunction,
                                                         returnResources, countSampling, gradientPrecision=gradientPrecision, testerFunction=verifyFunction,
                                                         badEnergyFunction=badEnergyFunction,
                                                         goodEnergyFunction=goodEnergyFunction)
                    i = 0
                    while i < its:
                        clive = next(simulator)
                        if returnResources:
                            thing = clive[0:2]
                            newStep = clive[2]
                        else:
                            thing = clive[0]
                            newStep = clive[1]
                        if newStep < i:  # if we branched
                            for k in [x for x in range(i + 1) if x >= newStep]:
                                oldFile = H3_TRAJECTORIES_DATA_DIR + makeFilenameAdaptive(startingCoords, startingVels,
                                                                                          badGradientFunction,
                                                                                          goodGradientFunction, k,
                                                                                          returnResources, countSampling,
                                                                                          gradientPrecision,
                                                                                          testerFunction=verifyFunction,
                                                                                          badEnergyFunction=badEnergyFunction,
                                                                                          goodEnergyFunction=goodEnergyFunction)
                                newFile = H3_TRAJECTORIES_DATA_DIR + makeFilenameAdaptive(startingCoords, startingVels,
                                                                                          badGradientFunction,
                                                                                          goodGradientFunction, k,
                                                                                          returnResources, countSampling,
                                                                                          gradientPrecision,
                                                                                          testerFunction=verifyFunction,
                                                                                          badEnergyFunction=badEnergyFunction,
                                                                                          goodEnergyFunction=goodEnergyFunction,
                                                                                          branch=True)
                                os.rename(oldFile, newFile)
                        i = newStep
                        filePath = H3_TRAJECTORIES_DATA_DIR + makeFilenameAdaptive(startingCoords, startingVels,
                                                                                   badGradientFunction, goodGradientFunction, i,
                                                                                   returnResources, countSampling,
                                                                                   gradientPrecision,
                                                                                   testerFunction=verifyFunction,
                                                                                   badEnergyFunction=badEnergyFunction,
                                                                                   goodEnergyFunction=goodEnergyFunction)
                        with open(filePath, 'wb') as f:
                            pickle.dump(thing, f)

        print('Done {}'.format(gradientPrecision))
    return


def geom1(q1,q2):
    return [('H',(0.,0.,0.)),('H',(0.,0.,q1)),('He',(0.,0.,q2 + 0.5*q1))]
def geom2(q1,q2):
    return [('H',(0.,0.,0.)),('He',(0.,0.,q1)),('H',(0.,0.,q2 + 0.5*q1))]
def geom3(q1,q2):
    return [('He',(0.,0.,0.)),('H',(0.,0.,q1)),('H',(0.,0.,q2 + 0.5*q1))]

paths = ['/home/andrew/data/molecular_dynamics/He1',
         '/home/andrew/data/molecular_dynamics/He2',
         '/home/andrew/data/molecular_dynamics/He3']

HeMasses = [[0.5,(4./3.)],
            [(4./5.),(5./6.)],
            [(4./5.),(5./6.)]]
geomFunctions = [geom1,geom2,geom3]

def panic(its=300,
          geometryFunction=trajectories.colinearH3JacobiGeometry,
          startingCoords=[0.71,1.555],
          startingVels=[0.,-0.1],
          masses=trajectories.H3JacobiMasses,
          multiplicity=1,
          unadaptiveGradientFunctions = [trajectories.openFermionHartreeFock,
                                         trajectories.openFermionGradientFunction,
                                         #bomd.openFermionGradientVQE,
                                         trajectories.openFermionCCSD],
          badGradientFunctions=[],
          goodGradientFunctions=[],
          returnResources=True,
          countSampling=True,
          gradientPrecisions=[],
          verifyFunctionsGradient=[AdaptiveIntegrator.compareGradients],
          verifyFunctionsEnergy=[],
          energyFunctions={trajectories.openFermionHartreeFock: trajectories.openFermionHartreeFockEnergy,
                           trajectories.openFermionGradientVQE: trajectories.openFermionEnergyVQE,
                           trajectories.openFermionGradientFunction: trajectories.openFermionPotentialFunction,
                           trajectories.openFermionCCSD: trajectories.openFermionCCSDEnergy,
                           trajectories.openFermionMP2: trajectories.openFermionMP2Energy}):





    charPath = [#(+1,'/home/andrew/data/molecular_dynamics/H3+/'),
                (-1,'/home/andrew/data/molecular_dynamics/H3-/')]
    chargesMultiplicities = [(+1,2,'+'),(0,1,''),(-1,2,'-')]

    REPORT_FREQUENCY = datetime.timedelta(seconds=60)
    lastReport = datetime.datetime.now()
    startTime = lastReport
    unAdaptiveResults = []
    fullNumTrials = its * len(unadaptiveGradientFunctions)
    doneTrials = 0
    for i123,pathToFix in enumerate(paths):
        geometryFunction = geomFunctions[i123]
        masses=HeMasses[i123]
        for j513,thing in enumerate(chargesMultiplicities):
            charge = thing[0]
            multiplicity = thing[1]
            charStr = thing[2]
            path = pathToFix + charStr + '/'
            for thisGradientFunction in unadaptiveGradientFunctions:
                result = []
                initialGeometry = GeometrySpec.GeometrySpec(geometryFunction, startingCoords, startingVels, masses,
                                                            multiplicity, charge=charge)
                simulator = trajectories.run(initialGeometry, thisGradientFunction, returnResources, countSampling)
                # firstFile = makeFilenameH3Jacobi(startingCoords,startingVels,thisGradientFunction,0,returnResources,countSampling)
                # firstPath = H3_TRAJECTORIES_DATA_DIR + firstFile
                # with open(firstPath,'wb') as f:
                #     pickle.dump(simulator,f)
                for i in range(its + 1):
                    thing = next(simulator)
                    result.append(thing)
                    doneTrials += 1
                    if datetime.datetime.now() - lastReport > REPORT_FREQUENCY:
                        fracDone = doneTrials / float(fullNumTrials)
                        estTime = (datetime.datetime.now() - startTime) * ((1. / fracDone) - 1.)
                        print('{}  {}% DONE, EST TIME TO COMPLETE: {}'.format(datetime.datetime.now(), fracDone * 100,
                                                                              estTime))
                        lastReport = datetime.datetime.now()
                    thisFile = makeFilename(startingCoords, startingVels, thisGradientFunction, i, returnResources,
                                            countSampling)
                    thisPath = path + thisFile
                    with open(thisPath, 'wb') as f:
                        pickle.dump(thing, f)
                unAdaptiveResults.append(result)
    return

def compareGradientFunctionsFileAdaptiveH3Big(its=300,
                                              geometryFunction=trajectories.colinearH3JacobiGeometry,
                                              startingCoords=[0.71,1.555],
                                              startingVels=[0.,0.],
                                              masses=trajectories.H3JacobiMasses,
                                              multiplicity=2,
                                              unadaptiveGradientFunctions = [trajectories.openFermionHartreeFock,
                                                                             trajectories.openFermionGradientFunction,
                                                                             trajectories.openFermionGradientVQE,
                                                                             trajectories.openFermionCCSD,
                                                                             trajectories.openFermionMP2],
                                              badGradientFunctions=[trajectories.openFermionHartreeFock,
                                                                    trajectories.openFermionCCSD,
                                                                    trajectories.openFermionMP2],
                                              goodGradientFunctions=[trajectories.openFermionGradientVQE,
                                                                     trajectories.openFermionGradientFunction],
                                              returnResources=True,
                                              countSampling=True,
                                              gradientPrecisions=[1e-08,1e-09,1e-07,1e-06,1e-05,1e-04,5e-08,5e-07,5e-09,0.016,0.001,0.01],
                                              verifyFunctionsGradient=[AdaptiveIntegrator.compareGradients],
                                              verifyFunctionsEnergy=[AdaptiveIntegrator.compareEnergy],
                                              energyFunctions={trajectories.openFermionHartreeFock: trajectories.openFermionHartreeFockEnergy,
                                                               trajectories.openFermionGradientVQE: trajectories.openFermionEnergyVQE,
                                                               trajectories.openFermionGradientFunction: trajectories.openFermionPotentialFunction,
                                                               trajectories.openFermionCCSD: trajectories.openFermionCCSDEnergy,
                                                               trajectories.openFermionMP2: trajectories.openFermionMP2Energy},
                                              quantumChemPackage='pyscf'):






    REPORT_FREQUENCY = datetime.timedelta(seconds=60)
    lastReport = datetime.datetime.now()
    startTime = lastReport
    unAdaptiveResults = []
    fullNumTrials = its * len(unadaptiveGradientFunctions)
    doneTrials = 0
    for gradientPrecision in [0]:
        for thisGradientFunction in unadaptiveGradientFunctions:
            result = []
            initialGeometry = GeometrySpec.GeometrySpec(geometryFunction, startingCoords, startingVels, masses,
                                                        multiplicity)
            simulator = trajectories.run(initialGeometry, thisGradientFunction, returnResources, countSampling)
            # firstFile = makeFilenameH3Jacobi(startingCoords,startingVels,thisGradientFunction,0,returnResources,countSampling)
            # firstPath = H3_TRAJECTORIES_DATA_DIR + firstFile
            # with open(firstPath,'wb') as f:
            #     pickle.dump(simulator,f)
            for i in range(its + 1):
                thing = next(simulator)
                result.append(thing)
                doneTrials += 1
                if datetime.datetime.now() - lastReport > REPORT_FREQUENCY:
                    fracDone = doneTrials / float(fullNumTrials)
                    estTime = (datetime.datetime.now() - startTime) * ((1. / fracDone) - 1.)
                    print('{}  {}% DONE, EST TIME TO COMPLETE: {}'.format(datetime.datetime.now(), fracDone * 100,
                                                                          estTime))
                    lastReport = datetime.datetime.now()
                thisFile = makeFilename(startingCoords, startingVels, thisGradientFunction, i, returnResources,
                                        countSampling)
                thisPath = H3_TRAJECTORIES_DATA_DIR + thisFile
                with open(thisPath, 'wb') as f:
                    pickle.dump(thing, f)
            unAdaptiveResults.append(result)
    for gradientPrecision in gradientPrecisions:
        for verifyFunction in verifyFunctionsGradient:
            for badGradientFunction in badGradientFunctions:
                for goodGradientFunction in goodGradientFunctions:
                    initialGeometry = GeometrySpec.GeometrySpec(geometryFunction, startingCoords, startingVels, masses,
                                                                multiplicity)
                    simulator = trajectories.runAdaptive(initialGeometry, badGradientFunction, goodGradientFunction,
                                                         returnResources, countSampling, gradientPrecision=gradientPrecision, testerFunction=verifyFunction)
                    i = 0
                    while i < its:
                        clive = next(simulator)
                        if returnResources:
                            thing = clive[0:2]
                            newStep = clive[2]
                        else:
                            thing = clive[0]
                            newStep = clive[1]
                        #thing, newStep = next(simulator)
                        if newStep < i:  # if we branched
                            for k in [x for x in range(i + 1) if x >= newStep]:
                                oldFile = H3_TRAJECTORIES_DATA_DIR + makeFilenameAdaptive(startingCoords, startingVels,
                                                                                          badGradientFunction,
                                                                                          goodGradientFunction, k,
                                                                                          returnResources, countSampling,
                                                                                          gradientPrecision,testerFunction=verifyFunction,
                                                                                          )
                                newFile = H3_TRAJECTORIES_DATA_DIR + makeFilenameAdaptive(startingCoords, startingVels,
                                                                                          badGradientFunction,
                                                                                          goodGradientFunction, k,
                                                                                          returnResources, countSampling,
                                                                                          gradientPrecision,testerFunction=verifyFunction,
                                                                                          branch=True)
                                os.rename(oldFile, newFile)
                        i = newStep
                        filePath = H3_TRAJECTORIES_DATA_DIR + makeFilenameAdaptive(startingCoords, startingVels,
                                                                                   badGradientFunction, goodGradientFunction, i,
                                                                                   returnResources, countSampling,
                                                                                   gradientPrecision,testerFunction=verifyFunction,
                                                                                   )
                        with open(filePath, 'wb') as f:
                            pickle.dump(thing, f)




        for verifyFunction in verifyFunctionsEnergy:
            for badGradientFunction in badGradientFunctions:
                for goodGradientFunction in goodGradientFunctions:
                    badEnergyFunction=energyFunctions[badGradientFunction]
                    goodEnergyFunction=energyFunctions[goodGradientFunction]
                    initialGeometry = GeometrySpec.GeometrySpec(geometryFunction, startingCoords, startingVels, masses,
                                                                multiplicity)
                    simulator = trajectories.runAdaptive(initialGeometry, badGradientFunction, goodGradientFunction,
                                                         returnResources, countSampling, gradientPrecision=gradientPrecision, testerFunction=verifyFunction,
                                                         badEnergyFunction=badEnergyFunction,
                                                         goodEnergyFunction=goodEnergyFunction)
                    i = 0
                    while i < its:
                        clive = next(simulator)
                        if returnResources:
                            thing = clive[0:2]
                            newStep = clive[2]
                        else:
                            thing = clive[0]
                            newStep = clive[1]
                        if newStep < i:  # if we branched
                            for k in [x for x in range(i + 1) if x >= newStep]:
                                oldFile = H3_TRAJECTORIES_DATA_DIR + makeFilenameAdaptive(startingCoords, startingVels,
                                                                                          badGradientFunction,
                                                                                          goodGradientFunction, k,
                                                                                          returnResources, countSampling,
                                                                                          gradientPrecision,
                                                                                          testerFunction=verifyFunction,
                                                                                          badEnergyFunction=badEnergyFunction,
                                                                                          goodEnergyFunction=goodEnergyFunction)
                                newFile = H3_TRAJECTORIES_DATA_DIR + makeFilenameAdaptive(startingCoords, startingVels,
                                                                                          badGradientFunction,
                                                                                          goodGradientFunction, k,
                                                                                          returnResources, countSampling,
                                                                                          gradientPrecision,
                                                                                          testerFunction=verifyFunction,
                                                                                          badEnergyFunction=badEnergyFunction,
                                                                                          goodEnergyFunction=goodEnergyFunction,
                                                                                          branch=True)
                                os.rename(oldFile, newFile)
                        i = newStep
                        filePath = H3_TRAJECTORIES_DATA_DIR + makeFilenameAdaptive(startingCoords, startingVels,
                                                                                   badGradientFunction, goodGradientFunction, i,
                                                                                   returnResources, countSampling,
                                                                                   gradientPrecision,
                                                                                   testerFunction=verifyFunction,
                                                                                   badEnergyFunction=badEnergyFunction,
                                                                                   goodEnergyFunction=goodEnergyFunction)
                        with open(filePath, 'wb') as f:
                            pickle.dump(thing, f)

        print('Done {}'.format(gradientPrecision))
    return


def compareGradientFunctionsFileIndependentH3Big(its=300,
                                                 geometryFunction=trajectories.colinearH3JacobiGeometry,
                                                 startingCoords=[0.741,2.357],
                                                 startingVels=[0.,-0.2],
                                                 masses=trajectories.H3JacobiMasses,
                                                 multiplicity=2,
                                                 unadaptiveGradientFunctions = [],
                                                 badGradientFunctions=[trajectories.openFermionHartreeFock],
                                                 goodGradientFunctions=[trajectories.openFermionGradientFunction],
                                                 returnResources=True,
                                                 countSampling=True,
                                                 gradientPrecisions=[1e-08,1e-09,1e-07,1e-06,1e-05,1e-04,5e-08,5e-07,5e-09,0.016,0.001,0.01],
                                                 verifyFunctionsGradient=[
                                                     IndependentAdaptiveIntegrator.compareGradients],
                                                 verifyFunctionsEnergy=[],
                                                 energyFunctions={trajectories.openFermionHartreeFock: trajectories.openFermionHartreeFockEnergy,
                                                                  trajectories.openFermionGradientVQE: trajectories.openFermionEnergyVQE,
                                                                  trajectories.openFermionGradientFunction: trajectories.openFermionPotentialFunction,
                                                                  trajectories.openFermionCCSD: trajectories.openFermionCCSDEnergy,
                                                                  trajectories.openFermionMP2: trajectories.openFermionMP2Energy},
                                                 quantumChemPackage='pyscf'):






    REPORT_FREQUENCY = datetime.timedelta(seconds=60)
    lastReport = datetime.datetime.now()
    startTime = lastReport
    unAdaptiveResults = []
    fullNumTrials = its * len(unadaptiveGradientFunctions)
    doneTrials = 0
    for gradientPrecision in [0]:
        for thisGradientFunction in unadaptiveGradientFunctions:
            result = []
            initialGeometry = GeometrySpec.GeometrySpec(geometryFunction, startingCoords, startingVels, masses,
                                                        multiplicity)
            simulator = trajectories.run(initialGeometry, thisGradientFunction, returnResources, countSampling)
            # firstFile = makeFilenameH3Jacobi(startingCoords,startingVels,thisGradientFunction,0,returnResources,countSampling)
            # firstPath = H3_TRAJECTORIES_DATA_DIR + firstFile
            # with open(firstPath,'wb') as f:
            #     pickle.dump(simulator,f)
            for i in range(its + 1):
                thing = next(simulator)
                result.append(thing)
                doneTrials += 1
                if datetime.datetime.now() - lastReport > REPORT_FREQUENCY:
                    fracDone = doneTrials / float(fullNumTrials)
                    estTime = (datetime.datetime.now() - startTime) * ((1. / fracDone) - 1.)
                    print('{}  {}% DONE, EST TIME TO COMPLETE: {}'.format(datetime.datetime.now(), fracDone * 100,
                                                                          estTime))
                    lastReport = datetime.datetime.now()
                thisFile = makeFilename(startingCoords, startingVels, thisGradientFunction, i, returnResources,
                                        countSampling)
                thisPath = H3_TRAJECTORIES_DATA_DIR + thisFile
                with open(thisPath, 'wb') as f:
                    pickle.dump(thing, f)
            unAdaptiveResults.append(result)
    for gradientPrecision in gradientPrecisions:
        for verifyFunction in verifyFunctionsGradient:
            for badGradientFunction in badGradientFunctions:
                for goodGradientFunction in goodGradientFunctions:
                    initialGeometry = GeometrySpec.GeometrySpec(geometryFunction, startingCoords, startingVels, masses,
                                                                multiplicity)
                    simulator = trajectories.runIndependent(initialGeometry, badGradientFunction, goodGradientFunction,
                                                            returnResources, countSampling, gradientPrecision=gradientPrecision, testerFunction=verifyFunction)
                    i = 0
                    while i < its:
                        clive = next(simulator)
                        if returnResources:
                            thing = clive[0:2]
                            newStep = clive[2]
                        else:
                            thing = clive[0]
                            newStep = clive[1]
                        #thing, newStep = next(simulator)
                        if newStep < i:  # if we branched
                            for k in [x for x in range(i + 1) if x >= newStep]:
                                oldFile = H3_TRAJECTORIES_DATA_DIR + makeFilenameAdaptive(startingCoords, startingVels,
                                                                                          badGradientFunction,
                                                                                          goodGradientFunction, k,
                                                                                          returnResources, countSampling,
                                                                                          gradientPrecision,testerFunction=verifyFunction,
                                                                                          )
                                newFile = H3_TRAJECTORIES_DATA_DIR + makeFilenameAdaptive(startingCoords, startingVels,
                                                                                          badGradientFunction,
                                                                                          goodGradientFunction, k,
                                                                                          returnResources, countSampling,
                                                                                          gradientPrecision,testerFunction=verifyFunction,
                                                                                          branch=True)
                                os.rename(oldFile, newFile)
                        i = newStep
                        filePath = H3_TRAJECTORIES_DATA_DIR + makeFilenameAdaptive(startingCoords, startingVels,
                                                                                   badGradientFunction, goodGradientFunction, i,
                                                                                   returnResources, countSampling,
                                                                                   gradientPrecision,testerFunction=verifyFunction,
                                                                                   )
                        with open(filePath, 'wb') as f:
                            pickle.dump(thing, f)




        for verifyFunction in verifyFunctionsEnergy:
            for badGradientFunction in badGradientFunctions:
                for goodGradientFunction in goodGradientFunctions:
                    badEnergyFunction=energyFunctions[badGradientFunction]
                    goodEnergyFunction=energyFunctions[goodGradientFunction]
                    initialGeometry = GeometrySpec.GeometrySpec(geometryFunction, startingCoords, startingVels, masses,
                                                                multiplicity)
                    simulator = trajectories.runIndependent(initialGeometry, badGradientFunction, goodGradientFunction,
                                                            returnResources, countSampling, gradientPrecision=gradientPrecision, testerFunction=verifyFunction,
                                                            badEnergyFunction=badEnergyFunction,
                                                            goodEnergyFunction=goodEnergyFunction)
                    i = 0
                    while i < its:
                        clive = next(simulator)
                        if returnResources:
                            thing = clive[0:2]
                            newStep = clive[2]
                        else:
                            thing = clive[0]
                            newStep = clive[1]
                        if newStep < i:  # if we branched
                            for k in [x for x in range(i + 1) if x >= newStep]:
                                oldFile = H3_TRAJECTORIES_DATA_DIR + makeFilenameAdaptive(startingCoords, startingVels,
                                                                                          badGradientFunction,
                                                                                          goodGradientFunction, k,
                                                                                          returnResources, countSampling,
                                                                                          gradientPrecision,
                                                                                          testerFunction=verifyFunction,
                                                                                          badEnergyFunction=badEnergyFunction,
                                                                                          goodEnergyFunction=goodEnergyFunction)
                                newFile = H3_TRAJECTORIES_DATA_DIR + makeFilenameAdaptive(startingCoords, startingVels,
                                                                                          badGradientFunction,
                                                                                          goodGradientFunction, k,
                                                                                          returnResources, countSampling,
                                                                                          gradientPrecision,
                                                                                          testerFunction=verifyFunction,
                                                                                          badEnergyFunction=badEnergyFunction,
                                                                                          goodEnergyFunction=goodEnergyFunction,
                                                                                          branch=True)
                                os.rename(oldFile, newFile)
                        i = newStep
                        filePath = H3_TRAJECTORIES_DATA_DIR + makeFilenameAdaptive(startingCoords, startingVels,
                                                                                   badGradientFunction, goodGradientFunction, i,
                                                                                   returnResources, countSampling,
                                                                                   gradientPrecision,
                                                                                   testerFunction=verifyFunction,
                                                                                   badEnergyFunction=badEnergyFunction,
                                                                                   goodEnergyFunction=goodEnergyFunction)
                        with open(filePath, 'wb') as f:
                            pickle.dump(thing, f)

        print('Done {}'.format(gradientPrecision))
    return

def cartesianToXYZ(cartesian):
    output = '3\n\n'
    for atom in cartesian:
        thisLine = '{}  {}  {}  {}\n'.format(atom[0],atom[1][0],atom[1][1],atom[1][2])
        output += thisLine
    return output



def thing3(its=300,
                                                 geometryFunction=trajectories.colinearH3JacobiGeometry,
                                                 startingCoords=[0.741,2.357],
                                                 startingVels=[0.,-0.2],
                                                 masses=trajectories.H3JacobiMasses,
                                                 multiplicity=2,
                                                 unadaptiveGradientFunctions = [trajectories.openFermionTaperedPartitionedGradientVQE],
                                                 badGradientFunctions=[],
                                                 goodGradientFunctions=[],
                                                 returnResources=True,
                                                 countSampling=True,
                                                 gradientPrecisions=[1e-08],
                                                 verifyFunctionsGradient=[],
                                                 verifyFunctionsEnergy=[],
                                                 energyFunctions={},
                                                 quantumChemPackage='pyscf',
                                                **kwargs):






    REPORT_FREQUENCY = datetime.timedelta(seconds=60)
    lastReport = datetime.datetime.now()
    startTime = lastReport
    unAdaptiveResults = []
    fullNumTrials = its * len(unadaptiveGradientFunctions)
    doneTrials = 0
    for gradientPrecision in [0]:
        for thisGradientFunction in unadaptiveGradientFunctions:
            result = []
            initialGeometry = GeometrySpec.GeometrySpec(geometryFunction, startingCoords, startingVels, masses,
                                                        multiplicity)
            simulator = trajectories.run(initialGeometry, thisGradientFunction, returnResources, countSampling,**kwargs)
            # firstFile = makeFilenameH3Jacobi(startingCoords,startingVels,thisGradientFunction,0,returnResources,countSampling)
            # firstPath = H3_TRAJECTORIES_DATA_DIR + firstFile
            # with open(firstPath,'wb') as f:
            #     pickle.dump(simulator,f)
            for i in range(its + 1):
                thing = next(simulator)
                result.append(thing)
                doneTrials += 1
                if datetime.datetime.now() - lastReport > REPORT_FREQUENCY:
                    fracDone = doneTrials / float(fullNumTrials)
                    estTime = (datetime.datetime.now() - startTime) * ((1. / fracDone) - 1.)
                    print('{}  {}% DONE, EST TIME TO COMPLETE: {}'.format(datetime.datetime.now(), fracDone * 100,
                                                                          estTime))
                    lastReport = datetime.datetime.now()
                thisFile = makeFilename(startingCoords, startingVels, thisGradientFunction, i, returnResources,
                                        countSampling)
                thisPath = H3_TRAJECTORIES_DATA_DIR + thisFile
                with open(thisPath, 'wb') as f:
                    pickle.dump(thing, f)
            unAdaptiveResults.append(result)
    for gradientPrecision in gradientPrecisions:
        for verifyFunction in verifyFunctionsGradient:
            for badGradientFunction in badGradientFunctions:
                for goodGradientFunction in goodGradientFunctions:
                    initialGeometry = GeometrySpec.GeometrySpec(geometryFunction, startingCoords, startingVels, masses,
                                                                multiplicity)
                    simulator = trajectories.runIndependent(initialGeometry, badGradientFunction, goodGradientFunction,
                                                            returnResources, countSampling, gradientPrecision=gradientPrecision, testerFunction=verifyFunction)
                    i = 0
                    while i < its:
                        clive = next(simulator)
                        if returnResources:
                            thing = clive[0:2]
                            newStep = clive[2]
                        else:
                            thing = clive[0]
                            newStep = clive[1]
                        #thing, newStep = next(simulator)
                        if newStep < i:  # if we branched
                            for k in [x for x in range(i + 1) if x >= newStep]:
                                oldFile = H3_TRAJECTORIES_DATA_DIR + makeFilenameAdaptive(startingCoords, startingVels,
                                                                                          badGradientFunction,
                                                                                          goodGradientFunction, k,
                                                                                          returnResources, countSampling,
                                                                                          gradientPrecision,testerFunction=verifyFunction,
                                                                                          )
                                newFile = H3_TRAJECTORIES_DATA_DIR + makeFilenameAdaptive(startingCoords, startingVels,
                                                                                          badGradientFunction,
                                                                                          goodGradientFunction, k,
                                                                                          returnResources, countSampling,
                                                                                          gradientPrecision,testerFunction=verifyFunction,
                                                                                          branch=True)
                                os.rename(oldFile, newFile)
                        i = newStep
                        filePath = H3_TRAJECTORIES_DATA_DIR + makeFilenameAdaptive(startingCoords, startingVels,
                                                                                   badGradientFunction, goodGradientFunction, i,
                                                                                   returnResources, countSampling,
                                                                                   gradientPrecision,testerFunction=verifyFunction,
                                                                                   )
                        with open(filePath, 'wb') as f:
                            pickle.dump(thing, f)




        for verifyFunction in verifyFunctionsEnergy:
            for badGradientFunction in badGradientFunctions:
                for goodGradientFunction in goodGradientFunctions:
                    badEnergyFunction=energyFunctions[badGradientFunction]
                    goodEnergyFunction=energyFunctions[goodGradientFunction]
                    initialGeometry = GeometrySpec.GeometrySpec(geometryFunction, startingCoords, startingVels, masses,
                                                                multiplicity)
                    simulator = trajectories.runIndependent(initialGeometry, badGradientFunction, goodGradientFunction,
                                                            returnResources, countSampling, gradientPrecision=gradientPrecision, testerFunction=verifyFunction,
                                                            badEnergyFunction=badEnergyFunction,
                                                            goodEnergyFunction=goodEnergyFunction)
                    i = 0
                    while i < its:
                        clive = next(simulator)
                        if returnResources:
                            thing = clive[0:2]
                            newStep = clive[2]
                        else:
                            thing = clive[0]
                            newStep = clive[1]
                        if newStep < i:  # if we branched
                            for k in [x for x in range(i + 1) if x >= newStep]:
                                oldFile = H3_TRAJECTORIES_DATA_DIR + makeFilenameAdaptive(startingCoords, startingVels,
                                                                                          badGradientFunction,
                                                                                          goodGradientFunction, k,
                                                                                          returnResources, countSampling,
                                                                                          gradientPrecision,
                                                                                          testerFunction=verifyFunction,
                                                                                          badEnergyFunction=badEnergyFunction,
                                                                                          goodEnergyFunction=goodEnergyFunction)
                                newFile = H3_TRAJECTORIES_DATA_DIR + makeFilenameAdaptive(startingCoords, startingVels,
                                                                                          badGradientFunction,
                                                                                          goodGradientFunction, k,
                                                                                          returnResources, countSampling,
                                                                                          gradientPrecision,
                                                                                          testerFunction=verifyFunction,
                                                                                          badEnergyFunction=badEnergyFunction,
                                                                                          goodEnergyFunction=goodEnergyFunction,
                                                                                          branch=True)
                                os.rename(oldFile, newFile)
                        i = newStep
                        filePath = H3_TRAJECTORIES_DATA_DIR + makeFilenameAdaptive(startingCoords, startingVels,
                                                                                   badGradientFunction, goodGradientFunction, i,
                                                                                   returnResources, countSampling,
                                                                                   gradientPrecision,
                                                                                   testerFunction=verifyFunction,
                                                                                   badEnergyFunction=badEnergyFunction,
                                                                                   goodEnergyFunction=goodEnergyFunction)
                        with open(filePath, 'wb') as f:
                            pickle.dump(thing, f)

        print('Done {}'.format(gradientPrecision))
    return