from yaferp.bomd import trajectories


def analyseColinearH3(filepath, reverseGeometryFunction = trajectories.colinearH3JacobiCoordinates):
    cartesians = cartesianCoordinates(filepath)
    result = [colinearParameters(x,reverseGeometryFunction) for x in cartesians]
    return result

def colinearParameters(cartesian,parameterFunction):
    internals = colinearInternals(cartesian)
    return parameterFunction(*internals)

def colinearInternals(cartesian):
    return extractZAxis(cartesian)[1:]

def extractZAxis(cartesian):
    return[x[2] for x in cartesian]

def cartesianCoordinates(filepath):
    with open(filepath,'r',encoding='utf-8') as f:
        raw = f.readlines()
    cartesians = grabCartesians(raw)
    transformedCartesians = [putFirstAtomAtOrigin(x) for x in cartesians]
    return transformedCartesians

def bohrToAngstrom(x):
    return 0.529177211 * x

def putFirstAtomAtOrigin(cartesian):
    offset = cartesian[0]
    result = [[x[i] - offset[i] for i in range(3)] for x in cartesian]
    return result

def grabCartesians(raw):
    cartesiansRaw = findCartesians(raw)
    cartesiansProcessed = [parseCartesianBlock(x) for x in cartesiansRaw]
    return cartesiansProcessed

def findCartesians(raw):
    rawIter = iter(raw)
    stuff = []
    thisLine = next(rawIter)
    try:
        while thisLine:
            if 'Cartesian coordinates: (bohr)' in thisLine:
                thisBlock = []
                thisLine = next(rawIter)
                while 'MW cartesian velocity: (sqrt(amu)*bohr/sec)' not in thisLine and 'TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TR' not in thisLine:
                    thisBlock.append(thisLine)
                    thisLine = next(rawIter)
                if not 'TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TR' in thisLine:
                    stuff.append(thisBlock)
            thisLine = next(rawIter)
    except StopIteration:
        return stuff

def parseCartesianBlock(block):
    cartesianProcessed = [parseCartesianLine(x) for x in block]
    return cartesianProcessed

def parseCartesianLine(line):
    splitLine = line.split()
    oldCoords = [splitLine[i] for i in [3,5,7]]
    resultBohr = [float(x.replace('D','E')) for x in oldCoords]
    result = [bohrToAngstrom(x) for x in resultBohr]
    return result