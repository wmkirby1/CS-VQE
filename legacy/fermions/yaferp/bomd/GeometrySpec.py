import copy

import numpy

from yaferp.bomd.trajectoriesConstants import DR, DTHETA, ATOMIC_NUMBERS


class GeometrySpec:
    def __init__(self,
                 geometryFunction,
                 initialGeometryParameters,
                 initialVelocities,
                 masses,
                 multiplicity = 1,
                 charge = 0,
                 geometryParameterTypes=None,
                 dr=DR,
                 dtheta=DTHETA
                 ):
        self.geometryFunction = geometryFunction
        self.coordinates = numpy.array(initialGeometryParameters)
        self.cartesian = geometryFunction(*initialGeometryParameters)
        if geometryParameterTypes is None:
            geometryParameterTypes = [0] * len(self.coordinates)
        self.coordinateTypes = geometryParameterTypes
        self.velocity = numpy.array(initialVelocities)
        self.masses = masses
        self.multiplicity = multiplicity
        self.charge = charge
        self.dr=dr
        self.dtheta=dtheta
        self.numElectrons = self._numElectrons_()

    def _numElectrons_(self):
        atomicNumbers = [ATOMIC_NUMBERS[x[0]] for x in self.cartesian]
        result = sum(atomicNumbers) - self.charge
        return result

    def copy(self):
        return copy.deepcopy(self)

    def update(self,newCoordinates):
        self.coordinates = [x.real for x in newCoordinates]
        self.cartesian = self.geometryFunction(*newCoordinates)
        return

    def peturb(self,coordinateIndex,peturbation):
        newCoords = self.coordinates
        newCoords[coordinateIndex] += peturbation
        self.coordinates = newCoords
        self.cartesian = self.geometryFunction(*newCoords)
        return self

    def peturbedCopy(self,coordinateIndex,peturbation):
        newGeometry = copy.deepcopy(self)
        newGeometry = newGeometry.peturb(coordinateIndex,peturbation)
        return newGeometry

    def peturbedGeometry(self,coordinateIndex):
        #if thisCoordinatesT
        thisCoordinateType = self.coordinateTypes[coordinateIndex]
        #newCoordinatesUp = copy.deepcopy(self.coordinates)
        #newCoordinatesDown = copy.deepcopy(self.coordinates)
        #if thisCoordinateType > 0:
        #    newCoordinatesUp += DTHETA
        #    newCoordinatesDown -= DTHETA
        #else:
        #    newCoordinatesUp += DR
        #    newCoordinatesDown -= DR

        if thisCoordinateType > 0:
            peturbationUp = self.dtheta
            peturbationDown = -1. * self.dtheta
        else:
            peturbationUp = self.dr
            peturbationDown = -1. * self.dr
        upGeometry = self.peturbedCopy(coordinateIndex,peturbationUp)
        downGeometry = self.peturbedCopy(coordinateIndex,peturbationDown)



        #upGeometry = GeometrySpec(self.geometryFunction,newCoordinatesUp,None,self.masses,self.coordinateTypes)
        #downGeometry = GeometrySpec(self.geometryFunction,newCoordinatesDown,None,self.masses,self.coordinateTypes)
        return (upGeometry,downGeometry)

    def allPeturbedGeometries(self):
        return [self.peturbedGeometry(i) for i in range(len(self.coordinates))]

    def cartesianWithCOM(self,comX,comY,comZ):
        result = []
        for atom in self.cartesian:
            thisThing = (atom[0],(atom[1][0]+comX,atom[1][1]+comY,atom[1][2]+comZ))
            result.append(thisThing)
        return result

    def __repr__(self):
        return '__'.join(['{}_{}_{}_{}'.format(x[0],x[1][0],x[1][1],x[1][2]) for x in self.cartesian])