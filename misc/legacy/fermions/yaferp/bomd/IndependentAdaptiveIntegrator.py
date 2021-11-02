from yaferp.bomd import trajectoriesConstants
import copy
import numpy
from yaferp.bomd.AdaptiveIntegrator import AdaptiveIntegrator
from yaferp.bomd.VerletIntegrator import VerletIntegrator

def compareGradients(badGradientFunction,goodGradientFunction,resourceEstimator,geometries,precision):
    #oldResourceEstimator = copy.deepcopy(resourceEstimator)
    energyDifferencesBad = [badGradientFunction(x[0],x[1],x[2],resourceEstimator=resourceEstimator) for x in geometries]
    energyDifferencesGood = [goodGradientFunction(x[0],x[1],x[2],resourceEstimator=resourceEstimator) for x in geometries]
    energyDifferenceDifferences = [abs(energyDifferencesBad[i] - energyDifferencesGood[i]) for i in range(len(energyDifferencesBad))]
    tests = [x > precision for x in energyDifferenceDifferences]

    #result = (True in tests)
    #if result: #EXTREMELY CLUNKY HACK TO REMOVE DUPLICATION - if
    #    resourceEstimator.duplicate(oldResourceEstimator)

    return tests



class IndependentAdaptiveIntegrator(AdaptiveIntegrator):

    def __init__(self,
                 currentGeometrySpec,
                 badGradientFunction,
                 goodGradientFunction,
                 dr=trajectoriesConstants.DR,
                 dtheta=trajectoriesConstants.DTHETA,
                 dt=trajectoriesConstants.DT,
                 checkFrequency = trajectoriesConstants.CHECK_FREQUENCY,
                 gradientPrecision = trajectoriesConstants.GRADIENT_PRECISION,
                 resourceEstimator=None,
                 testerFunction = compareGradients,
                 badEnergyFunction = None,
                 goodEnergyFunction = None):
        self.hackyFlag = True
        super().__init__(currentGeometrySpec,
                         badGradientFunction,
                         goodGradientFunction,
                         dr,
                         dtheta,
                         dt,
                         checkFrequency,
                         gradientPrecision,
                         resourceEstimator,
                         testerFunction,
                         badEnergyFunction,
                         goodEnergyFunction)
        self.hackyFlag = False
        self.gradientFunction = [goodGradientFunction] * len(self.x)

        self.a = self.acceleration()
        self.store()

        self.gradientFunction = [badGradientFunction] * len(self.x)
        if self.resourceEstimator is not None:
            storedResourceEstimator = copy.deepcopy(self.resourceEstimator)
        self.a = self.acceleration()
        if self.resourceEstimator is not None:
            self.resourceEstimator = storedResourceEstimator
        return

    def evolve(self, timestep=None):
        if timestep is None:
            timestep = self.dt

        if ((self.step + 1) % self.checkFrequency) == 0:
            areEstimatesBad = self.verify(resourceCounting='mixed')
            isUsingGood = [x  == self.goodGradientFunction for x in self.gradientFunction]
            if True in [areEstimatesBad[i] and (not isUsingGood[i]) for i in range(len(areEstimatesBad))]:
                self.reset()
                for i,x in enumerate(areEstimatesBad):
                    if x:
                        self.gradientFunction[i] = self.goodGradientFunction
            else:
                self.gradientFunction = [self.goodGradientFunction]* len(self.x)
                if self.resourceEstimator is not None:
                    storedResourceEstimator = copy.deepcopy(self.resourceEstimator)
                self.a = self.acceleration()
                if self.resourceEstimator is not None:
                    self.resourceEstimator = storedResourceEstimator
                self.store()
                for i,x in enumerate(areEstimatesBad):
                    if not x:
                        self.gradientFunction[i] = self.badGradientFunction

        VerletIntegrator.evolve(self,timestep)

        return

    def verify(self,resourceCounting='mixed'):
        peturbedGeometries = self.baseGeometry.allPeturbedGeometries()
        geometries = [(self.baseGeometry,x[0],x[1]) for x in peturbedGeometries]
        distanceFunction = self.verificationFunction
        if resourceCounting == 'mixed':
            result = []
            for i, geometry in enumerate(geometries):
                if self.gradientFunction[i] == self.goodGradientFunction:
                    storedResourceEstimator = copy.deepcopy(self.resourceEstimator)
                    thing = distanceFunction(self.badGradientFunction,self.goodGradientFunction,self.resourceEstimator,[geometry],self.precision)[0]
                    result.append(thing)
                    self.resourceEstimator = storedResourceEstimator
                else:
                    thing = distanceFunction(self.badGradientFunction,self.goodGradientFunction,self.resourceEstimator,[geometry],self.precision)[0]
                    result.append(thing)
        return result

    def acceleration(self,geometry=None):
        if geometry is None:
            geometry = self.baseGeometry
        peturbedGeometries = geometry.allPeturbedGeometries()
        fullPeturbedGeometries = [(geometry,x[0],x[1]) for x in peturbedGeometries]
        if self.hackyFlag:
            forces = [self.force(fullPeturbedGeometries[i],self.gradientFunction) for i in range(len(fullPeturbedGeometries))]
        else:
            forces = [self.force(fullPeturbedGeometries[i],self.gradientFunction[i]) for i in range(len(fullPeturbedGeometries))]
        a = [forces[i]/geometry.masses[i] for i in range(len(peturbedGeometries))]
        arrayA = numpy.array(a) * 0.262549964 #return in angstrom/fs^2
        return arrayA

