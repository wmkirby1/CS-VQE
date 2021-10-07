from yaferp.bomd import trajectoriesConstants
from yaferp.bomd.VerletIntegrator import VerletIntegrator
import copy


def gradientDifferences(badGradientFunction,goodGradientFunction,geometries):
    energyDifferencesBad = [badGradientFunction(x[0], x[1], x[2]) for x in
                            geometries]
    energyDifferencesGood = [goodGradientFunction(x[0], x[1], x[2]) for x in
                             geometries]
    energyDifferenceDifferences = [abs(energyDifferencesBad[i] - energyDifferencesGood[i]) for i in
                                   range(len(energyDifferencesBad))]
    return energyDifferenceDifferences


def compareGradients(badGradientFunction,goodGradientFunction,resourceEstimator,geometries,precision):
    oldResourceEstimator = copy.deepcopy(resourceEstimator)
    energyDifferencesBad = [badGradientFunction(x[0],x[1],x[2],resourceEstimator=resourceEstimator) for x in geometries]
    energyDifferencesGood = [goodGradientFunction(x[0],x[1],x[2],resourceEstimator=resourceEstimator) for x in geometries]
    energyDifferenceDifferences = [abs(energyDifferencesBad[i] - energyDifferencesGood[i]) for i in range(len(energyDifferencesBad))]
    tests = [x > precision for x in energyDifferenceDifferences]

    result = (True in tests)
    if result: #EXTREMELY CLUNKY HACK TO REMOVE DUPLICATION - if
        resourceEstimator.duplicate(oldResourceEstimator)

    return result

def compareEnergy(badEnergyFunction,goodEnergyFunction,resourceEstimator,geometry,precision):
    energyBad = badEnergyFunction(geometry,resourceEstimator=resourceEstimator)
    energyGood = goodEnergyFunction(geometry,resourceEstimator=resourceEstimator)
    energyDifference = abs(energyGood - energyBad)
    result = energyDifference > precision
    return result

VERIFICATION_FUNCTIONS_TO_DISTANCE_FUNCTIONS = {compareGradients:gradientDifferences}


class AdaptiveIntegrator(VerletIntegrator):
    def __init__(self,
                 currentGeometrySpec,
                 badGradientFunction,
                 goodGradientFunction,
                 dr=trajectoriesConstants.DR,
                 dtheta=trajectoriesConstants.DTHETA,
                 dt=trajectoriesConstants.DT,
                 checkFrequency=trajectoriesConstants.CHECK_FREQUENCY,
                 gradientPrecision=trajectoriesConstants.GRADIENT_PRECISION,
                 resourceEstimator=None,
                 testerFunction=compareGradients,
                 badEnergyFunction=None,
                 goodEnergyFunction=None):
        self.badGradientFunction = badGradientFunction
        self.goodGradientFunction = goodGradientFunction
        self.lastGoodV = currentGeometrySpec.velocity
        self.lastGoodX = currentGeometrySpec.coordinates
        self.lastGoodStep = 0
        self.checkFrequency = checkFrequency
        self.precision = gradientPrecision
        self.verificationFunction = testerFunction
        if (badEnergyFunction is not None and goodEnergyFunction is None) or (goodEnergyFunction is None and badEnergyFunction is not None):
            raise ValueError('need either both bad and good energy functions or neither')
        self.badEnergyFunction = badEnergyFunction
        self.goodEnergyFunction = goodEnergyFunction
        self.lastGoodA = None
        super().__init__(currentGeometrySpec=currentGeometrySpec,
                        gradientFunction=self.badGradientFunction,
                        dr=dr,
                        dtheta=dtheta,
                        dt=dt,
                        resourceEstimator=resourceEstimator)



    def verificationValues(self):
        peturbedGeometries = self.baseGeometry.allPeturbedGeometries()
        geometries = [(self.baseGeometry,x[0],x[1]) for x in peturbedGeometries]
        distanceFunction = VERIFICATION_FUNCTIONS_TO_DISTANCE_FUNCTIONS[self.verificationFunction]
        result = distanceFunction(self.badGradientFunction,self.goodGradientFunction,geometries)
        return result


    def verify(self,geometry=None,storeGood=True):
        if geometry is None:
            geometry = self.baseGeometry

        if self.badEnergyFunction is not None:
            result = self.verificationFunction(self.badEnergyFunction,self.goodEnergyFunction,self.resourceEstimator,geometry,self.precision)
        else:
            peturbedGeometries = geometry.allPeturbedGeometries()
            geometries = [(geometry,x[0],x[1]) for x in peturbedGeometries]
            result = self.verificationFunction(self.badGradientFunction,self.goodGradientFunction,self.resourceEstimator,geometries,self.precision)
        return result

    def reset(self):
        self.step = self.lastGoodStep
        self.t = self.step * self.dt
        self.x = self.lastGoodX
        self.v = self.lastGoodV
        self.resetAccel()
        return

    def resetAccel(self):
        if self.lastGoodA is None:
            self.a = self.acceleration()
        else:
            self.a = self.lastGoodA
        return


    def store(self,storeAccel=True):
        self.lastGoodStep = self.step
        self.lastGoodX = self.x
        self.lastGoodV = self.v
        self.lastGoodA = self.a
        return

    def storeAccel(self,wipe=False):

        self.lastGoodA = self.a
        if wipe:
            self.lastGoodA = None
        return

    def storeGoodAccel(self,free=True):
        if free and self.resourceEstimator is not None:
            storedResourceEstimator = copy.deepcopy(self.resourceEstimator)
        storedGradientFunction = self.gradientFunction
        self.gradientFunction = self.goodGradientFunction
        self.lastGoodA = self.acceleration()
        self.gradientFunction = storedGradientFunction
        if free and self.resourceEstimator is not None:
            self.resourceEstimator.duplicate(storedResourceEstimator)
        return


    def evolve(self,timestep=None):
        if timestep is None:
            timestep = self.dt

        if ((self.step + 1) % self.checkFrequency) == 0:
            isUsingGood = self.gradientFunction == self.goodGradientFunction
            if isUsingGood:
                self.store()
                self.storeAccel()
                self.gradientFunction = self.badGradientFunction
            else:
                #storedResourceEstimator = copy.deepcopy(self.resourceEstimator)
                isEstimateBad = self.verify()
                if not isEstimateBad:
                    self.store()
                    self.storeGoodAccel(free=True)
                else:
                    self.reset()
                    #self.resourceEstimator.duplicate(storedResourceEstimator)
                    self.gradientFunction = self.goodGradientFunction

         #   if isUsingGood or (not isEstimateBad):
         #       self.store()
         #   if (not isUsingGood) and (not isEstimateBad):
         #       self.storeAccel(wipe=True)


        #    if (not isUsingGood) and isEstimateBad:
        #            self.reset()
        #            self.gradientFunction = self.goodGradientFunction

            '''
            if isUsingGood:
                

            
            isUsingGood = self.gradientFunction == self.goodGradientFunction
            if isRedoNeeded:
                if not isUsingGood:
                    self.reset()
                    self.gradientFunction = self.goodGradientFunction

            else:
                if isUsingGood:
                    self.gradientFunction = self.badGradientFunction

            '''
        super().evolve(timestep)
        '''
        if (self.step % self.checkFrequency) == 0:
            
            if (not isRedoNeeded) or isUsingGood:
                self.store()
        '''
        return

