import copy

import numpy

from yaferp.bomd.trajectoriesConstants import DR, DTHETA, DT


class VerletIntegrator:
    def __init__(self,
                 currentGeometrySpec=None,gradientFunction=None,dr=DR,dtheta=DTHETA,dt=DT,resourceEstimator=None, **kwargs):
        self.baseGeometry = copy.deepcopy(currentGeometrySpec)
        #self.potentialFunction = potentialFunction
        self.gradientFunction = gradientFunction
        self.t = 0.
        self.v = currentGeometrySpec.velocity
        self.x = currentGeometrySpec.coordinates
        self.dr = dr
        self.dtheta = dtheta
        self.dt = dt
        self.step = 0
        self.oplistsStored = True
        self.resourceEstimator = resourceEstimator
        self.config = kwargs
        self.a = self.acceleration()

    def accelerationByPotential(self,geometry=None):
        #TODO: allow for curvilinear coords
        if geometry is None:
            geometry = self.baseGeometry
        peturbedGeometries = geometry.allPeturbedGeometries()
        a = []
        #currentEnergy = self.potentialFunction(self.baseGeometry)
        for i,thisPair in enumerate(peturbedGeometries):
            pds = [self.potentialFunction(x) for x in thisPair]
            force=(-1. * (pds[0] - pds[1])/(2. * self.dr))  #ASSUMES RECTILINEAR HERE!
            thisA = force/geometry.masses[i]
            a.append(thisA)

        arrayA = numpy.array(a)
        return arrayA

    def force(self,geometries,gradientFunction=None):
        if gradientFunction is None:
            gradientFunction = self.gradientFunction
        ''' geometry0 is unpeturbed, geometriy 1 is plus, geometry 2 is minus'''
        if self.config: #backwards compat - TODO make gradient functions inherit from a single class or something
            energyDifference = gradientFunction(geometries[0],geometries[1],geometries[2],resourceEstimator=self.resourceEstimator,**self.config)
        else:
            energyDifference = gradientFunction(geometries[0],geometries[1],geometries[2],resourceEstimator=self.resourceEstimator)
        '''
        if self.oplistsStored:
            energyDifference= self.gradientFunction(geometries[0],geometries[1],geometries[2],returnOplists=True)
        else:
            energyDifference = self.gradientFunction(geometries[0],geometries[1],geometries[2],returnOplists=False)
        '''
        force = (-1. * energyDifference / (2.*self.dr))
        return force

    def acceleration(self,geometry=None):
        if geometry is None:
            geometry = self.baseGeometry
        peturbedGeometries = geometry.allPeturbedGeometries()
        fullPeturbedGeometries = [(geometry,x[0],x[1]) for x in peturbedGeometries]
        #a = []
        #pool = multiprocessing.Pool()
        forces = [self.force(x) for x in fullPeturbedGeometries]
        #forces = pool.map(self.force,fullPeturbedGeometries)
        #pool.close()
        #pool.join()
        a = [forces[i]/geometry.masses[i] for i in range(len(peturbedGeometries))]
        #for i, thisPair in enumerate(peturbedGeometries):
            #energyDifference = self.gradientFunction(geometry,thisPair[0],thisPair[1])
            #force = (-1. * energyDifference / (2.*DR))
            #thisA = force/geometry.masses[i]
            #a.append(thisA)
        arrayA = numpy.array(a) * 0.262549964 #return in angstrom/fs^2
        return arrayA


    def evolve(self,timestep=None):
        if timestep is None:
            timestep = self.dt
        self.step += 1
        self.t += timestep
        vHalf = self.v + (timestep * 0.5 * self.a)
        self.x = self.x + (timestep * vHalf)
        self.baseGeometry.update(self.x)
        newA = self.acceleration()
        self.v = self.v + (0.5 * (self.a + newA) * timestep)
        self.a = newA
        return