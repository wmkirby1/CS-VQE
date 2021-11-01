import warnings
TERM_PRECISION = 0.0016
class ResourceEstimator():
    def __init__(self,countSampling=False,precision=TERM_PRECISION):
        self.totalGates = 0
        self.totalCircuits = 0
        self.totalSlices = 0
        self.countSampling = countSampling
        self.precision = precision
        self.previousSamples = 0
        self.precisionSquared = precision * precision
        return


    def _addCirquitNoSampling_(self,cirquit):
        totalGates = sum([len(x) for x in cirquit])
        self.totalGates += totalGates
        self.totalSlices += len(cirquit)
        self.totalCircuits += 1
        return

    def _addCirquitSampling_(self,cirquit,opterm=None,repeatSamples=False):
        oneRunGates = sum([len(x) for x in cirquit])
        oneRunSlices = len(cirquit)

        if repeatSamples:
            samples = self.previousSamples
        else:
            coefficientMag = abs(opterm[0])
            samples = (coefficientMag * coefficientMag) / self.precisionSquared
            self.previousSamples = samples

        self.totalGates += (samples * oneRunGates)
        self.totalSlices += (samples * oneRunSlices)
        self.totalCircuits += samples
        return

    def addCirquit(self,cirquit,opterm=None,repeatSamples=False):
        if self.countSampling:
            if opterm is None and not repeatSamples:
                raise ValueError("No opterm provided to sampling resourceestimator")
            else:
                self._addCirquitSampling_(cirquit,opterm,repeatSamples)

        else:
            if opterm is not None:
                warnings.warn("Opterm provided to resourceestimator but resourceestimator is not counting samples.")
            self._addCirquitNoSampling_(cirquit)

        return

    def duplicate(self,other):
        self.totalGates = other.totalGates
        self.totalCircuits = other.totalCircuits
        self.totalSlices = other.totalSlices
        self.countSampling = other.countSampling
        self.precision = other.precision
        self.previousSamples = other.previousSamples
        self.precisionSquared = self.precision * self.precision
        return