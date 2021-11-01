import numpy
cimport numpy
INDEXTYPE = numpy.int32
ctypedef numpy.int32_t INDEXTYPE_t

def oneXOnState(numpy.ndarray[INDEXTYPE_t, ndim=1] indices,xIndex):
    cdef unsigned int xIndexStatic = xIndex
    cdef unsigned int qubitMask = 1<<xIndexStatic 
    cdef unsigned int maxIndex = len(indices)
    cdef unsigned int i       
    for i in range(maxIndex):
        indices[i] = indices[i] ^ qubitMask
    return(indices)