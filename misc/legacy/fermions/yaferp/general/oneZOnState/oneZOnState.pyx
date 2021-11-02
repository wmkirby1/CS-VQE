import numpy
cimport numpy
DATATYPE = numpy.complex128
INDEXTYPE = numpy.int32
ctypedef numpy.complex128_t DATATYPE_t
ctypedef numpy.int32_t INDEXTYPE_t

def oneZOnState(numpy.ndarray[DATATYPE_t, ndim=1] data, numpy.ndarray[INDEXTYPE_t, ndim=1] indices,zIndex):
    cdef unsigned int zIndexStatic = zIndex
    cdef unsigned int qubitMask = 1<<zIndexStatic 
    cdef unsigned int maxIndex = len(data)
    cdef unsigned int i       
    for i in range(maxIndex):
        if indices[i] & qubitMask:
            data[i] = -data[i]
    return(data)