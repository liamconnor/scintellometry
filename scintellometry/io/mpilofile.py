""" module to MPI-Open LOFAR files and have them act like python files"""
import numpy as np
from mpi4py import MPI

class mpilofile(object):
    def __init__(self, comm, fname):
        self.fh = MPI.File.Open(comm, fname, amode=MPI.MODE_RDONLY)

    def seek(self, offset):
        self.fh.Seek(offset)
 
    def read(self, size):
        z = np.zeros(size, dtype='i1')
        self.fh.Iread([z, MPI.BYTE])
        return z
 
    def close(self):
        self.fh.Close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.fh.Close()

    def __enter__(self):
        return self
 
