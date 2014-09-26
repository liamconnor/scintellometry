from __future__ import division

import numpy as np
import os
from astropy.table import Table

from mpi4py import MPI

class multifile(object):
    def __init__(self, sequence_file, raw_voltage_files, recsize=2**25,
                 comm=None):
        if comm is None:
            self.comm = MPI.COMM_SELF
        else:
            self.comm = comm
        self.sequence_file = sequence_file
        self.sequence = Table(np.loadtxt(sequence_file, np.int32),
                              names=['seq', 'raw'])
        self.sequence.sort('seq')
        self.fh_raw = []
        # MPI.File.Open doesn't handle files with ":"'s, track tmp files
        self.fh_links = []
        for raw in raw_voltage_files:
             fname, islnk = good_name(raw)
             self.fh_raw.append(MPI.File.Open(self.comm, fname,
                                              amode=MPI.MODE_RDONLY))
             if islnk:
                 self.fh_links.append(fname)
        self.recsize = recsize
        self.index = 0
        self.seq0 = self.sequence['seq'][0]

    def seek(self, offset):
        assert offset % self.recsize == 0
        self.index = offset // self.recsize
        for i, fh in enumerate(self.fh_raw):
            fh.Seek(np.count_nonzero(self.sequence['raw'][:self.index] == i) *
                    self.recsize)

    def close(self):
        for fh in self.fh_raw:
            fh.Close()
        for fh in self.fh_links:
            if os.path.exists(fh):
                os.unlink(fh)    

    def read(self, size):
        assert size == self.recsize
        if self.index == len(self.sequence):
            raise EOFError
        self.index += 1
        # so far, only works for continuous data, so ensure we're not missing
        # any sequence numbers
        if self.sequence['seq'][self.index-1] != self.index-1 + self.seq0:
            raise IOError("multifile sequence numbers have to be consecutive")
        i = self.sequence['raw'][self.index-1]
        # MPI read requires the buffer
        z = np.zeros(size, dtype='i1')
        self.fh_raw[i].Iread(z)
        return z

    def __repr__(self):
        return ("<open multifile raw_voltage_files {} "
                "using sequence file '{}' at index {}>"
                .format(self.fh_raw, self.sequence_file, self.index))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

def good_name(f):
    """ 
    MPI.File.Open can't process files with colons. 
    This routine checks for such cases and creates a well-named link to the file.
    
    Returns (good_name, islink)
    """
    if f is None: return f

    fl = f
    newlink = False
    if ':' in f:
        #fl = tempfile.mktemp(prefix=os.path.basename(f).replace(':','_'), dir='/tmp')
        fl = os.path.join('/tmp', os.path.dirname(f).replace('/','_') + '__' + os.path.basename(f).replace(':','_'))
        if not os.path.exists(fl):
            try:
                os.symlink(f, fl) 
            except(OSError):
                pass
            newlink = True
    return fl, newlink
