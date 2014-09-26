#           _____   ____   _____ _    _ _____ __  __ ______ 
#     /\   |  __ \ / __ \ / ____| |  | |_   _|  \/  |  ____|
#    /  \  | |__) | |  | | |    | |__| | | | | \  / | |__   
#   / /\ \ |  _  /| |  | | |    |  __  | | | | |\/| |  __|  
#  / ____ \| | \ \| |__| | |____| |  | |_| |_| |  | | |____ 
# /_/    \_\_|  \_\\____/ \_____|_|  |_|_____|_|  |_|______|

from __future__ import division
import os

import numpy as np
from scipy.fftpack import fftfreq, fftshift
from astropy.time import Time, TimeDelta
import astropy.units as u
from mpi4py import MPI

from . import MultiFile, header_defaults


class AROCHIMEData(MultiFile):

    telescope = 'arochime'

    def __init__(self, raw_files, blocksize,
                 samplerate, fedge, fedge_at_top, dtype='cu4bit,cu4bit',
                 comm=None):
        """ARO data aqcuired with a CHIME correlator containts 1024
        channels over the 400MHz BW, 2 polarizations, and 2 unsigned 8-byte ints for 
        real and imaginary for each timestamp. 
        """

        self.meta = eval(open(raw_files[0] + '.meta').read())
        nchan = self.meta['nfreq']
        self.time0 = Time(self.meta['stime'], format='unix')
        self.npol = self.meta['ninput']
        self.samplerate = samplerate
        self.fedge = fedge
        self.fedge_at_top = fedge_at_top
        f = fftshift(fftfreq(nchan, (2./samplerate).to(u.s).value)) * u.Hz
        if fedge_at_top:
            self.frequencies = fedge - (f-f[0])
        else:
            self.frequencies = fedge + (f-f[0])
        
        self.dtsample = (nchan * 2 / samplerate).to(u.s)
        if comm.rank == 0:
            print("In AROCHIMEData, calling super")
            print("Start time: ", self.time0.iso)
#        self.indices = np.zeros(2000, dtype=np.int)
        self.files = raw_files
        self.filesize = os.path.getsize(self.files[0])
        self.current_file_number = None
        super(AROCHIMEData, self).__init__(raw_files, blocksize, dtype, nchan,
                                           comm=comm)

    def open(self, files, file_number=0):
        if file_number == self.current_file_number:
            return
        if self.current_file_number is not None:
            self.fh_raw.Close()
        self.fh_raw = MPI.File.Open(self.comm, files[file_number], amode=MPI.MODE_RDONLY)
        self.current_file_number = file_number

    def close(self):
        self.fh_raw.Close()

    # def read(self, size):
    #     """Read size bytes, returning an ndarray with np.int8 dtype.

    #     Incorporate information from multiple underlying files if necessary.
    #     The individual file pointers are assumed to be pointing at the right
    #     locations, i.e., just before data that will be read here.
    #     """
    #     if size % self.recordsize != 0:
    #         raise ValueError("Cannot read a non-integer number of records")

    #     z = np.empty(size, dtype=np.uint8)
        
    #     self.fh_raw.Iread(z)
    #     self.offset += size
    #     return z

    def read(self, size):
        """Read size bytes, returning an ndarray with np.int8 dtype.

        Incorporate information from multiple underlying files if necessary.
        The current file pointer are assumed to be pointing at the right
        locations, i.e., just before the first bit of data that will be read.
        """

        if size % self.recordsize != 0:
            raise ValueError("Cannot read a non-integer number of records")

        # ensure we do not read beyond end
        size = min(size, len(self.files) * self.filesize - self.offset)
        if size <= 0:
            raise EOFError('At end of file in AROCHIME.read')

        # allocate buffer for MPI read
        z = np.empty(size, dtype=np.int8)

        # read one or more pieces
        iz = 0
        while(iz < size):
            block, already_read = divmod(self.offset, self.filesize)
            fh_size = min(size - iz, self.filesize - already_read)
            self.fh_raw.Iread(z[iz:iz+fh_size])
            self._seek(self.offset + fh_size)
            iz += fh_size

        return z

    def _seek(self, offset):
         assert offset % self.recordsize == 0
         file_number = offset // self.filesize
         file_offset = offset % self.filesize
         self.open(self.files, file_number)
         self.fh_raw.Seek(file_offset)
         self.offset = offset
        

    def ntint(self, nchan):
        return self.setsize


    # def record_read(self, count):
    #     data = fromfile_arochime(self, np.uint8, count)
    #     return data.reshape(-1, self.nchan, self.npol*2).astype(np.float32).view('c8,c8').squeeze()

# GMRT defaults for psrfits HDUs
# Note: these are largely made-up at this point
header_defaults['arochime'] = {
    'PRIMARY': {'TELESCOP':'AROCHIME',
                'IBEAM':1, 'FD_POLN':'LIN',
                'OBS_MODE':'SEARCH',
                'ANT_X':0, 'ANT_Y':0, 'ANT_Z':0, 'NRCVR':1,
                'FD_HAND':1, 'FD_SANG':0, 'FD_XYPH':0,
                'BE_PHASE':0, 'BE_DCC':0, 'BE_DELAY':0,
                'TRK_MODE':'TRACK',
                'TCYCLE':0, 'OBSFREQ':300, 'OBSBW':100,
                'OBSNCHAN':0, 'CHAN_DM':0,
                'EQUINOX':2000.0, 'BMAJ':1, 'BMIN':1, 'BPA':0,
                'SCANLEN':1, 'FA_REQ':0,
                'CAL_FREQ':0, 'CAL_DCYC':0, 'CAL_PHS':0, 'CAL_NPHS':0,
                'STT_IMJD':54000, 'STT_SMJD':0, 'STT_OFFS':0},
    'SUBINT': {'INT_TYPE': 'TIME',
               'SCALE': 'FluxDen',
               'POL_TYPE': 'AABB',
               'NPOL':1,
               'NBIN':1, 'NBIN_PRD':1,
               'PHS_OFFS':0,
               'NBITS':1,
               'ZERO_OFF':0, 'SIGNINT':0,
               'NSUBOFFS':0,
               'NCHAN':1,
               'CHAN_BW':1,
               'DM':0, 'RM':0, 'NCHNOFFS':0,
               'NSBLK':1}}


def read_start_time(filename):
    """
    Reads in arochime .meta file as a dictionary and gets start time.
    
    Parameters
    ----------
    filename: str
         full path to .meta file

    Returns
    -------
    start_time: Time object
         Unix time of observation start
    """
    f = eval(open(filename).read()) 
    return Time(f['stime'], format='unix')

    

def read_timestamp_file(filename, utc_offset=5.5*u.hr):
    """Read timestamps from GMRT timestamp file.

    Parameters
    ----------
    filename : str
        full path to the timestamp file
    utc_offset : Quantity or TimeDelta
        offset from UTC, subtracted from the times in the timestamp file.
        Default: 5.5*u.hr

    Returns
    -------
    indices : array of int
        list of indices (alternating 0 and 1) into the two raw data files
    timestamps : Time array
        UTC times associated with the data blocks
    gsb_start : Time
        UTC time at which the GMRT software correlator was started

    Notes
    -----

    A typical first line of a timestamp file is:

    2014 01 20 02 28 10 0.811174 2014 01 20 02 28 10 0.622453760 5049 1

    Here, the first set is the time as given by the PC that received the
    block, the second that from GPS.  This is followed by a sequence number
    and a sub-integration number.  These should increase monotonically.

    The code checks that the time difference PC-GPS is (roughly) constant,
    and that the first sequence was triggered on an integer GPS minute.

    The actual data are stored in two interleaved streams, and the routine
    returns a Time array that is twice the length of the time-stamp file,
    having interpolated the times for the second data stream.
    """

    utc_offset = TimeDelta(utc_offset)

    str2iso = lambda str: '{}-{}-{}T{}:{}:{}'.format(
        str[:4], str[5:7], str[8:10], str[11:13], str[14:16], str[17:19])
    dtype = np.dtype([('pc', 'S19'), ('pc_frac', np.float),
                      ('gps', 'S19'), ('gps_frac', np.float),
                      ('seq', np.int), ('sub', np.int)])

    timestamps = np.genfromtxt(filename, dtype=dtype,
                               delimiter=(19, 10, 20, 12, 5, 2),  # col lengths
                               converters={0: str2iso, 2: str2iso})

    # check if last line was corrupted
    if timestamps[-1]['sub'] < 0:
        timestamps = timestamps[:-1]

    # should have continuous series, of subintegrations at least
    assert np.all(np.diff(timestamps['sub']) % 8 == 1)  # either 1 or -7

    pc_times = (Time(timestamps['pc'], scale='utc', format='isot') +
                TimeDelta(timestamps['pc_frac'], format='sec') - utc_offset)
    gps_times = (Time(timestamps['gps'], scale='utc', format='isot') +
                 TimeDelta(timestamps['gps_frac'], format='sec') - utc_offset)

    gps_pc = gps_times - pc_times
    assert np.allclose(gps_pc.sec, gps_pc[0].sec, atol=5.e-3)

    # GSB should have started on whole minute
    gsb_start = gps_times[0] - timestamps[0]['seq'] * (gps_times[1] -
                                                       gps_times[0])
#    assert '00.000' in gsb_start.isot

    # still, the sequence can have holes of 8, which need to be filled
    seq = timestamps['seq']
    dseq = np.diff(seq)
    holes = np.where(dseq > 1)
    # hole points to time just before hole
    for hole in holes[0][::-1]:  # reverse order since we are adding stuff
        hole_dt = gps_times[hole+1] - gps_times[hole]
        hole_frac = np.arange(1, dseq[hole], dtype=np.int) / float(dseq[hole])
        hole_times = gps_times[hole] + hole_frac * hole_dt
        gps_times = Time([gps_times[:hole+1], hole_times,
                          gps_times[hole+1:]])
        seq = np.hstack((seq[:hole+1], -np.ones(len(hole_frac)), seq[hole+1:]))

    # time differences between subsequent samples should now be (very) similar
    dt = gps_times[1:] - gps_times[:-1]
    assert np.allclose(dt.sec, dt[0].sec, atol=1.e-5)

    indices = np.repeat([[0,1]], len(gps_times), axis=0)
    # double the number of timestamps
    times = Time(np.repeat(gps_times.jd1, 2), np.repeat(gps_times.jd2, 2),
                 format='jd', scale='utc', precision=9)

    times = times + indices.flatten() * (dt[0] / 2.)
    # mark bad indices
    indices[seq < 0] = np.array([-1,-1])

    return indices.flatten(), times, gsb_start
