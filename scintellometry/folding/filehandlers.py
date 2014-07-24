""" classes to handle ARO, LOFAR, and GMRT data in a consistent way """

from __future__ import division

import numpy as np
import os
import re
import warnings

from scipy.fftpack import fftfreq, fftshift
from astropy import units as u
from astropy.time import Time, TimeDelta
from fromfile import fromfile
from h5py import File as HDF5File
try:
    from mpi4py import MPI
except ImportError:
    pass
from psrfits_tools import psrFITS


# size in bytes of records read from file (simple for ARO: 1 byte/sample)
def dtype_itemsize(dtype):
    bps = {'ci1': 2, '(2,)ci1': 4, '4bit': 0.5}.get(dtype, None)
    if bps is None:
        bps = np.dtype(dtype).itemsize
    return bps

# default properties for various telescopes
header_defaults = {}

# hdf5 dtype conversion
_lofar_dtypes = {'float': '>c8', 'int8': 'ci1'}


class MultiFile(psrFITS):

    def __init__(self, files=None, blocksize=None, dtype=None, nchan=None,
                 comm=None):
        if comm is None:
            self.comm = MPI.COMM_SELF
        else:
            self.comm = comm
        if files is not None:
            self.open(files)
        # parameters for fold:
        if blocksize is not None:
            self.blocksize = blocksize
        if dtype is not None:
            self.dtype = dtype
        if nchan is not None:
            self.nchan = nchan
        self.itemsize = dtype_itemsize(self.dtype)
        self.recordsize = self.itemsize * self.nchan
        assert self.blocksize % self.recordsize == 0
        self.setsize = self.blocksize // self.recordsize

        super(MultiFile, self).__init__(hdus=['SUBINT'])
        self.set_hdu_defaults(header_defaults[self.telescope])

    def set_hdu_defaults(self, dictionary):
        for hdu in dictionary:
            self[hdu].header.update(dictionary[hdu])

    def open(self, files):
        # MPI.File.Open doesn't handle files with ":"
        self.fh_raw = []
        self.fh_links = []
        for raw in files:
            fname, islnk = good_name(os.path.abspath(raw))
            self.fh_raw.append(MPI.File.Open(self.comm, fname,
                                             amode=MPI.MODE_RDONLY))
            if islnk:
                self.fh_links.append(fname)
        self.offset = 0

    def close(self):
        for fh in self.fh_raw:
            fh.Close()
        for fh in self.fh_links:
            if os.path.exists(fh):
                os.unlink(fh)

    def read(self, size):
        """Read size bytes, returning an ndarray with np.int8 dtype.

        Incorporate information from multiple underlying files if necessary.
        The individual file pointers are assumed to be pointing at the right
        locations, i.e., just before data that will be read here.
        """
        if size % self.recordsize != 0:
            raise ValueError("Cannot read a non-integer number of records")

        # ensure we do not read beyond end
        size = min(size, len(self.indices) * self.blocksize - self.offset)
        if size <= 0:
            raise EOFError('At end of file in MultiFile.read')

        # allocate buffer for MPI read
        z = np.empty(size, dtype=np.int8)

        # read one or more pieces
        iz = 0
        while(iz < size):
            block, already_read = divmod(self.offset, self.blocksize)
            fh_size = min(size - iz, self.blocksize - already_read)
            fh_index = self.indices[block]
            if fh_index >= 0:
                self.fh_raw[fh_index].Iread(z[iz:iz+fh_size])
            else:
                z[iz:iz+fh_size] = 0
            self.offset += fh_size
            iz += fh_size

        return z

    def seek(self, offset):
        """Move filepointers to given offset

        Parameters
        ----------
        offset : float, Quantity, TimeDelta, Time, or str (iso-t)
            If float, in units of bytes
            If Quantity in time units or TimeDelta, interpreted as offset from
                start time, and converted to nearest record
            If Time, calculate offset from start time and convert
        """
        if isinstance(offset, Time):
            offset = offset-self.time0
        elif isinstance(offset, str):
            offset = Time(offset, scale='utc') - self.time0

        try:
            offset = offset.to(self.dtsample.unit)
        except AttributeError:
            pass
        except u.UnitsError:
            offset = int(offset.to(u.byte).value)
        else:
            offset = (offset/self.dtsample).to(u.dimensionless_unscaled)
            offset = int(round(offset)) * self.recordsize
        self._seek(offset)

    def _seek(self, offset):
        if offset % self.recordsize != 0:
            raise ValueError("Cannot offset to non-integer number of records")
        # determine index in units of the blocksize
        block, extra = divmod(offset, self.blocksize)
        if block > len(self.indices):
            raise EOFError('At end of file in MultiFile.read')
        # check how many of the indices preceding the block were in each file
        indices = self.indices[:block]
#        fh_offsets = np.bincount(indices[indices >= 0],
#                                 minlength=len(self.fh_raw)) * self.blocksize
        fh_offsets = np.bincount(indices[indices >= 0], minlength=1)
        # add the extra bytes to the correct file
        print("yes", self.indices, block, self.indices[block], extra)
        if self.indices[block] >= 0:
            fh_offsets[self.indices[block]] += extra

        # actual seek in files
        for fh, fh_offset in zip(self.fh_raw, fh_offsets):
            fh.Seek(fh_offset)

        self.offset = offset

    def tell(self, offset=None, unit=None):
        if offset is None:
            offset = self.offset

        if unit is None:
            return offset

        if isinstance(unit, str) and unit == 'time':
            return self.time(offset)

        return (offset * u.byte).to(
            unit, equivalencies=[(u.Unit(self.recordsize * u.byte),
                                  u.Unit(self.dtsample))])

    def time(self, offset=None):
        """Get time corresponding to the current (or given) offset"""
        if offset is None:
            offset = self.offset
        if offset % self.recordsize != 0:
            warnings.warn("Offset for which time is requested is not "
                          "integer multiple of record size.")
        return self.time0 + self.tell(offset, u.day)

    # ARO and GMRT (LOFAR_Pcombined overwrites this)
    def seek_record_read(self, offset, count):
        """Read count samples starting from offset (also in samples)"""
        self.seek(offset)
        return self.record_read(count)

    def record_read(self, count):
        return fromfile(self, self.dtype,
                        count).reshape(-1, self.nchan).squeeze()

    def nskip(self, date, time0=None):
        """
        Return the number of records needed to skip from start of
        file to iso timestamp 'date'.

        Optionally:
        time0 : use this start time instead of self.time0
                either a astropy.time.Time object or string in 'utc'
        """
        time0 = self.time0 if time0 is None else Time(time0, scale='utc')
        dt = Time(date, scale='utc') - time0
        nskip = int(round((dt / self.dtsample / self.setsize)
                          .to(u.dimensionless_unscaled)))
        return nskip

    def ntimebins(self, t0, t1):
        """
        determine the number of timebins between UTC start time 't0'
        and end time 't1'
        """
        t0 = Time(t0, scale='utc')
        t1 = Time(t1, scale='utc')
        nt = ((t1-t0) / self.dtsample /
              (self.setsize)).to(u.dimensionless_unscaled).value
        return np.ceil(nt).astype(int)

    # for use in context manager ("with <MultiFile> as fh:")
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
#           _____   ____      ___   
#     /\   |  __ \ / __ \     | |    
#    /  \  | |__) | |  | | ___| |__  
#   / /\ \ |  _  /| |  | |/ __| '_ \ 
#  / ____ \| | \ \| |__| | (__| | | |
# /_/    \_\_|  \_\\____/ \___|_| |_

class AROchdata(MultiFile):

    telescope = 'AROch'

    def __init__(self, raw_files, comm=None, blocksize=2**24,
                 refloat=True, samplerate=200.*u.MHz, dtype='cu4bit'):

        self.nchan = 512
        self.dtype=dtype
        self.fedge = (100*u.Hz).to(u.MHz)
        self.fedge_at_top = (200*u.Hz).to(u.MHz)
        self.blocksize = blocksize
        self.samplerate = samplerate
        self.recordsize = 1 # Not actually sure what this is
        self.dtsample = (1./samplerate).to(u.s)
        self.comm = MPI.COMM_SELF
        self.fh_raw = [MPI.File.Open(self.comm, '/scratch/p/pen/klages/ARO/pulsar_140613_b1957p20_v0/0000000.dat')]
        self.time0 = Time(0, format='unix')
        self.setsize = 10
        self.indices = np.array([0])
        self.fh_links = []
        self.offset = 0
        self.frequencies = np.linspace(self.fedge_at_top, self.fedge, self.nchan)

    def ntint(self, nchan):
        """
        number of samples in a frequency bin
        this is baseband data so need to know number of channels we're making
        """
        return self.setsize // (2*nchan)
    def ntimebins(self, t0, t1):
        """
        determine the number of timebins between UTC start time 't0'
        and end time 't1'
        """
        t0 = Time(t0, scale='utc')
        t1 = Time(t1, scale='utc')
        nt = ((t1-t0) / self.dtsample /
              (self.setsize)).to(u.dimensionless_unscaled).value
        return np.ceil(nt).astype(int)

    def __repr__(self):
        return ("<open lofar polarization pair {}>"
                .format(self.fh_raw))

#    def close(self):
#        for fh in self.fh_raw:
#            print(fh)
#            fh.close()

header_defaults['AROch'] = {
    'PRIMARY': {'TELESCOP':'LOFAR',
                'IBEAM':1, 'FD_POLN':'LIN',
                'OBS_MODE':'SEARCH',
                'ANT_X':0, 'ANT_Y':0, 'ANT_Z':0, 'NRCVR':1,
                'FD_HAND':1, 'FD_SANG':0, 'FD_XYPH':0,
                'BE_PHASE':0, 'BE_DCC':0, 'BE_DELAY':0,
                'TRK_MODE':'TRACK',
                'TCYCLE':0, 'OBSFREQ':300, 'OBSBW':100,
                'OBSNCHAN':20, 'CHAN_DM':0,
                'EQUINOX':2000.0, 'BMAJ':1, 'BMIN':1, 'BPA':0,
                'SCANLEN':1, 'FA_REQ':0,
                'CAL_FREQ':0, 'CAL_DCYC':0,
                'CAL_PHS':0, 'CAL_NPHS':0,
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


#    ____  ____   ___
#   /    ||    \ /   \
#  |  o  ||  D  )     |
#  |     ||    /|  O  |
#  |  _  ||    \|     |
#  |  |  ||  .  \     |
#  |__|__||__|\_|\___/
#
class AROdata(MultiFile):

    telescope = 'aro'

    def __init__(self, sequence_file, raw_voltage_files, blocksize=2**25,
                 dtype='4bit', samplerate=200.*u.MHz,
                 utc_offset=-4.*u.hr, comm=None):

        self.sequence_file = sequence_file
        seq, indices = np.loadtxt(sequence_file, np.int32, unpack=True)

        # seq starts counting at 2 for some reason
        seq -= 2
        self.indices = -np.ones(seq.max() + 1, dtype=np.int64)
        self.indices[seq] = indices

        # get start date from sequence filename
        arodate = re.search('\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',
                            os.path.basename(sequence_file))
        if arodate:
            isot = arodate.group()
            # convert time to UTC; dates given in EDT
            self.time0 = Time(isot, scale='utc') - utc_offset
            # ARO time is off by two 32MiB record or 128Misamples
            self.time0 -= (2.**27/samplerate).to(u.s)
        else:
            self.time0 = None

        self.fedge = 200. * u.MHz
        self.fedge_at_top = True
        self.samplerate = samplerate
        self.dtsample = (1./samplerate).to(u.s)

        super(AROdata, self).__init__(raw_voltage_files, blocksize, dtype, 1,
                                      comm=comm)
        # update headers for fun
        self['PRIMARY'].header['DATE-OBS'] = self.time0.iso
        self[0].header.update('TBIN', (1./samplerate).to('s').value),

    def ntint(self, nchan):
        """
        number of samples in a frequency bin
        this is baseband data so need to know number of channels we're making
        """
        return self.setsize // (2*nchan)

    def __repr__(self):
        return ("<open MultiFile raw_voltage_files {} "
                "using sequence file '{}' at offset {}>"
                .format(self.fh_raw, self.sequence_file, self.offset))


header_defaults['aro'] = {
    'PRIMARY': {'TELESCOP': 'Algonquin',
                'IBEAM': 1,
                'FD_POLN': 'LIN',
                'OBS_MODE': 'SEARCH',
                'ANT_X': 0, 'ANT_Y': 0, 'ANT_Z': 0,
                'NRCVR': 1,
                'FD_HAND': 1, 'FD_SANG': 0, 'FD_XYPH': 0,
                'BE_PHASE': 0, 'BE_DCC': 0, 'BE_DELAY': 0,
                'TRK_MODE': 'TRACK',
                'TCYCLE': 0, 'OBSFREQ': 300, 'OBSBW': 100,
                'OBSNCHAN': 20, 'CHAN_DM': 0,
                'EQUINOX': 2000.0,
                'BMAJ': 1, 'BMIN': 1, 'BPA': 0,
                'SCANLEN': 1, 'FA_REQ': 0,
                'CAL_FREQ': 0, 'CAL_DCYC': 0,
                'CAL_PHS': 0, 'CAL_NPHS': 0,
                'STT_IMJD': 54000, 'STT_SMJD': 0, 'STT_OFFS': 0},
    'SUBINT': {'INT_TYPE': 'TIME',
               'SCALE': 'FluxDen',
               'POL_TYPE': 'AABB',
               'NPOL': 1,
               'NBIN': 1,
               'NBIN_PRD': 1,
               'PHS_OFFS': 0,
               'NBITS': 1,
               'ZERO_OFF': 0,
               'SIGNINT': 0,
               'NSUBOFFS': 0,
               'NCHAN': 1,
               'CHAN_BW': 1,
               'DM': 0, 'RM': 0,
               'NCHNOFFS': 0,
               'NSBLK': 1}}


#   _       ___   _____   ____  ____
#  | |     /   \ |     | /    ||    \
#  | |    |     ||   __||  o  ||  D  )
#  | |___ |  O  ||  |_  |     ||    /
#  |     ||     ||   _] |  _  ||    \
#  |     ||     ||  |   |  |  ||  .  \
#  |_____| \___/ |__|   |__|__||__|\_|
#
class LOFARdata(MultiFile):

    telescope = 'lofar'

    def __init__(self, raw_files, comm=None, blocksize=2**20,
                 refloat=True):
        """
        Initialize a lofar observation, tracking/joining the two polarizations.
        We also parse the corresponding HDF5 files to initialize:
        nchan, samplerate, fwidth

        Parameters
        ----------
        raw_files : list
            full paths to *.raw files; the associated *.h5 files are assumed
            to be in the same directory.
        comm : MPI communicator
        blocksize : int or None
            Preferred number of bytes *per channel* and *per file* that are
            read in one go.
            Default: 2**20, or *20*2*npol=40|80 MB, or 2**18 samples (float)
        refloat : Bool
            Whether to convert compressed lofar data (stored as int1) back
            to float using the associated scale factors.  If False, simply
            use the integer data, ignoring the scale factors.  Default: True

        """
        self.nfh = len(raw_files)
        # sanity check: one or two polarisations
        assert self.nfh == 2 or self.nfh == 4
        self.npol = self.nfh // 2
        # read the HDF5 files and get useful data
        lofar_dtype = None
        for ifile, raw_file in enumerate(raw_files):
            try:
                h5 = HDF5File(raw_file.replace('.raw', '.h5'), 'r')
            except IOError:  # no h5 file, treat as all zeros
                raw_files[ifile] = '/dev/zero'
                continue

            # get attributes
            saps = sorted([i for i in h5.keys() if 'SUB_ARRAY_POINTING' in i])
            s0 = h5[saps[0]]
            beams = sorted([i for i in s0.keys() if 'BEAM' in i])
            b0 = s0[beams[0]]
            stokes = sorted([i for i in b0.keys()
                             if 'STOKES' in i and 'i2f' not in i])
            st0 = b0[stokes[0]]
            if lofar_dtype is None:  # only do this once; files should be same
                try:
                    lofar_dtype = st0.attrs['DATATYPE']
                    string_key_ok = True
                except KeyError:  # BGQ doesn't like strings in attributes
                    string_key_ok = False
                    lofar_dtype = ('int8' if any('i2f' in i for i in b0.keys())
                                   else 'float')

                if lofar_dtype == 'int8' and refloat:
                    # uncompress to float on-the-fly; ensure we look like
                    # two float per polarisation = complex
                    dtype = 'c8'  # no ">c8" since we produce these ourselves
                    # define lists to be filled below
                    self.compressed_block_size = [None]*len(raw_files)
                    self.scales = [None]*len(raw_files)
                else:
                    dtype = _lofar_dtypes[lofar_dtype]
                    # signal that no scaling is needed
                    self.scales = False

                if self.npol == 2:  # two polarisations -> two complex numbers
                    dtype = dtype + ',' + dtype

            if lofar_dtype == 'int8' and refloat:
                # size over which scaling was derived; this is *per file*
                diginfo = b0['{0}_i2f'.format(stokes[0])]
                self.compressed_block_size[ifile] = diginfo.attrs[
                    '{0}_recsize'.format(stokes[0])]
                # associated scales
                self.scales[ifile] = np.array(b0['{0}_i2f'.format(stokes[0])])

            if hasattr(self, 'frequencies'):  # no need to do more than once
                continue

            self.frequencies = (b0['COORDINATES']['COORDINATE_1']
                                .attrs['AXIS_VALUES_WORLD'] * u.Hz).to(u.MHz)
            fwidth = b0.attrs['SUBBAND_WIDTH']
            samplerate = b0.attrs['SAMPLING_RATE']
            try:
                self.time0 = Time(s0.attrs['EXPTIME_START_UTC']
                                  .replace('Z',''), scale='utc')
                self.fwidth = u.Quantity(
                    fwidth, b0.attrs['CHANNEL_WIDTH_UNIT']).to(u.MHz)
                self.samplerate = u.Quantity(
                    samplerate, b0.attrs['SAMPLING_RATE_UNIT']).to(u.MHz)
            except KeyError:
                time0 = Time(s0.attrs['EXPTIME_START_MJD'], format='mjd',
                             scale='utc', precision=3)
                # should start on whole minute
                assert time0.isot.endswith('00.000')
                # ensure we have the time to double-float rounding precision
                self.time0 = Time(time0.isot, scale='utc')
                self.fwidth = (fwidth * u.Hz).to(u.MHz)
                self.samplerate = (samplerate * u.Hz).to(u.MHz)

        self.fedge = self.frequencies[0]
        self.fedge_at_top = False
        self.dtsample = (1./self.fwidth).to(u.s)

        nchan = len(self.frequencies)  # = st0.attrs['NOF_SUBBANDS']
        super(LOFARdata, self).__init__(raw_files, blocksize*nchan,
                                        dtype, nchan, comm=comm)
        # update some of the hdu data
        self['PRIMARY'].header['DATE-OBS'] = self.time0.isot
        self[0].header.update('TBIN', (1./self.samplerate).to('s').value)

    def ntint(self, nchan):
        """
        number of samples in an integration
        Lofar data is already channelized so we assert
        nchan is the same
        """
        assert(nchan == self.nchan)
        return self.setsize

    def read(self, size):
        """
        read 'size' bytes of the LOFAR data; returns the two streams
        interleaved, such that one has complex numbers (either complex64
        or ci1, i.e., using two signed one-byte integers).

        For compressed (int8) lofar data, the data are decompressed if
        self.scale has been set (see refloat in initializer).
        """
        if not self.scales:
            z = np.empty(size, dtype='i1').reshape(self.nfh, -1,
                                                   self.itemsize // self.nfh)
            for fh, buf in zip(self.fh_raw, z):
                fh.Iread([buf, MPI.BYTE])
        else:  # rescaling compressed integer data
            # offset and size in the compressed file
            read_offset = self.offset // self.itemsize
            read_size = size // self.itemsize
            # create float output array (real, imag), possibly times two
            z = np.empty(read_size * self.nfh, dtype='f4').reshape(
                self.nfh, -1, self.nchan)
            # and a buffer to receive the compressed data
            buf = np.empty(read_size, dtype='i1').reshape(-1, self.nchan)
            for ifh, fh in enumerate(self.fh_raw):
                if self.scales[ifh] is None:  # non-existing file
                    z[ifh] = 0.
                    continue

                fh.Iread([buf, MPI.BYTE])
                # multiply with the scale appropriate for each part of the
                # buffer (for smaller reads, this will be a single value)
                for iscale in range(read_offset //
                                    self.compressed_block_size[ifh],
                                    (read_offset + read_size) //
                                    self.compressed_block_size[ifh] + 1):
                    # find range for which scale was determined
                    i0 = iscale * self.compressed_block_size[ifh]
                    i1 = (iscale + 1) * self.compressed_block_size[ifh]
                    # get part that overlaps with the buffer
                    i0 = max(0, (i0 - read_offset) // self.nchan)
                    i1 = min(buf.shape[0], (i1 - read_offset) // self.nchan)
                    # apply scales
                    z[ifh, i0:i1] = buf[i0:i1] * self.scales[ifh][iscale, :]
            z = z.reshape(self.nfh, -1, 1)

        self.offset += size
        # return what can be interpreted as a byte stream
        return z.transpose(1, 0, 2).ravel()

    def _seek(self, offset):
        """Offset by the given number of bytes"""
        if offset % self.recordsize != 0:
            raise ValueError("Cannot offset to non-integer number of records")
        # can have two or four combined files, i.e., divide by that number
        # to get the total number of corresponding bytes in each file
        # if compressed from float32 to int8: correct for additional factor 4.
        if self.scales is None:
            assert offset % self.nfh == 0
            for fh in self.fh_raw:
                fh.Seek(offset // self.nfh)
        else:
            assert offset % (self.nfh * 4) == 0
            for fh in self.fh_raw:
                fh.Seek(offset // self.nfh // 4)
        self.offset = offset

    def __repr__(self):
        return ("<open lofar polarization pair {}>"
                .format(self.fh_raw))


class LOFARdata_Pcombined(MultiFile):
    """
    convenience class to combine multiple subbands, making them act
    as a single file.
    """
    telescope = 'lofar'

    def __init__(self, raw_files_list, comm=None, **kwargs):
        """
        A list of tuples, to be 'concatenated' together
        (as returned by observations.obsdata[telescope].file_list(obskey) )
        """
        self.per_channel_blocksize = kwargs.pop('blocksize', 2**18)
        super(LOFARdata_Pcombined, self).__init__(raw_files_list, comm=comm)
        self.fbottom = self.frequencies[0]
        self.fedge = self.frequencies[0]
        self.fedge_at_top = False
        # update some of the hdu data
        self['PRIMARY'].header['DATE-OBS'] = self.time0.isot
        self['PRIMARY'].header.update('TBIN',
                                      (1./self.samplerate).to('s').value)
        self['PRIMARY'].header.update('NCHAN', self.nchan)

    def open(self, raw_files_list):
        self.fh_raw = [LOFARdata(raw_files, comm=self.comm,
                                 blocksize=getattr(self,
                                                   'per_channel_blocksize',
                                                   2**18))
                       for raw_files in raw_files_list]
        self.fh_links = []
        # make sure basic properties of the files are the same
        for prop in ['dtype', 'itemsize', 'time0', 'samplerate',
                     'fwidth', 'dtsample', 'npol']:
            props = [fh.__dict__[prop] for fh in self.fh_raw]
            if prop == 'time0':
                props = [p.isot for p in props]
            assert len(set(props)) == 1
            self.__setattr__(prop, self.fh_raw[0].__dict__[prop])

        self.blocksize = sum([fh.blocksize for fh in self.fh_raw])
        self.recordsize = sum([fh.recordsize for fh in self.fh_raw])
        self.frequencies = u.Quantity(np.hstack([fh.frequencies.value
                                                 for fh in self.fh_raw]),
                                      self.fh_raw[0].frequencies.unit)
        self.nchan = len(self.frequencies)
        self.offset = 0

    def close(self):
        for fh in self.fh_raw:
            fh.close()

    def ntint(self, *args):
        """
        number of samples in an integration
        Lofar data is already channelized so we assert
        nchan is the same
        """
        # LOFAR is already channelized, we accept args for generalization
        return self.setsize

    def record_read(self, size):
        assert size % self.recordsize == 0
        nrecords = size // self.recordsize
        raw = np.hstack([fh.record_read(nrecords * fh.recordsize)
                         for fh in self.fh_raw])
        self.offset += size
        return raw

    def _seek(self, offset):
        assert offset % self.recordsize == 0
        nrecoff = offset // self.recordsize
        for fh in self.fh_raw:
            fh._seek(nrecoff * fh.recordsize)
        self.offset = offset

    def seek_record_read(self, offset, size):
        """
        LOFARdata_Pcombined class opens a lot of filehandles.
        This routine tries to minimize file seeks
        """
        assert offset % self.recordsize == 0 and size % self.recordsize == 0
        nrecoff = offset // self.recordsize
        nrecords = size // self.recordsize
        raw = np.hstack([fh.seek_record_read(nrecoff * fh.recordsize,
                                             nrecords * fh.recordsize)
                         for fh in self.fh_raw])
        self.offset = offset + size
        return raw

    def __repr__(self):
        return ("<open (concatenated) lofar sets from {} to {}>"
                .format(self.fh_raw[0].fh_raw[0], self.fh_raw[-1].fh_raw[-1]))

# LOFAR defaults for psrfits HDUs
header_defaults['lofar'] = {
    'PRIMARY': {'TELESCOP':'LOFAR',
                'IBEAM':1, 'FD_POLN':'LIN',
                'OBS_MODE':'SEARCH',
                'ANT_X':0, 'ANT_Y':0, 'ANT_Z':0, 'NRCVR':1,
                'FD_HAND':1, 'FD_SANG':0, 'FD_XYPH':0,
                'BE_PHASE':0, 'BE_DCC':0, 'BE_DELAY':0,
                'TRK_MODE':'TRACK',
                'TCYCLE':0, 'OBSFREQ':300, 'OBSBW':100,
                'OBSNCHAN':20, 'CHAN_DM':0,
                'EQUINOX':2000.0, 'BMAJ':1, 'BMIN':1, 'BPA':0,
                'SCANLEN':1, 'FA_REQ':0,
                'CAL_FREQ':0, 'CAL_DCYC':0,
                'CAL_PHS':0, 'CAL_NPHS':0,
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


#    ____  ___ ___  ____  ______
#   /    ||   |   ||    \|      |
#  |   __|| _   _ ||  D  )      |
#  |  |  ||  \_/  ||    /|_|  |_|
#  |  |_ ||   |   ||    \  |  |
#  |     ||   |   ||  .  \ |  |
#  |___,_||___|___||__|\_| |__|
#
class GMRTdata(MultiFile):

    telescope = 'gmrt'

    def __init__(self, timestamp_file, raw_files, blocksize, nchan,
                 samplerate, fedge, fedge_at_top, dtype='ci1',
                 utc_offset=5.5*u.hr, comm=None):
        """GMRT phased data stored in blocks holding 0.25 s worth of data,
        separated over two streams (each with 0.125s).  For 16MHz BW, each
        block is 4 MiB with 2Mi complex samples split in 256 or 512 channels;
        complex samples consist of two signed ints (custom 'ci1' dtype).
        """
        self.samplerate = samplerate
        self.fedge = fedge
        self.fedge_at_top = fedge_at_top
        f = fftshift(fftfreq(nchan, (2./samplerate).to(u.s).value)) * u.Hz
        if fedge_at_top:
            self.frequencies = fedge - (f-f[0])
        else:
            self.frequencies = fedge + (f-f[0])

        self.timestamp_file = timestamp_file

        if comm.rank == 0:
            print("In GMRTdata, just before read_timestamp_file({0}, {1})"
                  .format(timestamp_file, utc_offset))
        self.indices, self.timestamps, self.gsb_start = read_timestamp_file(
            timestamp_file, utc_offset)
        self.time0 = self.timestamps[0]
        # GMRT time is off by one 32MB record ---- remove for now
        # self.time0 -= (2.**25/samplerate).to(u.s)

        self.dtsample = (nchan * 2 / samplerate).to(u.s)
        if comm.rank == 0:
            print("In GMRTdata, calling super")
        super(GMRTdata, self).__init__(raw_files, blocksize, dtype, nchan,
                                       comm=comm)

    def ntint(self, nchan):
        return self.setsize

    def __repr__(self):
        return ("<open two raw_voltage_files {} "
                "using timestamp file '{}' at index {} (time {})>"
                .format(self.fh_raw, self.timestamp_file, self.offset,
                        self.time().iso))

# GMRT defaults for psrfits HDUs
# Note: these are largely made-up at this point
header_defaults['gmrt'] = {
    'PRIMARY': {'TELESCOP':'GMRT',
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
    assert '00.000' in gsb_start.isot

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


#   __ __  ______  ____  _     _____
#  |  |  ||      ||    || |   / ___/
#  |  |  ||      | |  | | |  (   \_
#  |  |  ||_|  |_| |  | | |___\__  |
#  |  :  |  |  |   |  | |     /  \ |
#  |     |  |  |   |  | |     \    |
#   \__,_|  |__|  |____||_____|\___|
#
def good_name(f):
    """
    MPI.File.Open can't process files with colons.
    This routine checks for such cases and creates a well-named
    link to the file.

    Returns (good_name, islink)
    """
    if f is None:
        return f

    fl = f
    newlink = False
    if ':' in f:
        #fl = tempfile.mktemp(prefix=os.path.basename(f)
        # .replace(':','_'), dir='/tmp')
        fl = os.path.join('/tmp', os.path.dirname(f).replace('/','_') +
                          '__' + os.path.basename(f).replace(':','_'))
        if not os.path.exists(fl):
            try:
                os.symlink(f, fl)
            except(OSError):
                pass
            newlink = True
    return fl, newlink
