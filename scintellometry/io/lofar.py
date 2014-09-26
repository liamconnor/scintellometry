#   _       ___   _____   ____  ____
#  | |     /   \ |     | /    ||    \
#  | |    |     ||   __||  o  ||  D  )
#  | |___ |  O  ||  |_  |     ||    /
#  |     ||     ||   _] |  _  ||    \
#  |     ||     ||  |   |  |  ||  .  \
#  |_____| \___/ |__|   |__|__||__|\_|
#
from __future__ import division

import numpy as np

from h5py import File as HDF5File
try:
    from mpi4py import MPI
except ImportError:
    pass

import astropy.units as u
from astropy.time import Time

from . import MultiFile, header_defaults

# hdf5 dtype conversion
_lofar_dtypes = {'float': '>c8', 'int8': 'ci1'}


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
