#    ____  ____   ___
#   /    ||    \ /   \
#  |  o  ||  D  )     |
#  |     ||    /|  O  |
#  |  _  ||    \|     |
#  |  |  ||  .  \     |
#  |__|__||__|\_|\___/
#
from __future__ import division

import os
import re
import numpy as np
import astropy.units as u
from astropy.time import Time

from . import MultiFile, header_defaults


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
