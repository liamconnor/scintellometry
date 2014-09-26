#    ____  ___ ___  ____  ______
#   /    ||   |   ||    \|      |
#  |   __|| _   _ ||  D  )      |
#  |  |  ||  \_/  ||    /|_|  |_|
#  |  |_ ||   |   ||    \  |  |
#  |     ||   |   ||  .  \ |  |
#  |___,_||___|___||__|\_| |__|
#
from __future__ import division

import numpy as np
from scipy.fftpack import fftfreq, fftshift
from astropy.time import Time, TimeDelta
import astropy.units as u

from . import MultiFile, header_defaults


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
