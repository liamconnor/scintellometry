from __future__ import division, print_function

import argparse
import numpy as np
from astropy.time import Time

from scintellometry.folding.fold import normalize_counts
from scintellometry.folding.pmap import pmap
from scintellometry.folding import correlate
from scintellometry.io import AROdata, LOFARdata_Pcombined, GMRTdata

from observations import obsdata

from mpi4py import MPI

MAX_RMS = 4.

def rfi_filter_raw(raw, nchan):
    # note this should accomodate all data (including's lofar raw = complex)
    rawbins = raw.reshape(-1, 2**11*nchan)  # note, this is view!
    std = rawbins.std(-1, keepdims=True)
    ok = std < MAX_RMS
    rawbins *= ok
    return raw, ok

def main_correlate(tel1, date1, tel2, date2, nchan, tstart, tend, dedisperse,
                   do_foldspec, ntbin, ngate,
                   do_waterfall, ntw_min,
                   save_xcorr, verbose=0):
    """
    the main cross-correlation routine
    """
    comm = MPI.COMM_WORLD
    if comm.size > 1 and save_xcorr:
        if comm.rank == 0:
	    print("Warning, h5py mpio is sometimes slow. Consider disabling save_xcorr")
	# save_xcorr = False
    # observing parameters
    t0 = Time(tstart, scale='utc')
    t1 = Time(tend, scale='utc')

    Obs = obsdata()
    obskey1 = Obs[tel1].nearest_observation(date1)
    obskey2 = Obs[tel2].nearest_observation(date2)
    psr1 = Obs[tel1][obskey1]['src']
    psr2 = Obs[tel2][obskey2]['src']
    files1 = Obs[tel1].file_list(obskey1)
    files2 = Obs[tel2].file_list(obskey2)

    assert psr1 == psr2
    if comm.rank == 0:
        print("forming visibilities from (telescope, observation_key) = \n"
              "\t ({0}, {1}) and ({2}, {3}), source {4}".format(tel1, obskey1, tel2, obskey2, psr1))
    dm = Obs['psrs'][psr1]['dm']
    with LOFARdata_Pcombined(*files1, comm=comm) as fh1,\
            GMRTdata(*files2, comm=comm) as fh2:
        phasepol1 = Obs['lofar'][obskey1].get_phasepol(fh1.time0, rphase=None)
        phasepol2 = Obs['gmrt'][obskey2].get_phasepol(fh2.time0, rphase=None)
        nt = min(fh1.ntimebins(t0, t1), fh2.ntimebins(t0, t1))
        # out = (foldspec, icount, waterfall)
        out = correlate.correlate(fh1, fh2, dm=dm, nchan=nchan, ngate=ngate,
                                  ntbin=ntbin, nt=nt, ntw=ntw_min,
                                  t0=t0, t1=t1, dedisperse=dedisperse,
                                  phasepol=(phasepol1, phasepol2),
                                  do_waterfall=do_waterfall,
                                  do_foldspec=do_foldspec,
                                  save_xcorr=save_xcorr,
                                  comm=comm)
        myfoldspec = out[0]
        myicount = out[1]
        mywaterfall = out[2]

    savepref = "{0}{1}_{2}chan{3}ntbin".format(tel1[0], tel2[0], nchan, ntbin)
    dt = t1 - t0
    if do_waterfall:
        waterfall = np.zeros_like(mywaterfall)
        comm.Reduce(mywaterfall, waterfall, op=MPI.SUM, root=0)
        if comm.rank == 0:
            # waterfall = normalize_counts(waterfall)
            np.save("{0}waterfall_{1}+{2:08}sec.npy"
                    .format(savepref, t0, dt.sec), waterfall)

    if do_foldspec:
        foldspec = np.zeros_like(myfoldspec)
        icount = np.zeros_like(myicount)
        comm.Reduce(myfoldspec, foldspec, op=MPI.SUM, root=0)
        comm.Reduce(myicount, icount, op=MPI.SUM, root=0)
        if comm.rank == 0:
            fname = ("{0}foldspec_{1}+{2:08}sec.npy")
            iname = ("{0}icount_{1}+{2:08}sec.npy")
            np.save(fname.format(savepref, t0, dt.sec), foldspec)
            np.save(iname.format(savepref, t0, dt.sec), icount)

            # get normalized flux in each bin (where any were added)
            f2 = normalize_counts(foldspec, icount)
            foldspec1 = f2.sum(axis=2)
            fluxes = foldspec1.sum(axis=0)
            foldspec3 = f2.sum(axis=0)

            with open('{0}flux_{1}+{2:08}sec.dat'
                      .format(savepref, t0, dt.sec), 'w') as f:
                for i, flux in enumerate(fluxes):
                    f.write('{0:12d} {1:12.9g}\n'.format(i + 1, flux))

    plots = True
    if plots and comm.rank == 0:
        if do_waterfall:
            w = waterfall.copy()
            try:
                pmap('{0}waterfall_{1}+{2:08}sec.pgm'
                     .format(savepref, t0, dt.sec), w, 1, verbose=True)
            except:
                pass
        if do_foldspec:
            pmap('{0}folded_{1}+{2:08}sec.pgm'
                 .format(savepref, t0, dt.sec), foldspec1, 0, verbose)
            # TODO: Note, I (aaron) don't think this works for LOFAR data
            # since nchan=20, but we concatenate several subband files
            # together, so f2.nchan = N_concat * nchan
            # It should work for my "new" LOFAR_Pconcate file class
            pmap('{0}foldedbin_{1}+{2:08}sec.pgm'
                 .format(savepref, t0, dt.sec),
                 f2.transpose(0, 2, 1).reshape(nchan, -1), 1, verbose)
            pmap('{0}folded3_{1}+{2:08}sec.pgm'
                 .format(savepref, t0, dt.sec), foldspec3, 0, verbose)


def CL_parser():
    parser = argparse.ArgumentParser(
        prog='corr.py',
        description='Crosscorrelate two datasets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--save_xcorr', type=bool, default=True,
        help='Save the x-correlations to an hdf5 file')

    t_parser = parser.add_argument_group(
        "Telescope related parameters \n"
        "Specify the two telescopes and the observation date")

    t_parser.add_argument(
        '--tel1', type=str, default='lofar',
        help='The first telescope.'
        )
    t_parser.add_argument(
        '--date1', type=str, default='2013-07-25',
        help='The date of observation. Should be an observation key in'
             'Observations.obsdata[tel1]'
        )
    t_parser.add_argument(
        '--tel2', type=str, default='gmrt',
        help='The second telescope.'
        )
    t_parser.add_argument(
        '--date2', type=str, default='2013-07-25',
        help='The date of observation. Should be an observation key in'
             'Observations.obsdata[tel2]'
        )

    d_parser = parser.add_argument_group(
        "Data processing parameter")
    d_parser.add_argument('-t0', '--starttime', type=str,
                          default='2013-07-25T22:25:01.0',
                          help='iso-t timestamp within observation run to '
                          'start processing.')
    d_parser.add_argument('-t1', '--endtime', type=str,
                          default='2013-07-25T22:25:12.0',
                          help='iso-t timestamp within observation run to '
                          'end processing.')
    d_parser.add_argument(
        '-nc', '--nchan', type=int, default=512,
        help="Number of channels in folded spectrum.")
    d_parser.add_argument(
        '--dedisperse', type=str, default='incoherent',
        help="One of ['None', 'coherent', 'by-channel', 'incoherent'].\n"
             "Note: None really does nothing.")

    f_parser = parser.add_argument_group("folding related parameters")
    f_parser.add_argument(
        '-f','--do_foldspec', type=bool, default=True,
        help="Produce a folded spectrum")
    f_parser.add_argument(
        '-ng', '--ngate', type=int, default=512,
        help="number of bins over the pulsar period.")
    f_parser.add_argument(
        '-nb', '--ntbin', type=int, default=12,
        help="number of time bins the time series is split into for folding.")

    w_parser = parser.add_argument_group("Waterfall related parameters")
    w_parser.add_argument(
        '-w', '--do_waterfall', type=bool, default=True,
        help="Produce a waterfall plot.")
    w_parser.add_argument(
        '-nwm', '--ntw_min', type=int, default=10200,
        help="number of samples to combine for waterfall")

    parser.add_argument('-v', '--verbose', action='append_const', const=1)
    return parser.parse_args()

if __name__ == '__main__':
    args = CL_parser()
    args.verbose = 0 if args.verbose is None else sum(args.verbose)

    main_correlate(args.tel1, args.date1, args.tel2, args.date2, args.nchan,
                   args.starttime, args.endtime, args.dedisperse,
                   args.do_foldspec, args.ntbin, args.ngate,
                   args.do_waterfall, args.ntw_min,
                   args.save_xcorr,
                   args.verbose)
