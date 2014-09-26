from __future__ import division, print_function

import numpy as np
from numpy.polynomial import Polynomial
import astropy.units as u
from astropy.time import Time, TimeDelta

from scintellometry.folding.fold_aro2 import fold
from scintellometry.folding.pmap import pmap
from scintellometry.folding.multifile import multifile

from mpi4py import MPI

MAX_RMS = 4.2


def rfi_filter_raw(raw):
    rawbins = raw.reshape(-1, 1048576)  # note, this is view!
    rawbins *= (rawbins.std(-1, keepdims=True) < MAX_RMS)
    return raw


def rfi_filter_power(power):
    return np.clip(power, 0., MAX_RMS**2 * power.shape[-1])

def normalize_counts(q, count=None):
    """ normalize routines for waterfall and foldspec data """
    if count is None:
        nonzero = np.isclose(q, np.zeros_like(q)) # == 0.
        qn = q
    else:
        nonzero = count > 0
        qn = np.where(nonzero, q/count, 0.)
    qn -= np.where(nonzero,
                   np.sum(qn, 1, keepdims=True) /
                   np.sum(nonzero, 1, keepdims=True), 0.)
    return qn

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    # pulsar parameters
    psr = 'B1919+21'
    # psr = 'B2016+28'
    # psr = 'B0329+54'
    # psr = 'B0823+26'
    # psr = 'J1810+1744'
    date_dict = {'B0823+26': '2013-07-24T15:06:16',
                 'B1919+21': '2013-07-25T18:14:20',
                 # 'J1810+1744': '2013-07-27T16:55:17'}
                 'J1810+1744': '2013-07-26T16:30:37'}

    dm_dict = {'B0329+54': 26.833 * u.pc / u.cm**3,
               'B0823+26': 19.454 * u.pc / u.cm**3,
               'J1810+1744': 39.659298 * u.pc / u.cm**3,
               'B1919+21': 12.455 * u.pc / u.cm**3,
               'B1957+20': 29.11680*1.001 * u.pc / u.cm**3,
               'B2016+28': 14.172 * u.pc / u.cm**3,
               'noise': 0. * u.pc / u.cm**3}
    phasepol_dict = {'2013-07-27T16:55:17':  # J1810+1744
                     Polynomial([-1252679.1986725251,
                                 601.39629721056895,
                                 -6.6664639926379228e-06,
                                 -3.005404797321569e-10,
                                 1.3404520057431192e-13,
                                 3.5632030706667189e-18,
                                 -1.0874017282180807e-21,
                                 -1.8089896985287676e-26,
                                 4.803545433801123e-30,
                                 1.4787240038933893e-35,
                                 -1.1792841185454315e-38,
                                 2.6298912108944255e-43]),
                     '2013-07-26T16:30:37':  # J1810+1744
                     Polynomial([-4307671.0917832768,
                                 601.37394786958396,
                                 -5.7640759068738662e-06,
                                 6.2468664899676703e-10,
                                 1.1429714466878334e-13,
                                 -7.5191478615746773e-18,
                                 -7.4658136316940933e-22,
                                 -1.5804755712584567e-26,
                                 1.3208008604369681e-29,
                                 -9.5396362858203809e-34,
                                 2.7444696554344206e-38,
                                 -2.9004096379523196e-43]),
                     # 'B1919+21':  # B1919+21/ 2013-07-25T18:14:20
                     # Polynomial([0.5, 0.7477741603725])}
                     'B1919+21': 'data/polycob1919+21_aro.dat'}

    dt = date_dict[psr]
    dm = dm_dict[psr]
    phasepol = phasepol_dict[dt] if dt in phasepol_dict else phasepol_dict[psr]

    # fndir1 = '/mnt/b/algonquin/'
    fnbase = '/mnt/data-pen1/pen/njones/VLBI_july212013'
    disk_no = [2, 1, 3]
    node = 9
    seq_file = ('{0}/hdd{1}_node{2}/sequence.{3}.3.dat'
                .format(fnbase, disk_no[0], node, dt))
    raw_files = ['{0}/hdd{1}_node{2}/raw_voltage.{3}.{4}.dat'
                 .format(fnbase, disk_no[i], node, dt, i) for i in range(3)]

    #***TODO: apply LOFAR polyphase instead
    nchan = 512  # frequency channels to make
    ngate = 512  # number of bins over the pulsar period
    recsize = 2**25  # 32MB sets
    ntint = recsize*2//(2 * nchan)  # number of samples after FFT
    # total_size = sum(os.path.getsize(fil) for fil in raw_files)
    # nt = total_size // recsize
    nt = 1800  # each 32MB set has 2*2**25/2e8=0.33554432 s, so 180 -> ~1 min

    nskip = 0

    ntbin = 12  # number of bins the time series is split into for folding
    ntw = min(10200, nt*ntint)  # number of samples to combine for waterfall

    samplerate = 200 * u.MHz

    fedge = 200. * u.MHz
    fedge_at_top = True

    fref = 150. * u.MHz  # ref. freq. for dispersion measure

    # convert time to UTC; dates given in EDT
    time0 = Time(date_dict[psr], scale='utc') + 4*u.hr

    verbose = 'very'
    do_waterfall = True
    do_foldspec = True
    dedisperse = 'incoherent'

    with multifile(seq_file, raw_files, comm=comm) as fh1:

        if not isinstance(phasepol, Polynomial):
            from astropy.utils.data import get_pkg_data_filename
            from pulsar.predictor import Polyco

            polyco_file = get_pkg_data_filename(phasepol)
            polyco = Polyco(polyco_file)
            phasepol = polyco.phasepol(time0, rphase='fraction', t0=time0,
                                       time_unit=u.second, convert=True)
            nskip = int(round(
                ((Time('2013-07-25T22:15:00', scale='utc') - time0) /
                 (recsize * 2 / samplerate)).to(u.dimensionless_unscaled)))
            if verbose:
                print("Using start time {0} and phase polynomial {1}"
                      .format(time0, phasepol))
                print("Skipping {0} records and folding {1} records to cover "
                      "time span {2} to {3}"
                      .format(nskip, nt,
                              time0 + nskip * recsize * 2 / samplerate,
                              time0 + (nskip+nt) * recsize * 2 / samplerate))

        myfoldspec, myicount, mywaterfall = fold(
            fh1, '4bit', samplerate, fedge, fedge_at_top, nchan,
            nt, ntint, nskip, ngate, ntbin, ntw, dm, fref, phasepol,
            dedisperse=dedisperse, do_waterfall=do_waterfall,
            do_foldspec=do_foldspec, verbose=verbose, progress_interval=1,
            rfi_filter_raw=rfi_filter_raw,
            rfi_filter_power=None, comm=comm)  # rfi_filter_power)

    if do_waterfall:
        waterfall = np.zeros_like(mywaterfall)
        comm.Reduce(mywaterfall, waterfall, op=MPI.SUM, root=0)

        if comm.rank == 0:
            waterfall = normalize_counts(waterfall)
        np.save("aro{0}waterfall_{1}.npy".format(psr, node), waterfall)

    if do_foldspec:
        foldspec = np.zeros_like(myfoldspec)
        icount = np.zeros_like(myicount)
        comm.Reduce(myfoldspec, foldspec, op=MPI.SUM, root=0)
        comm.Reduce(myicount, icount, op=MPI.SUM, root=0)
        if comm.rank == 0:
            np.save("aro{0}foldspec_{1}".format(psr, node), foldspec)
            np.save("aro{0}icount_{1}".format(psr, node), icount)
            # get normalised flux in each bin (where any were added)
            f2 = normalize_counts(foldspec, icount)
            foldspec1 = f2.sum(axis=2)
            fluxes = foldspec1.sum(axis=0)
            foldspec3 = f2.sum(axis=0)

            with open('aro{0}flux_{1}.dat'.format(psr, node), 'w') as f:
                for i, flux in enumerate(fluxes):
                    f.write('{0:12d} {1:12.9g}\n'.format(i+1, flux))

    plots = True
    if plots and comm.rank == 0:
        if do_waterfall:
            w = waterfall.copy()
            pmap('aro{0}waterfall_{1}.pgm'.format(psr, node),
                 w, 1, verbose=True)
        if do_foldspec:
            pmap('aro{0}folded_{1}.pgm'.format(psr, node),
                 foldspec1, 0, verbose)
            pmap('aro{0}foldedbin_{1}.pgm'.format(psr, node),
                 f2.transpose(0,2,1).reshape(nchan,-1), 1, verbose)
            pmap('aro{0}folded3_{1}.pgm'.format(psr, node),
                 foldspec3, 0, verbose)
