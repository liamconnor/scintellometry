from __future__ import division

from __future__ import division, print_function

import numpy as np
from numpy.polynomial import Polynomial
import astropy.units as u
from astropy.time import Time

from scintellometry.folding.twofile import twofile
from scintellometry.folding.fold_gmrt_phased import fold
from scintellometry.folding.pmap import pmap

from mpi4py import MPI

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
    date = '26jul2013'
    # psr = 'B2016+28'
    # psr = 'B0329+54'
    # psr = 'B0823+26'
    # psr = 'J1810+1744'
    # psr = 'B2111+46'
    psrname = ['0809+74','1508+55','1957+20','1919+21']
    dm_dict = {'B0329+54': 26.833 * u.pc / u.cm**3,
               'B0823+26': 19.454 * u.pc / u.cm**3,
               'J1810+1744': 39.659298 * u.pc / u.cm**3,
               'B1919+21': 12.455 * u.pc / u.cm**3,
               'B1957+20': 29.11680*1.001 * u.pc / u.cm**3,
               'B2016+28': 14.172 * u.pc / u.cm**3,
               'B2111+46': 141.26 * u.pc / u.cm**3,
               'noise': 0. * u.pc / u.cm**3}
    phasepol_dict = {'B0329+54': Polynomial([0., 1.399541538720]),
                     # 'B1919+21': Polynomial([0.5, 0.7477741603725]),
                     'B1919+21': 'data/polycob1919+21_gmrt.dat',
                     'B2016+28': Polynomial([0., 1.7922641135652]),
                     'B2111+46': 'data/polycob2111+46_gmrt.dat'}

    dm = dm_dict[psr]
    phasepol = phasepol_dict[psr]

    file_template = ('/mnt/data-pen1/bahmanya/tape_6/temp1/phased_array/'
                     'node{0:2d}/' + date + '/' + psr.lower() + '.raw')
    timestamp_file = file_template.format(33) + '.timestamp'
    file_template += '.Pol-{1:1s}{2:1d}.dat'
    node = 33
    pol = 'R'
    file1 = file_template.format(node, pol, 1)
    file2 = file_template.format(node, pol, 2)

    # frequency samples in a block; every sample is two bytes: real, imag
    recsize = 2**22
    nchan = 512
    ngate = 512  # number of bins over the pulsar period
    ntint = recsize // (2 * nchan)  # no. of bytes -> real,imag -> channels
    nt = 500  # 5 min/16.667 MHz=2400*(recsize / 2)
    ntbin = 5  # number of bins the time series is split into for folding
    ntw = min(170, nt*ntint)  # number of samples to combine for waterfall
    # 170 /(100.*u.MHz/6.) * 512 = 0.0052224 s = 256 bins/pulse

    samplerate = 100.*u.MHz / 3.

    fedge = 156. * u.MHz
    fedge_at_top = True

    fref = 150. * u.MHz  # ref. freq. for dispersion measure

    verbose = True
    do_waterfall = True
    do_foldspec = True
    dedisperse = 'incoherent'

    with twofile(timestamp_file, [file1, file2], comm=comm) as fh1:
        if verbose:
            print("Start time = {}; gsb start = {}"
                  .format(fh1.timestamps[0], fh1.gsb_start))

        if not isinstance(phasepol, Polynomial):
            from astropy.utils.data import get_pkg_data_filename
            from pulsar.predictor import Polyco

            polyco_file = get_pkg_data_filename(phasepol)
            polyco = Polyco(polyco_file)
            time0 = fh1.timestamps[0]
            # GMRT time is off by 1 second
            time0 -= (2.**24/(100*u.MHz/6.)).to(u.s)
            # time0 -= 1. * u.s
            phasepol = polyco.phasepol(time0, rphase='fraction', t0=time0,
                                       time_unit=u.second, convert=True)
            nskip = int(round(
                ((Time('2013-07-25T22:15:00', scale='utc') - time0) /
                 (recsize / samplerate)).to(u.dimensionless_unscaled)))
            if verbose:
                print("Using start time {0} and phase polynomial {1}"
                      .format(time0, phasepol))
                print("Skipping {0} records and folding {1} records to cover "
                      "time span {2} to {3}"
                      .format(nskip, nt,
                              time0 + nskip * recsize / samplerate,
                              time0 + (nskip+nt) * recsize / samplerate))

        else:
            nskip = 0
        myfoldspec, myicount, mywaterfall = fold(fh1, np.int8, samplerate,
                                    fedge, fedge_at_top, nchan,
                                    nt, ntint, nskip,
                                    ngate, ntbin, ntw, dm, fref, phasepol,
                                    dedisperse=dedisperse,
                                    do_waterfall=do_waterfall,
                                    do_foldspec=do_foldspec,
                                    verbose=verbose, progress_interval=1, comm=comm)

    if do_waterfall:
        waterfall = np.zeros_like(mywaterfall)
        comm.Reduce(mywaterfall, waterfall, op=MPI.SUM, root=0)
        if comm.rank == 0:
            waterfall = normalize_counts(waterfall)
            np.save("gmrt{0}waterfall_{1}.npy".format(psr, pol), waterfall)

    if do_foldspec:
        foldspec = np.zeros_like(myfoldspec)
        comm.Reduce(myfoldspec, foldspec, op=MPI.SUM, root=0)
        icount = np.zeros_like(myicount)
        comm.Reduce(myicount, icount, op=MPI.SUM, root=0)
        if comm.rank == 0:
            foldspec2 = normalize_counts(foldspec, icount)
            np.save("gmrt{0}foldspec2_{1}".format(psr, pol), foldspec2)
            f2 = foldspec2.copy()
            foldspec1 = f2.sum(axis=2)
            fluxes = foldspec1.sum(axis=0)
            foldspec3 = f2.sum(axis=0)

            with open('gmrt{0}flux_{1}.dat'.format(psr, pol), 'w') as f:
                for i, flux in enumerate(fluxes):
                    f.write('{0:12d} {1:12.9g}\n'.format(i+1, flux))

    plots = True
    if plots and comm.rank == 0:
        if do_waterfall:
            w = waterfall.copy()
            pmap('gmrt{0}waterfall_{1}.pgm'.format(psr, pol),
                 w, 1, verbose=True)
        if do_foldspec:
            pmap('gmrt{0}folded_{1}.pgm'.format(psr, pol),
                 foldspec1, 0, verbose)
            pmap('gmrt{0}foldedbin_{1}.pgm'.format(psr, pol),
                 f2.transpose(0,2,1).reshape(nchan,-1), 1, verbose)
            pmap('gmrt{0}folded3_{1}.pgm'.format(psr, pol),
                 foldspec3, 0, verbose)

