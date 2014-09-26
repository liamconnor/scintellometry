from __future__ import division, print_function

import numpy as np
from numpy.polynomial import Polynomial
import astropy.units as u

from scintellometry.folding.fold_lofar import fold
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
    # psr = 'B2016+28'
    # psr = 'B1957+20'
    # psr = 'B0329+54'
    # psr = 'J1810+1744'
    # psr = 'B2111+46'  # not on disk yet

    dm_dict = {'B0329+54': 26.833 * u.pc / u.cm**3,
               'J1810+1744': 39.659298 * u.pc / u.cm**3,
               'B1919+21': 12.455 * u.pc / u.cm**3,
               'B2016+28': 14.172 * u.pc / u.cm**3,
               'B2111+46': 141.26 * u.pc / u.cm**3}
    phasepol_dict = {'B0329+54': Polynomial([0., 1.399541538720]),
                     'J1810+1744': Polynomial([5123935.3179235281,
                                               601.3858344512422,
                                               -6.8670334150772988e-06,
                                               1.6851467436247837e-10,
                                               1.4924190280848832e-13,
                                               3.681791676784501e-18,
                                               3.4408214917205562e-22,
                                               2.3962705401172674e-25,
                                               2.7467843239802234e-29,
                                               1.3396130966170961e-33,
                                               3.0840132342990634e-38,
                                               2.7633775352567796e-43]),
                     # 'B1919+21': Polynomial([0.5, 0.7477741603725]),
                     'B1919+21': 'data/polycob1919+21_lofar.dat',
                     'B2016+28': Polynomial([0., 1.7922641135652]),
                     'B2111+46': 'data/polycob2111+46_lofar.dat'}

    dm = dm_dict[psr]
    phasepol = phasepol_dict[psr]

    file_dict = {
        'J1810+1744': '/mnt/data-pen1/jhessels/J1810+1744/L166111/L166111',
        'B1919+21': '/mnt/data-pen1/jhessels/B1919+21/L166109/L166109',
        'B2111+46': '/mnt/data-pen1/jhessels/B2111+46/DOESNOTEXIST'}

    file_fmt = file_dict[psr] + '_SAP000_B000_S{S:1d}_P{P:03d}_bf.raw'
    # frequency channels that file contains
    nchan = 20
    # TODO: adjust so we start at common start time
    # to match GMRT, need to skip 68.16556 s or 13313586 samples
    recsize = 4 * nchan * 2**16  # ~5 MB
    ntint = recsize // (4 * nchan)  # ntint/fwidth = 0.33554432 s (32MB @ ARO)
    nt = 180  # number of sets to fold: 180*ntint/fwidth~1 minutes
    ntbin = 6  # number of bins the time series is split into for folding
    ngate = 512  # for 1919 # number of bins over the pulsar period
    ntw = min(1020, nt*ntint)  # number of samples to combine for waterfall
    # 1020 / (200.*u.MHz/1024.) = 0.0052224 s

    samplerate = 200. * u.MHz
    fwidth = samplerate / 1024.

    fref = 150. * u.MHz  # ref. freq. for dispersion measure

    coherent = False
    verbose = True
    do_waterfall = False
    do_foldspec = True

    foldspecs = []
    icounts = []
    waterfalls = []
    S = [0,1]

    for P in range(7, 12):  # GMRT overlap: 7--11; full range 0--19
        file1 = file_fmt.format(S=S[0], P=P)
        file2 = file_fmt.format(S=S[1], P=P)

        if not isinstance(phasepol, Polynomial):
            from h5py import File as HDF5File
            from astropy.utils.data import get_pkg_data_filename
            from astropy.time import Time
            from pulsar.predictor import Polyco

            polyco_file = get_pkg_data_filename(phasepol)
            polyco = Polyco(polyco_file)
            h0 = HDF5File(file1.replace('.raw', '.h5'), 'r')
            time0 = Time(h0['SUB_ARRAY_POINTING_000']
                         .attrs['EXPTIME_START_UTC'].replace('Z',''),
                         scale='utc')
            fbottom = (h0['SUB_ARRAY_POINTING_000']['BEAM_000']
                       ['COORDINATES']['COORDINATE_1']
                       .attrs['AXIS_VALUES_WORLD'][0] * u.Hz).to(u.MHz)
            # time0 = Time('2013-07-25T22:12:00.000000000',
            #             scale='utc')
            h0.close()
            phasepol = polyco.phasepol(time0, rphase='fraction', t0=time0,
                                       time_unit=u.second, convert=True)
            nskip = int(round(
                ((Time('2013-07-25T22:15:00', scale='utc') - time0) /
                 (ntint / fwidth)).to(u.dimensionless_unscaled)))
            if verbose:
                print("Using start time {0} and phase polynomial {1}"
                      .format(time0, phasepol))
                print("Skipping {0} records and folding {1} records to cover "
                      "time span {2} to {3}"
                      .format(nskip, nt,
                              time0 + nskip * ntint / fwidth,
                              time0 + (nskip+nt) * ntint / fwidth))
        else:
            fbottom = fwidth*(563.+P*20)

        if verbose:
            print("Doing P={:03d}, fbottom={}".format(P, fbottom))

        myf, myi, mywf = fold(file1, file2, '>f4',
                              fbottom, fwidth, nchan,
                              nt, ntint, nskip,
                              ngate, ntbin, ntw, dm, fref, phasepol,
                              coherent=coherent, do_waterfall=do_waterfall,
                              do_foldspec=do_foldspec,
                              verbose=verbose, progress_interval=1, comm=comm)

        if do_foldspec:
            foldspec = np.zeros_like(myf)
            comm.Reduce(myf, foldspec, op=MPI.SUM, root=0)
            icount = np.zeros_like(myi)
            comm.Reduce(myi, icount, op=MPI.SUM, root=0)

            # save this spectrum
            if comm.rank == 0:
                foldspecs.append(foldspec)
                icounts.append(icount)
                this_f2 = normalize_counts(foldspec, icount)

                np.save("lofar{0}foldspec2{1}_{2}{3}.npy".format(psr, P, *S), this_f2)

        if do_waterfall:
            wf = np.zeros_like(mywf)
            comm.Reduce(mywf, wf, op=MPI.SUM, root=0)
            if comm.rank == 0:
                waterfalls.append(wf)

    if do_waterfall and comm.rank == 0:
        waterfall = np.concatenate(waterfalls, axis=0)
        waterfall = normalize_counts(waterfall)
        np.save("lofar{0}waterfall_{1}{2}.npy".format(psr, *S), waterfall)

    if do_foldspec and comm.rank == 0:
        foldspec2 = np.concatenate(foldspecs, axis=0)
        icounts2 = np.concatenate(icounts, axis=0)
        this_foldspec2 = normalize_counts(foldspec2, icounts2)
        np.save("lofar{0}foldspec2_{1}{2}.npy".format(psr, *S), foldspec2)
        np.save("lofar{0}icounts2_{1}{2}.npy".format(psr, *S), icounts2)

        f2 = this_foldspec2.copy()
        foldspec1 = f2.sum(axis=2)
        fluxes = foldspec1.sum(axis=0)
        foldspec3 = f2.sum(axis=0)

        f = open('lofar{0}flux_{1}{2}.dat'.format(psr, *S), 'w')
        for i, flux in enumerate(fluxes):
            f.write('{0:12d} {1:12.9g}\n'.format(i+1, flux))
        f.close()

    plots = True
    if plots and comm.rank == 0:
        if do_waterfall:
            w = waterfall.copy()
            pmap('lofar{0}waterfall_{1}{2}.pgm'.format(psr, *S),
                 w, 1, verbose=True)
        if do_foldspec:
            pmap('lofar{0}folded_{1}{2}.pgm'.format(psr, *S),
                 foldspec1, 0, verbose)
            pmap('lofar{0}foldedbin_{1}{2}.pgm'.format(psr, *S),
                 f2.transpose(0,2,1).reshape(nchan,-1), 1, verbose)
            pmap('lofar{0}folded3_{1}{2}.pgm'.format(psr, *S),
                 foldspec3, 0, verbose)
