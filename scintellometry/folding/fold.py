from __future__ import division, print_function

from inspect import getargspec
import numpy as np
import os
import astropy.units as u
import astropy.io.fits as FITS

try:
    # do *NOT* use on-disk cache; blue gene doesn't work; slower anyway
    # import pyfftw
    # pyfftw.interfaces.cache.enable()
    from pyfftw.interfaces.scipy_fftpack import (rfft, rfftfreq,
                                                 fft, ifft, fftfreq)
    _fftargs = {'threads': int(os.environ.get('OMP_NUM_THREADS', 2)),
                'planner_effort': 'FFTW_ESTIMATE'}
except(ImportError):
    print("Consider installing pyfftw: https://github.com/hgomersall/pyFFTW")
    # use FFT from scipy, since unlike numpy it does not cast up to complex128
    from scipy.fftpack import rfft, rfftfreq, fft, ifft, fftfreq
    _fftargs = {}

dispersion_delay_constant = 4149. * u.s * u.MHz**2 * u.cm**3 / u.pc


def fold(fh, comm, samplerate, fedge, fedge_at_top, nchan,
         nt, ntint, ngate, ntbin, ntw, dm, fref, phasepol,
         dedisperse='incoherent',
         do_waterfall=True, do_foldspec=True, verbose=True,
         progress_interval=100, rfi_filter_raw=None, rfi_filter_power=None,
         return_fits=False):
    """
    FFT data, fold by phase/time and make a waterfall series

    Folding is done from the position the file is currently in

    Parameters
    ----------
    fh : file handle
        handle to file holding voltage timeseries
    comm: MPI communicator or None
        will use size, rank attributes
    samplerate : Quantity
        rate at which samples were originally taken and thus double the
        band width (frequency units)
    fedge : float
        edge of the frequency band (frequency units)
    fedge_at_top: bool
        whether edge is at top (True) or bottom (False)
    nchan : int
        number of frequency channels for FFT
    nt, ntint : int
        total number nt of sets, each containing ntint samples in each file
        hence, total # of samples is nt*ntint, with each sample containing
        a single polarisation
    ngate, ntbin : int
        number of phase and time bins to use for folded spectrum
        ntbin should be an integer fraction of nt
    ntw : int
        number of time samples to combine for waterfall (does not have to be
        integer fraction of nt)
    dm : float
        dispersion measure of pulsar, used to correct for ism delay
        (column number density)
    fref: float
        reference frequency for dispersion measure
    phasepol : callable
        function that returns the pulsar phase for time in seconds relative to
        start of the file that is read.
    dedisperse : None or string (default: incoherent).
        None, 'incoherent', 'coherent', 'by-channel'.
        Note: None really does nothing
    do_waterfall, do_foldspec : bool
        whether to construct waterfall, folded spectrum (default: True)
    verbose : bool or int
        whether to give some progress information (default: True)
    progress_interval : int
        Ping every progress_interval sets
    return_fits : bool (default: False)
        return a subint fits table for rank == 0 (None otherwise)

    """
    assert dedisperse in (None, 'incoherent', 'by-channel', 'coherent')
    assert nchan % fh.nchan == 0
    if dedisperse == 'by-channel':
        oversample = nchan // fh.nchan
        assert ntint % oversample == 0
    else:
        oversample = 1

    if dedisperse == 'coherent' and fh.nchan > 1:
        raise ValueError("For coherent dedispersion, data must be "
                         "unchannelized before folding.")

    if comm is None:
        mpi_rank = 0
        mpi_size = 1
    else:
        mpi_rank = comm.rank
        mpi_size = comm.size

    npol = getattr(fh, 'npol', 1)
    assert npol == 1 or npol == 2
    if verbose > 1 and mpi_rank == 0:
        print("Number of polarisations={}".format(npol))

    # initialize folded spectrum and waterfall
    # TODO: use estimated number of points to set dtype
    if do_foldspec:
        foldspec = np.zeros((ntbin, nchan, ngate, npol**2), dtype=np.float32)
        icount = np.zeros((ntbin, nchan, ngate), dtype=np.int32)
    else:
        foldspec = None
        icount = None

    if do_waterfall:
        nwsize = nt*ntint//ntw
        waterfall = np.zeros((nwsize, nchan, npol**2), dtype=np.float64)
    else:
        waterfall = None

    if verbose and mpi_rank == 0:
        print('Reading from {}'.format(fh))

    nskip = fh.tell()/fh.blocksize
    if nskip > 0:
        if verbose and mpi_rank == 0:
            print('Starting {0} blocks = {1} bytes out from start.'
                  .format(nskip, nskip*fh.blocksize))

    dt1 = (1./samplerate).to(u.s)
    # need 2*nchan real-valued samples for each FFT
    if fh.telescope == 'lofar':
        dtsample = fh.dtsample
    else:
        dtsample = nchan * 2 * dt1
    tstart = dtsample * ntint * nskip

    # pre-calculate time delay due to dispersion in coarse channels
    # for channelized data, frequencies are known

    if fh.nchan == 1:
        if fedge_at_top:
            freq = fedge - rfftfreq(nchan*2, dt1.value)[::2] * u.Hz
        else:
            freq = fedge + rfftfreq(nchan*2, dt1.value)[::2] * u.Hz

        freq_in = freq
    else:
        # input frequencies may not be the ones going out
        freq_in = fh.frequencies
        if oversample == 1:
            freq = freq_in
        else:
            if fedge_at_top:
                freq = (freq_in[:, np.newaxis] - u.Hz *
                        fftfreq(oversample, dtsample.value))
            else:
                freq = (freq_in[:, np.newaxis] + u.Hz *
                        fftfreq(oversample, dtsample.value))
    ifreq = freq.ravel().argsort()

    # pre-calculate time offsets in (input) channelized streams
    dt = dispersion_delay_constant * dm * (1./freq_in**2 - 1./fref**2)

    if dedisperse in ['coherent', 'by-channel']:
        # pre-calculate required turns due to dispersion
        if fedge_at_top:
            fcoh = (freq_in[np.newaxis,:] - u.Hz *
                    fftfreq(ntint, dtsample.value)[:, np.newaxis])
        else:
            fcoh = (freq_in[np.newaxis,:] + u.Hz *
                    fftfreq(ntint, dtsample.value)[:, np.newaxis])

        # set frequency relative to which dispersion is coherently corrected
        if dedisperse == 'coherent':
            _fref = fref
        else:
            _fref = freq_in[np.newaxis, :]
        # (check via eq. 5.21 and following in
        # Lorimer & Kramer, Handbook of Pulsar Astrono
        dang = (dispersion_delay_constant * dm * fcoh *
                (1./_fref-1./fcoh)**2) * u.cycle

        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            dd_coh = np.exp(dang * 1j).conj().astype(np.complex64)

        # add dimension for polarisation
        dd_coh = dd_coh[..., np.newaxis]

    for j in xrange(mpi_rank, nt, mpi_size):
        if verbose and j % progress_interval == 0:
            print('#{:4d}/{:4d} is doing {:6d}/{:6d}; time={:18.12f}'.format(
                mpi_rank, mpi_size, j+1, nt,
                (tstart+dtsample*j*ntint).value))  # time since start

        # just in case numbers were set wrong -- break if file ends
        # better keep at least the work done
        try:
            # ARO/GMRT return int-stream,
            # LOFAR returns complex64 (count/nchan, nchan)
            # LOFAR "combined" file class can do lots of seeks, we minimize
            # that with the 'seek_record_read' routine
            raw = fh.seek_record_read(int((nskip+j)*fh.blocksize),
                                      fh.blocksize)
        except(EOFError, IOError) as exc:
            print("Hit {0!r}; writing pgm's".format(exc))
            break
        if verbose >= 2:
            print("#{:4d}/{:4d} read {} items"
                  .format(mpi_rank, mpi_size, raw.size), end="")

        if npol == 2:  # multiple polarisations
            raw = raw.view(raw.dtype.fields.values()[0][0])

        if fh.nchan == 1:  # raw.shape=(ntint*npol)
            raw = raw.reshape(-1, npol)
        else:              # raw.shape=(ntint, nchan*npol)
            raw = raw.reshape(-1, fh.nchan, npol)

        if rfi_filter_raw is not None:
            raw, ok = rfi_filter_raw(raw)
            if verbose >= 2:
                print("... raw RFI (zap {0}/{1})"
                      .format(np.count_nonzero(~ok), ok.size), end="")

        if np.can_cast(raw.dtype, np.float32):
            vals = raw.astype(np.float32)
        else:
            assert raw.dtype.kind == 'c'
            vals = raw

        if fh.nchan == 1:
            # have real-valued time stream; if we need some coherent
            # dedispersion, do FT of whole thing, otherwise to output channels
            ftchan = nchan if dedisperse == 'incoherent' else len(vals)//2
            vals = rfft(vals.reshape(-1, ftchan*2, npol), axis=1,
                        overwrite_x=True, **_fftargs)
            # rfft: Re[0], Re[1], Im[1], ..., Re[n/2-1], Im[n/2-1], Re[n/2]
            # re-order to normal fft format; make it like Numerical Recipes:
            # Re[0], Re[n], Re[1], Im[1], .... (channel 0 is junk anyway)
            vals = np.hstack((vals[:, 0], vals[:, -1],
                              vals[:, 1:-1])).view(np.complex64)
            # for incoherent, vals.shape=(ntint, nchan, npol) -> OK
            # for others, have           (1, ntint*nchan, npol)
            # reshape(nchan, ntint) gives rough as slowly varying -> .T
            if dedisperse != 'incoherent':
                fine = vals.reshape(nchan, -1, npol).transpose(1, 0, 2)
                # now have fine.shape=(ntint, nchan, npol)

        else:  # data already channelized
            if dedisperse == 'by-channel':
                fine = fft(vals, axis=0, overwrite_x=True, **_fftargs)
                # have fine.shape=(ntint, fh.nchan, npol)

        if dedisperse in ['coherent', 'by-channel']:
            fine *= dd_coh
            # rechannelize to output channels
            if oversample > 1 and dedisperse == 'by-channel':
                # fine.shape=(ntint*oversample, chan_in, npol)
                #           =(coarse,fine,fh.chan, npol)
                #  -> reshape(oversample, ntint, fh.nchan, npol)
                # want (ntint=fine, fh.nchan, oversample, npol) -> .transpose
                fine = (fine.reshape(oversample, -1, fh.nchan, npol)
                        .transpose(1, 2, 0, 3)
                        .reshape(-1, nchan, npol))
            # now, for both,     fine.shape=(ntint, nchan, npol)
            vals = ifft(fine, axis=0, overwrite_x=True, **_fftargs)
            # vals[time, chan, pol]
            if verbose >= 2:
                print("... dedispersed", end="")

        if npol == 1:
            power = vals.real**2 + vals.imag**2
        else:
            p0 = vals[..., 0]
            p1 = vals[..., 1]
            power = np.empty(vals.shape[:-1] + (4,), np.float32)
            power[..., 0] = p0.real**2 + p0.imag**2
            power[..., 1] = p0.real*p1.real + p0.imag*p1.imag
            power[..., 2] = p0.imag*p1.real - p0.real*p1.imag
            power[..., 3] = p1.real**2 + p1.imag**2

        if verbose >= 2:
            print("... power", end="")

        if rfi_filter_power is not None:
            power = rfi_filter_power(power)
            print("... power RFI", end="")

        # current sample positions in stream
        isr = j*(ntint // oversample) + np.arange(ntint // oversample)

        if do_waterfall:
            # loop over corresponding positions in waterfall
            for iw in xrange(isr[0]//ntw, isr[-1]//ntw + 1):
                if iw < nwsize:  # add sum of corresponding samples
                    waterfall[iw, :] += np.sum(power[isr//ntw == iw],
                                               axis=0)[ifreq]
            if verbose >= 2:
                print("... waterfall", end="")

        if do_foldspec:
            ibin = (j*ntbin) // nt  # bin in the time series: 0..ntbin-1

            # times since start
            tsample = (tstart + isr*dtsample*oversample)[:, np.newaxis]
            # correct for delay if needed
            if dedisperse in ['incoherent', 'by-channel']:
                # tsample.shape=(ntint/oversample, nchan_in)
                tsample = tsample - dt

            phase = (phasepol(tsample.to(u.s).value.ravel())
                     .reshape(tsample.shape))
            # corresponding PSR phases
            iphase = np.remainder(phase*ngate, ngate).astype(np.int)

            for k, kfreq in enumerate(ifreq):  # sort in frequency while at it
                iph = iphase[:, (0 if iphase.shape[1] == 1
                                 else kfreq // oversample)]
                # sum and count samples by phase bin
                for ipow in xrange(npol**2):
                    foldspec[ibin, k, :, ipow] += np.bincount(
                        iph, power[:, kfreq, ipow], ngate)
                icount[ibin, k, :] += np.bincount(
                    iph, power[:, kfreq, 0] != 0., ngate)

            if verbose >= 2:
                print("... folded", end="")

        if verbose >= 2:
            print("... done")

    if verbose >= 2 or verbose and mpi_rank == 0:
        print('#{:4d}/{:4d} read {:6d} out of {:6d}'
              .format(mpi_rank, mpi_size, j+1, nt))

    if npol == 1:
        if do_foldspec:
            foldspec = foldspec.reshape(foldspec.shape[:-1])
        if do_waterfall:
            waterfall = waterfall.reshape(waterfall.shape[:-1])

    return foldspec, icount, waterfall


class Folder(dict):
    """
    convenience class to populate many of the 'fold' arguments
    from the psrfits headers of a datafile

    """
    def __init__(self, fh, **kwargs):

        # get the required arguments to 'fold'
        fold_args = getargspec(fold)
        fold_argnames = fold_args.args
        fold_defaults = fold_args.defaults
        # and set the defaults
        Nargs = len(fold_args.args)
        Ndefaults = len(fold_defaults)
        for i, v in enumerate(fold_defaults):
            self[fold_argnames[Nargs - Ndefaults + i]] = v

        # get some defaults from fh (may be overwritten by kwargs)
        self['samplerate'] = fh.samplerate
        # ??? (1./fh['SUBINT'].header['TBIN']*u.Hz).to(u.MHz)
        self['fedge'] = fh.fedge
        self['fedge_at_top'] = fh.fedge_at_top
        self['nchan'] = fh['SUBINT'].header['NCHAN']
        self['ngate'] = fh['SUBINT'].header['NBIN_PRD']

        # update arguments with passed kwargs
        for k in kwargs:
            if k in fold_argnames:
                self[k] = kwargs[k]
            else:
                print("{} not needed for fold routine".format(k))
        # warn of missing, skipping fh and comm
        missing = [k for k in fold_argnames[2:] if k not in self]
        if len(missing) > 0:
            print("Missing 'fold' arguments: {}".format(missing))

    def __call__(self, fh, comm=None):
        return fold(fh, comm=comm, **self)


def normalize_counts(q, count=None):
    """ normalize routines for waterfall and foldspec data """
    if count is None:
        nonzero = np.isclose(q, np.zeros_like(q))  # == 0.
        qn = q
    else:
        nonzero = count > 0
        qn = np.where(nonzero, q/count, 0.)
    # subtract mean profile (pulsar phase always last dimension)
    qn -= np.where(nonzero,
                   np.sum(qn, -1, keepdims=True) /
                   np.sum(nonzero, -1, keepdims=True), 0.)
    return qn

# if return_fits and mpi_rank == 0:
#     # subintu HDU
#     # update table columns
#     # TODO: allow multiple polarizations
#     npol = 1
#     newcols = []
#     # FITS table creation difficulties...
#     # assign data *after* 'new_table' creation
#     array2assign = {}
#     tsubint = ntint*dtsample
#     for col in fh['subint'].columns:
#         attrs = col.copy().__dict__
#         # remove non-init args
#         for nn in ['_pseudo_unsigned_ints', '_dims', '_physical_values',
#                    'dtype', '_phantom', 'array']:
#             attrs.pop(nn, None)

#         if col.name == 'TSUBINT':
#             array2assign[col.name] = np.array(tsubint)
#         elif col.name == 'OFFS_SUB':
#             array2assign[col.name] = np.arange(ntbin) * tsubint
#         elif col.name == 'DAT_FREQ':
#             # TODO: sort from lowest freq. to highest
#             # ('DATA') needs sorting as well
#             array2assign[col.name] = freq.to(u.MHz).value.astype(np.double)
#             attrs['format'] = '{0}D'.format(freq.size)
#         elif col.name == 'DAT_WTS':
#             array2assign[col.name] = np.ones(freq.size, dtype=np.float32)
#             attrs['format'] = '{0}E'.format(freq.size)
#         elif col.name == 'DAT_OFFS':
#             array2assign[col.name] = np.zeros(freq.size*npol,
#                                               dtype=np.float32)
#             attrs['format'] = '{0}E'.format(freq.size*npol)
#         elif col.name == 'DAT_SCL':
#             array2assign[col.name] = np.ones(freq.size*npol,
#                                              dtype=np.float32)
#             attrs['format'] = '{0}E'.format(freq.size)
#         elif col.name == 'DATA':
#             array2assign[col.name] = np.zeros((ntbin, npol, freq.size,
#                                                ngate), dtype='i1')
#             attrs['dim'] = "({},{},{})".format(ngate, freq.size, npol)
#             attrs['format'] = "{0}I".format(ngate*freq.size*npol)
#         newcols.append(FITS.Column(**attrs))
#     newcoldefs = FITS.ColDefs(newcols)

#     oheader = fh['SUBINT'].header.copy()
#     newtable = FITS.new_table(newcoldefs, nrows=ntbin, header=oheader)
#     # update the 'subint' header and create a new one to be returned
#     # owing to the structure of the code (MPI), we need to assign
#     # the 'DATA' outside of fold.py
#     newtable.header.update('NPOL', 1)
#     newtable.header.update('NBIN', ngate)
#     newtable.header.update('NBIN_PRD', ngate)
#     newtable.header.update('NCHAN', freq.size)
#     newtable.header.update('INT_UNIT', 'PHS')
#     newtable.header.update('TBIN', tsubint.to(u.s).value)
#     chan_bw = np.abs(np.diff(freq.to(u.MHz).value).mean())
#     newtable.header.update('CHAN_BW', chan_bw)
#     if dedisperse in ['coherent', 'by-channel', 'incoherent']:
#         newtable.header.update('DM', dm.value)
#     # finally assign the table data
#     for name, array in array2assign.iteritems():
#         try:
#             newtable.data.field(name)[:] = array
#         except ValueError:
#             print("FITS error... work in progress",
#                   name, array.shape, newtable.data.field(name)[:].shape)

#     phdu = fh['PRIMARY'].copy()
#     subinttable = FITS.HDUList([phdu, newtable])
#     subinttable[1].header.update('EXTNAME', 'SUBINT')
#     subinttable['PRIMARY'].header.update('DATE-OBS', fh.time0.isot)
#     subinttable['PRIMARY'].header.update('STT_IMJD', int(fh.time0.mjd))
#     subinttable['PRIMARY'].header.update(
#         'STT_SMJD', int(str(fh.time0.mjd - int(fh.time0.mjd))[2:])*86400)

# return subinttable
