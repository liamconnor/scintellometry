""" work in progress: need to do lofar-style waterfall and foldspec """
from __future__ import division, print_function

import numpy as np
import astropy.units as u

from reduction import reduce, CL_parser

MAX_RMS = 4.
_fref = 150. * u.MHz  # ref. freq. for dispersion measure


def rfi_filter_raw(raw, nchan):
    # note this should accomodate all data (including's lofar raw = complex)
    rawbins = raw.reshape(-1, 2**11*nchan)  # note, this is view!
    std = rawbins.std(-1, keepdims=True)
    ok = std < MAX_RMS
    rawbins *= ok
    return raw, ok


def rfi_filter_power(power):
    return np.clip(power, 0., MAX_RMS**2 * power.shape[-1])


if __name__ == '__main__':
    args = CL_parser()
    args.verbose = 0 if args.verbose is None else sum(args.verbose)
    if args.fref is None:
        args.fref = _fref

    if args.rfi_filter_raw:
        args.rfi_filter_raw = rfi_filter_raw
    else:
        args.rfi_filter_raw = None

    if args.reduction_defaults == 'lofar':
        args.telescope = 'lofar'
        # already channelized, determined from filehandle
        # (previously args.nchan = 20)
        args.ntw_min = 1020
        args.waterfall = False
        args.verbose += 1
        args.rfi_filter_raw = None

    elif args.reduction_defaults == 'aro':
        # do little, most args are already set to aro.py defaults
        args.telescope = 'aro'
        args.ntw_min = 1020
        args.verbose += 1
        args.rfi_filter_raw = rfi_filter_raw

    elif args.reduction_defaults == 'gmrt':
        args.telescope = 'gmrt'
        args.nchan = 512
        args.ngate = 512
        args.ntbin = 5
        # 170 /(100.*u.MHz/6.) * 512 = 0.0052224 s = 256 bins/pulse
        args.ntw_min = 170
        args.rfi_filter_raw = None
        args.verbose += 1
    reduce(
        args.telescope, args.date, tstart=args.tstart, tend=args.tend,
        nchan=args.nchan, ngate=args.ngate, ntbin=args.ntbin,
        ntw_min=args.ntw_min, rfi_filter_raw=args.rfi_filter_raw,
        do_waterfall=args.waterfall, do_foldspec=args.foldspec,
        dedisperse=args.dedisperse, fref=args.fref, verbose=args.verbose)
