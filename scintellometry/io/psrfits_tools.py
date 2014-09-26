""" routine to create PSRFITS files """

"""
To create a (for example) a 'SUBINT' bintable:
headers, coldefs = PSRFITS_hdus()

t = FITS.BinTableHDU(header=headers['SUBINT'])
for col in coldefs['SUBINT']:
     t.columns.add_col(col)

"""
import numpy as np
import os
from astropy.io import fits

# fits.column.Column  kwargs conversions
_tabledata = {'TTYPE#':'name',
              'TFORM#':'format',
              'TUNIT#':'unit',
              'TNULL#':'null',
              'TSCAL#':'bscale',
              'TZERO#':'tzero',
              'TDISP#':'disp',
              'TBCOL#':'start',
              'TDIM#':'dim'}

# conversion from FITS data formats to numpy
_dataconv = {'L':'|b1', 'A':'a', 'B':'|u1', 'I': 'i1', 'J':'i2', 'K': 'i4',
             'E':'f4', 'D':'f8', 'C':'c8', 'M':'c16'}


def psrFITS_hdus(fitsdef=None):
    """
    open the PSRfits definition file and return a dictionary
    of fits headers, and a dictionary of ColDefs keyed by 'EXTNAME' or 'PRIMARY'

    Args:
    fitsdef : text file of the PSRfits definition (default: fitsdef.txt)

    Returns:
    headers, coldefs

    """
    if fitsdef is None:
        fitsdef = os.path.join(os.path.dirname(__file__), 'fitsdef.txt')

    fh = open(fitsdef, 'r')
    data = fh.read().splitlines()

    hdus = {}
    coldefs = {}
    # header cards
    hcards = []
    # column cards
    coldef = {}
    ccards = []
    # for coldefs
    ncol = 0
    for line in data:
        if line[0] == '#': continue

        card = fits.Card.fromstring(line)
        try:
            card.verify(option='silentfix')
        except(fits.VerifyError):
            # Note:
            # PSRFITS def has a lot of '*' which are not handled by pyFITS
            pass 

        # table entries
        if card.keyword in _tabledata.keys():
            # a BinTable column
            # map from TTYPE/TFORM/... to fits.Column kwargs
            key = _tabledata[card.keyword]

            if key == 'name' and len(coldef) >= 2:
                # reset the Column definition on each new TTYPE
                ccards.append(fits.Column(**coldef))
                coldef = {}
            coldef[key] = card.value

        elif card.keyword == 'END':
            if 'EXTNAME' in [c.keyword for c in hcards]:
                name = [c.value for c in hcards if c.keyword == 'EXTNAME'][0]
            else:
                name = 'PRIMARY'
            hdus[name] = fits.Header(cards=hcards)
            hcards = []

            if len(coldef) >= 1:
                ccards.append(fits.Column(**coldef))
                coldefs[name] = fits.ColDefs(ccards)
                coldef = {}
                ccards = []

        # standard header cards
        else:
            if card.value == '*':
                card.value = 0
            hcards.append(card)

    return hdus, coldefs
        

_hdefs, _coldefs = psrFITS_hdus()
_extnames = [v.get('EXTNAME',None) for v in _hdefs.values()\
             if v.get('EXTNAME',None) is not None]

class psrFITS(fits.HDUList):
    """ class to help make raw telescope files act like FITS files """
    def __init__(self, hdus=[]):
        hdulist = [fits.PrimaryHDU(header=_hdefs['PRIMARY'])]
        fits.HDUList.__init__(self, hdus=hdulist)
        for extname in hdus:
            if extname == 'PRIMARY': continue
            self.add_hdu(extname)

    def add_hdu(self, extname):
        """ add one of the PSRFITs-defined HDUs """
        hdr = _hdefs[extname].copy()
        name = hdr.pop('EXTNAME')
        t = fits.BinTableHDU(name=name, header=hdr)
        for col in _coldefs[extname]:
            t.columns.add_col(col)
        self.append(t)
        
def fits2numpy_fmts(fmts):
    """ convenience routine to convert FITS formats to numpy dtypes """
    nfmts = []
    for f in fmts:
        for key, val in _dataconv.iteritems():
            if key in f:
                n = f.strip(key)
                nfmts.append(n + val)
    return nfmts
                
