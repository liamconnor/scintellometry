from __future__ import division, print_function

import numpy as np

shift40 = np.array([4,0], np.int8)
shift76543210 = np.array([7,6,5,4,3,2,1,0], np.int8)
msblsb_bits = np.array([-16, 15], np.int8)
twopiby256 = 2.*np.pi / 256.

NP_DTYPES = {'1bit': 'i1', '4bit': 'i1', 'nibble': 'i1',
             'ci1': '2i1', 'ci1,ci1': '2i1,2i1', 'cu4bit,cu4bit': '2u1'}


def fromfile(file, dtype, count, verbose=False):
    """Read count bytes, with type dtype which can be bits.

    Calls np.fromfile but handles some special dtype's:
    'ci1'  : complex number stored as two signed 1-byte integers
             returns count/2 np.complex64 samples
    '1bit' : Unfold for 1-bit sampling (LSB first),
             returns 8*count np.int8 samples, with values of +1 or -1
    '4bit' : Unfold for 4-bit sampling (LSB first)
             returns 2*count np.int8 samples, with -8 < value < 7
    'nibble' : Unfold for Ue-Li's 8-bit complex numbers
             returns count np.complex64 samples, with amplitudes in 4 lsb,
             as sqrt(-2*log(1-((unsigned-4bit/16+0.5)/16.))
             and phase in msb, ((signed-4bit)+0.5) * 2 * pi
             **NOT FINISHED YET** **NEEDS SCALING**
    """

    np_dtype = NP_DTYPES.get(dtype, dtype)
    if verbose:
        print("Reading {} units of dtype={}".format(count, np_dtype))
    # go via direct read to ensure we can read from gzip'd files
    raw = file.read(count)
    # but MultiFile returns 1-byte ndarray; viewing much faster than fromstring
    try:
        raw = raw.view(dtype=np_dtype)
    except:
        raw = np.fromstring(raw, dtype=np_dtype)

    if raw.shape[0] != count // np.dtype(np_dtype).itemsize:
        raise EOFError('In fromfile, got {0} items, expected {1}'
                       .format(raw.shape[0],
                               count // np.dtype(np_dtype).itemsize))

    if np_dtype is dtype:
        return raw
    elif dtype.startswith('ci1'):
        return raw.astype('f4').view(dtype.replace('ci1', 'c8')).squeeze()
    elif dtype == '1bit':
        # For a given int8 byte containing bits 76543210
        # left_shift(byte[:,np.newaxis], shift76543210):
        #    [0xxxxxxx, 10xxxxxx, 210xxxxx, 3210xxxx,
        #     43210xxx, 543210xx, 6543210x, 76543210]
        split = np.left_shift(raw[:,np.newaxis], shift76543210).flatten()
        # right_shift(..., 6):
        #    [0000000x, 11111110, 22222221, 33333332,
        #     44444443, 55555554, 66666665, 77777776]
        # so least significant bits go first.
        np.right_shift(split, 6, split)  # explicitly give output for speedup
        # | 1 -> value becomes +1 or -1
        split |= 1
        return split
    elif dtype == '4bit':
        # For a given int8 byte containing bits 76543210
        # left_shift(byte[:,np.newaxis], shift40):  [3210xxxx, 76543210]
        split = np.left_shift(raw[:,np.newaxis], shift40).flatten()
        # right_shift(..., 4):                      [33333210, 77777654]
        # so least significant bits go first.
        np.right_shift(split, 4, split)  # explicitly give output for speedup
        return split
    elif dtype.startswith('cu4bit'):
        # For a given int8 byte containing bits 76543210, with real as 
        # 4 most significant bits and imaginary as 4 least significant bits:
        # right_shift(byte[:,np.newaxis], shift40):  [xxxx7654, 76543210]
#        raw = raw.flatten()
#        raw_re = ((raw >> 4).astype(np.int8) - 8)
#        raw_im = ((raw & 0xf).astype(np.int8) - 8)

#        data = np.zeros([2*len(raw_re)], dtype=np.int8)
#        data[::2] = raw_re
#        data[1::2] = raw_im

        split = np.right_shift(raw.reshape(-1, 1), shift40).flatten()
        split &= 0xf
        # now ----7654, ----3210
        split -= 8
        return split.astype('f4').view(dtype.replace('cu4bit', 'c8')).squeeze()
    elif dtype == 'nibble':
        # For a given int8 byte containing bits 76543210
        split = np.bitwise_and(raw[:,np.newaxis], msblsb_bits)
        # now interpretation:
        # lsb=amplitude=0..15; msb=phase=16*(-8..7)=-128..112
        # calculate sqrt(-2*log(1-((idf/16+0.5)/16.)); inplace for speed
        iph = split[:,0]
        idf = split[:,1]
        # idf += 0.5
        # idf /= 16.
        # idf = np.subtract(1., idf, out=idf)
        # idf = np.log(idf, out=idf)
        # idf *= 2.
        # amp = np.sqrt(idf, out=idf)
        amp = (idf.astype(np.float32)+0.5) / 16.
        amp = np.sqrt(-2.*np.log(1.-amp))
        phase = (iph.astype(np.float32) + 8.) * twopiby256
        return amp * np.exp(1.j * np.pi * phase)
    else:
        # this should never happen, but just in case...
        raise TypeError('data type "{}" not understood (but in NP_DTYPES!)'
                        .format(dtype))
