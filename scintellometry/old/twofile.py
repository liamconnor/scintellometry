import numpy as np

from astropy.time import Time, TimeDelta

from mpi4py import MPI

class twofile(object):
    def __init__(self, timestamp_file, files, recsize=2**22,
                 utc_offset=TimeDelta(5.5*3600, format='sec'),
                 comm=None):
        if comm is None:
            self.comm = MPI.COMM_SELF
        else:
            self.comm = comm
        self.timestamp_file = timestamp_file
        self.indices, self.timestamps, self.gsb_start = read_timestamp_file(
            timestamp_file, utc_offset)
        self.fh_raw = [MPI.File.Open(self.comm, raw, amode=MPI.MODE_RDONLY) for raw in files]
        self.recsize = recsize
        self.index = 0

    def seek(self, offset):
        assert offset % self.recsize == 0
        self.index = offset // self.recsize
        for i, fh in enumerate(self.fh_raw):
            fh.Seek(np.count_nonzero(self.indices[:self.index] == i) *
                    self.recsize)

    def close(self):
        for fh in self.fh_raw:
            fh.Close()

    def read(self, size):
        assert size == self.recsize
        if self.index == len(self.indices):
            raise EOFError
        self.index += 1
        # print('reading from {}, t={}'.format(
        #     self.fh_raw[self.indices[self.index-1]],
        #     self.timestamps[self.index-1]))
        z = np.zeros(size, dtype='i1')
        self.fh_raw[self.indices[self.index-1]].Iread(z)
        return z

    @property
    def time(self):
        return self.timestamps[self.index]

    def __repr__(self):
        return ("<open two raw_voltage_files {} "
                "using timestamp file '{}' at index {} (time {})>"
                .format(self.fh_raw, self.timestamp_file, self.index,
                        self.timestamps[self.index]))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def read_timestamp_file(filename, utc_offset):
    pc_times = []
    gps_times = []
    ist_utc = TimeDelta(utc_offset)
    prevseq = prevsub = -1
    with open(filename) as fh:
        line = fh.readline()
        while line != '':
            strings = ('{}-{}-{}T{}:{}:{} {} {}-{}-{}T{}:{}:{} {} {} {}'
                       .format(*line.split())).split()
            seq = int(strings[4])
            sub = int(strings[5])
            if prevseq > 0:
                assert seq == prevseq+1
                assert sub == (prevsub+1) % 8
            prevseq, prevsub = seq, sub

            time = (Time([strings[0], strings[2]], scale='utc') +
                    TimeDelta([float(strings[1]), float(strings[3])],
                              format='sec')) - ist_utc
            pc_times += [time[0]]
            gps_times += [time[1]]  # add half a step below

            line = fh.readline()

    indices = np.array(len(gps_times)*[0, 1], dtype=np.int8)

    pc_times = Time(pc_times)
    gps_times = Time(gps_times)
    gps_pc = gps_times - pc_times
    assert np.allclose(gps_pc.sec, gps_pc[0].sec, atol=1.e-3)
    dt = gps_times[1:] - gps_times[:-1]
    assert np.allclose(dt.sec, dt[0].sec, atol=1.e-5)
    gsb_start = gps_times[-1] - seq * dt[0]  # should be whole minute
    assert '00.000' in gsb_start.isot

    timestamps = Time([t + i * dt[0] / 2. for t in gps_times for i in (0,1)])

    return indices, timestamps, gsb_start
