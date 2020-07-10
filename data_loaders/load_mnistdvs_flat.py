import struct
import tables
from data_loaders.events_timeslices import *
import os

dcll_folder = os.path.dirname(__file__)


class SequenceGenerator(object):
    def __init__(self,
                 filename=os.path.join(dcll_folder, '../data/mnist_dvs_events.hdf5'),
                 group='train',
                 batch_size=32,
                 chunk_size=500,
                 n_inputs=1352,
                 dt=1000):

        self.group = group
        self.dt = dt
        self.n_inputs = n_inputs
        f = tables.open_file(filename, 'r', swmr=True, libver="latest")
        self.stats = f.root.stats

        if self.group =='train':
            self.grp1 = f.root.train
        else:
            self.grp1 = f.root.test
        self.num_classes = 10
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.T = int(chunk_size * 1000 / dt)


    def next(self):
        if self.group != 'test':
            dat, lab = next(
                self.grp1,
                batch_size=self.batch_size,
                chunk_size=self.chunk_size,
                T=self.T,
                n_classes=self.num_classes,
                n_inputs=self.n_inputs,
                dt=self.dt)
        else:
            dat, lab = next_all(
                self.grp1,
                chunk_size=self.chunk_size,
                T=self.T,
                n_classes=self.num_classes,
                n_inputs=self.n_inputs,
                dt=self.dt)

        return dat, lab


def next(hdf5_group, batch_size=32, chunk_size=500, T=80, n_classes=10, n_inputs=1352, dt=1000):
    batch = np.empty([batch_size, T, n_inputs], dtype='float')
    batch_idx = np.random.choice(len(hdf5_group['labels'][:]), [batch_size], replace=False)
    batch_idx_l = np.empty(batch_size, dtype='int')

    times = hdf5_group['time'][:]
    for i, b in (enumerate(batch_idx)):
        batch_idx_l[i] = hdf5_group['labels'][b, 0]
        start_time = hdf5_group['labels'][b, 1]
        end_time = hdf5_group['labels'][b, 2]

        curr = get_event_slice(times, hdf5_group['data'], start_time, end_time, chunk_size, n_inputs=n_inputs, dt=dt)
        if len(curr) < T:
            batch[i][:len(curr)] = curr
        else:
            batch[i] = curr[:T]

    return batch.swapaxes(0, 1), expand_targets(one_hot(batch_idx_l, n_classes), T).astype('float')


def next_all(hdf5_group, chunk_size=500, T=80, n_classes=10, n_inputs=1352, dt=1000):
    batch_size = len(hdf5_group['labels'][:])
    batch = np.empty([batch_size, T, n_inputs], dtype='float')
    batch_idx_l = np.empty(batch_size, dtype='int')

    times = hdf5_group['time'][:]
    for i, labels in enumerate(hdf5_group['labels'][:]):
        label = labels[0]
        batch_idx_l[i] = label
        start_time = labels[1]
        end_time = labels[2]

        curr = get_event_slice(times, hdf5_group['data'], start_time, end_time, chunk_size, n_inputs=n_inputs, dt=dt)
        if len(curr) < T:
            batch[i][:len(curr)] = curr
        else:
            batch[i] = curr[:T]

    batch = batch.swapaxes(0, 1)
    return batch, expand_targets(one_hot(batch_idx_l, n_classes), T)



def get_event_slice(times, addrs, start_time, end_time, chunk_size, n_inputs=676, dt=1000):
    try:
        idx_beg = find_first(times, start_time)
        idx_end = find_first(times[idx_beg:], min(end_time, start_time + chunk_size * dt)) + idx_beg
        return chunk_evs_pol(times[idx_beg:idx_end], addrs[idx_beg:idx_end], deltat=dt, n_inputs=n_inputs)
    except IndexError:
        print("Empty batch found, returning -1")
        raise
        return -1


def chunk_evs_pol(times, addrs, deltat=1000, n_inputs=676):
    t_start = times[0]
    ts = range(t_start, times[-1], deltat)
    chunks = np.zeros([len(ts), n_inputs], dtype='int8')
    idx_start = 0
    idx_end = 0

    for i, t in enumerate(ts):
        idx_end += find_first(times[idx_end:], t)
        if idx_end > idx_start:
            ee = addrs[idx_start:idx_end]
            pol, x, y = ee[:, 2], ee[:, 0], ee[:, 1]
            print((1 + pol)/2)
            print(type(pol))
            addr = ((1 + pol)/2 + 2 * (x * 26 + y)).astype(int)
            try:
                chunks[i, addr] = 1
            except:
                print(i, t, addrs.shape, idx_start, idx_end)
                print(i, (1 + pol)/2 + 2 * addr)
                raise IndexError
        idx_start = idx_end
    return chunks


def gather_aedat(directory, start_id, end_id):
    if not os.path.isdir(directory):
        raise FileNotFoundError("MNIST-DVS Dataset not found, looked at: {}".format(directory))

    dirs = [r'/' + dir_ for dir_ in os.listdir(directory)]
    fns = [[] for _ in range(10)]

    for i in range(10):
        for j in range(start_id, end_id):
            for dir_ in dirs:
                if dir_.find(str(i)) != -1:
                    fns[i].append(directory + dir_ + '/scale4' + '/' + ('mnist_%d_scale04_' % i) + "{0:04d}".format(j) + '.aedat')
    return fns


def aedat_to_events(datafile, last_ts=0, min_pxl_value=48, max_pxl_value=73):
    # constants
    aeLen = 8  # 1 AE event takes 8 bytes
    readMode = '>II'  # struct.unpack(), 2x ulong, 4B+4B
    xmask = 0x00fe
    ymask = 0x7f00
    pmask = 0x1

    aerdatafh = open(datafile, 'rb')
    k = 0  # line number
    p = 0  # pointer, position on bytes
    statinfo = os.stat(datafile)

    length = statinfo.st_size

    # header
    lt = aerdatafh.readline()
    while lt and lt[:1] == b'#':
        p += len(lt)
        k += 1
        lt = aerdatafh.readline()
        continue

    # variables to parse
    events = []
    # read data-part of file
    aerdatafh.seek(p)
    s = aerdatafh.read(aeLen)
    p += aeLen

    while p < length:
        addr, ts = struct.unpack(readMode, s)

        # parse event's data
        x_addr = 128 - 1 - ((xmask & addr) >> 1)
        y_addr = ((ymask & addr) >> 8)
        a_pol = 1 - 2 * (addr & pmask)

        if (x_addr >= min_pxl_value) & (x_addr <= max_pxl_value) & (y_addr >= min_pxl_value) & (y_addr <= max_pxl_value):
            events.append([ts, x_addr - min_pxl_value, y_addr - min_pxl_value, a_pol])

        aerdatafh.seek(p)
        s = aerdatafh.read(aeLen)
        p += aeLen

    events = np.column_stack(events).astype(np.int64)
    events[0, :] += last_ts + 100000
    label = int(datafile[-20])

    print(events[0, 0], events[0, -1])

    return events, label


def create_events_hdf5(path_to_hdf5, path_to_data=r'/users/k1804053/processed_polarity'):
    fns_train = gather_aedat(path_to_data, 1, 901)
    fns_test = gather_aedat(path_to_data, 901, 1001)

    hdf5_file = tables.open_file(path_to_hdf5, 'w')
    train = hdf5_file.create_group(where=hdf5_file.root, name='train')
    train_times_array = hdf5_file.create_earray(where=hdf5_file.root.train, name='time', atom=tables.Atom.from_dtype(np.dtype('int64')), shape=(0,))
    train_data_array = hdf5_file.create_earray(where=hdf5_file.root.train, name='data', atom=tables.Atom.from_dtype(np.dtype('int64')), shape=(0, 3))
    train_labels_array = hdf5_file.create_earray(where=hdf5_file.root.train, name='labels', atom=tables.Atom.from_dtype(np.dtype('int64')), shape=(0, 3))

    print("processing training data...")

    last_ts = 0

    labels_count = {i: 0 for i in range(10)}

    for i, digit in enumerate(fns_train):
        for file in digit:
            events, label = aedat_to_events(file, last_ts)

            train_labels_array.append(np.array([i, events[0, 0], events[0, -1]], dtype=np.int64)[None, :])
            train_times_array.append(events[0, :])
            train_data_array.append(events[1:, :].T)

            labels_count[i] += 1
            last_ts = events[0, -1]

    test = hdf5_file.create_group(where=hdf5_file.root, name='test')
    test_times_array = hdf5_file.create_earray(where=hdf5_file.root.test, name='time', atom=tables.Atom.from_dtype(np.dtype('int64')), shape=(0,))
    test_data_array = hdf5_file.create_earray(where=hdf5_file.root.test, name='data', atom=tables.Atom.from_dtype(np.dtype('int64')), shape=(0, 3))
    test_labels_array = hdf5_file.create_earray(where=hdf5_file.root.test, name='labels', atom=tables.Atom.from_dtype(np.dtype('int64')), shape=(0, 3))

    print("processing testing data...")
    last_ts = 0
    for i, digit in enumerate(fns_test):

        for file in digit:
            events, label = aedat_to_events(file, last_ts)

            test_labels_array.append(np.array([label, events[0, 0], events[0, -1]])[None, :])
            test_times_array.append(events[0, :])
            test_data_array.append(events[1:, :].T)

            last_ts = events[0, -1]

    stats = np.array([i/sum(labels_count.values()) for i in labels_count.values()])
    test_data_array = hdf5_file.create_array(where=hdf5_file.root, name='stats', atom=tables.Atom.from_dtype(stats.dtype), obj=stats)

    hdf5_file.close()


def create_data(path_to_hdf5=os.path.join(dcll_folder, '../data/mnist_dvs_events.hdf5'), path_to_data=None, batch_size=64, chunk_size=500, n_inputs=1352, dt=1000):
    print(path_to_hdf5)
    if os.path.exists(path_to_hdf5):
        print("File {} exists: not re-converting data".format(path_to_hdf5))
    elif (not os.path.exists(path_to_hdf5)) & (path_to_data is not None):
        print("converting MNIST-DVS to h5file")
        create_events_hdf5(path_to_hdf5, path_to_data)
    else:
        print('Either an hdf5 file or MNIST DVS data must be specified')

    strain = SequenceGenerator(filename=path_to_hdf5, group='train', batch_size=batch_size, chunk_size=chunk_size, n_inputs=n_inputs, dt=dt)
    stest = SequenceGenerator(filename=path_to_hdf5, group='test', batch_size=batch_size, chunk_size=chunk_size, n_inputs=n_inputs, dt=dt)
    return strain, stest
