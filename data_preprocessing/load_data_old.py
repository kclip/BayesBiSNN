import os
import glob

import numpy as np
import torch
import tables

from .misc import *


def get_batch_example(hdf5_group, idxs, batch_size, T=80, classes=[0], size=[1, 26, 26], dt=1000, x_max=1, polarity=True):
    data = np.zeros([len(idxs), T] + size, dtype='float')
    labels = hdf5_group.labels[idxs, 0]
    start_times = hdf5_group.labels[idxs, 1]
    end_times = hdf5_group.labels[idxs, 2]

    times = hdf5_group.time[:]

    idx_beg = np.array([find_first(times, start_times[i]) for i in range(len(start_times))])
    idx_end = np.array([find_first(times, min(end_times[i], start_times[i] + T * dt)) for i in range(len(idx_beg))])

    addrs = hdf5_group.data[:max(idx_end) + 1].astype('int')

    curr = chunk_evs_pol(times=times[:max(idx_end) + 1], addrs=addrs, batch_size=batch_size,
                         idx_beg=idx_beg, idx_end=idx_end, T=T, size=size, dt=dt, x_max=x_max, polarity=polarity)
    if len(curr) < T:
        data[:, :curr.shape[1]] = curr
    else:
        data = curr[:, :T]

    return torch.FloatTensor(data), torch.FloatTensor(make_output_from_labels(labels, T, classes))


def chunk_evs_pol(times, addrs, batch_size, idx_beg, idx_end, T, dt=1000, size=[2, 304, 240], x_max=1, polarity=True):
    if set(addrs[:100, 2]) == set([-1, 1]):  # Polarities are either [0, 1] or [-1, 1], we set them to [0, 1]
        addrs[:, 2] = ((1 + addrs[:, 2]) / 2).astype('uint8')

    ts = [np.arange(times[idx_beg[i]], times[idx_end[i]], dt) for i in range(len(idx_beg))]
    shortest_len = min([len(t_s) for t_s in ts])
    ts = np.array([t_s[:shortest_len] for t_s in ts])

    batch = np.zeros([batch_size, shortest_len] + size, dtype='float')
    bucket_start = idx_beg
    bucket_end = idx_beg

    for i in range(ts.shape[-1]):
        t = ts[:, i]
        bucket_end = bucket_end + np.array([find_first(times[bucket_end[j]:], t[j]) for j in range(len(bucket_end))])

        ee = [addrs[np.arange(beg, end)] for (beg, end) in zip(bucket_start, bucket_end)]

        for s in range(batch_size):
            evts = ee[s]
            pol, x, y = evts[:, 2], evts[:, 0], evts[:, 1]
            try:
                if len(size) == 3:
                    batch[s, i, pol, x, y] = 1.
                elif len(size) == 2:
                    batch[s, i, pol, (x * x_max + y).astype(int)] = 1.
                elif len(size) == 1:
                    if polarity:
                        batch[s, i, (pol + 2 * (x * x_max + y)).astype(int)] = 1.
                    else:
                        batch[s, i, (x * x_max + y).astype(int)] = 1.
            except:
                i_max = np.argmax((pol + 2 * (x * x_max + y)))
                print(x[i_max], y[i_max], pol[i_max])
                raise IndexError

        bucket_start = bucket_end

    if shortest_len > T:
        batch = batch[:, :T]

    return batch



