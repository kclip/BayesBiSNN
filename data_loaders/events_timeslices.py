#!/bin/python
#-----------------------------------------------------------------------------
# File Name : event_timeslices.py
# Author: Emre Neftci
#
# Creation Date : 
# Last Modified : Thu 16 May 2019 02:13:09 PM PDT
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 

from __future__ import print_function
import bisect
import numpy as np
from scipy.sparse import coo_matrix as sparse_matrix

def expand_targets(targets, T=500, burnin=0):
    y = np.tile(targets.copy(), [T, 1, 1])
    y[:burnin] = 0
    return y

def one_hot(mbt, num_classes):
    out = np.zeros([mbt.shape[0], num_classes])
    out[np.arange(mbt.shape[0], dtype='int'),mbt.astype('int')] = 1
    return out

def find_first(a, tgt):
    return bisect.bisect_left(a, tgt)
