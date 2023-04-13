"""
Created on 09/11/2021

@author: fsommer

Module Description:

File to test options in splitting large scan-arrays into smaller chunks and store them on disk instead of memory

"""
import numpy as np
import h5py
import dask.dataframe as dd

from tempfile import mkdtemp
import os.path as path

import Tilda.Service.Formatting as Form

# REFERENCE
nOfTracks = 1
nOfPmts = 6
nOfScans = 1
nOfSteps = 100
nOfBins = 400000

print('starting test, data size for one track will be {} bytes (= {} Gb)'.format(nOfBins*nOfSteps*nOfPmts*4, nOfBins*nOfSteps*nOfPmts*4*1e-9))

try:
    time_res = np.array([np.random.random_integers(0, 10000, (nOfPmts, nOfSteps, nOfBins)) for tr in range(nOfTracks)], dtype=np.int32)

    rebinned = Form.rebin_single_track_spec_data(time_res[0], [], 20, resolution_xml_ns=10)
    print('Done Rebinning')
except Exception as e:
    print(e)

# Try with hdf5
filenameh5 = path.join(mkdtemp(), 'newfile')

# create on hdf5 for each step
for pmt in range(nOfPmts):
    for step in range(nOfSteps):
        hf = h5py.File('{}_{}_{}.dat'.format(filenameh5, pmt, step), 'w')
        stepdata = np.random.random_integers(0, 10000, (nOfBins))
        hf.create_dataset('pmt_{}_step_{}'.format(pmt, step), data=stepdata)
        hf.close()

# load the hdf5s into dask
df = dd.read_hdf('{}_*.dat'.format(filenameh5), '/x')


# Now try a memmap
filename = path.join(mkdtemp(), 'newfile.dat')
fp = np.memmap(filename, dtype='int32', mode='w+', shape=(nOfPmts, nOfSteps, nOfBins))

for pmt in range(nOfPmts):
    for step in range(nOfSteps):
        stepdata = np.random.random_integers(0, 10000, (nOfBins))
        fp[pmt, step, :] = stepdata[:]
        fp.flush()

print(fp[0, 0, 0])


def create_np_like_by_looping(function):
    1
