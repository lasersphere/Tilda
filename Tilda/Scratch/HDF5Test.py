"""
Created on 

@author: simkaufm

Module Description:  module to find out if HDF5 would be an alternative to pickle.
"""

import h5py

import numpy as np

import Tilda.Service.Scan.draftScanParameters as DraftDict
from Tilda.Driver.DataAcquisitionFpga.TimeResolvedSequencerDummy import TimeResolvedSequencer as TrsDummy

# test with dict
drft_dict = DraftDict.draftScanDict
print(drft_dict)

#  serialize dict with JSON then store it.
# store_dict_to = 'E:\\Workspace\\deleteMe\\hdf5_pipe_test.hdf5pipedat'
# f = h5py.File(store_dict_to, 'w')
#     json.dump(drft_dict, f)
# f.close()
#
# with open(store_dict_to, 'r') as f:
#     ret = json.load(f)
#     print(ret)
#     print(type(ret))
#
# print('draft dict == loaded dict ? : ', drft_dict == ret)

# Test with raw data
trs = TrsDummy()
trs.data_builder(drft_dict, 0)
trs.artificial_build_data = np.asarray(trs.artificial_build_data)
print(trs.artificial_build_data)

store_raw_data_to = 'E:\\Workspace\\deleteMe\\hdf5_raw_data_test.hdf5'
f_raw = h5py.File(store_raw_data_to, 'w')
f_raw.create_dataset('raw_data', data=trs.artificial_build_data)
f_raw.close()

f_load = h5py.File(store_raw_data_to, 'r')
print(f_load['raw_data'].value)

# with open(store_raw_data_to, 'r') as f:
#     ret_raw = json.load(f)
#     print(ret_raw)
#     print(type(ret_raw))
#
# print('build data == loaded data? ? : ', trs.artificial_build_data == ret_raw)

