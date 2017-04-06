"""
Created on 

@author: simkaufm

Module Description: module to find out if JSON would be an alternative to pickle.
"""

import json
import h5py
import numpy as np
import timeit
import pickle
from copy import deepcopy

import Service.Scan.draftScanParameters as DraftDict
from Driver.DataAcquisitionFpga.TimeResolvedSequencerDummy import TimeResolvedSequencer as TrsDummy

# test with dict
drft_dict = DraftDict.draftScanDict
print(drft_dict)

store_dict_to = 'E:\\Workspace\\deleteMe\\JSon_pipe_test.pipedat'
with open(store_dict_to, 'w') as f:
    json.dump(drft_dict, f)
f.close()

with open(store_dict_to, 'r') as f:
    ret = json.load(f)
    print(ret)
    print(type(ret))

print('draft dict == loaded dict ? : ', drft_dict == ret)

# Test with raw data
trs = TrsDummy()
trs.data_builder(drft_dict, 0)
trs.artificial_build_data = np.asarray(trs.artificial_build_data)
for i in range(0, 9):
    trs.artificial_build_data = np.append(trs.artificial_build_data, trs.artificial_build_data)
trs.artificial_build_data = trs.artificial_build_data[0:500000]  # typical max arrays size
print(trs.artificial_build_data.size, trs.artificial_build_data)


def load_save_numpy():
    store_raw_data_to = 'E:\\Workspace\\deleteMe\\Numpy_raw_data_test.npraw'
    np.savetxt(store_raw_data_to, trs.artificial_build_data, fmt='%d', delimiter='\t')

    with open(store_raw_data_to, 'r') as f:
        ret_raw = np.loadtxt(f, dtype=np.uint32)
    f.close()
    return ret_raw


def load_save_pickle():
    store_raw_data_to = 'E:\\Workspace\\deleteMe\\pickle_raw_data_test.raw'
    with open(store_raw_data_to, 'wb') as f:
        pickle.dump(trs.artificial_build_data, f)
    f.close()

    with open(store_raw_data_to, 'rb') as f:
        ret_raw = pickle.load(f)
    f.close()
    return ret_raw


def load_save_h5py():
    store_raw_data_to = 'E:\\Workspace\\deleteMe\\hdf5_raw_data_test.hdf5'
    f_raw = h5py.File(store_raw_data_to, 'w')
    f_raw.create_dataset('raw_data', data=trs.artificial_build_data)
    f_raw.close()

    f_load = h5py.File(store_raw_data_to, 'r')
    ret = deepcopy(f_load['raw_data'].value)
    f_load.close()
    # print(f_load['raw_data'].value)
    return ret


def load_save_json():
    store_raw_data_to = 'E:\\Workspace\\deleteMe\\json_raw_data_test.jraw'
    j_dat = trs.artificial_build_data.tolist()
    with open(store_raw_data_to, 'w') as f:
        json.dump(j_dat, f)

    with open(store_raw_data_to, 'r') as read_f:
        read_j = json.load(read_f)
        ret = np.array(read_j)

    return ret

# # print('build data == loaded data? ? : ', trs.artificial_build_data == ret_raw)
# print('build data == loaded data? ? : ', np.alltrue(trs.artificial_build_data == load_save_h5py()))
# print('json, build data == loaded data? ? : ', np.alltrue(trs.artificial_build_data == load_save_json()))

# if __name__ == '__main__':
#     # ret = load_save_pickle()
#     # print(ret == trs.artificial_build_data, ret)
#     # for 500000 elements (typicla max array size:
#     print('numpy took:',
#           timeit.timeit("load_save_numpy()", setup="from __main__ import load_save_numpy", number=10))
#     # 5000 ms/save_load, filesize: 5MB
#     print('pickle took:',
#           timeit.timeit("load_save_pickle()", setup="from __main__ import load_save_pickle", number=1000))
#     # 6.5 ms/save_load, filesize: 1.9MB
#     print('hdf5 took:',
#           timeit.timeit("load_save_h5py()", setup="from __main__ import load_save_h5py", number=1000))
#     # 6.0 ms/save_load, filesize: 1.9MB
#     print('json took:',
#           timeit.timeit("load_save_json()", setup="from __main__ import load_save_json", number=10))
#     # 1284 ms/save_load, filesize: 5.6MB
