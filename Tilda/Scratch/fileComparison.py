"""
module for comparing the time resolution of two files
"""

import os

import numpy as np

from Tilda.PolliFit.Measurement import XMLImporter

file_dir = 'C:/TildaDebug/sums'
sample_file = 'test_trsdummy_run243.xml'
test_files = os.listdir(file_dir)
print(os.listdir(file_dir))

sample_meas = XMLImporter(os.path.join(file_dir, sample_file))
res_lis = []
not_equal = []
for comp_file in test_files:
    if comp_file != sample_file:
        comp_meas = XMLImporter(os.path.join(file_dir, comp_file))
        same = np.array_equal(comp_meas.time_res_zf, sample_meas.time_res_zf)
        res_lis.append((comp_file, same))
        if not same:
            not_equal.append((comp_file, same))
print('all')
print(res_lis)
print('not equal:')
print(not_equal)
