"""
Created on 21/07/2016

@author: sikaufma

Module Description:

Analysis module to investigate what happens when kepco scan fails.

"""


import os
import numpy as np
import Tilda.Service.FileOperations.FolderAndFileHandling as FileHandle
import Tilda.Service.Formating as Form
import Tilda.Service.VoltageConversions.VoltageConversions as VCon

np.set_printoptions(threshold=np.inf)
raw_base = 'D:\\Sn_beamtime_Tilda_active_data\\raw'
good_kepco = os.path.join(raw_base, '_Kepco_fl1_kepco_029_000.raw')
bad_kepco = os.path.join(raw_base, '_Kepco_fl1_kepco_031_000.raw')

good_kepco_data_stream = FileHandle.loadPickle(good_kepco)
good_kepco_data_stream = np.asarray([Form.split_32b_data(i) if isinstance(i, int) else i for i in good_kepco_data_stream])

print(good_kepco_data_stream)

print('____________________________________________ \n \n \n')

bad_kepco_data_stream = FileHandle.loadPickle(bad_kepco)
bad_kepco_data_stream = np.asarray([Form.split_32b_data(i) if isinstance(i, int) else i for i in bad_kepco_data_stream])
print(bad_kepco_data_stream)
