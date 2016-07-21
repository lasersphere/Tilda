"""
Created on 21/07/2016

@author: sikaufma

Module Description:

Analysis module to investigate what happens when kepco scan fails.

"""


import os
import numpy as np
import Service.FileOperations.FolderAndFileHandling as FileHandl
import Service.Formating as Form
import Service.VoltageConversions.VoltageConversions as VCon

raw_base = 'D:\\Sn_beamtime_Tilda_active_data\\raw'
good_kepco = os.path.join(raw_base, 'kepco_Kepco_fl1_track0_000.raw')
bad_kepco = os.path.join(raw_base, 'kepco_Kepco_fl1_track0_002.raw')

good_kepco_data_stream = FileHandl.loadPickle(good_kepco)
good_kepco_data_stream = np.asarray([Form.split_32b_data(i) if isinstance(i, int) else i for i in good_kepco_data_stream])

print(good_kepco_data_stream)

print('____________________________________________ \n \n \n')

bad_kepco_data_stream = FileHandl.loadPickle(bad_kepco)
bad_kepco_data_stream = np.asarray([Form.split_32b_data(i) if isinstance(i, int) else i for i in bad_kepco_data_stream])
print(bad_kepco_data_stream)
