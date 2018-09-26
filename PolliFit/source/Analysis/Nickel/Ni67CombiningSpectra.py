"""
Created on 

@author: simkaufm

Module Description:  This will be used to combine certain files of Ni67 because this was a partially tracked measurement.

"""

import os

import Tools
from Measurement import MeasLoad

workdir = 'R:\\Projekte\\COLLAPS\\Nickel\\Measurement_and_Analysis_Simon\\Ni_workspace'

datafolder = os.path.join(workdir, 'Ni_April2016_mcp')

db = os.path.join(workdir, 'Ni_workspace.sqlite')

files_67 = Tools.fileList(db, '67_Ni')
print(files_67)

single_peak_files_left = ['67Ni_no_protonTrigger_leftest_peak_Run061.mcp',
                          '67Ni_no_protonTrigger_leftest_peak_Run063.mcp']
# original files:
# single_peak_files_middle = ['67Ni_no_protonTrigger_middle_peak_Run062.mcp',
#                             '67Ni_no_protonTrigger_middle_peak_Run64.mcp',
#                             '67Ni_no_protonTrigger_middle_peak_Run65.mcp']
# added run064 & run065 to a new file:
single_peak_files_middle = ['67Ni_no_protonTrigger_middle_peak_Run062.mcp',
                            '67Ni_no_protonTrigger_middle_peak_Run64_sum_Run065.xml']

single_peak_files_left_full_path = [os.path.join(datafolder, file) for file in single_peak_files_left]
single_peak_files_middle_full_path = [os.path.join(datafolder, file) for file in single_peak_files_middle]

runs = ['narrow_gate', 'wide_gate']
runs = [runs[0]]


meas_left_pk = [MeasLoad.load(selected_file, db, raw=True) for selected_file in single_peak_files_left_full_path]
meas_middle_pk = [MeasLoad.load(selected_file, db, raw=True) for selected_file in single_peak_files_middle_full_path]
# print(meas_singl.activePMTlist)
# fit = InteractiveFit(os.path.split(selected_file)[1], db, 'narrow_gate_67_Ni')

''' now fiddeling those together '''
# adding files with same x-axis to a new one:
# new = TildaTools.add_specdata(meas_middle_pk[-2], [(1, meas_middle_pk[-1])],
#                               datafolder, '67Ni_no_protonTrigger_middle_peak_Run64_sum_Run065')
# TildaTools.create_scan_dict_from_spec_data()
# TildaTools.save_spec_data()

# adding the two tracks to a new file:
# combine run061 & run062 and run063 & (run064 + run065) each to a new file:

# for i in [0, 1]:
#     filename = ['67Ni_sum_Run061_Run062.xml', '67Ni_sum_Run063_Run064_Run065.xml'][i]
#     new_file_path = os.path.join(datafolder, filename)
#     meas_left_pk[i].nrTracks = 2
#     meas_left_pk[i].cts.append(meas_middle_pk[i].cts[0])
#     for tr_ind in range(meas_left_pk[i].nrTracks):
#         meas_left_pk[i].cts[tr_ind] = np.array(meas_left_pk[i].cts[tr_ind])
#     meas_left_pk[i].x.append(meas_middle_pk[i].x[0])
#     if i != 0:  # for xml files it is named active_pmt_list.
#         meas_left_pk[i].activePMTlist.append(meas_middle_pk[i].active_pmt_list[0])
#     else:
#         meas_left_pk[i].activePMTlist.append(meas_middle_pk[i].activePMTlist[0])
#     meas_left_pk[i].nrScalers.append(meas_middle_pk[i].nrScalers[0])
#     meas_left_pk[i].nrScans.append(meas_middle_pk[i].nrScans[0])
#     meas_left_pk[i].accVolt = np.mean([meas_left_pk[i].accVolt, meas_middle_pk[i].accVolt])
#     meas_left_pk[i].offset = np.mean([meas_left_pk[i].offset, meas_middle_pk[i].offset])
#     # delete first scalers, because they are not used and the tracked measurements don't have them.
#     for tr_ind in range(meas_left_pk[i].nrTracks):
#         meas_left_pk[i].cts[tr_ind] = meas_left_pk[i].cts[tr_ind][4:]
#         meas_left_pk[i].activePMTlist[tr_ind] = meas_left_pk[i].activePMTlist[tr_ind][4:]
#         meas_left_pk[i].nrScalers[tr_ind] -= 4
#
#     scan_d = TildaTools.create_scan_dict_from_spec_data(meas_left_pk[i], new_file_path)
#     TildaTools.createXmlFileOneIsotope(scan_d, 'cs', new_file_path)
#     TildaTools.save_spec_data(meas_left_pk[i], scan_d)
