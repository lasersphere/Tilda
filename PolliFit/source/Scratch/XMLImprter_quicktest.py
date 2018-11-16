"""
Created on 

@author: simkaufm

Module Description:
"""
import os
from Measurement.XMLImporter import XMLImporter
import TildaTools as TiTs


workdir = 'E:\\Workspace\\OwnCloud\\Projekte\\COLLAPS\\Nickel\\' \
          'Measurement_and_Analysis_Simon\\Ni_workspace2017\\Ni_2017'

datafolder = os.path.join(workdir, 'sums')

db = os.path.join(workdir, 'Ni_2017.sqlite')

files = os.listdir(datafolder)
# ni58_files = [file for file in files if '58_Ni' in file and '.xml' in file and 'trs' in file]
ni58_files = ['58_Ni_trs_run086.xml', '58_Ni_trs_run087.xml', '58_Ni_trs_run111.xml',
              '58_Ni_trs_run112.xml', '58_Ni_trs_run229.xml', '58_Ni_trs_run230.xml']
# ni67_files = [file for file in files if '67_Ni' in file and '.xml' in file and 'trs' in file]
ni67_files = ['67_Ni_trs_run204.xml', '67_Ni_trs_run204_sum204_207.xml', '67_Ni_trs_run207.xml',
              '67_Ni_trs_run243.xml', '67_Ni_trs_run247.xml', '67_Ni_trs_run248.xml']

ni58_files = [os.path.normpath(os.path.join(datafolder, f)) for f in ni58_files]
ni67_files = [os.path.normpath(os.path.join(datafolder, f)) for f in ni67_files]

meas = XMLImporter(ni58_files[1])
for i in range(20, 54, 4):
    TiTs.calc_bunch_width_relative_to_peak_height(meas, i, show_plt=False)
TiTs.calc_bunch_width_relative_to_peak_height(meas, i + 4, show_plt=True)




