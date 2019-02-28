"""
Created on 15.02.19

@author: simkaufm

Module Description: get the mid time of all file by using the information from the tilda passive files
"""


import os
import datetime
import sqlite3

import Tools
import Analyzer
import TildaTools as TiTs
from Measurement.XMLImporter import XMLImporter
from Measurement.MCPImporter import MCPImporter
from Analysis.Nickel.NiCombineTildaPassiveAndMCP import find_tipa_file_to_mcp_file

''' settings '''
workdir = 'E:\\Workspace\\OwnCloud\\Projekte\\COLLAPS\\Nickel\\Measurement_and_Analysis_Simon\\Ni_workspace'

mcp_file_folder = os.path.join(workdir, 'Ni_April2016_mcp')
tipa_file_folder = os.path.join(workdir, 'TiPaData')
tipa_files = sorted([file for file in os.listdir(tipa_file_folder) if file.endswith('.xml')])
tipa_files = [os.path.join(tipa_file_folder, file) for file in tipa_files]

db16 = os.path.join(workdir, 'Ni_workspace.sqlite')
Tools.add_missing_columns(db16)

# old:
# runs = ['narrow_gate', 'wide_gate']
# runs = [runs[0]]
runs = ['wide_gate_asym']

# isotopes = ['%s_Ni' % i for i in range(58, 71)]
# isotopes.remove('69_Ni')
# isotopes.remove('67_Ni')  # 67 need extrac treatment due to tracked measurement
stables = ['58_Ni', '60_Ni', '61_Ni', '62_Ni', '64_Ni']
isotopes = ['67_Ni']

''' get all mcp and tipa files '''

mcp_files = []
for iso in isotopes:
    mcp_files += Tools.fileList(db16, iso)
print(mcp_files)
db_dates = Analyzer.get_date_date_err_to_files(db16, mcp_files)  # each tuple (file, date, dateErrIns)

tipa_files = []
for mcp_file in mcp_files:
    if '.xml' in mcp_file:
        found_tipa = os.path.join(mcp_file_folder, mcp_file)
    else:
        loadedtipa_meas, found_tipa = find_tipa_file_to_mcp_file(mcp_file)
    tipa_files += found_tipa,

''' load all tipa files -> mid_time will be calculated from XMLImporter '''
time_fmt = '%Y-%m-%d %H:%M:%S'
tipa_mid_dates = []  # str of date according to the time format: time_fmt = '%Y-%m-%d %H:%M:%S'
tipa_date_errs_in_s = []

mcp_f_failed_list = []
for mcp_f, tipa_f, db_date in zip(mcp_files, tipa_files, db_dates):
    mid_date_err = db_date[2]
    mid_date = db_date[1]
    t_to_print = None
    if tipa_f is not None:
        t_to_print = os.path.split(tipa_f)[1]
    # print(mcp_f, t_to_print, mid_date, mid_date_err, type(mid_date_err))
    if tipa_f is not None and isinstance(mid_date_err, str):
        tipa_meas = XMLImporter(tipa_f)
        mid_date_err = 0.0
        if tipa_meas.date_d > 0:  # otherwise anyhow the mid date determination failed.
            mid_date_err = tipa_meas.date_d
            mid_date = tipa_meas.date
        if mid_date_err > 0:
            # only update if this is a real mid time, otherwise leave untouched, need to be dealt elsewhere
            con = sqlite3.connect(db16)
            cur = con.cursor()
            cur.execute('''UPDATE Files SET date = ?, errDateInS = ? WHERE file = ?''',
                        (mid_date, mid_date_err, mcp_f))
            con.commit()
            con.close()
    if mid_date_err is "" or mid_date_err == 0 or isinstance(mid_date_err, str):
        print('Warning, mcp_file %s is not midTime!' % mcp_f)
        mcp_f_failed_list += mcp_f,

print('failed mcp files:')
for each in mcp_f_failed_list:
    print(each)
#
# print('dates:')
# for file, date, daterr in db_dates:
#     print(file, date, daterr)
