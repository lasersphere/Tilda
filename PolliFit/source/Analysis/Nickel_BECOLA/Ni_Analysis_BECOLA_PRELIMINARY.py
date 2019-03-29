"""
Created on 2018-12-19

@author: fsommer

Module Description:  Analysis of the Nickel Data from BECOLA taken on 13.04.-23.04.2018
"""

import ast
import math
import os
import sqlite3
from datetime import datetime
import re

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import Analyzer
import BatchFit
import MPLPlotter
import Physics
import TildaTools as TiTs
import Tools
from KingFitter import KingFitter

''' working directory: '''
workdir = 'C:\\DEVEL\\Analysis\\Ni_Analysis\\XML_Data'
''' data folder '''
datafolder = os.path.join(workdir, 'Sums')
''' database '''
db = os.path.join(workdir, 'Ni_Becola.sqlite')
Tools.add_missing_columns(db)

runs = ['CEC_AsymVoigt', 'CEC_AsymVoigt_60', 'CEC_AsymVoigt_56']

isotopes = ['%sNi' % i for i in range(55, 60)]
isotopes.remove('57Ni')
isotopes.remove('59Ni')

odd_isotopes = [iso for iso in isotopes if int(iso[:2]) % 2]
even_isotopes = [iso for iso in isotopes if int(iso[:2]) % 2 == 0]
stables = ['58Ni', '60Ni', '61Ni', '62Ni', '64Ni']

''' Masses '''
# # Reference:   'The Ame2016 atomic mass evaluation: (II). Tables, graphs and references'
# #               Chinese Physics C Vol.41, No.3 (2017) 030003
# #               Meng Wang, G. Audi, F.G. Kondev, W.J. Huang, S. Naimi, Xing Xu
# masses = {
#     '55Ni': (54951330.0, 0.8),
#     '56Ni': (55942127.9, 0.5),
#     '57Ni': (56939791.5, 0.6),
#     '58Ni': (57935341.8, 0.4),
#     '59Ni': (58934345.6, 0.4),
#     '60Ni': (59930785.3, 0.4)
#      }
# # Write masses to db:
# con = sqlite3.connect(db)
# cur = con.cursor()
# for iso, mass_tupl in masses.items():
#     cur.execute('''UPDATE Isotopes SET mass = ?, mass_d = ? WHERE iso = ? ''',
#                 (mass_tupl[0] * 10 ** -6, mass_tupl[1] * 10 ** -6, iso))
# con.commit()
# con.close()

''' Moments, Spin '''
# Reference:    "Table of Nuclear Magnetic Dipole and Electric Quadrupole Moments",
#               IAEA Nuclear Data Section, INDC(NDS)-0658, February 2014,
#               N.J.Stone
#               p.36
# magnetic dipole moment µ in units of nuclear magneton µn
# electric Quadrupolemoment Q in units of barn
# Format: {'xxNi' : (IsoMass_A, IsoSpin_I, IsoDipMom_µ, IsoDipMomErr_µerr, IsoQuadMom_Q, IsoQuadMomErr_Qerr)}
nuclear_spin_and_moments = {
    '55Ni': (55, -3/2, 0.98, 0.03, 0, 0),
    '57Ni': (57, -3/2, -0.7975, 0.0014, 0, 0)
    # even isotopes 56, 58, 60 Ni have Spin 0 and since they are all even-even nucleons also the moments are zero
}

''' A and B Factors '''
# Reference:

''' restframe transition frequency '''
# Reference: ??
# NIST: observed wavelength air 352.454nm corresponds to 850586060MHz
# upper lvl 28569.203cm-1; lower lvl 204.787cm-1
# resulting wavenumber 28364.416cm-1 corresponds to 850343800MHz
# KURUCZ database: 352.4535nm, 850344000MHz, 28364.424cm-1
# Some value I used in the excel sheet: 850347590MHz Don't remember where that came from...
restframe_trans_freq = 850343800

''' literature value IS 60-58'''
# Reference: ??
# isotope shift of Nickel-60 with respect to Nickel-58 (=fNi60-fNi58)
# Collaps 2017: 509.074(879)[7587] MHz
# Collaps 2016: 510.7(6)[95]MHz
# Steudel 1980: 0.01694(9) cm-1 corresponds to 507.8(27) MHz
literature_IS60vs58 = 510.7


''' Calibration runs '''
# Pick all 58Ni and 60Ni runs and fit.
# Basically this should do the same as batchfitting all runs in PolliFit
# So I will start by stealing the code from there and in a later version I might adapt the code for my analysis
"""
:params:    fileList: ndarray of str: names of files to be analyzed e.g. ['BECOLA_123.xml' 'BECAOLA_234.xml']
            db: str: path to database e.g. 'C:/DEVEL/Analysis/Ni_Analysis/XML_Data/Ni_Becola.sqlite'
            run: str: run as specified in database e.g.: 'CEC_AsymVoigt_60'
            x_as_voltage: bool: is unit of x-axis volts? e.g. True
            softw_gates_trs: None
            save_file_as: str: file format for saving results e.g. '.png'
:return: list, (shifts, shiftErrors, shifts_weighted_mean, statErr, systErr, rChi)
"""
###################
# select Ni58 files
con = sqlite3.connect(db)
cur = con.cursor()
cur.execute(
    '''SELECT file FROM Files WHERE type LIKE '58Ni%' ''')
files = cur.fetchall()
con.close()
# convert into np array
filelist = [f[0] for f in files]
filearray = np.array(filelist)
# do the batchfit for 58Ni
#BatchFit.batchFit(filearray, db, runs[0], x_as_voltage=True, softw_gates_trs=None, save_file_as='.png')
# get fitresults (center) vs run for 58
all_58_center_MHz = []
for files in filelist:
    con = sqlite3.connect(db)
    cur = con.cursor()
    # Get corresponding isotope
    cur.execute(
        '''SELECT type FROM Files WHERE file = ? ''', (files,))
    iso_type = cur.fetchall()[0][0]
    # Query fitresults for file and isotope combo
    cur.execute(
        '''SELECT pars FROM FitRes WHERE file = ? AND iso = ? ''', (files, iso_type))
    pars = cur.fetchall()
    con.close()
    parsdict = ast.literal_eval(pars[0][0])
    all_58_center_MHz.append(parsdict['center'][0])
runNos58 = []
for files in filelist:
    file_no = int(re.split('[_.]', files)[1])
    runNos58.append(file_no)

###################
# select Ni60 files
con = sqlite3.connect(db)
cur = con.cursor()
cur.execute(
    '''SELECT file FROM Files WHERE type LIKE '60Ni%' ''')
files = cur.fetchall()
con.close()
# convert into np array
filelist = [f[0] for f in files]
filearray = np.array(filelist)
# do the batchfit for 58Ni
#BatchFit.batchFit(filearray, db, runs[1], x_as_voltage=True, softw_gates_trs=None, save_file_as='.png')
# get fitresults (center) vs run for 60
all_60_center_MHz = []
for files in filelist:
    con = sqlite3.connect(db)
    cur = con.cursor()
    # Get corresponding isotope
    cur.execute(
        '''SELECT type FROM Files WHERE file = ? ''', (files,))
    iso_type = cur.fetchall()[0][0]
    # Query fitresults for file and isotope combo
    cur.execute(
        '''SELECT pars FROM FitRes WHERE file = ? AND iso = ? ''', (files, iso_type))
    pars = cur.fetchall()
    con.close()
    parsdict = ast.literal_eval(pars[0][0])
    all_60_center_MHz.append(parsdict['center'][0]-510)
runNos60 = []
for files in filelist:
    file_no = int(re.split('[_.]', files)[1])
    runNos60.append(file_no)

# plot center frequency in MHz for all 85Ni runs:
plt.plot(runNos60, all_60_center_MHz, '--o', color='red', label='60Ni - 510MHz')
plt.plot(runNos58, all_58_center_MHz, '--o', color='blue', label='58Ni')
plt.title('Center Frequency FitPar in MHz for all 58,60 Ni Runs')
plt.xlabel('run numbers')
plt.ylabel('center fit parameter [MHz]')
plt.legend(loc='best')
#plt.xticks(range(len(yData)), runNos, rotation=-30)
plt.show()

##################
# Calibration sets of 58/60Ni
calib_tuples = [(6191, 6192), (6207, 6208), (6224, 6225), (6232, 6233), (6242, 6243), (6253, 6254), (6258, 6259),
                (6269, 6270), (6284, 6285), (6294, 6295), (6301, 6302), (6310, 6311), (6313, 6312), (6323, 6324),
                (6340, 6342), (6356, 6357), (6362, 6363), (6395, 6396), (6417, 6419), (6467, 6466), (6501, 6502)]
calib_tuples_with_isoshift = []

for tuples in calib_tuples:
    # Get 58Nickel center fit parameter in MHz
    run58 = tuples[0]
    run58file = 'BECOLA_'+str(run58)+'.xml'
    con = sqlite3.connect(db)
    cur = con.cursor()
    # Get corresponding isotope
    cur.execute(
        '''SELECT type FROM Files WHERE file = ? ''', (run58file,))
    iso_type58 = cur.fetchall()[0][0]
    # Query fitresults for file and isotope combo
    cur.execute(
        '''SELECT pars FROM FitRes WHERE file = ? AND iso = ? ''', (run58file, iso_type58))
    pars58 = cur.fetchall()
    con.close()
    pars58dict = ast.literal_eval(pars58[0][0])
    center58 = pars58dict['center']

    # Get 60Nickel center fit parameter in MHz
    run60 = tuples[1]
    run60file = 'BECOLA_' + str(run60) + '.xml'
    con = sqlite3.connect(db)
    cur = con.cursor()
    # Get corresponding isotope
    cur.execute(
        '''SELECT type FROM Files WHERE file = ? ''', (run60file,))
    iso_type60 = cur.fetchall()[0][0]
    # Query fitresults for file and isotope combo
    cur.execute(
        '''SELECT pars FROM FitRes WHERE file = ? AND iso = ? ''', (run60file, iso_type60))
    pars60 = cur.fetchall()
    con.close()
    pars60dict = ast.literal_eval(pars60[0][0])
    center60 = pars60dict['center']

    # Calculate isotope shift of 60Ni with respect to 58Ni for this calibration point
    isoShift = center60[0]-center58[0]
    print('Isotope shift for calibration point with runs {} and {}: {}MHz'.format(tuples[0], tuples[1], isoShift))
    tuple_with_isoshift = tuples + (isoShift,)
    calib_tuples_with_isoshift.append(tuple_with_isoshift)

# plot isotope shift for all calibration points (can be removed later on):
calib_isoShift_yData = []
calib_point_runNos = []
for tuples in calib_tuples_with_isoshift:
    calib_isoShift_yData.append(tuples[2])
    calib_point_name = str(tuples[0])+'/'+str(tuples[1])
    calib_point_runNos.append(calib_point_name)

plt.plot(range(len(calib_isoShift_yData)), calib_isoShift_yData, '-o')
plt.xticks(range(len(calib_isoShift_yData)), calib_point_runNos, rotation=-30)
plt.title('Isotope Shift for all Calibration Points')
plt.xlabel('Run Numbers of Calibration Pairs')
plt.ylabel('Isotope Shift 60-58 Ni [MHz]')
plt.show()

# Calculate resonance DAC Voltage from the 'center' positions
calib_tuples_with_isoshift_and_calibrationvoltage = []
average_calib_voltage = 0
for tuples in calib_tuples_with_isoshift:
    # get filenames
    run58, run60, isoShift = tuples
    run58file = 'BECOLA_' + str(run58) + '.xml'
    run60file = 'BECOLA_' + str(run60) + '.xml'
    calib_point_dict = {run58file: {},
                        run60file: {}}

    # calculate centerDAC and get some usefull info
    for files, dicts in calib_point_dict.items(): # only 2 elements to iterate: 58 and 60
        con = sqlite3.connect(db)
        cur = con.cursor()
        # get laser frequency and accelVolt
        cur.execute(
            '''SELECT type, accVolt, laserFreq, colDirTrue FROM Files WHERE file = ? ''', (files,))
        iso, accVolt, laserFreq, colDirTrue = cur.fetchall()[0]
        # Query fitresults for file and isotope combo
        cur.execute(
            '''SELECT pars FROM FitRes WHERE file = ? AND iso = ? ''', (files, iso))
        pars = cur.fetchall()
        # get mass
        cur.execute(
            '''SELECT mass FROM Isotopes WHERE iso = ? ''', (iso,))
        isoMass = cur.fetchall()[0][0]
        con.close()

        # write to dict
        dicts['filename'] = files
        dicts['iso'] = iso
        dicts['accVolt'] = float(accVolt)
        dicts['laserFreq'] = float(laserFreq)
        dicts['colDirTrue'] = colDirTrue
        parsDict = ast.literal_eval(pars[0][0])
        center = parsDict['center'][0]
        dicts['center'] = float(center)
        dicts['isoMass'] = float(isoMass)

        # calculate resonance frequency
        dicts['resonanceFreq'] = restframe_trans_freq + dicts['center']
        # calculate relative velocity
        relVelocity = Physics.invRelDoppler(dicts['laserFreq'], dicts['resonanceFreq'])
        # calculate relativistic energy of the beam particles at resonance freq and thereby resonance Voltage
        centerE = Physics.relEnergy(relVelocity, dicts['isoMass'] * Physics.u)/Physics.qe
        # get DAC resonance voltage
        centerDAC = centerE - dicts['accVolt']

        dicts['centerDAC'] = centerDAC

    # do voltage calibration to literature IS
    accVolt = calib_point_dict[run58file]['accVolt'] # should be the same for 58 and 60
    voltage_list = np.arange(accVolt-100, accVolt+100)
    IS_perVolt_list = np.zeros(0) # isotope shift for an assumed voltage from voltage list
    for volt in voltage_list:
        # calculate velocity for 58 and 60
        velo58sign = -1 if calib_point_dict[run58file]['colDirTrue'] else 1
        velo58 = velo58sign * Physics.relVelocity((volt + calib_point_dict[run58file]['centerDAC'])*Physics.qe,
                                     calib_point_dict[run58file]['isoMass']*Physics.u)
        velo60sign = -1 if calib_point_dict[run60file]['colDirTrue'] else 1
        velo60 = velo60sign * Physics.relVelocity((volt + calib_point_dict[run60file]['centerDAC']) * Physics.qe,
                                     calib_point_dict[run60file]['isoMass']*Physics.u)
        # calculate resonance frequency for 58 and 60
        f_reso58 = Physics.relDoppler(calib_point_dict[run58file]['laserFreq'], velo58)
        f_reso60 = Physics.relDoppler(calib_point_dict[run60file]['laserFreq'], velo60)
        # calculate isotope shift
        isoShift = f_reso60 - f_reso58
        IS_perVolt_list = np.append(IS_perVolt_list, np.array(isoShift))

    # calibrate voltage by fitting line to plot
    IS_perVolt_list -= literature_IS60vs58
    fitpars0 = np.array([0.0, 0.0])
    def linfunc(x, m, b):
        return m*x+b
    fitres = curve_fit(linfunc, voltage_list, IS_perVolt_list, fitpars0)
    m = fitres[0][0]
    b = fitres[0][1]
    calibrated_voltage = -b/m
    tuple_withcalibvolt = tuples + (calibrated_voltage,)
    calib_tuples_with_isoshift_and_calibrationvoltage.append(tuple_withcalibvolt)

    average_calib_voltage += calibrated_voltage
    print(calibrated_voltage)

    # display calibration graph
    #plt.plot(voltage_list, IS_perVolt_list)
    #plt.scatter(calibrated_voltage, m * calibrated_voltage + b)
    #plt.title('Voltage Calibration for Calibration Tuple [Ni58:{}/Ni60:{}]'.format(run58, run60))
    #plt.xlabel('voltage [V]')
    #plt.ylabel('isotope shift [MHz]')
    #plt.show()
average_calib_voltage = average_calib_voltage/len(calib_tuples)

print(calib_tuples_with_isoshift_and_calibrationvoltage)
print(average_calib_voltage)

# display all voltage calibrations
# plot isotope shift for all calibration points (can be removed later on):
calib_voltages = []
calib_point_runNos = []
for tuples in calib_tuples_with_isoshift_and_calibrationvoltage:
    calib_voltages.append(tuples[3])
    calib_point_name = str(tuples[0])+'/'+str(tuples[1])
    calib_point_runNos.append(calib_point_name)

plt.plot(range(len(calib_voltages)), calib_voltages, '-o')
plt.plot(range(len(calib_voltages)), [29850]*len(calib_voltages), '-o', color='red')
plt.ylim(bottom=29840)
plt.xticks(range(len(calib_voltages)), calib_point_runNos, rotation=-30)
plt.title('Calibrated Voltage for all Calibration Tuples')
plt.xlabel('Run Numbers of Calibration Pairs')
plt.ylabel('Voltage [V]')
plt.show()

# Write calibrations to XML database
print('Updating db with new voltages now...')
for entries in calib_tuples_with_isoshift_and_calibrationvoltage:
    calibration_name = str(entries[0]) + 'w' +  str(entries[1])
    file58 = 'BECOLA_' + str(entries[0]) + '.xml'
    file58_newType = '58Ni_cal' + calibration_name
    file60 = 'BECOLA_' + str(entries[1]) + '.xml'
    file60_newType = '60Ni_cal' + calibration_name
    new_voltage = entries[3]

    # Update 'Files' in db
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute('''UPDATE Files SET accVolt = ?, type = ? WHERE file = ? ''', (new_voltage, file58_newType, file58))
    cur.execute('''UPDATE Files SET accVolt = ?, type = ? WHERE file = ? ''', (new_voltage, file60_newType, file60))
    con.commit()
    con.close()

    # Calculate differential Doppler shift for re-assigning center fit pars
    diff_Doppler_58 = Physics.diffDoppler(restframe_trans_freq, new_voltage, 58)
    diff_Doppler_60 = Physics.diffDoppler(restframe_trans_freq, new_voltage, 60)
    # Create new isotopes in db
    con = sqlite3.connect(db)
    cur = con.cursor()
    # create new 58 calibration isotope
    cur.execute('''SELECT * FROM Isotopes WHERE iso = ? ''', ('58Ni',))  # get original isotope to copy from
    mother_isopars = cur.fetchall()
    center58 = mother_isopars[0][4]
    new_center58 = center58 + (29850 - new_voltage) * diff_Doppler_58
    isopars_lst = list(mother_isopars[0])  # change into list to replace some values
    isopars_lst[0] = file58_newType
    isopars_lst[4] = new_center58
    new_isopars = tuple(isopars_lst)
    cur.execute('''INSERT OR REPLACE INTO Isotopes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                new_isopars)
    # create new 60 calibration isotope
    cur.execute('''SELECT * FROM Isotopes WHERE iso = ? ''', ('60Ni',))  # get original isotope to copy from
    mother_isopars = cur.fetchall()
    center60 = mother_isopars[0][4]
    new_center60 = center60 + (29850 - new_voltage) * diff_Doppler_60
    isopars_lst = list(mother_isopars[0])  # change into list to replace some values
    isopars_lst[0] = file60_newType
    isopars_lst[4] = new_center60
    new_isopars = tuple(isopars_lst)
    print(new_isopars)
    cur.execute('''INSERT OR REPLACE INTO Isotopes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                new_isopars)
    con.commit()
    con.close()

print('...db update completed!')


###################
# select Ni56 files
con = sqlite3.connect(db)
cur = con.cursor()
cur.execute(
    '''SELECT file FROM Files WHERE type LIKE '56Ni%' ''')
files = cur.fetchall()
con.close()
# convert into np array
filelist56 = [f[0] for f in files]
filearray56 = np.array(filelist56)

# attach the Ni56 runs to some calibration point(s) and adjust voltage plus create new isotope with adjusted center
files56_withReference_tuples = []  # tuples of (56file, (58reference, 60reference))
# hand-assigned calibration runs
files56_withReference_tuples_handassigned = [(6199, (6191, 6192)), (6202, (6191, 6192)), (6203, (6191, 6192)), (6204, (6191, 6192)),
                                             (6211, (6224, 6225)), (6213, (6224, 6225)), (6214, (6224, 6225)),
                                             (6238, (6242, 6243)), (6239, (6242, 6243)), (6240, (6242, 6243)),
                                             (6251, (6253, 6254)), (6252, (6253, 6254))]
files56_withReference_tuples_handassigned_V2 = [(6202, (6207, 6208)), (6203, (6207, 6208)), (6204, (6207, 6208)),
                                             (6211, (6207, 6208)), (6213, (6207, 6208)), (6214, (6207, 6208)),
                                             (6238, (6242, 6243)), (6239, (6242, 6243)), (6240, (6242, 6243)),
                                             (6251, (6253, 6254)), (6252, (6253, 6254))]
for files in filearray56:
    # extract file number
    file_no = int(re.split('[_.]',files)[1])
    # find nearest calibration tuple
    #nearest_calib = (0, 0)
    #for calibs in calib_tuples:
        #nearest_calib = calibs if abs(calibs[0] - file_no) < abs(nearest_calib[0]-file_no) else nearest_calib
    #files56_withReference_tuples.append((file_no, (nearest_calib[0], nearest_calib[1])))
    # navigate to 58Ni reference file in db
    # calib_file_58 = 'BECOLA_'+str(nearest_calib[0])+'.xml'
    calibration_tuple = ()  # for hand assigned
    for refs in files56_withReference_tuples_handassigned:  # for hand assigned
        if refs[0] == file_no:
            calibration_tuple = (refs[1][0], refs[1][1])
    calib_file_58 = 'BECOLA_' + str(calibration_tuple[0]) + '.xml'  # for hand assigned
    con = sqlite3.connect(db)
    cur = con.cursor()
    # extract voltage from calibration
    cur.execute(
        '''SELECT accVolt FROM Files WHERE file = ? ''', (calib_file_58, ))
    accVolt_calib = cur.fetchall()[0][0]
    # write new voltage for 56Ni file + create name and insert new isotope type
    # calibration_name = str(nearest_calib[0]) + 'w' + str(nearest_calib[1])
    calibration_name = str(calibration_tuple[0]) + 'w' + str(calibration_tuple[1])
    file56_newType = '56Ni_cal'+ calibration_name
    cur.execute('''UPDATE Files SET accVolt = ?, type = ? WHERE file = ? ''', (accVolt_calib, file56_newType, files))
    con.commit()
    con.close()
    # calculate center shift and create new isotope for fitting
    # Calculate differential Doppler shift for re-assigning center fit pars
    diff_Doppler_56 = Physics.diffDoppler(restframe_trans_freq, accVolt_calib, 56)
    # Create new isotopes in db
    con = sqlite3.connect(db)
    cur = con.cursor()
    # create new 56 calibrated isotope
    cur.execute('''SELECT * FROM Isotopes WHERE iso = ? ''', ('56Ni',))  # get original isotope to copy from
    mother_isopars = cur.fetchall()
    center56 = mother_isopars[0][4]
    new_center56 = center56 + (29850 - accVolt_calib) * diff_Doppler_56
    isopars_lst = list(mother_isopars[0])  # change into list to replace some values
    isopars_lst[0] = file56_newType
    isopars_lst[4] = new_center56
    new_isopars = tuple(isopars_lst)
    print(new_isopars)
    cur.execute('''INSERT OR REPLACE INTO Isotopes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                new_isopars)
    con.commit()
    con.close()

files56_withReference_tuples = files56_withReference_tuples_handassigned

# do the batchfit for 56Ni
BatchFit.batchFit(filearray56, db, runs[2], x_as_voltage=True, softw_gates_trs=None, save_file_as='.png')

# calculate isotope shift between 56file and reference
files56_withReference_andIsoshift_tuples = []
for files56 in filearray56:
    # extract file number
    file_no = int(re.split('[_.]', files56)[1])
    # Get 56Nickel center fit parameter in MHz
    con = sqlite3.connect(db)
    cur = con.cursor()
    # Get corresponding isotope
    cur.execute(
        '''SELECT type FROM Files WHERE file = ? ''', (files56,))
    iso_type56 = cur.fetchall()[0][0]
    # Query fitresults for file and isotope combo
    cur.execute(
        '''SELECT pars FROM FitRes WHERE file = ? AND iso = ? ''', (files56, iso_type56))
    pars56 = cur.fetchall()
    con.close()
    pars56dict = ast.literal_eval(pars56[0][0])
    center56 = pars56dict['center'] # tuple of (center frequency, Uncertainty?, Fixed?)

    # Get reference 58Nickel center fit parameter in MHz,
    calibration_tuple = ()
    ref58_file = ''
    for refs in files56_withReference_tuples:
        if refs[0] == file_no:
            calibration_tuple = (refs[1][0], refs[1][1])
            ref58_file = 'BECOLA_'+ str(refs[1][0])+'.xml'
    con = sqlite3.connect(db)
    cur = con.cursor()
    # Get corresponding isotope
    cur.execute(
        '''SELECT type FROM Files WHERE file = ? ''', (ref58_file,))
    iso_type58 = cur.fetchall()[0][0]
    # Query fitresults for file and isotope combo
    cur.execute(
        '''SELECT pars FROM FitRes WHERE file = ? AND iso = ? ''', (ref58_file, iso_type58))
    pars58 = cur.fetchall()
    con.close()
    pars58dict = ast.literal_eval(pars58[0][0])
    center58 = pars58dict['center']

    # calculate isotope shift of 56 nickel with reference to 58 nickel
    isoShift56 = center56[0] - center58[0]
    print('Isotope shift of Ni56 run {} with respect to Ni58 run{} is: {}MHz'.
          format(file_no, calibration_tuple, isoShift56))
    files56_withReference_andIsoshift_tuples.append((file_no, calibration_tuple, isoShift56))


# plot isotope shift for all 56 nickel runs (can be removed later on):
ni56_isoShift_yData = []
ni56_point_runNos = []
for tuples in files56_withReference_andIsoshift_tuples:
    ni56_isoShift_yData.append(tuples[2])
    ni56_point_name = str(tuples[0])
    ni56_point_runNos.append(ni56_point_name)

ni56_isoShift_alt_yData = -525.8169055478309, -523.0479365515923, -525.2808338260361, -533.4630219328083,\
                          -540.3585627829973, -521.3067663175245, -509.42569032109384, -511.40285471674554,\
                          -511.1400904483909, -508.7950760887162, -511.4211280594908

plt.plot(range(len(ni56_isoShift_yData)), ni56_isoShift_yData, '-o', label='preferred')
plt.plot(range(len(ni56_isoShift_yData)), ni56_isoShift_alt_yData, 'r-o', label='alternative')
plt.xticks(range(len(ni56_isoShift_yData)), ni56_point_runNos, rotation=-30)
plt.title('Isotope Shift Ni 56-58 for all runs')
plt.xlabel('Run Number')
plt.ylabel('Isotope Shift  [MHz]')
plt.legend(loc='lower right')
plt.show()