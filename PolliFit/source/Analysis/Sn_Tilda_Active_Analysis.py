"""
Created on 13.07.2016

@author: simkaufm

Module Description: Script to analyse the files gained via tilda active during the Sn run @COLLAPS
"""
import os

import Analyzer
import BatchFit
import Tools

workdir = "R:\Projekte\COLLAPS\Sn\Measurement_and_Analysis_Simon\Sn_beamtime_Tilda_active_data"
db = os.path.join(workdir, os.path.split(workdir)[1] + '.sqlite')

# run = 'Run0'

# ''' Kepco Scans: '''
# fl_1 = Tools.fileList(db, 'Kepco_fl1')
# fl_2 = Tools.fileList(db, 'Kepco_fl2')
# fl_3 = Tools.fileList(db, 'kepco_fl3')
#
# fl_1.pop(fl_1.index('Kepco_fl1_kepco_001.xml'))  # grounding pin was inserted
#
# fl_1 = [os.path.join(workdir, 'sums', i) for i in fl_1]
# fl_2 = [os.path.join(workdir, 'sums', i) for i in fl_2]
# fl_3 = [os.path.join(workdir, 'sums', i) for i in fl_3]
# print('fl3', fl_3)
# kepco_flat = fl_1 + fl_2 + fl_3
#
# ''' correct the x-axis: '''
# # this needs to be done, because the stepsize and number of steps was unfortunately choosen in that way,
# # that the resulting DAC-Register (0 + 328 * 800 = 262400) was larger than the maximum (2 ** 18 - 1 = 262143)
# # therefore the last step was coerced to 262143 which results in a smaller stepsize for this one.
# # this can be corrected by taking the x-axis in units of dac-registers and than converting it to voltages
# # inside the conversion the dac_register gets coerced to the range 0, 262143.
#
# def correct_x_and_fit(path):
#     file = os.path.split(path)[1]
#     meas = XMLImporter(path, False)
#     meas.x[0] = np.asarray([VCon.get_voltage_from_18bit(meas.x[0][i]) for i, j in enumerate(meas.x[0])])
#     meas.preProc(db)
#     if np.isnan(meas.cts[0]).sum() > 0:
#         print('nan detected, deleting nan')
#         meas.x[0] = meas.x[0][:-1]  # deleting first element from x axis because this one was not measured.
#         new_cts = meas.cts[0][0][:-1]  # deleting last element, because this should be the nan
#         new_err = meas.err[0][0][:-1]  # deleting last element, because this should be the nan
#         meas.cts = [[[]]]
#         meas.err = [[[]]]
#         meas.cts[0][0] = new_cts
#         meas.err[0][0] = new_err
#     spec = Straight()
#     spec.evaluate(meas.x[0][-1], (0, 1))
#     fitter = SPFitter(spec, meas, ([0], 0))
#     fitter.fit()
#     plot.plotFit(fitter)
#     plot.get_current_axes().legend([file])
#
#     plot.show(True)
#     # Create and save graph
#     fig = os.path.splitext(path)[0] + run + '.png'
#     plot.plotFit(fitter)
#     plot.save(fig)
#     plot.clear()
#
#     result = fitter.result()
#
#     con = sqlite3.connect(db)
#     cur = con.cursor()
#
#     for r in result:
#         # Only one unique result, according to PRIMARY KEY, thanks to INSERT OR REPLACE
#         cur.execute('''INSERT OR REPLACE INTO FitRes (file, iso, run, rChi, pars)
#             VALUES (?, ?, ?, ?, ?)''', (file, r[0], run, fitter.rchi, repr(r[1])))
#     con.commit()
#     con.close()
#     print("Finished fitting", file)
#
# for i in kepco_flat:
#     path = i
#     correct_x_and_fit(path)
#
# ''' insert results from kepco scan and wavelength to other files: '''
# con = sqlite3.connect(db)
# cur = con.cursor()
# cur.execute(
#     '''UPDATE Files SET laserFreq=662929863.205568,
#     voltDivRatio="{'accVolt':10000, 'offset':991.6}",
#     lineMult=0.050417, lineOffset=0.000009, accVolt=4.0024713'''
# )
# con.commit()
# con.close()

''' Sn 119 '''
iso = '119_Sn'
run = 'AllPmtsManGatesSn119'
files_sn119 = Tools.fileList(db, iso)[:-1]  # not the flipped one
BatchFit.batchFit(files_sn119, db, run)
Analyzer.combineRes(iso, 'center', run, db)
Analyzer.combineRes(iso, 'Au', run, db)
ints0 = Analyzer.combineRes(iso, 'Int0', run, db)
ints1 = Analyzer.combineRes(iso, 'Int1', run, db)

int_rel = []
for i, j in enumerate(ints0[3][1]):
    int_rel.append(ints1[3][1][i] / j)
print(int_rel)


''' Sn 124 '''
iso = '124_Sn'
files_sn124 = Tools.fileList(db, iso)
files_sn124.pop(files_sn124.index('124_Sn_trs_000.xml'))  # timing was wrong
files_sn124.pop(files_sn124.index('124_Sn_trs_001.xml'))  # timing was wrong
run = 'AllPmtsManGatesSn124'
BatchFit.batchFit(files_sn124, db, run)
Analyzer.combineRes(iso, 'center', run, db)




''' Sn 126 '''
iso = '126_Sn'
files_sn126 = Tools.fileList(db, iso)
run = 'AllPmtsManGatesSn126'
BatchFit.batchFit(files_sn126, db, run)
Analyzer.combineRes(iso, 'center', run, db)


''' isotope shifts: '''
run = 'AllPmtsManGatesSn119'
Analyzer.combineShift('119_Sn', run, db)
run = 'AllPmtsManGatesSn126'
Analyzer.combineShift('126_Sn', run, db)




