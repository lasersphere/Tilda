"""
Created on 

@author: simkaufm

Module Description: Module to read offset, accvolat and DAC-Start&-Stopp values from MCP file
and copy them to the corresponding Tilda Passive file.
"""

import ast
import logging
import os
import sys
from datetime import datetime

import Physics
import TildaTools as TiTs
import Tools
from Measurement.MeasLoad import load

logging.basicConfig(level='DEBUG', format='%(message)s', stream=sys.stdout)
sys.setrecursionlimit(2000)

''' working directory: '''

# workdir = 'R:\\Projekte\\COLLAPS\\Nickel\\Measurement_and_Analysis_Simon\\Ni_workspace'
workdir = 'E:\\Workspace\\OwnCloud\\Projekte\\COLLAPS\\Nickel\\Measurement_and_Analysis_Simon\\Ni_workspace'

mcp_file_folder = os.path.join(workdir, 'Ni_April2016_mcp')
tipa_file_folder = os.path.join(workdir, 'TiPaData')
tipa_files = sorted([file for file in os.listdir(tipa_file_folder) if file.endswith('.xml')])
tipa_files = [os.path.join(tipa_file_folder, file) for file in tipa_files]

db = os.path.join(workdir, 'Ni_workspace.sqlite')

# old:
# runs = ['narrow_gate', 'wide_gate']
# runs = [runs[0]]
runs = ['wide_gate_asym']

isotopes = ['%s_Ni' % i for i in range(58, 71)]
isotopes.remove('69_Ni')
isotopes.remove('67_Ni')  # 67 need extrac treatment due to tracked measurement
stables = ['58_Ni', '60_Ni', '61_Ni', '62_Ni', '64_Ni']

# isotopes = ['58_Ni']  # to shorten things here for now.
mcp_files = []
for iso in isotopes:
    mcp_files += [os.path.join(mcp_file_folder, each) for each in Tools.fileList(db, iso)]


def find_file(search, load_meas=True):
    for file in tipa_files:
        if search in file:
            if load_meas:
                return TiTs.load_xml(file), file
            else:  # faster
                return None, file
    print('Warning! Searchterm: %s not found in tipa files!' % search)
    return None, None


def find_mcp(search):
    for file in mcp_files:
        if search in file and 'KepcoScan' not in file:
            meas_mcp_raw = load(file, db, raw=True)
            meas_mcp = load(file, db, raw=False)
            return meas_mcp_raw, meas_mcp, file
    print('Warning! Searchterm: %s not found in mcp files!' % search)
    return None, None, None


def write_mcp_to_tipa(search_tuple):
    meas_mcp_raw, meas_mcp, mcp_path = find_mcp(str(search_tuple[0]))
    xml_file, found_path = find_file(str(search_tuple[1]))

    if found_path and mcp_path:  # check for not none
        xml_file.find('header').find('accVolt').text = str(meas_mcp_raw.accVolt)
        xml_file.find('header').find('isotope').text = str(meas_mcp_raw.type)
        xml_file.find('header').find('laserFreq').text = str(Physics.wavenumber(meas_mcp.laserFreq / 2))
        xml_file.find('tracks').find('track0').find('header').find('dacStartVoltage').text = str(meas_mcp_raw.x[0][0])
        xml_file.find('tracks').find('track0').find('header').find('dacStopVoltage').text = str(meas_mcp_raw.x[0][-1])
        xml_file.find('tracks').find('track0').find('header').find('postAccOffsetVolt').text = str(meas_mcp_raw.offset)
        xml_file.find('tracks').find('track0').find('header').find('postAccOffsetVoltControl').text = str(
            int(meas_mcp_raw.post_acc_offset_volt_control[0]))
        print(meas_mcp_raw.offset)
        TiTs.save_xml(xml_file, found_path)
        return True
    else:
        return False


# print([('%03d' % i, '%03d' % (i + 11)) for i in range(37, 53)])
# first one always is MCP run number second one always is TiPa run number as stated in the lablog.
# (mcp_run_num, tipa_run_num)
search_tuple_list = [('010', '013'), ('011', '014'), ('012', '015'), ('013', '016'), ('014', '017'),
                     ('015', '020'), ('017', '021'),
                     ('018', '024'), ('019', '025'), ('020', '026'), ('022', '028'),
                     ('024', '029'), ('025', '030'), ('026', '032'), ('027', '033'), ('028', '036'), ('029', '037'),
                     ('030', '038'),
                     ('032', '042'), ('033', '043'), ('034', '000'), ('035', '046'),
                     ('036', '047'), ('037', '048'), ('038', '049'), ('039', '050'), ('040', '051'), ('041', '052'),
                     ('042', '053'), ('043', '054'), ('045', '056'), ('047', '057'),
                     ('048', '058'), ('049', '059'),
                     ('050', '060'), ('051', '061'), ('052', '062'), ('053', '063'),
                     ('054', '000'), ('055', '064'), ('056', '065'), ('057', '066'), ('058', '067'), ('059', '068'),
                     ('060', '069'),
                     ('066', '000'), ('067', '077'), ('068', '078'), ('069', '081'), ('070', '082'), ('071', '083'),
                     ('n72', '084'), ('073', '086'), ('074', '087'), ('075', '088'), ('076', '089'), ('077', '090'),
                     ('078', '091'), ('079', '092'), ('080', '093'), ('081', '094'), ('082', '095'), ('083', '000'),
                     ('084', '097'), ('086', '100'), ('087', '101'), ('088', '102'), ('089', '103'),
                     ('092', '106'), ('093', '107'), ('094', '108'), ('095', '109'),
                     ('096', '110'), ('097', '111'), ('100', '114'), ('101', '115'),
                     ('102', '000'), ('103', '116'), ('104', '117'), ('105', '118'), ('106', '119'),
                     # ('107', '122'), ('108', '122'),   # Tilda was kept running...
                     ('109', '124'), ('110', '125'), ('111', '126'), ('112', '127'), ('113', '128'),
                     ('114', '129'), ('115', '130'), ('116', '131'), ('117', '132'), ('118', '133'), ('119', '134'),
                     ('121', '136'), ('122', '000'), ('123', '138'), ('124', '139'), ('125', '140'),
                     ('126', '141'), ('127', '000'), ('129', '000'), ('130', '144'), ('131', '145'),
                     ('132', '146'), ('133', '000'), ('134', '147'), ('135', '148'), ('136', '149'), ('137', '000'),
                     ('138', '150'), ('139', '151'), ('140', '152'), ('141', '000'), ('142', '154'), ('143', '155'),
                     ('144', '156'), ('145', '158'), ('146', '159'), ('147', '160'), ('148', '161'), ('149', '162'),
                     ('150', '165'), ('151', '166'), ('152', '167'), ('153', '169'), ('154', '170'), ('155', '172'),
                     ('156', '000'), ('157', '000'), ('158', '175'), ('159', '177'), ('160', '179'), ('161', '180'),
                     ('162', '181'), ('163', '182'), ('164', '184'), ('165', '185'), ('166', '186'), ('167', '187'),
                     ('168', '188'), ('169', '189'), ('170', '190'), ('171', '191'), ('172', '192'), ('173', '193'),
                     ('174', '194'), ('175', '195'), ('176', '196'), ('177', '197'), ('178', '198'), ('179', '199'),
                     ('180', '200'), ('181', '201'), ('182', '202'), ('183', '203'), ('184', '204'), ('185', '205'),
                     ('188', '208'), ('189', '209'), ('190', '210'),
                     ('192', '212'), ('193', '213'), ('194', '000'), ('195', '214'), ('197', '216'),
                     ('198', '217'), ('199', '000'), ('200', '220'), ('201', '000'), ('202', '223'), ('203', '224'),
                     ('204', '225'), ('205', '226'), ('206', '227'), ('208', '229'), ('209', '230'),
                     ('210', '232'), ('211', '233'), ('212', '234'), ('213', '235'), ('214', '236'), ('215', '237'),
                     ('216', '238'), ('217', '000'), ('218', '239'), ('219', '240'), ('220', '241'), ('221', '242'),
                     ('222', '243'), ('223', '244'), ('224', '245'), ('225', '246'), ('226', '247'), ('227', '248'),
                     ('228', '251'), ('229', '252'), ('230', '000'), ('231', '253'), ('232', '255'), ('233', '257'),
                     ('234', '258'), ('235', '259'), ('236', '260'), ('237', '261'), ('238', '262'), ('239', '263'),
                     ('240', '264'), ('241', '265'), ('242', '266'), ('243', '267'), ('244', '268'), ('245', '269'),
                     ('246', '270'), ('247', '271'), ('248', '272'), ('249', '273'), ('250', '274'), ('251', '000'),
                     ('252', '275'), ('253', '276'), ('254', '277'), ('255', '278'), ('256', '279'), ('257', '280'),
                     ('258', '281'), ('259', '282'), ('260', '283'), ('261', '284'), ('262', '285'), ('263', '286'),
                     ('264', '287'), ('265', '288'), ('266', '289'), ('267', '290')]

start = datetime.now()

# flat_selected_tipa_files = [each[1] for each in search_tuple_list if each[1] != '000']
# for tipa_f in tipa_files:
#     direc, file = os.path.split(tipa_f)
#     if 'tipa' in file:
#         compare_str = file[8:-4]
#         if compare_str not in flat_selected_tipa_files:
#             # print(compare_str)
#             print('%s\t\tis removed because there is no matching mcp file' % file)
#             os.remove(tipa_f)
# print(flat_selected_tipa_files)
#
# unhappy_list = []
# for tupl_ind, search_tuple in enumerate(search_tuple_list):
#     err = 'unknown'
#
#     if int(search_tuple[1]):  # do not do this '000'
#         try:
#             answer = write_mcp_to_tipa(search_tuple)
#         except Exception as err:
#             answer = False
#             print('error: %s' % err)
#         if not answer:
#             unhappy_list.append((search_tuple, err))
#     step_done = datetime.now()
#     tupl_per_sec = (tupl_ind + 1) / (step_done - start).total_seconds()
#     tuples_left = len(search_tuple_list) - (tupl_ind + 1)
#     timeleft = tuples_left / tupl_per_sec
#     print('---- current search tuple: %s ---- %s completed ---- %s to do ---- %.2f timeleft ----' % (
#         search_tuple, (tupl_ind + 1), tuples_left, timeleft))
#
# print('the following search tuples where not happy: \n')
# print(unhappy_list)


# helper to type in the tuples:
# from PyQt5 import QtWidgets
#
# if __name__ == '__main__':
#     app = QtWidgets.QApplication(sys.argv)
#     all = [(0, 107)]
#     for i in range(93, 268):
#         res_int = QtWidgets.QInputDialog.getInt(QtWidgets.QInputDialog(),
#                                                 'get the pair', 'mcp_Run%03d' % i, int(all[-1][1]) + 1)
#         # print(res_int)
#         if res_int[1]:
#             all.append(('%03d' % i, '%03d' % res_int[0]))
#         else:
#             all.append(('%03d' % i, '000'))
#     print(all)
#     app.exec_()

''' check for abnormalities in time structure '''
# get all relevant files first from the used config in the db
configs = {}
for iso in isotopes:
    ret = TiTs.select_from_db(db, 'config', 'Combined', [['iso', 'run', 'parname'], [iso, runs[0], 'shift']])
    if ret is not None:
        ret = ret[0][0]
        configs[iso] = ast.literal_eval(ret)
        # print(configs[iso])

flat_configs = {}  # each file only once, just to check time structure!
for iso_name, conf in sorted(configs.items()):
    flat_configs[iso_name] = []
    for each in conf:
        for file in each:
            for tipa_f in file:
                if tipa_f not in flat_configs[iso_name]:
                    flat_configs[iso_name] += tipa_f,
    # print(iso_name, flat_configs[iso_name])

all_files = TiTs.select_from_db(db, 'file', 'Files', addCond='ORDER BY date')
flat_configs['all'] = []
for each in all_files:
    if '.mcp' in each[0] and 'epco' not in each[0]:
        flat_configs['all'] += each[0],


def find_tipa_file_to_mcp_file(mcp_file_str):
    # print('finding: ', mcp_file_str)
    run_str = mcp_file_str.split('.')[0].split('Run')[1]  # '010' etc.
    try:
        for mcp_run_str, tipa_run_str in search_tuple_list:
            if run_str in mcp_run_str:
                tipa_meas, tipa_file_n = find_file(tipa_run_str, load_meas=False)
                return tipa_meas, tipa_file_n
    except Exception as e:
        print('error: %s converting run str: %s, ' % (e, run_str))
        return None, None
    return None, None

# flat_configs_tipa = {}
# for isotope, flat_conf in sorted(flat_configs.items()):
#     flat_configs_tipa[isotope] = []
#     for mcp_f in flat_conf:
#         if mcp_f and not 'release' in mcp_f:
#             tipa_meas, tipa_file_n = find_tipa_file_to_mcp_file(mcp_f)
#             if tipa_file_n is not None:
#                 # tipa_file_n = tipa_file_n.split('\\')[1]
#                 flat_configs_tipa[isotope] += (tipa_file_n, mcp_f),
#                 tipa_file_n = os.path.split(tipa_file_n)[1]
#             print('%s\t%s\t%s' % (isotope, mcp_f, tipa_file_n))

''' plot the time structure and save it as a .jpg '''
# from Interface.LiveDataPlottingUi.LiveDataPlottingUi import TRSLivePlotWindowUi
# from Service.AnalysisAndDataHandling.DisplayData import DisplayData
#
# from PyQt5 import QtWidgets, QtGui
# from Measurement.XMLImporter import XMLImporter
# from PyQt5.QtCore import *
# import time
#
#
# app = QtWidgets.QApplication(sys.argv)
# ui = QtWidgets.QMainWindow()
# label = QtWidgets.QLabel()
# ui.setCentralWidget(label)
# ui.show()
# # copy = flat_configs_tipa
# # flat_configs_tipa = {}
# # flat_configs_tipa['58_Ni'] = copy['58_Ni']
# storage_path = 'E:\\Workspace\\OwnCloud\\Projekte\\COLLAPS\\Nickel\\Measurement_and_Analysis_Simon\\Ni_workspace\\TimeStructureAnalysis'
#
# win = []  # all windows need to be stored when wanting to keep them open.
# # file_names = []
# for iso, files in sorted(flat_configs_tipa.items()):
#     # if iso in ['58_Ni', '59_Ni', '60_Ni', '61_Ni', '62_Ni', '63_Ni',
#     #            '64_Ni', '65_Ni', '66_Ni', '67_Ni', '68_Ni', '70_Ni']:
#     if iso in ['65_Ni']:
#         cur_dir = os.path.join(storage_path, iso)
#         if not os.path.isdir(cur_dir):
#             os.mkdir(cur_dir)
#         # files = [files[-1]]
#         for tipa_f, mcp_f in files:
#             logging.info('working on: %s' % tipa_f)
#             # label.setText('working on: %s' % tipa_f)
#             try:
#                 cur_win = TRSLivePlotWindowUi(tipa_f, subscribe_as_live_plot=False, application=app)
#                 spec = XMLImporter(tipa_f, softw_gates=(db, runs[0]))
#                 spec.softBinWidth_ns = [100]
#                 disp_data = DisplayData(tipa_f, cur_win, x_as_volt=True, loaded_spec=spec)
#                 cur_win.set_time_range(padding=0.2)
#                 win += cur_win,   # if one wants to see all the windows comment this in.
#                 f_name = iso + '_real_iso_' + spec.type + '_mcp_file_' + mcp_f.split('.')[0] + '_tipa_file_' + os.path.split(tipa_f)[1]
#                 f_name = f_name.split('.')[0]
#                 f_name += '.jpg'
#                 f_name_full = os.path.join(cur_dir, f_name)
#                 # file_names += f_name_full,
#
#                 cur_win.export_screen_shot(f_name_full, quality=100)
#                 cur_win.close()
#
#             except Exception as e:
#                 print(e)
# label.setText('Done!')
# app.deleteLater()
# app.exec_()
# app.closeAllWindows()


''' direct comparison of files MCP - TIPA ~ 0 in order to have the same counts in both DAQs '''

# until MCP run 107 only scaler 0 of TIPA-Files can be used due to broken NIM-electronics

# db_both = os.path.join(os.path.split(db)[0], 'Ni_workspace_mcp_and_tilda.sqlite')
#
#
# def load_mcp_tipa_from_search_tuple(search_tuple, softw_gates):
#     meas_tipa = None
#     meas_mcp = None
#     try:
#         for file in tipa_files:
#             if search_tuple[1] in file and '_trs_' not in file:
#                 meas_tipa = load(file, db_both, raw=True, softw_gates=softw_gates)
#         if meas_tipa is None:
#             print('Warning! Searchterm: %s not found in tipa files!' % search_tuple[1])
#         for file in mcp_files:
#             if search_tuple[0] in file and 'KepcoScan' not in file:
#                 meas_mcp = load(file, db_both, raw=True)
#         if meas_mcp is None:
#             print('Warning! Searchterm: %s not found in mcp files!' % search_tuple[0])
#         meas_tipa.softw_gates = [softw_gates]
#         meas_tipa = TiTs.gate_specdata(meas_tipa)
#         meas_tipa.cts[0][0] = np.asarray(meas_tipa.cts[0][0], dtype=np.int32) - np.asarray(meas_mcp.cts[0][8], dtype=np.int32)  # scaler 8 in mcp is scaler 0 in tipa
#         return meas_tipa, meas_mcp
#     except Exception as e:
#         print('error while loading search tuple: %s, error: %s' % (search_tuple, e))
#         return None, None
#
#
# def optimize_gates3(tipa_meas, mcp_meas, t_max_incr=1, t_min_incr=1, working_t_max=True):
#     max_gate_len = 10
#     min_gate_len = 4
#     max_reps = 500
#     result_list = []
#     result_list.append((99999999999999, t_max_incr, t_min_incr, working_t_max))
#     init_gates = tipa_meas.softw_gates
#     v_min_opti = init_gates[0][0][0]
#     v_max_opti = init_gates[0][0][1]
#     TiTs.gate_specdata(tipa_meas)
#     tipa_meas.cts[0][0] -= mcp_meas.cts[0][8]
#     sum_sc0 = np.sum(tipa_meas.cts[0][0])
#     sum_abs_sc0 = np.sum(np.abs(tipa_meas.cts[0][0]))  # should be 0
#     t_min_incr *= np.sign(sum_sc0) * -1  # start with increasing t_min if tipa-mcp < 0
#     t_max_incr *= np.sign(sum_sc0)  # start with decreasing t_max if tipa-mcp < 0
#     result_list.append((sum_abs_sc0, t_max_incr, t_min_incr, working_t_max))
#
#     for reps in range(0, max_reps):
#         init_gates = tipa_meas.softw_gates
#         t_min_1_opti = float(init_gates[0][0][2])
#         t_max_1_opti = float(init_gates[0][0][3])
#         if reps == 0:
#             print('initial absolute sum is: %s' % sum_abs_sc0)
#             print('initial software gates are: %s' % tipa_meas.softw_gates)
#         gate_dif = t_max_1_opti - t_min_1_opti
#         if abs(t_max_incr) == 0.01 and abs(t_min_incr) == 0.01 and min_gate_len < gate_dif < max_gate_len:  # we did not use bigger gates than 10µs
#             print('maximum precision reached, abs_sum is: %s, iterations needed: %s' % (sum_abs_sc0, reps))
#             return tipa_meas
#         if working_t_max:
#             # print('working t_max, now: %s before: %s t_min: %.2f t_max: %.3f' % (
#             #     result_list[-1][0], result_list[-2][0], t_min_1_opti, t_max_1_opti))
#             if result_list[-1][0] < result_list[-2][0]:  # keep changing t_max
#                 if gate_dif > max_gate_len:  # gate is too long, shorten it.
#                     t_max_incr = - 1 * max(abs(gate_dif - max_gate_len) / 4, 0.02)
#                     # print('gate was too long (%.2f), shorten it, t_max_incr: %.2f' % (gate_dif, t_max_incr))
#                 else:
#                     if gate_dif < min_gate_len:  # gate is too short, increase it
#                         t_max_incr = max(abs(gate_dif - min_gate_len) / 4, 0.02)
#                         # print('gate was too short (%.2f), make it longer, t_max_incr: %.2f' % (gate_dif, t_max_incr))
#                 t_max_1_opti += t_max_incr
#             else:  # undo last changes and change direction
#                 t_max_1_opti -= t_max_incr
#                 t_max_incr *= -0.85  # turn the increment factor around
#                 t_max_incr = np.sign(t_max_incr) * max(0.02, abs(t_max_incr))
#                 working_t_max = False
#         else:
#             # print('working t_MIN, now: %s before: %s t_min: %.2f t_max: %.3f' % (
#             #     result_list[-1][0], result_list[-2][0], t_min_1_opti, t_max_1_opti))
#             if result_list[-1][0] < result_list[-2][0]:  # keep working on t_min
#                 if gate_dif > max_gate_len:  # gate is too long, shorten it.
#                     t_min_incr = max(abs(gate_dif - max_gate_len) / 4, 0.02)
#                     # print('gate was too long (%.2f), shorten it, t_min_incr: %.2f' % (gate_dif, t_min_incr))
#                 else:
#                     if gate_dif < min_gate_len:  # gate is too short, increase it
#                         t_min_incr = - 1 * max(abs(gate_dif - min_gate_len) / 4, 0.02)
#                         # print('gate was too short (%.2f), make it longer, t_min_incr: %.2f' % (gate_dif, t_min_incr))
#                 t_min_1_opti += t_min_incr
#             else:  # stop working on  t_min
#                 t_min_1_opti -= t_min_incr
#                 working_t_max = True
#                 t_min_incr *= -0.85  # turn the increment factor around
#                 t_min_incr = np.sign(t_min_incr) * max(0.02, abs(t_min_incr))
#
#         tipa_meas.softw_gates = [[[v_min_opti, v_max_opti, t_min_1_opti, t_max_1_opti],
#                                   [v_min_opti, v_max_opti, t_min_1_opti + 50, t_max_1_opti + 50],
#                                   [v_min_opti, v_max_opti, t_min_1_opti, t_max_1_opti],
#                                   [v_min_opti, v_max_opti, t_min_1_opti + 50, t_max_1_opti + 50]]]
#         TiTs.gate_specdata(tipa_meas)
#         tipa_meas.cts[0][0] -= mcp_meas.cts[0][8]
#         sum_abs_sc0 = np.sum(np.abs(tipa_meas.cts[0][0]))  # should be 0
#         result_list.append((sum_abs_sc0, t_max_incr, t_min_incr, working_t_max))
#     print('max iterations reached!')
#     print('asolute sum is: %s' % sum_abs_sc0)
#     print('software gates are: %s' % tipa_meas.softw_gates)
#     return tipa_meas
#
# # trouble makers:
# search_tuple_list = [
#     ('150', '165'), ('149', '162'), ('082', '095'), ('109', '124'), ('110', '125'), ('169', '189'), ('144', '156'),
#     ('182', '202'), ('170', '190'), ('153', '169'), ('121', '136'), ('154', '170'), ('074', '087'), ('075', '088'),
#     ('148', '161'), ('103', '116'), ('101', '115'), ('123', '138'), ('145', '158'), ('189', '209'), ('018', '024'),
#     ('073', '086'), ('100', '114'), ('181', '201'), ('159', '177'), ('106', '119'), ('193', '213'), ('077', '090'),
#     ('135', '148'), ('160', '179'), ('177', '197'), ('192', '212')
# ]
#
# results = []
# errors = []
#
# pictur_path = os.path.join(workdir, 'TiPaVsMCPPlots')
#
# # starting_vals = [
# #     (0, 10),
# #     (0, 40),
# #     (10, 20), (10, 30), (10, 40),
# #     (15, 25), (5, 15), (25, 35),
# #     (12, 22), (8, 18),
# #     (20, 30), (20, 40),
# # ]
#
# starting_vals = [
#     (0, 20), (0, 30),
#     (5, 10), (5, 20),
#     (10, 25), (20, 35),
#     (15, 25), (6, 16), (17, 27), (19, 29),
#     (15, 20), (15, 30), (15, 40),
#     (20, 25), (20, 35),
#     (25, 35),
#     (30, 40)
# ]
#
# start_time = datetime.now()
# v_min = -10
# v_max = 10
# t_min_1 = starting_vals[0][0]
# t_max_1 = starting_vals[0][1]
# t_min_2 = t_min_1 + 0.5
# t_max_2 = t_max_1 + 0.5
#
# for tupl_ind, search_tuple in enumerate(search_tuple_list):
#     if search_tuple[1] != '000':
#         best_start_val_result = []
#         sum_abs_sc0_before = 999999999999999999
#         meas_loaded, mcp_loaded = load_mcp_tipa_from_search_tuple(
#             search_tuple, [[v_min, v_max, t_min_1, t_max_1], [v_min, v_max, t_min_2, t_max_2],
#                            [v_min, v_max, t_min_1, t_max_1], [v_min, v_max, t_min_2, t_max_2]])
#         if meas_loaded is not None and mcp_loaded is not None:
#             for start_val_ind, start_val_tupl in enumerate(starting_vals):
#                 try:
#                     print('--------- starting with search tuple: %s %s ----------' % search_tuple)
#                     v_min = -10
#                     v_max = 10
#                     t_min_1 = start_val_tupl[0]
#                     t_max_1 = start_val_tupl[1]
#                     t_min_2 = t_min_1 + 0.5
#                     t_max_2 = t_max_1 + 0.5
#                     meas_loaded.softw_gates = [[[v_min, v_max, t_min_1, t_max_1], [v_min, v_max, t_min_2, t_max_2],
#                                                 [v_min, v_max, t_min_1, t_max_1], [v_min, v_max, t_min_2, t_max_2]]]
#
#                     tipa_meas_opti = optimize_gates3(meas_loaded, mcp_loaded, t_max_incr=5, t_min_incr=5)
#                     sum_abs_sc0 = np.sum(np.abs(tipa_meas_opti.cts[0][0]))
#                     if sum_abs_sc0 < sum_abs_sc0_before:
#                         print('overwriting sum! sum now: %s sum before: %s tuple: %s' % (
#                             sum_abs_sc0, sum_abs_sc0_before, start_val_tupl))
#                         tipa_meas = deepcopy(tipa_meas_opti)
#                         best_start_values = deepcopy(start_val_tupl)
#                         sum_abs_sc0_before = deepcopy(sum_abs_sc0)
#
#                 except Exception as e:
#                     print('error on search_tuple: %s : e' % search_tuple)
#                     errors.append(search_tuple)
#             try:
#                 mean = np.mean(tipa_meas.cts[0][0])
#                 total_cts_mcp = np.sum(mcp_loaded.cts[0][0])
#                 total_cts_tipa = np.sum(
#                     mcp_loaded.getArithSpec([0], -1)[1] + tipa_meas.getArithSpec([0], -1)[1], dtype=np.int32)
#                 sum_abs_sc0 = np.sum(np.abs(tipa_meas.cts[0][0]))
#                 deviation_percent = 100 * sum_abs_sc0 / total_cts_mcp
#                 results.append((search_tuple, total_cts_mcp, sum_abs_sc0, mean, deviation_percent,
#                                 tipa_meas.softw_gates[0][0][2], tipa_meas.softw_gates[0][0][3],
#                                 best_start_values[0], best_start_values[1]))
#                 print('%s\t%s\t%s\t%.4f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f' % results[-1])
#                 fig = MPLPlotter.plt.figure(1, (8, 8))
#                 fig.patch.set_facecolor('white')
#                 upper_ax = MPLPlotter.plt.axes([0.15, 0.35, 0.8, 0.6])
#                 dif_plot_ax = MPLPlotter.plt.axes([0.15, 0.1, 0.8, 0.2], sharex=upper_ax)
#                 dif_plot_ax.get_xaxis().get_major_formatter().set_useOffset(False)
#
#                 mcp_plot = upper_ax.plot(
#                     mcp_loaded.getArithSpec([0], -1)[0], mcp_loaded.getArithSpec([0], -1)[1],
#                     'r-', label='McpRun%s cts: %s' % (search_tuple[0], total_cts_mcp))
#                 tipa_plot = upper_ax.plot(
#                     mcp_loaded.getArithSpec([0], -1)[0],
#                     mcp_loaded.getArithSpec([0], -1)[1] + tipa_meas.getArithSpec([0], -1)[1],
#                     'g-', label='TiPaRun%s cts: %s' % (search_tuple[1], total_cts_tipa))
#
#                 dif_plot = dif_plot_ax.plot(tipa_meas.getArithSpec([0], -1)[0], tipa_meas.getArithSpec([0], -1)[1],
#                                             label='TiPa%s-Mcp%s cts: %s' % (search_tuple[1],
#                                                                             search_tuple[0],
#                                                                             sum_abs_sc0))
#                 MPLPlotter.plt.xlabel('DAC voltage [V]')
#                 MPLPlotter.plt.ylabel('remaining cts')
#                 upper_ax.set_ylabel('cts [a.u.]')
#
#                 plots = [dif_plot[0], tipa_plot[0], mcp_plot[0]]
#                 print(plots)
#                 labels = [pl.get_label() for pl in plots]
#
#                 from matplotlib.patches import Rectangle
#
#                 extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
#                 plots.append(extra)
#
#                 labels.append('t_min: %.2f, t_max: %.2f, dif/total_mcp: %.2f %%, mean: %.3f ' % (
#                     tipa_meas.softw_gates[0][0][2], tipa_meas.softw_gates[0][0][3], deviation_percent, mean))
#                 box = upper_ax.get_position()
#                 upper_ax.set_position([box.x0, box.y0, box.width, box.height * 0.7])
#                 upper_ax.legend(plots, labels, loc='upper center', bbox_to_anchor=(0.5, 1.45))
#                 picname = str(search_tuple).replace(', ', '_TiPa_Run').replace('(', 'MCP_Run').replace(')', '').replace('\'',
#                                                                                                                         '') + '.png'
#                 MPLPlotter.save(os.path.join(pictur_path, picname))
#                 MPLPlotter.clear()
#                 step_done = datetime.now()
#                 tupl_per_sec = (tupl_ind + 1) / (step_done - start).total_seconds()
#                 tuples_left = len(search_tuple_list) - (tupl_ind + 1)
#                 timeleft = tuples_left / tupl_per_sec
#                 print('---- current search tuple: %s ---- %s completed ---- %s to do ---- %.2f timeleft ----' % (
#                     search_tuple, (tupl_ind + 1), tuples_left, timeleft))
#             except Exception as e:
#                 print('error on search_tuple: %s : e' % search_tuple)
#                 errors.append(search_tuple)
#         else:
#             errors.append(search_tuple)
#
# stop_time = datetime.now()
# print('total execution time [s]: ' + str(stop_time - start_time))
#
# print('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' % ('search_tuple', 'total_cts', 'Tipa-MCP_absolute_sum',
#                                      'Tipa-MCP_mean', 'Tipa-MCP_absolute_sum/total_cts * 100',
#                                      't_min_gate_mus', 't_max_gate_mus', 't_min_start', 't_max_start'))
# for res_tpl in results:
#     print('%s\t%s\t%s\t%.4f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f' % res_tpl)
#
# print('tuples with error: \n %s ' % errors)
# app = QtWidgets.QApplication([""])

# ok = True
# ok_2 = True
# while ok and ok_2:
#     MPLPlotter.plot(meas_loaded.getArithSpec([0], -1))
#     MPLPlotter.show(True)
#     inp_par = QtWidgets.QInputDialog()
#     t_min_1, ok = QtWidgets.QInputDialog.getDouble(inp_par, 'get_t_min', 't_min:', t_min_1)
#     t_max_1, ok_2 = QtWidgets.QInputDialog.getDouble(inp_par, 'get_t_max', 't_max:', t_max_1)
#     meas_loaded.softw_gates = [[[v_min, v_max, t_min_1, t_max_1], [v_min, v_max, t_min_2, t_max_2],
#                                 [v_min, v_max, t_min_1, t_max_1], [v_min, v_max, t_min_2, t_max_2]]]
#     TiTs.gate_specdata(meas_loaded)
#     meas_loaded.cts[0][0] -= mcp_loaded.cts[0][8]
#
# sys.exit(app.exec)

best_gates = os.path.join(workdir, 'Excel_stuff/difference_Tipa-MCP.txt')


def write_gates_to_tipa_xml(search_tuple, t_min, t_max):
    xml_file, found_path = find_file(str(search_tuple[1]))
    v_min = -10
    v_max = 10
    softw_gates = [[v_min, v_max, t_min, t_max], [v_min, v_max, t_min + 0.5, t_max + 0.5],
                    [v_min, v_max, t_min, t_max], [v_min, v_max, t_min + 0.5, t_max + 0.5]]

    if found_path:  # check for not none
        xml_file.find('tracks').find('track0').find('header').find('softwGates').text = str(softw_gates)
        TiTs.save_xml(xml_file, found_path)
        return True
    else:
        return False
# err_gates = []
# file = open(best_gates, 'r')
# line = file.readline()
# while line:
#     line = file.readline()
#     line = line.replace(',', '.').replace('. ', ', ').replace('\t', ', ')
#     line = '[' + line + ']'
#     line = ast.literal_eval(line)
#     print(line)
#     if write_gates_to_tipa_xml(line[0], line[5], line[6]):
#         print('saving ok')
#     else:
#         err_gates.append(line)
#         print('broke')
#
# print('lines with error: %s' % err_gates)

