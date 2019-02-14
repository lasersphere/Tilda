"""
Created on 03.12.18

@author: simkaufm

Module Description:  used to find all bunch lengths of the involved files
and compare these between 2016 and 2017
results are used in the origin file .../Owncloud/User/Simon/Doktorarbeit/Arbeit/origin/isotope_shifts
imported there by hand from the Measurement and Analysis folders
    .../OwnCloud/Projekte/COLLAPS/Nickel/Measurement_and_Analysis_Simon/Ni_workspace/timing_info
    .../OwnCloud/Projekte/COLLAPS/Nickel/Measurement_and_Analysis_Simon/Ni_workspace2017/Ni_2017/timing_info
"""

import os
import sys
import numpy as np
import ast
from PyQt5 import QtWidgets
import csv
import matplotlib.pyplot as plt

import Physics
import Analyzer as Anal
import TildaTools as TiTs
from Measurement.XMLImporter import XMLImporter
import Service.Formating as Form

''' settings: '''

create_bunch_len_txt_16 = False  # create text file by reading all data files and interpreting the bunch length
add_time_rchi_16 = 0.0
create_config_sum_file_16 = False
create_mean_iso_bun_len_txt_file_16 = False
plot_time_struct_16 = False  # plot the time projection for scaler 0 for seperate 58 Ni files
normalize_on_scans = False  # false is better...
plot_gauss = True
create_bunch_len_txt_17 = False
add_time_rchi_17 = add_time_rchi_16
create_config_sum_file_17 = False
create_mean_iso_bun_len_txt_file_17 = False

''' working directory: '''

workdir17 = 'E:\\Workspace\\OwnCloud\\Projekte\\COLLAPS\\Nickel\\' \
            'Measurement_and_Analysis_Simon\\Ni_workspace2017\\Ni_2017'

datafolder17 = os.path.join(workdir17, 'sums')

time_info_folder17 = os.path.join(workdir17, 'timing_info')
if not os.path.isdir(time_info_folder17):
    os.mkdir(time_info_folder17)

db17 = os.path.join(workdir17, 'Ni_2017.sqlite')

run_hot_cec_17 = 'AsymVoigtHotCec'
run_hot_cec_exp_17 = 'VoigtAsy'
normal_run_17 = 'AsymVoigt'
final_2017_run = '2017_Experiment'  # this will be the joined analysis of hot CEC and normal run!

run2016_final_db = '2016Experiment'
exp_liang_run16 = '2016ExpLiang'
exp_liang_run17 = '2017ExpLiang'
steudel_1980_run = 'Steudel_1980'
bradley_2016_3_ev_run = '2016_Bradley_3eV'

# for plotting:
run_colors = {
    normal_run_17: (0.4, 0, 0),  # dark red
    run_hot_cec_17: (0.6, 0, 0),  # dark red
    run2016_final_db: (0, 0, 1),  # blue
    steudel_1980_run: (0, 0.6, 0.5),  # turqoise
    exp_liang_run16: (0.6, 0.4, 1),  # purple
    exp_liang_run17: (1, 0.5, 0.3),  # orange
    run_hot_cec_exp_17: (0.4, 0, 0),  # dark red
    bradley_2016_3_ev_run: (0, 0.3, 1),
    final_2017_run: (1, 0, 0)  # red
}

run_markes = {
    normal_run_17: 's',  # square
    run_hot_cec_17: 's',  # square
    run2016_final_db: 'o',  # circle
    steudel_1980_run: 'o',  # circle
    exp_liang_run16: 'D',  # diamond
    exp_liang_run17: 'D',  # diamond
    run_hot_cec_exp_17: 's',
    bradley_2016_3_ev_run: '^',  # triangle up
    final_2017_run: 's'
}

run_comments = {
    normal_run_17: ' 2017 Simon (runs 83-end)',
    run_hot_cec_17: ' 2017 hot CEC (runs 62-82) Simon',
    run_hot_cec_exp_17: '2017 hot CEC (runs 62-82) Simon',
    run2016_final_db: '2016 Simon',
    steudel_1980_run: '1980 Steudel',
    exp_liang_run16: '2016 Liang',
    exp_liang_run17: '2017 Liang',
    bradley_2016_3_ev_run: '2016 Bradley',
    final_2017_run: '2017 Simon'
}

isotopes_17 = ['%s_Ni' % i for i in range(58, 71)]
isotopes_17.remove('59_Ni')  # not measured in 2017
isotopes_17.remove('63_Ni')  # not measured in 2017
isotopes_17.remove('69_Ni')  # not measured in 2017
# isotopes = ['58_Ni']

odd_isotopes = [iso for iso in isotopes_17 if int(iso[:2]) % 2]
even_isotopes = [iso for iso in isotopes_17 if int(iso[:2]) % 2 == 0]
stables = ['58_Ni', '60_Ni', '61_Ni', '62_Ni', '64_Ni']

# isotopes = ['64_Ni']


# 2016 database etc.
workdir2016 = 'E:\\Workspace\\OwnCloud\\Projekte\\COLLAPS\\Nickel\\Measurement_and_Analysis_Simon\\Ni_workspace'

datafolder_mcp_2016 = os.path.join(workdir2016, 'Ni_April2016_mcp')
datafolder_tipa_2016 = os.path.join(workdir2016, 'TiPaData')

time_info_folder2016 = os.path.join(workdir2016, 'timing_info')
if not os.path.isdir(time_info_folder2016):
    os.mkdir(time_info_folder2016)

db2016 = os.path.join(workdir2016, 'Ni_workspace.sqlite')
runs2016 = ['wide_gate_asym', 'wide_gate_asym_67_Ni']
final_2016_run = 'wide_gate_asym'

isotopes2016 = ['%s_Ni' % i for i in range(58, 71)]  # 58, 71
isotopes2016.remove('67_Ni')  # not relevant here since no TIPA data was acquired for Nickel 67
isotopes2016.remove('69_Ni')
isotopes2016.remove('70_Ni')  # not relevant here since no TIPA data was acquired for Nickel 70
odd_isotopes2016 = [iso for iso in isotopes2016 if int(iso[:2]) % 2]

""" get file numbers: """


def get_file_number_from_file_str(file_str, mass_index, end_result_len, app=None):
    numbers = []
    number_str = ''
    for i, letter in enumerate(file_str):
        if letter.isdigit():  # is either 0-9
            number_str += letter
        else:  # not a digit
            if number_str.isdigit():  # convert the consecutive number
                numbers += [number_str]
            number_str = ''  # reset number str
    if isinstance(mass_index, list):
        # [0] etc. .
        numbers = [val for n, val in enumerate(numbers) if n not in mass_index]
    if end_result_len > 0:
        # user want to check if the correct amount of integers is found
        if len(numbers) != end_result_len:
            # does not match, require user input!
            print('warning: ')
            print('file', file_str, 'nums', numbers)
            if app is None:
                app = QtWidgets.QApplication(sys.argv)

            print('opening dial:')
            text, ok_pressed = QtWidgets.QInputDialog.getText(None, 'Warning',
                                                              '%s has more or less than %s numbers: %s \n'
                                                              ' please write the desired file number(s) here'
                                                              'still as a list of strings please!:' %
                                                              (file_str, end_result_len, numbers),
                                                              QtWidgets.QLineEdit.Normal,
                                                              str(numbers)
                                                              )
            if ok_pressed:
                try:
                    numbers = ast.literal_eval(text)
                except Exception as e:
                    print('could not convert %s, error is: %s' % (text, e))
            # make sure it has the right length in the end!
            if len(numbers) > end_result_len:
                numbers = numbers[:end_result_len]
                print('warning, still incorrect amount of numbers! Will use %s fo file: %s' % (numbers, file_str))
            elif len(numbers) < end_result_len:
                numbers = numbers * (end_result_len // len(numbers) + 1)
                numbers = numbers[:end_result_len]
                print('warning, still incorrect amount of numbers! Will use %s fo file: %s' % (numbers, file_str))
    return numbers, app


def get_file_numbers(file_list, mass_index=[0], end_result_len=1, app=None, user_overwrite={}):
    """
    get all file numbers (=conescutive integer numbers)
    in the filenames that are listed in file_list.
    :param file_list: list, like  ['62_Ni_trs_run071.xml', ...]
    :param mass_index: list, indice of all expected mass numbers, that will be removed from the output.
    if the mass number is wanted -> use mass_index=[-1]
    for 62_Ni_trs_run071.xml
     -> mass_index=[0] -> [[71]]
     -> mass_index=None -> [[62, 71]]
     :param end_result_len: int, desired amount of numbers to be found, as a cross check.
     use -1/0 if you don't care
     :param user_overwrite: dict, key is orig. filenum, value is str that will be put as file_num_str
     e.g.  {'60_Ni_trs_run113_sum114.xml': ['113+114']}
     helpful to avoid user input on runtime
    :return: list of all file numbers each still as string, that might be convertable to int,
     but can also be something like '123+124' by user choice.
    """
    file_nums = []
    for f in file_list:
        if f in user_overwrite.keys():
            file_nums += user_overwrite[f]
        else:
            file_num, app = get_file_number_from_file_str(f, mass_index, end_result_len, app)
            file_nums += file_num
    return file_nums


""" compare them """


def get_all_files_per_iso(db, iso, run):
    """ get all files relevant for the isotope shift """
    conf_staterrform_systerrform = TiTs.select_from_db(db, 'config, statErrForm, systErrForm', 'Combined',
                                                       [['iso', 'parname', 'run'], [iso, 'shift', run]],
                                                       caller_name=__name__)
    if conf_staterrform_systerrform:
        (config, statErrForm, systErrForm) = conf_staterrform_systerrform[0]
    else:
        return {}
    config = ast.literal_eval(config)
    ref_files = []
    iso_files = []
    for ref_bef, iso_f, ref_after in config:
        ref_files += ref_bef
        ref_files += ref_after
        iso_files += iso_f
    return config, iso_files, ref_files


# 2017
files_flat_2017 = {}  # sorted by isotope
configs_2017 = {}
for isotope in isotopes_17:
    config, iso_files, ref_files = get_all_files_per_iso(db17, isotope, final_2017_run)
    configs_2017[isotope] = config
    files_flat_2017[isotope] = iso_files
    if any(files_flat_2017.get('60_Ni', [])):
        files_flat_2017['60_Ni'] += ref_files
    else:
        files_flat_2017['60_Ni'] = ref_files

files_flat_2017['60_Ni'] = list(set(files_flat_2017['60_Ni']))
# TiTs.print_dict_pretty(files_flat_2017)
txt_file17 = os.path.join(time_info_folder17, 'bunch_lengths_mean.txt')

if create_bunch_len_txt_17:
    # get bunchlength for each files and store it in .txt file in roder nto to always have to import all the files etc.
    with open(txt_file17, 'wb') as f17:
        for iso in isotopes_17:
            for tilda_file in files_flat_2017[iso]:
                abs_path = os.path.join(datafolder17, tilda_file)
                meas = XMLImporter(abs_path)
                rebinning = 100  # in ns
                meas = Form.time_rebin_all_spec_data(meas, [rebinning] * meas.nrTracks, -1)
                meas = TiTs.gate_specdata(meas)
                new_meas, ret_dict = TiTs.calc_bunch_width_relative_to_peak_height(
                    meas, 25, show_plt=False, non_consectutive_time_bins_tolerance=10,
                    save_to_path=[os.path.join(time_info_folder17, os.path.splitext(meas.file)[0] + '.png'),
                                  os.path.join(time_info_folder17, os.path.splitext(meas.file)[0] + '.pdf')],
                    time_around_bunch=(3.0, 6.0), additional_time_rChi=add_time_rchi_17
                )
                b_lengths = np.mean(ret_dict['bunch_lenght_us'], 0)  # get mean over all tracks for each scaler
                # b_lengths = [item for sublist in b_lengths for item in sublist]
                # mean_b_lengths, err_mean_b_lengths, rChi = Anal.weightedAverage(b_lengths,
                #                                                                 [rebinning / 1000] * len(b_lengths))
                rel_scaler = 1  # only account sc0 because different scalers show different behaviour
                b_lengths_relevant_scalers = b_lengths[:rel_scaler]
                mean_b_lengths, err_mean_b_lengths, rChi = Anal.weightedAverage(
                    b_lengths_relevant_scalers, [rebinning / 1000] * len(b_lengths_relevant_scalers))
                gauss_res = ret_dict['gaussian_res']
                r_chis_file = []
                for tr_ind, gauss_res_tr in enumerate(gauss_res):
                    r_chis_file += [],
                    for gauss_res_tr_sc in gauss_res_tr:
                        rchisq_tr_sc = gauss_res_tr_sc[0][-1]
                        r_chis_file[tr_ind] += rchisq_tr_sc,
                rchisq_mean_sc_wise = np.mean(r_chis_file, 0)  # mean over all track scaler wise
                rchisq_mean = np.mean(rchisq_mean_sc_wise)
                f17.write(bytes('%s\t%.6f\t%.6f\t%.2f\n' % (
                    tilda_file, mean_b_lengths, err_mean_b_lengths, rchisq_mean), 'UTF-8'))

    f17.close()

overwrites = {'60_Ni_trs_run113_sum114.xml': ['113+114'],
              '60_Ni_trs_run192_sum193.xml': ['192+193'],
              '60_Ni_trs_run198_sum199_200.xml': ['198+199+200'],
              '60_Ni_trs_run117_sum118.xml': ['117+118'],
              '60_Ni_trs_run122_sum123.xml': ['122+123'],
              '67Ni_no_protonTrigger_3Tracks_Run191.mcp': ['191'],
              '67Ni_sum_Run061_Run062.xml': ['061+062'],
              '67Ni_sum_Run063_Run064_Run065.xml': ['063+064+065'],
              '67_Ni_sum_run243_and_run248.xml': ['243+248'],
              '60_Ni_trs_run239_sum240.xml': ['239+240'],
              '70Ni_protonTrigger_Run248_sum_252_254_259_265.xml': ['248+252+254+259+265'],
              '60_Ni_trs_run266_sum267.xml': ['266+267'],
              '70_Ni_trs_run268_plus_run312.xml': ['268+312']}

bunch_lenghts_2017_dict = {}  # 'filename': (len_us, err_len_us)

with open(txt_file17, 'r') as f17:
    lines = csv.reader(f17, delimiter='\t')
    for f, bun_len, err_bun_len, rchisq in lines:
        bunch_lenghts_2017_dict[f] = (float(bun_len), float(err_bun_len), float(rchisq))

# TiTs.print_dict_pretty(bunch_lenghts_2017_dict)
print(bunch_lenghts_2017_dict.keys())
print(Anal.get_date_date_err_to_files(db17, bunch_lenghts_2017_dict.keys()))
raise Exception

''' 2016 '''

all_file_flat_2016 = {iso16: [] for iso16 in isotopes2016}  # sorted by isotope
all_f = TiTs.select_from_db(db2016, 'file, type', 'Files', addCond='ORDER BY date')
for f, f_type in all_f:
    if '.mcp' in f and f_type in isotopes2016:
        all_file_flat_2016[f_type] += f,

files_flat_2016 = {}  # sorted by isotope
configs_2016 = {}
for isotope in isotopes2016:
    config, iso_files, ref_files = get_all_files_per_iso(db2016, isotope, final_2016_run)
    configs_2016[isotope] = config
    files_flat_2016[isotope] = iso_files
    if any(files_flat_2016.get('60_Ni', [])):
        files_flat_2016['60_Ni'] += ref_files
    else:
        files_flat_2016['60_Ni'] = ref_files

files_flat_2016['60_Ni'] = list(sorted(set(files_flat_2016['60_Ni'])))
TiTs.print_dict_pretty(files_flat_2016)

from Analysis.Nickel.NiCombineTildaPassiveAndMCP import find_tipa_file_to_mcp_file  # grml does import everything...

# print(files_flat_2016['58_Ni'])
# print(files_flat_2016['60_Ni'])
# print(os.path.split(find_tipa_file_to_mcp_file(files_flat_2016['60_Ni'][0])[1])[1])

txt_file16 = os.path.join(time_info_folder2016, 'bunch_lengths_mean.txt')

if create_bunch_len_txt_16:
    # get bunchlength for each files and store it in .txt file in roder nto to always have to import all the files etc.
    with open(txt_file16, 'wb') as f16:
        for iso in isotopes2016:
            for mcp_file in all_file_flat_2016[iso]:
                # abs_path_mcp = os.path.join(datafolder_mcp_2016, each)
                tipa_file = find_tipa_file_to_mcp_file(mcp_file)[1]
                print('working on iso: %s mcp_file: %s tipa_file: %s' % (iso, mcp_file, tipa_file))
                run_num = int(get_file_numbers([mcp_file])[0])
                # run_num = int(mcp_file.split('.')[0].split('Run')[1])
                print('mcp run number: %s' % run_num)
                if tipa_file:
                    tipa_file = os.path.split(tipa_file)[1]
                    abs_path_tipa = os.path.join(datafolder_tipa_2016, tipa_file)
                    meas = XMLImporter(abs_path_tipa)
                    rebinning = 100  # in ns
                    meas = Form.time_rebin_all_spec_data(meas, [rebinning] * meas.nrTracks, -1)
                    meas = TiTs.gate_specdata(meas)
                    new_meas, ret_dict = TiTs.calc_bunch_width_relative_to_peak_height(
                        meas, 25, show_plt=False, non_consectutive_time_bins_tolerance=10,
                        save_to_path=[os.path.join(
                            time_info_folder2016,
                            os.path.splitext(mcp_file)[0] + '_' + os.path.splitext(meas.file)[0] + '.png'),
                            os.path.join(
                                time_info_folder2016,
                                os.path.splitext(mcp_file)[0] + '_' + os.path.splitext(meas.file)[0] + '.pdf')
                        ],
                        time_around_bunch=(3.0, 10.0), additional_time_rChi=add_time_rchi_16

                    )
                    b_lengths = np.mean(ret_dict['bunch_lenght_us'], 0)  # get mean over all tracks for each scaler
                    # mean is ok, since all have same error and not weighting is needed.
                    rel_scaler = 1  # in 2016 only scaler 0 is valid because other pmts show very different behaviour,
                    #  probably due to some nim cabling
                    b_lengths_relevant_scalers = b_lengths[:rel_scaler]
                    mean_b_lengths, err_mean_b_lengths, rChi = Anal.weightedAverage(
                        b_lengths_relevant_scalers, [rebinning / 1000] * len(b_lengths_relevant_scalers))
                    gauss_res = ret_dict['gaussian_res']
                    scalers = ['sc0, sc1, sc2, sc3']
                    r_chis_file = []
                    for tr_ind, gauss_res_tr in enumerate(gauss_res):
                        r_chis_file += [],
                        for gauss_res_tr_sc in gauss_res_tr:
                            rchisq_tr_sc = gauss_res_tr_sc[0][-1]
                            r_chis_file[tr_ind] += rchisq_tr_sc,
                    rchisq_mean_sc_wise = np.mean(r_chis_file, 0)  # mean over all track scaler wise
                    rchisq_mean = np.mean(rchisq_mean_sc_wise[:rel_scaler])
                else:
                    mean_b_lengths = 0.0
                    err_mean_b_lengths = 0.0
                    rchisq_mean = 0.0
                f16.write(
                    bytes('%s\t%s\t%.6f\t%.6f\t%.2f\n'
                          % (mcp_file, tipa_file, mean_b_lengths, err_mean_b_lengths, rchisq_mean), 'UTF-8'))

    f16.close()

bunch_lenghts_2016_dict = {}  # 'filename': (len_us, err_len_us, tipa_f)

with open(txt_file16, 'r') as f16:
    lines = csv.reader(f16, delimiter='\t')
    for f, tipa_f, bun_len, err_bun_len, rchisq_m in lines:
        bunch_lenghts_2016_dict[f] = (float(bun_len), float(err_bun_len), float(rchisq_m), tipa_f)

TiTs.print_dict_pretty(bunch_lenghts_2016_dict)

if plot_time_struct_16:
    # plot some selected tiome projections for 60ni and 58ni from 2016 to
    #  see the distribution of the ions over time depending on th filling of iscool
    ni58Files = ['58Ni_no_protonTrigger_Run074.mcp',
                 '58Ni_no_protonTrigger_Run028.mcp',
                 '58Ni_no_protonTrigger_Run149.mcp',
                 '58Ni_no_protonTrigger_Run210.mcp',
                 '58Ni_no_protonTrigger_Run212.mcp']
    ni60Files = ['60Ni_no_protonTrigger_Run018.mcp',
                 '60Ni_no_protonTrigger_Run076.mcp',
                 # '60Ni_no_protonTrigger_Run027.mcp',  # very similar to 019
                 '60Ni_no_protonTrigger_Run019.mcp',
                 '60Ni_no_protonTrigger_Run035.mcp',
                 '60Ni_no_protonTrigger_Run084.mcp',
                 '60Ni_no_protonTrigger_Run089.mcp']
    ni61Files_bl0 = [
        '61Ni_no_protonTrigger_Run014.mcp',
        '61Ni_no_protonTrigger_Run010.mcp',
        '61Ni_no_protonTrigger_Run012.mcp']  # no need to show since similar bunch length then 014
    ni61Files_bl1 = [
        '61Ni_no_protonTrigger_Run120.mcp',
        '61Ni_no_protonTrigger_Run121.mcp',
        '61Ni_no_protonTrigger_Run123.mcp',
        # '61Ni_no_protonTrigger_Run124.mcp'   # very similar to 123
    ]  # some heavily overloaded files in 61 Ni
    ni61Files_all = [
        '61Ni_no_protonTrigger_Run014.mcp',
        '61Ni_no_protonTrigger_Run010.mcp',
        '61Ni_no_protonTrigger_Run012.mcp',
        '61Ni_no_protonTrigger_Run120.mcp',
        '61Ni_no_protonTrigger_Run121.mcp',
        '61Ni_no_protonTrigger_Run123.mcp',
        # '61Ni_no_protonTrigger_Run124.mcp'
    ]
    # iso_file_list = [ni58Files]
    # iso_file_list = [ni58Files, ni60Files]
    iso_file_list = [ni58Files, ni60Files, ni61Files_bl0, ni61Files_bl1]  #, ni61Files_all]
    isos_name = ['58_Ni', '60_Ni', '61_Ni_bl0', '61_Ni_bl1', '61_Ni']
    lims = [(52, 64), (52, 72), (49, 73), (50, 72), (50, 72)]
    for i, file_list in enumerate(iso_file_list):
        fig = plt.figure(1, (16, 16), facecolor='w')
        if plot_gauss:
            plt_axes = fig.add_axes([0.1, 0.40, 0.85, 0.58])
            res_axes = fig.add_axes([0.1, 0.1, 0.85, 0.25], sharex=plt_axes)
        else:
            plt_axes = fig.add_axes([0.1, 0.1, 0.85, 0.85])

        for mcp_file in file_list:
            if bunch_lenghts_2016_dict.get(mcp_file, False):
                print('working on mcp_file: %s' % mcp_file)
                b_len, b_len_err, rchisq_mean, tipa_file = bunch_lenghts_2016_dict[mcp_file]
                abs_path_tipa = os.path.join(datafolder_tipa_2016, tipa_file)
                meas = XMLImporter(abs_path_tipa)
                rebinning = 100  # in ns
                meas = Form.time_rebin_all_spec_data(meas, [rebinning] * meas.nrTracks, -1)
                meas = TiTs.gate_specdata(meas)
                if normalize_on_scans or mcp_file in [
                    '60Ni_no_protonTrigger_Run018.mcp',
                    '61Ni_no_protonTrigger_Run120.mcp'] or mcp_file in ni61Files_bl0:
                    num_of_scans = meas.nrScans[0]  # number of scans
                    if mcp_file == '60Ni_no_protonTrigger_Run018.mcp':
                        num_of_scans = 2
                    elif mcp_file == '61Ni_no_protonTrigger_Run120.mcp':
                        num_of_scans = 1 / 5
                    elif mcp_file in ni61Files_bl0:
                        num_of_scans = 100  # too many zeros for plotting
                    for tr_ind, tr_t_proj in enumerate(meas.t_proj):
                        for sc_ind, tr_sc_t_proj in enumerate(tr_t_proj):
                            meas.t_proj[tr_ind][sc_ind] = tr_sc_t_proj / num_of_scans
                else:
                    num_of_scans = 1  # number of scans set to one in order not to normalize on number of scans
                y = meas.t_proj[0][0] / num_of_scans # tr0 sc0
                x = meas.t[0]
                # print('file: %s skew: %s' % (mcp_file, scStats.skew(y)))
                # for rchisq calc only use area around peak
                new_meas, ret_dict = TiTs.calc_bunch_width_relative_to_peak_height(
                    meas, 25, show_plt=False, non_consectutive_time_bins_tolerance=10,
                    save_to_path='',  # do not save here
                    time_around_bunch=(3.0, 10.0), additional_time_rChi=add_time_rchi_16, fit_gaussian=plot_gauss
                )
                print('gaussian_res: ', ret_dict.get('gaussian_res'))
                gauss_off_res = ret_dict.get('gaussian_res')[0][0]  # tr0, sc0 for these examples
                mean_fit = gauss_off_res[0][0]
                err_mean_fit = gauss_off_res[1][0]
                sigma_fit = gauss_off_res[0][1]
                err_sigma_fit = gauss_off_res[1][1]
                amp_fit = gauss_off_res[0][2]
                off_fit = gauss_off_res[0][3]
                rChiSqFit = gauss_off_res[0][4]
                y_gauss = [Physics.gaussian_offset(x_i, mean_fit, sigma_fit, amp_fit, off_fit) / num_of_scans for x_i in x]
                residuals = [y[ind] - y_gauss[ind]
                             for ind, x_i in enumerate(x)]

                mcp_num = get_file_numbers([mcp_file])[0]
                tipa_num = get_file_numbers([tipa_file], mass_index=None)[0]
                cur_plt = plt_axes.plot(
                    x, y,
                    label='bunch length %.2f(%.0f)µs    run%s (tipa%s)\nGaussian: tof: %.2f(%.0f)    rChiSq: %.2f'
                          % (b_len, b_len_err * 100, mcp_num, tipa_num, mean_fit, err_mean_fit * 100, rChiSqFit),
                    linewidth=2)
                if plot_gauss:
                    res_plot = res_axes.plot(x, residuals, linewidth=2, color=cur_plt[-1].get_c())
                    gauss_plt = plt_axes.plot(x, y_gauss,
                                              label='', color=cur_plt[-1].get_c(), linewidth=2)
                save_to = os.path.join(time_info_folder2016, '%s.txt' % mcp_file.split('.')[0])

                with open(save_to, 'wb') as f:
                    f.write(bytes(
                        '%s\t%s\t%s\t%s\n' % ('t', 'cts', 'gauss', 'residuals'), 'UTF-8'))
                    for ind, cur_x in enumerate(x):
                        f.write(bytes(
                            '%.2f\t%.2f\t%.5f\t%.5f\n' % (cur_x, y[ind], y_gauss[ind], residuals[ind]), 'UTF-8'))
                f.close()

        font_s = 20
        plt_axes.legend(fontsize=font_s)
        [ax.set_xlabel('time of flight / µs', fontsize=font_s) for ax in fig.axes]
        plt_axes.set_ylabel('cts / a.u.', fontsize=font_s)
        if plot_gauss:
            res_axes.set_ylabel('residuals', fontsize=font_s)
        [ax.tick_params(axis='both', labelsize=font_s) for ax in fig.axes]
        [ax.locator_params(axis='y', nbins=6) for ax in fig.axes]
        [ax.set_xlim(lims[i][0], lims[i][1]) for ax in fig.axes]
        # plt.show(block=True)
        store_to = os.path.join(time_info_folder2016, '%s_bunch_length_evolution%s%s%s.pdf'
                                % (isos_name[i], '_gauss' if plot_gauss else '',
                                   '_pm_%d_us' % add_time_rchi_16,
                                   '_normalized_on_scans' if normalize_on_scans else ''))
        print('saving to: ', store_to)
        fig.savefig(store_to)
        fig.clear()
    #
''' now combine them '''

combined_txt_16 = os.path.join(time_info_folder2016, 'configs_time_info16.txt')
combined_mean_abs_len_txt_16 = os.path.join(time_info_folder2016, 'configs_time_info16_mean_abs.txt')
combined_mean_rel_len_txt_16 = os.path.join(time_info_folder2016, 'configs_time_info16_mean_rel.txt')
combined_dict_mean_16 = {}
with open(combined_txt_16, 'wb' if create_config_sum_file_16 else 'r') as combf16:
    header = '#A\tfiles(A)\ttipa_files(A)\tbunch_len(us)\tbunch_len_mean(us)\terr_bunch_len_mean(us)\tfiles(60)' \
             '\ttipa_files(60)\tbunch_lens(us)\tbunch_len_mean(us)\terr_bunch_len_mean(us)' \
             '\tratio A/60\terr_ratio A/60\tshift\terr_Shift\trChiSq\n'
    if create_config_sum_file_16:
        combf16.write(bytes(header, 'UTF-8'))
    for iso in isotopes2016:
        # A_2016 | file_2016(A) | b_len_2016 / us | err_b_len_2016 / us |
        #  files_2016(60) | b_len_60_2016 / us | err_b_len_60_2016 / us | ratio  b_len_2016 / b_len_60_2016 |
        #  err( b_len_2016 / b_len_60_2016)
        # + same for 2017, but in different file...
        mass = int(iso[:2])
        mass_16 = mass - 0.05
        combined_dict_mean_16[iso] = [mass_16, [], [], [], []]  # bunch_lens, err bunch_lens, ratios, err ratios
        if iso == '60_Ni':
            files = files_flat_2016[iso]
            for ref_file in files:

                ref_file_nums = get_file_numbers([ref_file])
                iso_len, err_iso_len, iso_rchisq_m, tipa_file = bunch_lenghts_2016_dict.get(ref_file, (0.0, 0.0, 0.0, ''))
                if iso_len > 0.0:
                    ref_tipa_file_nums = get_file_numbers([tipa_file], mass_index=None)

                    iso_file_nums = ref_file_nums
                    iso_bunch_lens = [iso_len]
                    iso_bun_len_mean = iso_len
                    iso_err_bun_len_mean = err_iso_len

                    ref_bunch_lens = [iso_len]
                    ref_bun_len_mean = iso_len
                    ref_err_bun_len_mean = err_iso_len

                    ratio = 1.0
                    err_ratio = 0.0

                    to_print = '%.2f\t%s\t%s\t%s\t%.5f\t%.5f\t%s\t%s\t%s\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n' \
                               % (mass_16,
                                  str(iso_file_nums)[1:-1], str(iso_tipa_file_nums)[1:-1], str(iso_bunch_lens)[1:-1],
                                  iso_bun_len_mean, iso_err_bun_len_mean,
                                  str(ref_file_nums)[1:-1], str(ref_tipa_file_nums)[1:-1], str([iso_len])[1:-1],
                                  iso_len, err_iso_len,
                                  ratio, err_ratio, 0.0, 0.0, iso_rchisq_m
                                  )
                    if create_config_sum_file_16:
                        combf16.write(bytes(to_print, 'UTF-8'))
                    combined_dict_mean_16[iso][1] += iso_len,  # bunch lens
                    combined_dict_mean_16[iso][2] += err_iso_len,  # err bunch lens
                    combined_dict_mean_16[iso][3] += 0.0,  # ratios
                    combined_dict_mean_16[iso][4] += 0.0,  # err ratios

        else:
            for refs_bef, iso_f, refs_after in configs_2016[iso]:
                iso_bunch_lens = []
                iso_err_bunch_lens = []
                iso_tipa_files = []
                for each in iso_f:
                    iso_len, err_iso_len, iso_rchisq_m, tipa_file = bunch_lenghts_2016_dict.get(each, (0.0, 0.0, 0.0, ''))
                    if iso_len > 0.0:  # some migth not exist, becaue not every .mcp file has tipa file
                        iso_bunch_lens += [iso_len]
                        iso_err_bunch_lens += [err_iso_len]
                        iso_tipa_files += [tipa_file]

                if any(iso_bunch_lens):
                    iso_bun_len_mean, iso_err_bun_len_mean, isorChi = Anal.weightedAverage(iso_bunch_lens,
                                                                                           iso_err_bunch_lens)
                    iso_file_nums = get_file_numbers(iso_f)
                    # print(iso_file_num)

                    iso_tipa_file_nums = get_file_numbers(iso_tipa_files, mass_index=None)
                    # print(iso_tipa_files)
                    # print(iso_tipa_file_nums)

                refs = refs_bef + refs_after
                ref_bunch_lens = []
                ref_err_bunch_lens = []
                ref_tipa_files = []
                for each in refs:
                    ref_len, err_ref_len, ref_rchisq_m, ref_tipa_file = bunch_lenghts_2016_dict.get(each,
                                                                                                    (0.0, 0.0, 0.0, ''))
                    if ref_len > 0.0:
                        ref_bunch_lens += [ref_len]
                        ref_err_bunch_lens += [err_ref_len]
                        ref_tipa_files += [ref_tipa_file]
                if any(ref_bunch_lens):
                    ref_bun_len_mean, ref_err_bun_len_mean, refrChi = Anal.weightedAverage(ref_bunch_lens,
                                                                                           ref_err_bunch_lens)
                    ref_file_nums = get_file_numbers(refs)

                    ref_tipa_file_nums = get_file_numbers(ref_tipa_files, mass_index=None)
                    # print(ref_tipa_files)
                    # print(ref_tipa_file_nums)

                    ratio = iso_bun_len_mean / ref_bun_len_mean
                    err_ratio = np.sqrt(
                        (iso_err_bun_len_mean / ref_bun_len_mean) ** 2 +
                        (iso_bun_len_mean * ref_err_bun_len_mean / ref_bun_len_mean ** 2) ** 2
                    )

                if any(iso_bunch_lens):
                    to_print = '%.2f\t%s\t%s\t%s\t%.5f\t%.5f\t%s\t%s\t%s\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n' \
                               % (mass_16,
                                  str(iso_file_nums)[1:-1], str(iso_tipa_file_nums)[1:-1], str(iso_bunch_lens)[1:-1],
                                  iso_bun_len_mean, iso_err_bun_len_mean,
                                  str(ref_file_nums)[1:-1], str(ref_tipa_file_nums)[1:-1], str(ref_bunch_lens)[1:-1],
                                  ref_bun_len_mean, ref_err_bun_len_mean,
                                  ratio, err_ratio, 0.0, 0.0, iso_rchisq_m
                                  )
                    if create_config_sum_file_16:
                        combf16.write(bytes(to_print, 'UTF-8'))

                    combined_dict_mean_16[iso][1] += iso_bun_len_mean,  # bunch lens
                    combined_dict_mean_16[iso][2] += iso_err_bun_len_mean,  # err bunch lens
                    combined_dict_mean_16[iso][3] += ratio,  # ratios
                    combined_dict_mean_16[iso][4] += err_ratio,  # err ratios

# print('result for combined:')
# TiTs.print_dict_pretty(combined_dict_mean_16)
if create_mean_iso_bun_len_txt_file_16:
    with open(combined_mean_abs_len_txt_16, 'wb') as abs_mean_f16:
        header = 'iso\tbunch len mean\terr bunch len mean\trChi\terr applied rChi\n'
        abs_mean_f16.write(bytes(header, 'UTF-8'))
        for iso in isotopes2016:
            if any(combined_dict_mean_16[iso][1]):
                val, err, rChi = Anal.weightedAverage(combined_dict_mean_16[iso][1], combined_dict_mean_16[iso][2])
                err_appl_rChi = Anal.applyChi(err, rChi)
                to_print = '%.2f\t%.5f\t%.5f\t%.5f\t%.5f\n' % (combined_dict_mean_16[iso][0],
                                                               val, err, rChi, err_appl_rChi)
                abs_mean_f16.write(bytes(to_print, 'UTF-8'))
    with open(combined_mean_rel_len_txt_16, 'wb') as rel_mean_f16:
        header = 'iso\tratio mean\terr ratio mean\trChi\terr applied rChi\n'
        rel_mean_f16.write(bytes(header, 'UTF-8'))
        for iso in isotopes2016:
            if iso != '60_Ni':
                if any(combined_dict_mean_16[iso][1]):
                    val, err, rChi = Anal.weightedAverage(combined_dict_mean_16[iso][3], combined_dict_mean_16[iso][4])
                    err_appl_rChi = Anal.applyChi(err, rChi)
                    to_print = '%.2f\t%.5f\t%.5f\t%.5f\t%.5f\n' % (combined_dict_mean_16[iso][0],
                                                                   val, err, rChi, err_appl_rChi)
                    rel_mean_f16.write(bytes(to_print, 'UTF-8'))

# now creat file for 2017:
combined_txt_17 = os.path.join(time_info_folder17, 'configs_time_info17.txt')
combined_mean_abs_len_txt_17 = os.path.join(time_info_folder17, 'configs_time_info17_mean_abs.txt')
combined_mean_rel_len_txt_17 = os.path.join(time_info_folder17, 'configs_time_info17_mean_rel.txt')
combined_dict_mean_17 = {}

with open(combined_txt_17, 'wb' if create_config_sum_file_17 else 'r') as combf17:
    header = '#A\tfiles(A)\tbunch_len(us)\tbunch_len_mean(us)\terr_bunch_len_mean(us)\tfiles(60)' \
             '\tbunch_lens(us)\tbunch_len_mean(us)\terr_bunch_len_mean(us)' \
             '\tratio A/60\terr_ratio A/60\tshift\terr_Shift\trChiSq\n'
    if create_config_sum_file_17:
        combf17.write(bytes(header, 'UTF-8'))
    for iso in isotopes_17:
        # A_2016 | file_2016(A) | b_len_2016 / us | err_b_len_2016 / us |
        #  files_2016(60) | b_len_60_2016 / us | err_b_len_60_2016 / us | ratio  b_len_2016 / b_len_60_2016 |
        #  err( b_len_2016 / b_len_60_2016)
        # + same for 2017, but in different file...
        mass = int(iso[:2])
        mass_17 = mass + 0.05
        combined_dict_mean_17[iso] = [mass_17, [], [], [], []]  # bunch_lens, err bunch_lens, ratios, err ratios
        if iso == '60_Ni':
            files = files_flat_2017[iso]
            for ref_file in files:

                ref_file_nums = get_file_numbers([ref_file], user_overwrite=overwrites)
                iso_len, err_iso_len, iso_rchisq = bunch_lenghts_2017_dict.get(ref_file, (0.0, 0.0))
                if iso_len > 0.0:

                    iso_file_nums = ref_file_nums
                    iso_bunch_lens = [iso_len]
                    iso_bun_len_mean = iso_len
                    iso_err_bun_len_mean = err_iso_len

                    ref_bunch_lens = [iso_len]
                    ref_bun_len_mean = iso_len
                    ref_err_bun_len_mean = err_iso_len

                    ratio = 1.0
                    err_ratio = 0.0

                    to_print = '%.2f\t%s\t%s\t%.5f\t%.5f\t%s\t%s\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n' \
                               % (mass_17,
                                  str(iso_file_nums)[1:-1], str(iso_bunch_lens)[1:-1],
                                  iso_bun_len_mean, iso_err_bun_len_mean,
                                  str(ref_file_nums)[1:-1], str(ref_bunch_lens)[1:-1],
                                  ref_bun_len_mean, ref_err_bun_len_mean,
                                  ratio, err_ratio, 0.0, 0.0, iso_rchisq
                                  )
                    if create_config_sum_file_17:
                        combf17.write(bytes(to_print, 'UTF-8'))
                    combined_dict_mean_17[iso][1] += iso_len,  # bunch lens
                    combined_dict_mean_17[iso][2] += err_iso_len,  # err bunch lens
                    combined_dict_mean_17[iso][3] += ratio,  # ratios
                    combined_dict_mean_17[iso][4] += err_ratio,  # err ratios
        else:
            for refs_bef, iso_f, refs_after in configs_2017[iso]:
                iso_bunch_lens = []
                iso_err_bunch_lens = []
                for each in iso_f:
                    iso_len, err_iso_len, iso_rchisq = bunch_lenghts_2017_dict.get(each, (0.0, 0.0, 0.0))
                    if iso_len > 0.0:  # some migth not exist, becaue not every .mcp file has tipa file
                        iso_bunch_lens += [iso_len]
                        iso_err_bunch_lens += [err_iso_len]

                if any(iso_bunch_lens):
                    iso_bun_len_mean, iso_err_bun_len_mean, isorChi = Anal.weightedAverage(iso_bunch_lens,
                                                                                           iso_err_bunch_lens)
                    iso_file_nums = get_file_numbers(iso_f, user_overwrite=overwrites)
                    # print(iso_file_num)

                refs = refs_bef + refs_after
                ref_bunch_lens = []
                ref_err_bunch_lens = []
                ref_tipa_files = []
                for each in refs:
                    ref_len, err_ref_len, ref_rchisq = bunch_lenghts_2017_dict.get(each, (0.0, 0.0, 0.0))
                    if ref_len > 0.0:
                        ref_bunch_lens += [ref_len]
                        ref_err_bunch_lens += [err_ref_len]
                if any(ref_bunch_lens):
                    ref_bun_len_mean, ref_err_bun_len_mean, refrChi = Anal.weightedAverage(ref_bunch_lens,
                                                                                           ref_err_bunch_lens)
                    ref_file_nums = get_file_numbers(refs, user_overwrite=overwrites)

                    ratio = iso_bun_len_mean / ref_bun_len_mean
                    err_ratio = np.sqrt(
                        (iso_err_bun_len_mean / ref_bun_len_mean) ** 2 +
                        (iso_bun_len_mean * ref_err_bun_len_mean / ref_bun_len_mean ** 2) ** 2
                    )

                    to_print = '%.2f\t%s\t%s\t%.5f\t%.5f\t%s\t%s\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n' \
                               % (mass_17,
                                  str(iso_file_nums)[1:-1], str(iso_bunch_lens)[1:-1],
                                  iso_bun_len_mean, iso_err_bun_len_mean,
                                  str(ref_file_nums)[1:-1], str(ref_bunch_lens)[1:-1],
                                  ref_bun_len_mean, ref_err_bun_len_mean,
                                  ratio, err_ratio, 0.0, 0.0, iso_rchisq
                                  )
                    if create_config_sum_file_17:
                        combf17.write(bytes(to_print, 'UTF-8'))
                    combined_dict_mean_17[iso][1] += iso_bun_len_mean,  # bunch lens
                    combined_dict_mean_17[iso][2] += iso_err_bun_len_mean,  # err bunch lens
                    combined_dict_mean_17[iso][3] += ratio,  # ratios
                    combined_dict_mean_17[iso][4] += err_ratio,  # err ratios

if create_mean_iso_bun_len_txt_file_17:
    with open(combined_mean_abs_len_txt_17, 'wb') as abs_mean_f17:
        header = 'iso\tbunch len mean\terr bunch len mean\trChi\terr applied rChi\n'
        abs_mean_f17.write(bytes(header, 'UTF-8'))
        for iso in isotopes_17:
            if any(combined_dict_mean_17[iso][1]):
                val, err, rChi = Anal.weightedAverage(combined_dict_mean_17[iso][1], combined_dict_mean_17[iso][2])
                err_appl_rChi = Anal.applyChi(err, rChi)
                to_print = '%.2f\t%.5f\t%.5f\t%.5f\t%.5f\n' % (combined_dict_mean_17[iso][0],
                                                               val, err, rChi, err_appl_rChi)
                abs_mean_f17.write(bytes(to_print, 'UTF-8'))
    with open(combined_mean_rel_len_txt_17, 'wb') as rel_mean_f17:
        header = 'iso\tratio mean\terr ratio mean\trChi\terr applied rChi\n'
        rel_mean_f17.write(bytes(header, 'UTF-8'))
        for iso in isotopes_17:
            if iso != '60_Ni':  # no ratio for 60 ni
                if any(combined_dict_mean_17[iso][1]):
                    val, err, rChi = Anal.weightedAverage(combined_dict_mean_17[iso][3], combined_dict_mean_17[iso][4])
                    err_appl_rChi = Anal.applyChi(err, rChi)
                    to_print = '%.2f\t%.5f\t%.5f\t%.5f\t%.5f\n' % (combined_dict_mean_17[iso][0],
                                                                   val, err, rChi, err_appl_rChi)
                    rel_mean_f17.write(bytes(to_print, 'UTF-8'))
