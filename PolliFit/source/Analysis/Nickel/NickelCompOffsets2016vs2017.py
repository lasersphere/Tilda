"""
Created on 

@author: simkaufm

Module Description: Module to compare and store the offsets from 2017 and 2017
"""

import os
import numpy as np

import Analyzer as Anal
from TildaTools import get_file_numbers

''' working directory: '''

workdir17 = 'E:\\Workspace\\OwnCloud\\Projekte\\COLLAPS\\Nickel\\' \
          'Measurement_and_Analysis_Simon\\Ni_workspace2017\\Ni_2017'

datafolder17 = os.path.join(workdir17, 'sums')

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

datafolder2016 = os.path.join(workdir2016, 'Ni_April2016_mcp')

db2016 = os.path.join(workdir2016, 'Ni_workspace.sqlite')
runs2016 = ['wide_gate_asym', 'wide_gate_asym_67_Ni']
final_2016_run = 'wide_gate_asym'


isotopes2016 = ['%s_Ni' % i for i in range(58, 71)]
isotopes2016.remove('69_Ni')
isotopes2016.remove('60_Ni')
odd_isotopes2016 = [iso for iso in isotopes2016 if int(iso[:2]) % 2]

""" get file numbers: """

""" compare them """


def compare_offset(db, run, iso, user_overwrite):
    to_print = []
    offsets, offset_errs, conf = Anal.combineShiftOffsetPerBunchDisplay(iso, run, db, show_plot=False)
    for i, each in enumerate(conf):
        avg_ref, err_ref, rChi_ref = Anal.weightedAverage(offsets[i][0], offset_errs[i][0])
        avg_iso, err_iso, rChi_iso = Anal.weightedAverage(offsets[i][1], offset_errs[i][1])
        mean_ref_err = Anal.applyChi(err_ref, rChi_ref)
        mean_iso_err = Anal.applyChi(err_iso, rChi_iso)
        ratio = avg_iso / avg_ref
        ratio_err = np.sqrt(
            (mean_iso_err / avg_ref) ** 2 +
            (mean_ref_err * avg_iso / avg_ref ** 2) ** 2
        )  # from error prop
        ref_files = each[0] + each[2]
        iso_files = each[1]

        ref_file_numbers = get_file_numbers(ref_files, mass_index=[0], user_overwrite=user_overwrite)
        iso_file_numbers = get_file_numbers(iso_files, mass_index=[0], user_overwrite=user_overwrite)

        ref_files_str = ''
        for i, each in enumerate(ref_file_numbers):
            new_l = ', \\newline ' if i and (i + 1) % 3 == 0 and not (i + 1) == len(ref_file_numbers) else ', '
            ref_files_str += each + new_l
        ref_files_str = ref_files_str[:-2]

        iso_files_str = ''
        for i, each in enumerate(iso_file_numbers):
            new_l = ', \\newline ' if i and (i + 1) % 3 == 0 and not (i + 1) == len(ref_file_numbers) else ', '
            iso_files_str += each + new_l
        iso_files_str = iso_files_str[:-2]

        mean_ref_err = max(0.01, mean_ref_err)  # error 0 makes no sense
        mean_iso_err = max(0.01, mean_iso_err)

        to_print += (
            iso_files_str, avg_iso, mean_iso_err * 100,
            ref_files_str, avg_ref, mean_ref_err * 100,
            ratio, ratio_err),
    all_ratios = [r[-2] for r in to_print]
    all_ratio_errs = [r[-1] for r in to_print]
    w_avg, w_err, rChi = Anal.weightedAverage(all_ratios, all_ratio_errs)
    w_er_final = Anal.applyChi(w_err, rChi)

    return to_print, w_avg, w_er_final


default_to_print = (
            '', 0, 0,
            0, 0, 0,
            0, 0)

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
str_to_print = ''
str_to_print_orig = ''
for iso in isotopes2016:
    mass = int(iso[:2])
    pr_16, w_avg_16, w_er_final_16 = compare_offset(db2016, final_2016_run, iso, overwrites)
    print('done with 2016 %s' % iso)
    if iso in isotopes_17:
        pr_17, w_avg_17, w_er_final_17 = compare_offset(db17, final_2017_run, iso, overwrites)
        print('done with 2017 %s' % iso)
    else:
        pr_17 = []
    if len(pr_16) > len(pr_17):
        dif = len(pr_16) - len(pr_17)
        pr_17 += [default_to_print] * dif
    elif len(pr_17) > len(pr_16):
        dif = len(pr_17) - len(pr_16)
        pr_16 += [default_to_print] * dif
    for i, each in enumerate(pr_16):
        print(each)
        print(pr_17[i])
        str_to_print += '%d & ' % mass
        if each[0]:
            str_to_print += '%s & %.2f(%.0f) & %s & %.2f(%.0f) & %.2f(%.0f) & ' % each
        else:
            str_to_print += ' &  &  &  & '
        if pr_17[i][0]:
            str_to_print += '%s & %.2f(%.0f) & %s & %.2f(%.0f) & %.2f(%.0f)\\\\ \n' % pr_17[i]
        else:
            str_to_print += ' &  &  &  & \\\\ \n'
    str_to_print += '\hline \n'
    
    # now also for .csv , e.g. origin
    print('starting with origin output')
    for i, each in enumerate(pr_16):
        print(each)
        print(pr_17[i])

        str_to_print_orig += '%d\t%.2f\t%.2f\t' % (mass, mass-0.05, mass+0.05)
        if each[0]:
            # undo latex stuff
            each = list(each)
            each[2] = each[2] / 100
            each[5] = each[5] / 100
            print('length of each: ', len(each), each)
            each = tuple(each)
            to_append = '%s\t%.5f\t%.5f\t%s\t%.5f\t%.5f\t%.5f\t%.5f\t' % each
            to_append = to_append.replace('\\newline ', '')
            to_append += '%.5f\t%.5f\t' % (w_avg_16, w_er_final_16)
            str_to_print_orig += to_append
        else:
            str_to_print_orig += '\t\t\t\t\t\t\t\t\t\t'
        if pr_17[i][0]:
            pr_17_i = list(pr_17[i])
            pr_17_i[2] = pr_17_i[2] / 100
            pr_17_i[5] = pr_17_i[5] / 100
            pr_17_i = tuple(pr_17_i)
            to_app = '%s\t%.5f\t%.5f\t%s\t%.5f\t%.5f\t%.5f\t%.5f\t' % pr_17_i
            to_app = to_app.replace('\\newline ', '')
            to_app += '%.5f\t%.5f \n' % (w_avg_17, w_er_final_17)
            str_to_print_orig += to_app
        else:
            str_to_print_orig += '\t\t\t\t\t\t\t\t\t \n'


print('\hline \hline')
print(' & 2016 & & & & & 2017 & & & & \\\\')
print('\hline')
print('A & file(A) & norm. offs.(A) & files(60) & norm. offs.(60) & ratio A/60 '
      '& file(A) & norm. offs.(A) & files(60) & norm. offs.(60) & ratio A/60 \\\\')
print('\hline')
print(str_to_print)


print('for origin:')
print('A\tA2016\tA2017\tfile(A)\tnorm. offs.(A)\terr. norm. offs.(A)\tfiles(60)\tnorm. offs.(60)'
      '\terr. norm. offs.(60)\tratio A/60\terr. ratio A/60\tmean ratio A/60\terr. mean ratio A/60'
      '\tfile(A)\tnorm. offs.(A)\terr. norm. offs.(A)\tfiles(60)\tnorm. offs.(60)'
      '\terr. norm. offs.(60)\tratio A/60\terr. ratio A/60\tmean ratio A/60\terr. mean ratio A/60')
print(str_to_print_orig)


#
# test = ['60_Ni_trs5_run068.xml', '60_Ni_trs_run069.xml',
#         '60_Ni_trs_run073.xml', '60_Ni_trs_run074.xml', '60_Ni_trs_run075.xml']
# print(get_file_numbers(test, mass_index=[0]))
