import os
import numpy as np
from KingFitter import KingFitter
import random
from time import clock, strftime, time
import collections
from joblib import Parallel, delayed

start = clock()
print('Start: ', start)
date = strftime("%d-%m-%Y, %H:%M:%S")

db = 'C:\\PythonProjects\\PolliFit\\test\\Project\\CaCombined_3DKing.sqlite'

'''performing a King fit analysis with values and uncertainties taken from F. Gebert's Ca paper'''
litval = {'42_Ca': [0.21, .007],
          '44_Ca': [0.29, .009],
          '48_Ca': [-0.005, .006]}

litvals = collections.OrderedDict(sorted(litval.items()))

'''All isotopes to calculate charge radii for'''
isotopeList = ['42_Ca', '44_Ca', '48_Ca']

print(litvals)

'''All runs that should be included to 3D KingPlot. It is important that all litvals isotopes have a corresponding shift
   in all runs.
   Run0 = D1 line
   Run2 = d -> p trans'''
runs = ['Run0', 'Run2']

alpha = 22
i_max = 1000 # runs
int_steps = 1000 #+- interval steps
max_int = 0.01 # +- maximum interval width around litval


'''Create protocol files & header lines'''
f_var_all = open('3DKing_var_all_' + str(int(time())) + '.txt', 'w')
f_var_all.write('3D King Plot Monte Carlo type analysis.\nThis file includes all varied dr^2 and the corresponding chi^2.\n'
                'Start: '+date + '\nAlpha: '+str(alpha)+', Runs: '+str(i_max)+', Max interval width +-: '+str(max_int)+', Interval steps +-: '+str(int_steps)+'\n---------------------\n')
# f_var_best = open('3DKing_var_best_'+ str(int(time()))+'.txt', 'w')
# f_var_best.write('3D King Plot Monte Carlo type analysis.\nThis file includes the best varied dr^2 and the corresponding chi^2\n'
#                  'Start: '+date + '\nAlpha: '+str(alpha)+', Circles: '+ str(a_max)+', Runs per Circle: '+str(i_max)+', Max interval width +-: '+str(max_int)+', Interval steps +-: '+str(int_steps)+'\n---------------------\n')
f_var_all.write('ID, chi^2, ')
# f_var_best.write('ID, chi^2, ')

for key in litvals.keys():
    f_var_all.write('dr^2 ' + key + ', ')
    # f_var_best.write('dr^2 ' + key + ', ')
f_var_all.write('F1, F1_d, k1, k1_d, Correlation factor1, F2, F2_d, k2, k2_d, Correlation factor2, Time\n')
# f_var_best.write('F1, F1_d, k1, k1_d, Correlation factor1, F2, F2_d, k2, k2_d, Correlation factor2, Time\n')

# No Projection needed
# f_proj_all_run0 = open('3DKing_proj_all_run0_'+ str(int(time()))+'.txt', 'w')
# f_proj_all_run0.write('3D King Plot Monte Carlo type analysis.\nThis file includes all projected dr^2 with corresponding chi^2 of Run0\n'
#                  'Start: '+date + '\nAlpha: '+str(alpha)+', Circles: '+ str(a_max)+', Runs per Circle: '+str(i_max)+', Max interval width +-: '+str(max_int)+', Interval steps +-: '+str(int_steps)+'\n---------------------\n')
# f_proj_best_run0 = open('3DKing_proj_best_run0_'+ str(int(time()))+'.txt', 'w')
# f_proj_best_run0.write('3D King Plot Monte Carlo type analysis.\nThis file includes the best projected dr^2 with corresponding chi^2 of Run0\n'
#                  'Start: '+date + '\nAlpha: '+str(alpha)+', Circles: '+ str(a_max)+', Runs per Circle: '+str(i_max)+', Max interval width +-: '+str(max_int)+', Interval steps +-: '+str(int_steps)+'\n---------------------\n')
# f_proj_all_run2 = open('3DKing_proj_all_run2_'+ str(int(time()))+'.txt', 'w')
# f_proj_all_run2.write('3D King Plot Monte Carlo type analysis.\nThis file includes all projected dr^2 with corresponding chi^2 of Run2\n'
#                  'Start: '+date + '\nAlpha: '+str(alpha)+', Circles: '+ str(a_max)+', Runs per Circle: '+str(i_max)+', Max interval width +-: '+str(max_int)+', Interval steps +-: '+str(int_steps)+'\n---------------------\n')
# f_proj_best_run2 = open('3DKing_proj_best_run2_'+ str(int(time()))+'.txt', 'w')
# f_proj_best_run2.write('3D King Plot Monte Carlo type analysis.\nThis file includes the best projected dr^2 with corresponding chi^2 of Run2\n'
#                  'Start: '+date + '\nAlpha: '+str(alpha)+', Circles: '+ str(a_max)+', Runs per Circle: '+str(i_max)+', Max interval width +-: '+str(max_int)+', Interval steps +-: '+str(int_steps)+'\n---------------------\n')
# f_proj_all_run0.write('ID, chi^2, ')
# f_proj_best_run0.write('ID, chi^2, ')
# f_proj_all_run2.write('ID, chi^2, ')
# f_proj_best_run2.write('ID, chi^2, ')
#
# for key in isotopeList:
#     f_proj_all_run0.write('dr^2 ' + key + ', Delta dr^2 ' + key + ', ')
#     f_proj_best_run0.write('dr^2 ' + key + ', Delta dr^2 ' + key + ', ')
#     f_proj_all_run2.write('dr^2 ' + key + ', Delta dr^2 ' + key + ', ')
#     f_proj_best_run2.write('dr^2 ' + key + ', Delta dr^2 ' + key + ', ')
#
# f_proj_all_run0.write('\n')
# f_proj_all_run2.write('\n')
# f_proj_best_run0.write('\n')
# f_proj_best_run2.write('\n')



def chiSq(modIS1, modIS1_err, modIS2, modIS2_err, modChr, litVals, litVals_err, a1, b1, a2, b2):
    chi1 = 0
    chi2 = 0
    chi3 = 0

    for i in range(len(modIS1)):
        chi1 = chi1 + ((modIS1[i] - (b1*modChr[i] + a1))/modIS1_err[i])**2
        #print('ch1: ', chi1, modIS1[i], b1*modChr[i] + a1, modIS1_err[i])

    for i in range(len(modIS2)):
        chi2 = chi2 + ((modIS2[i] - (b2*modChr[i] + a2))/modIS2_err[i])**2
        #print('ch2: ', chi2)

    for i in range(len(modChr)):
        chi3 = chi3 + (((litVals[i] - alpha) - modChr[i])/litVals_err[i])**2

    chiSq = chi1 + chi2 + chi3
    # print('Chi1: ', chi1)
    # print('Chi2: ', chi2)
    # print('Chi3: ', chi3)
    return chiSq



litChr = KingFitter(db, litvals=litvals, ref_run=runs[0])
litChr.calcRedVar(run=runs[0])
#print('X(ChargeR): ', litChr.x, litChr.xerr)

print('Running...')
line1 = KingFitter(db, litvals=litvals, ref_run=runs[0])
line1.showing = False

line2 = KingFitter(db, litvals=litvals, ref_run=runs[1])
line2.showing = False

def calc_step(i):
    if i == 0:
        varChr = litvals
        y_reset = True
    else:
        # varChr = litvals
        varChr = {'42_Ca': [0.21+max_int/int_steps*random.randint(-int_steps-1, int_steps), .007],
                  '44_Ca': [0.29+max_int/int_steps*random.randint(-int_steps-1, int_steps), .009],
                  '48_Ca': [-0.005+max_int/int_steps*random.randint(-int_steps-1, int_steps), .006]
                  }
        varChr = collections.OrderedDict(sorted(varChr.items()))
        y_reset=False

    line1.litvals = varChr
    line1.reset_y_values = y_reset
    line1.kingFit(run=runs[0], alpha=alpha, findBestAlpha=False, print_coeff=False, print_information=False, results_to_db=False)
        # line1.calcRedVar(run=runs[0], findBestAlpha=False, alpha=alpha)
        # line1.fit(run=runs[0], showplot=True, print_corr_coeff=True)
        # print('X(ChargeR): ', king.x, king.xerr)
        # print('Y(IS): ', king.y, king.yerr)
        # print('a, b:', king.a, king.b)

    line2.litvals = varChr
    line2.reset_y_values = y_reset
    line2.kingFit(run=runs[1], alpha=alpha, findBestAlpha=False, print_coeff=False, print_information=False, results_to_db=False)
        # line2.calcRedVar(run=runs[1], findBestAlpha=False, alpha=alpha)
        # line2.fit(run=runs[1], showplot=False, print_corr_coeff=True)
        # print('X(ChargeR): ', king2.x, king2.xerr)
        # print('Y(IS): ', king2.y, king2.yerr)
        # print('a, b:', king2.a, king2.b)

    cSq = chiSq(line1.y, line1.yerr, line2.y, line2.yerr, line1.x, litChr.x, litChr.xerr, line1.a, line1.b, line2.a,
                    line2.b)

        # cr_line1 = line1.calcChargeRadii(isotopes=isotopeList,run=runs[0], save_in_db=False, print_results=False, print_information=False)
        # cr_line2 = line2.calcChargeRadii(isotopes=isotopeList, run=runs[1], save_in_db=False, print_results=False, print_information=False)

        # print(cr_line1)
        # print(cr_line2)


        # print('cSq: ', cSq)

    f_var_all_string = str(i) + ', ' + str(cSq) + ', '
    for key in litvals.keys():
        f_var_all_string += str(varChr[key][0]) + ', '
    f_var_all_string += str(line1.b) + ', ' + str(line1.berr) + ', ' + str(line1.a) + ', ' + str(line1.aerr) + ', ' + str(line1.a_b_correlation) + ', ' + str(line2.b) + ', ' + str(line2.berr) + ', ' + str(line2.a) + ', ' + str(line2.aerr) + ', ' + str(line2.a_b_correlation) + ', ' + str(clock()) + '\n'

        # f_proj_all_run0_string = str(counter) + ', ' + str(cSq) + ', '
        # for key in isotopeList:
        #     f_proj_all_run0_string += str(cr_line1[key][0]) + ', ' + str(cr_line1[key][1]) + ', '
        # f_proj_all_run0_string += '\n'
        #
        # f_proj_all_run2_string = str(counter) + ', ' + str(cSq) + ', '
        # for key in isotopeList:
        #     f_proj_all_run2_string += str(cr_line2[key][0]) + ', ' + str(cr_line2[key][1]) + ', '
        # f_proj_all_run2_string += '\n'

        # if cSq < minChi[0]:
        #     minChi = [cSq, varChr, [line1.a, line1.b, line2.a, line2.b]]
        #     f_var_best_string = f_var_all_string
            # f_proj_best_run0_string = f_proj_all_run0_string
            # f_proj_best_run2_string = f_proj_all_run2_string

        # print('Var all string: ', f_var_all_string)

    f_var_all.write(f_var_all_string)
        # f_proj_all_run0.write(f_proj_all_run0_string)
        # f_proj_all_run2.write(f_proj_all_run2_string)

    print(i+1, "/", i_max)

# Parallel(n_jobs=-1, backend='threading')(delayed(calc_step)(i) for i in range(i_max))
for i in range(i_max):
    calc_step(i)

    # f_var_best.write(f_var_best_string)
    # f_proj_best_run0.write(f_proj_best_run0_string)
    # f_proj_best_run2.write(f_proj_best_run2_string)


f_var_all.close()
# f_var_best.close()
# f_proj_all_run0.close()
# f_proj_all_run2.close()
# f_proj_best_run0.close()
# f_proj_best_run2.close()


# print('MinChiSq: ', minChi[0])
# print('Best ChR: ', minChi[1])
# print('Best F, k: ', minChi[2])
# print('MaxChiSq: ', maxChi[0])
# print('Worst ChR: ', maxChi[1])
# print('Worst F, k: ', maxChi[2])
# print('LitVal:', litvals)

# f = open('3DKing.txt', 'a')
#
# f.write('\n'+str(minChi[0])+ ', ' +str(minChi[1]['106_Cd'][0])+ ', ' +str(minChi[1]['108_Cd'][0])
#         + ', ' +str(minChi[1]['110_Cd'][0])+ ', ' +str(minChi[1]['111_Cd'][0])+ ', ' +str(minChi[1]['112_Cd'][0])
#         + ', ' +str(minChi[1]['113_Cd'][0])+ ', ' +str(minChi[1]['116_Cd'][0])+', '+str(i))
#
# f.close()

'''Vielleicht doch immer bestes alpha suchen lassen?
   Nochmal checken ob calcChargeRadii auch nix ändert bzw. alles in der richtigen Reihenfolge in die DB geschrieben wird
   und ausgelesen wird.
'''

stop = clock()
print('Stop: ', stop)

