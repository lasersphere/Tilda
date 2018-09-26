"""
Created on 

@author: simkaufm

Module Description:  This should plot the radii of Nickel along the radii of copper, maybe Zn or Ga as well
"""
import os
import MPLPlotter
import Tools
import numpy as np

font_size = 16

''' Cu '''
# all infos from: Cu charge radii reveal a weak sub-shell effect at N=40
# @article{PhysRevC.93.064318,
#   title = {Cu charge radii reveal a weak sub-shell effect at $N=40$},
#   author = {Bissell, M. L. and Carette, T. and Flanagan, K. T. and Vingerhoets, P. and Billowes, J. and Blaum, K. and Cheal, B. and Fritzsche, S. and Godefroid, M. and Kowalska, M. and Kr\"amer, J. and Neugart, R. and Neyens, G. and N\"ortersh\"auser, W. and Yordanov, D. T.},
#   journal = {Phys. Rev. C},
#   volume = {93},
#   issue = {6},
#   pages = {064318},
#   numpages = {7},
#   year = {2016},
#   month = {Jun},
#   publisher = {American Physical Society},
#   doi = {10.1103/PhysRevC.93.064318},
#   url = {https://link.aps.org/doi/10.1103/PhysRevC.93.064318}
# }
cu_radii = {
    '58_Cu': (-0.833, 13, 91),
    '59_Cu': (-0.635, 9, 71),
    '60_Cu': (-0.511, 8, 57),
    '61_Cu': (-0.359, 6, 40),
    '62_Cu': (-0.293, 5, 33),
    '63_Cu': (-0.148, 1, 17),
    '64_Cu': (-0.116, 3, 13),
    '65_Cu': (0, 0, 0),
    '66_Cu': (0.033, 4, 12),
    '67_Cu': (0.115, 5, 18),
    '68_Cu': (0.133, 5, 31),
    '69_Cu': (0.238, 3, 34),
    '70_Cu': (0.271, 3, 44),
    # '70m1_Cu': (0.287, 11, 44),
    # '70m2_Cu': (0.323, 11, 44),
    '71_Cu': (0.407, 11, 44),
    '72_Cu': (0.429, 5, 55),
    '73_Cu': (0.523, 15, 58),
    '74_Cu': (0.505, 18, 72),
    '75_Cu': (0.546, 21, 80)
}

cu_Z = 29
cu_x = []
cu_y = []
cu_yerr = []
print('iso\t $\delta$ <r$^2$>[fm$^2$]')
for i in sorted(cu_radii.keys()):
    cu_x.append(int(str(i).split('_')[0]) - cu_Z)
    cu_y.append(cu_radii[i][0])
    cu_yerr.append(np.sqrt((cu_radii[i][1] / 1000) ** 2 + (cu_radii[i][2] / 1000) ** 2))
    print('%s\t%.3f(%.0f)[%.0f]' % (i, cu_radii[i][0], cu_radii[i][1], cu_radii[i][2]))

cu_x_odd = [each for each in cu_x if each % 2 != 0]
cu_y_odd = [each for i, each in enumerate(cu_y) if cu_x[i] % 2 != 0]
cu_y_odd_err = [each for i, each in enumerate(cu_yerr) if cu_x[i] % 2 != 0]
cu_x_even = [each for each in cu_x if each % 2 == 0]
cu_y_even = [each for i, each in enumerate(cu_y) if cu_x[i] % 2 == 0]
cu_y_even_err = [each for i, each in enumerate(cu_yerr) if cu_x[i] % 2 == 0]


''' working directory: '''

workdir = 'E:\\Workspace\\OwnCloud\\Projekte\\COLLAPS\\Nickel\\Measurement_and_Analysis_Simon\\Ni_workspace'

datafolder = os.path.join(workdir, 'Ni_April2016_mcp')

db = os.path.join(workdir, 'Ni_workspace.sqlite')

runs = ['wide_gate_asym', 'wide_gate_asym_67_Ni']

isotopes = ['%s_Ni' % i for i in range(58, 71)]
isotopes.remove('69_Ni')
odd_isotopes = [iso for iso in isotopes if int(iso[:2]) % 2]
even_isotopes = [iso for iso in isotopes if int(iso[:2]) % 2 == 0]
stables = ['58_Ni', '60_Ni', '61_Ni', '62_Ni', '64_Ni']

extracted = Tools.extract_from_combined([runs[0]], db, isotopes=isotopes, par='delta_r_square')[runs[0]]
print(extracted)

ni_Z = 28
x = []
y = []
yerr = []
print('iso\t $\delta$ <r$^2$>[fm$^2$]')
for i in isotopes:
    x.append(int(str(i).split('_')[0]) - ni_Z)
    y.append(extracted[i][0])
    yerr.append(extracted[i][2])
    print('%s\t%.3f(%.0f)' % (i, extracted[i][0], extracted[i][2] * 1000))
    # print("'"+str(i)+"'", ':[', np.round(finalVals[i][0],3), ','+ str(np.round(np.sqrt(finalVals[i][1]**2 + finalVals[i][2]**2),3))+'],')


x_odd = [each for each in x if each % 2 != 0]
y_odd = [each for i, each in enumerate(y) if x[i] % 2 != 0]
y_odd_err = [each for i, each in enumerate(yerr) if x[i] % 2 != 0]
x_even = [each for each in x if each % 2 == 0]
y_even = [each for i, each in enumerate(y) if x[i] % 2 == 0]
y_even_err = [each for i, each in enumerate(yerr) if x[i] % 2 == 0]


''' plotting '''

fig = MPLPlotter.plt.figure()
ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
# ax = plt.gca()
ax.set_ylabel(r'$\delta$ < r' + r'$^2$ > (fm $^2$) ', fontsize=font_size)
ax.set_xlabel('N', fontsize=font_size)

ax.axvline(x=40, color='g', linestyle='--')

MPLPlotter.plt.errorbar(x_even, y_even, y_even_err, fmt='ro', label='Ni even N', linestyle='-')
MPLPlotter.plt.errorbar(x_odd, y_odd, y_odd_err, fmt='r^', label='Ni odd N', linestyle='--')

MPLPlotter.plt.errorbar(cu_x_even, cu_y_even, cu_y_even_err, fmt='bd', label='Cu even N', linestyle='-')
MPLPlotter.plt.errorbar(cu_x_odd, cu_y_odd, cu_y_odd_err, fmt='b*', label='Cu odd N', linestyle='--')

MPLPlotter.plt.legend(loc=2)
ax.set_xmargin(0.05)
MPLPlotter.plt.margins(0.1)
MPLPlotter.plt.gcf().set_facecolor('w')
MPLPlotter.plt.xticks(fontsize=font_size)
MPLPlotter.plt.yticks(fontsize=font_size)
MPLPlotter.plt.show()
