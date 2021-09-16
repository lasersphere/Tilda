"""
Created on 17.10.2017

@author: simkaufm

Module Description:
This is a script to compare and display the measurements of the AO from the 7852R FPGA vs. the AD5781.
 This is not foreseen as a full analysis which has already ben done and can be found in the lab log,
  but to make a nice plot for the thesis.
"""

import Tools
from Measurement.SimpleImporter import SimpleImporter
from Measurement.XMLImporter import XMLImporter
from Spectra.Straight import Straight
from SPFitter import SPFitter
import os
import matplotlib.pyplot as plt
import numpy as np

workdir = 'E:\\Workspace\\OwnCloud\\Projekte\\TRIGA\\Measurements and' \
          ' Analysis_Simon\\KepcoScans und DAC Scans\\DACsComparisonForThesis'

db = os.path.join(workdir, 'DACsComparisonForThesis.sqlite')

data_dir = os.path.join(workdir, 'data')


''' fpga file '''
fpga_ao_file = os.path.join(data_dir, 'VoltagScan_FPGA_AO1_M1.txt')

# importing
fpga_meas = SimpleImporter(fpga_ao_file)
fpga_meas.err[0] = abs(fpga_meas.cts[0]) * 0.0015/100 + 0.004/100  # error of the agilent reading
fpga_meas.type = 'Kepco'
fpga_meas.x[0] = np.linspace(0, 2 ** 16, 1001)
print(fpga_meas.getNrSteps(-1))

# fitting
spec = Straight()
spec.evaluate(fpga_meas.x[0][-1], (0, 1))
fitter = SPFitter(spec, fpga_meas, [[0], -1])
fitter.fit()
plotdat = fitter.spec.toPlotE(0, 0, fitter.par)
fpga_residuals = fitter.calcRes() * 1000  # in mV
fpga_x, fpga_y, fpga_y_err = fitter.meas.getArithSpec([0], -1)   # New not tested yet...


''' AD5781 File '''

ad_file = os.path.join(data_dir, 'AD5781Ser1_kepco_007.xml')

# importing
ad_meas = XMLImporter(ad_file)
ad_spec = Straight()
ad_spec.evaluate(ad_meas.x[0][-1], (0, 1))

# fitting
ad_fitter = SPFitter(ad_spec, ad_meas, [[0], -1])
ad_fitter.fit()
ad_plotdat = ad_fitter.spec.toPlotE(0, 0, ad_fitter.par)
ad_residuals = ad_fitter.calcRes() * 1000  # in mV
ad_x, ad_y, ad_y_err = ad_fitter.meas.getArithSpec([0], -1)  # New not tested yet...


''' the plot: '''
fontsize_ticks = 20
# setup figure
fig = plt.figure(num=0, facecolor='w', figsize=(18, 9))
width = 0.35
height = 0.4
fpg_plt_left = 0.1
common_bottom_lower = 0.1
common_bottom_upper = common_bottom_lower + height
ad_plt_left = 0.5
fig.text(fpg_plt_left - 0.15 * fpg_plt_left,
         common_bottom_upper + height + 0.05, 'a)', fontsize=fontsize_ticks, weight='bold')
fig.text(ad_plt_left - 0.15 * fpg_plt_left,
         common_bottom_upper + height + 0.05, 'b)', fontsize=fontsize_ticks, weight='bold')

# add 4 axes
fpga_ax = fig.add_axes([fpg_plt_left, common_bottom_upper, width, height])
fpga_ax.xaxis.tick_top()
fpga_ax_res = fig.add_axes([fpg_plt_left, common_bottom_lower, width, height], sharex=fpga_ax)
ad5781_ax = fig.add_axes([ad_plt_left, common_bottom_upper, width, height])
ad5781_ax.xaxis.tick_top()
ad5781_ax.yaxis.tick_right()
ad5781_ax.yaxis.set_label_position("right")
ad5781_ax_res = fig.add_axes([ad_plt_left, common_bottom_lower, width, height], sharex=ad5781_ax)
ad5781_ax_res.yaxis.tick_right()
ad5781_ax_res.yaxis.set_label_position("right")
all_axes = [fpga_ax, fpga_ax_res, ad5781_ax, ad5781_ax_res]

# plot fpga stuff
fpga_ax.errorbar(fpga_x, fpga_y, fpga_y_err, label='FPGA A0', fmt='k.', markersize=10)
fpga_ax.plot(plotdat[0], plotdat[1], 'r', label='lin. fit', linewidth=3)
fpga_ax_res.plot(fpga_x, fpga_residuals, 'k.', label='FPGA A0 residuals', markersize=10)
fpga_ax.legend(loc=2, fontsize=fontsize_ticks)
fpga_ax.set_xlim(-4000, 2 ** 16 + 4000)
fpga_ax.set_ylim(-12.5, 12.5)
fpga_ax_res.set_ylim(-1.15, 1.15)
fpga_ax_res.axhline(20 / (2 ** 16 - 1) * 1000, label='+/-1 * LSB', linewidth=2)
fpga_ax_res.axhline(-20 / (2 ** 16 - 1) * 1000, linewidth=2)
fpga_ax_res.legend(loc=2, fontsize=fontsize_ticks)
fpga_ax_res.set_xlabel('DAC register bits', fontsize=fontsize_ticks)
fpga_ax.set_ylabel('measured voltage / V', fontsize=fontsize_ticks)
fpga_ax_res.set_ylabel('residuals / mV', fontsize=fontsize_ticks)
fpga_ax.set_xticks([0, 20000, 40000, 60000])

# plot ad5781 stuff
ad5781_ax.errorbar(ad_x, ad_y, ad_y_err, label='AD5781', fmt='k.', markersize=10)
ad5781_ax.plot(ad_plotdat[0], ad_plotdat[1], 'r', label='lin. fit', linewidth=3)
ad5781_ax_res.plot(ad_x, ad_residuals, 'k.', label='AD5781 residuals', markersize=10)
ad5781_ax.legend(loc=2, fontsize=fontsize_ticks)
ad5781_ax.set_xlim(-8000, 2 ** 18 + 8000)
ad5781_ax.set_ylim(-12.5, 12.5)
ad5781_ax_res.set_ylim(-0.025, 0.025)
ad5781_ax_res.axhline(20 / (2 ** 18 - 1) * 1000 / 4, label='+/-0.25 * LSB', linestyle='dashed', linewidth=2)
ad5781_ax_res.axhline(-20 / (2 ** 18 - 1) * 1000 / 4, linestyle='dashed', linewidth=2)
ad5781_ax_res.legend(loc=3, fontsize=fontsize_ticks)
ad5781_ax_res.set_xlabel('DAC register bits', fontsize=fontsize_ticks)
ad5781_ax.set_ylabel('measured voltage / V', fontsize=fontsize_ticks)
ad5781_ax_res.set_ylabel('residuals / mV', fontsize=fontsize_ticks)
ad5781_ax.set_xticks([0, 100000, 200000])

# change the fontsize for all axis
for ax in all_axes:
    ax.tick_params(labelsize=fontsize_ticks)

fig_name = os.path.join(workdir, 'ad5781_comp.png')
fig_name_pdf = os.path.join(workdir, 'ad5781_comp.pdf')
fig.savefig(fig_name, dpi=300)
fig.savefig(fig_name_pdf, dpi=300)

print(fpga_ax.get_yscale)
plt.show()


