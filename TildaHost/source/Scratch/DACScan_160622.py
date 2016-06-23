"""

Created on '21.06.2016'

@author:'simkaufm'

"""
import os
import Analyzer
import Tools
import BatchFit
import matplotlib.pyplot as plt

from InteractiveFit import InteractiveFit

# run = 'Run0'
run = 'Ni4071_PXI1Slot5'
# run = 'Voltage'

workdir = 'D:\DACScan_160622'
db = workdir + '\\' + os.path.basename(workdir) + '.sqlite'
print(db, os.path.isfile(db))

# Tools.crawl(db)

# files = Tools.fileList(db, 'DACV2_7_5_Dig')
# print(files)
#
# run = 'DACReg_DACV2'
# fits = BatchFit.batchFit(files, db, run, x_as_voltage=False)

# plot all residuals:
# for fit in fits:
#     data = fit.meas.getArithSpec(*fit.st)
#     res = fit.calcRes() * 1000  # go to mV
#     plt.plot(data[0], res)
# plt.ylabel('residuum [mV]')
# plt.xlabel('DAC Register')
# plt.show()
#
# run = 'Voltage'
# BatchFit.batchFit(files[1:], db, run, x_as_voltage=True)

# fit = InteractiveFit(files[2], db, run, block=False, x_as_voltage=True)
# fit.fit()

avgm, statErrm, systErrm, plotdatam = Analyzer.combineRes('Kepco', 'm', run, db, show_plot=True)
avgb, statErrb, systErrb, plotdatab = Analyzer.combineRes('Kepco', 'b', run, db, show_plot=True)
