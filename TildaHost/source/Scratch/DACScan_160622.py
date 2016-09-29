"""

Created on '21.06.2016'

@author:'simkaufm'

"""
import os

import matplotlib.pyplot as plt

import BatchFit
import Tools

# run = 'Run0'
run = 'Ni4071_PXI1Slot5'
# run = 'Voltage'

# workdir = 'D:\DACScan_160622'
workdir = 'R:\Projekte\TRIGA\Measurements and Analysis_Simon\KepcoScans und DAC Scans\switchBoxSettleTimeDependency\kepco_260916'
db = workdir + '\\' + os.path.basename(workdir) + '.sqlite'
print(db, os.path.isfile(db))

# Tools.crawl(db)

files = Tools.fileList(db, 'offsetTest')
files = files[18:]
# print(files)
files_5s_1 = files[:5]
files_2p5s = files[5:10]
files_0s = files[10:15]
files_5s_2 = files[15:20]
files_10s_1 = [files[20]]
files_5s_3 = files[23:28]
files_10s_2 = files[29:34]
files_5s_4 = files[35:40]
# print(files_5s_3)
#
# run = 'DACReg_DACV2'
fits = BatchFit.batchFit(files_5s_4, db, run, x_as_voltage=True)

# plot all residuals:
for fit in fits:
    data = fit.meas.getArithSpec(*fit.st)
    res = fit.calcRes() * 1000  # go to mV
    plt.plot(data[0], res, label=fit.meas.file)
plt.ylabel('residuum [mV]')
plt.xlabel('voltage')
plt.legend()
plt.show()
#
# run = 'Voltage'
# BatchFit.batchFit(files[1:], db, run, x_as_voltage=True)

# fit = InteractiveFit(files[2], db, run, block=False, x_as_voltage=True)
# fit.fit()

# avgm, statErrm, systErrm, plotdatam = Analyzer.combineRes('Kepco', 'm', run, db, show_plot=True)
# avgb, statErrb, systErrb, plotdatab = Analyzer.combineRes('Kepco', 'b', run, db, show_plot=True)
