"""

Created on '21.06.2016'

@author:'simkaufm'

This tool can extract the residua in mV from a bunch of dac_calibration runs.
Please adjust run, isotope, workdir in the first lines according to your runs!
"""
import os

import matplotlib.pyplot as plt

import BatchFit
import Tools

run = 'Agilent_3458A'
isotope = 'AD5791_dac_calibration'

workdir = 'C:\\Users\\FS_Desk\\ownCloud\\Projekte\\KOALA\\CalibrationAndTests\\DAC_20bit_tests'
db = workdir + '\\' + os.path.basename(workdir) + '.sqlite'
print(db, os.path.isfile(db))

# Tools.crawl(db)

files = [Tools.fileList(db, isotope)]
print(len(files), files[0])


for i, file_list in enumerate(files):
    # time = times[i]
    # print('Fitting all files for a settel tiem of %s s' % time)
    fits, error_files = BatchFit.batchFit(file_list, db, run, x_as_voltage=True)

    # plot all residuals:
    for i, fit in enumerate(fits):
        data = fit.meas.getArithSpec(*fit.st)
        res = fit.calcRes() * 1000  # go to mV
        plt.plot(data[0], res, label=fit.meas.file)
    plt.ylabel('residuum [mV]')
    plt.xlabel('voltage')
    fig = plt.gcf()
    # fig.canvas.set_window_title('settle time: %ss' % time)
    # plt.legend()
    plt.show()
# #
# run = 'Voltage'
# BatchFit.batchFit(files[1:], db, run, x_as_voltage=True)

# fit = InteractiveFit(files[2], db, run, block=False, x_as_voltage=True)
# fit.fit()

# avgm, statErrm, systErrm, plotdatam = Analyzer.combineRes('Kepco', 'm', run, db, show_plot=True)
# avgb, statErrb, systErrb, plotdatab = Analyzer.combineRes('Kepco', 'b', run, db, show_plot=True)
