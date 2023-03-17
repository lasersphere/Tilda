"""

Created on '21.06.2016'

@author:'simkaufm'

"""
import os
from Tilda.PolliFit import Tools, Analyzer

# run = 'Run0'
# run = 'Ni4071_PXI1Slot5'
# run = 'Voltage'

workdir = 'C:\\Users\\FS_Desk\\ownCloud\\Projekte\\KOALA\\CalibrationAndTests\\DAC_20bit_tests'
db = workdir + '\\' + os.path.basename(workdir) + '.sqlite'
print(db, os.path.isfile(db))

files = Tools.fileList(db, 'AD5791_dac_calibration')
#
run = 'Agilent_3458A'
#BatchFit.batchFit(files[1:], db, run, x_as_voltage=False)
#
# run = 'Voltage'
# BatchFit.batchFit(files[1:], db, run, x_as_voltage=True)

#fit = InteractiveFit(files[2], db, run, block=False, x_as_voltage=True)
#fit.fit()

avgm, statErrm, systErrm, rChim, plotdatam, axm = Analyzer.combineRes('Kepco', 'm', run, db, show_plot=True)
avgb, statErrb, systErrb, rChib, plotdatab, axb = Analyzer.combineRes('Kepco', 'b', run, db, show_plot=True)

print('## m ## avg: {}, stat.Err: {}, syst.Err: {}, rChi: {}'.format(avgm, statErrm, systErrm, rChim))
print('## b ## avg: {}, stat.Err: {}, syst.Err: {}, rChi: {}'.format(avgb, statErrb, systErrb, rChib))
