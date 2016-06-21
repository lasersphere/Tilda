"""

Created on '21.06.2016'

@author:'simkaufm'

"""
import os
import Analyzer
import Tools
import BatchFit
from InteractiveFit import InteractiveFit

# run = 'Run0'
# run = 'Ni4071_PXI1Slot5'
# run = 'Voltage'

workdir = 'R:\Projekte\TRIGA\Measurements and Analysis_Simon\KepcoScans und DAC Scans\DAC - AD5781\KepcoScans_160621'
db = workdir + '\\' + os.path.basename(workdir) + '.sqlite'
print(db, os.path.isfile(db))

files = Tools.fileList(db, 'DACV1_7_5_Dig')
#
run = 'DACReg'
# BatchFit.batchFit(files[1:], db, run, x_as_voltage=False)
#
# run = 'Voltage'
# BatchFit.batchFit(files[1:], db, run, x_as_voltage=True)

fit = InteractiveFit(files[2], db, run, block=False, x_as_voltage=True)
fit.fit()

avgm, statErrm, systErrm, plotdatam = Analyzer.combineRes('Kepco', 'm', run, db, show_plot=True)
avgb, statErrb, systErrb, plotdatab = Analyzer.combineRes('Kepco', 'b', run, db, show_plot=True)
