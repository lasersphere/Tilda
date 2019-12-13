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
import Analyzer


##############################
# Adjust run properties here #
##############################

# Select run and isotope of the calibrations
run = 'Agilent_3458A'
isotope = 'AD5791_dac_calibration'

# Set working directory and db to the calibration files
workdir = 'C:\\Users\\FS_Desk\\ownCloud\\Projekte\\KOALA\\CalibrationAndTests\\DAC_20bit_tests'
db = workdir + '\\' + os.path.basename(workdir) + '.sqlite'
print(db, os.path.isfile(db))


######################################
# DAC Calib Evaluation starting here #
######################################
# Get a list of all files matching 'isotope' in database
files = Tools.fileList(db, isotope)

# Do a batchfit on all these files, writing the results to FitRes in db.
BatchFit.batchFit(files[1:], db, run, x_as_voltage=False)
# Combine the fit results for m and b and create plots
avgm, statErrm, systErrm, rChim, plotdatam, axm = Analyzer.combineRes('Kepco', 'm', run, db, show_plot=True)
avgb, statErrb, systErrb, rChib, plotdatab, axb = Analyzer.combineRes('Kepco', 'b', run, db, show_plot=True)
# Print m and b results
print('## m ## avg: {}, stat.Err: {}, syst.Err: {}, rChi: {}'.format(avgm, statErrm, systErrm, rChim))
print('## b ## avg: {}, stat.Err: {}, syst.Err: {}, rChi: {}'.format(avgb, statErrb, systErrb, rChib))


# change format of 'files' list to list of lists for further use
files = [files]

# Make the residuals-plot
for i, file_list in enumerate(files):
    # fit again, this time x_as_voltage=True
    fits, error_files = BatchFit.batchFit(file_list, db, run, x_as_voltage=True)

    # plot all residuals:
    for i, fit in enumerate(fits):
        data = fit.meas.getArithSpec(*fit.st)
        res = fit.calcRes() * 1000  # go to mV
        plt.plot(data[0], res, label=fit.meas.file)
    plt.ylabel('residuum [mV]')
    plt.xlabel('dac register [bits]')
    fig = plt.gcf()
    plt.show()

################################
# Print final results for m, b #
################################
print('\n\n#######################\n# final results m & b #\n#######################')
print('## m ## avg: {}, stat.Err: {}, syst.Err: {}, rChi: {}'.format(avgm, statErrm, systErrm, rChim))
print('## b ## avg: {}, stat.Err: {}, syst.Err: {}, rChi: {}'.format(avgb, statErrb, systErrb, rChib))
