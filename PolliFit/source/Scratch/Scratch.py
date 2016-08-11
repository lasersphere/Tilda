'''
Created on 31.03.2014

@author: hammen
'''

from Measurement.SimpleImporter import SimpleImporter
import MPLPlotter as plot

from DBIsotope import DBIsotope
from SPFitter import SPFitter
from Spectra.FullSpec import FullSpec

import os
import numpy as np
import Tools
import BatchFit
import InteractiveFit

# path = "../test/cd_c_137data.txt"
# file = SimpleImporter(path)
# iso = DBIsotope('114_Mi', 'Mi-D0',  '../test/iso.sqlite')
# spec = FullSpec(iso)
#
# fit = SPFitter(spec, file, (0, -1))
#
# print(fit.spec.parAssign())
#
# fit.fit()
#
#
# plot.plotFit(fit)
# plot.show()
db = 'V:/User/Christian/databases/ALIVE/Ca.sqlite'
Tools.crawl(db)
BatchFit.batchFit(['2016-07-22_09-15-46_003.dat', '2016-07-22_10-12-57_22.dat'], db)
# InteractiveFit.InteractiveFit('2016-07-22_10-15-09_22.dat', db, 'Run0')