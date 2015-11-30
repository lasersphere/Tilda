"""

Created on '20.08.2015'

@author:'simkaufm'

"""

import numpy as np

import PyQtGraphPlotter as pyGpl
import Service.Scan.draftScanParameters as dftSc


pmtList = dftSc.draftScanDict['activeTrackPar']['activePmtList']
proc, rpg, win = pyGpl.init()
dftSc.draftScanDict['pipeInternals']['activeGraphicsWindow'] = win

plots = [pyGpl.addPlot(dftSc.draftScanDict['pipeInternals']['activeGraphicsWindow'], 'Pmt' + str(n)) for n in dftSc.draftScanDict['activeTrackPar']['activePmtList']]

trackd = dftSc.draftScanDict['activeTrackPar']
dacStart18Bit = trackd['dacStartRegister18Bit']
dacStepSize18Bit = trackd['dacStepSize18Bit']
nOfsteps = trackd['nOfSteps']
dacStop18Bit = dacStart18Bit + (dacStepSize18Bit * nOfsteps)
x = np.arange(dacStart18Bit, dacStop18Bit, dacStepSize18Bit)
y = np.random.random(size=(nOfsteps, len(pmtList)))

for i, j in enumerate(plots):
    print(i, j)
    j.plot(x, y[:, i], pen=(i, len(plots)))
#
input('anything to stop')
# p1 = pyGpl.addPlot(win, 'current')
# p2 = pyGpl.addPlot(win, 'sum')
# p3 = pyGpl.addPlot(win, 'sum2')
# dat = np.random.random(10)
# p1.plot(dat, pen=(255,0,0), name="Red curve")
# p2.plot(dat, pen=(255,0,0), name="Red curve")
# p2.plot(dat+5, pen=(0,255,0), name="Blue curve")

# print(proc, rpg, win)
