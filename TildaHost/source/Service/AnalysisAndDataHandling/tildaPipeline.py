"""

Created on '20.05.2015'

@author:'simkaufm'

"""

import Service.AnalysisAndDataHandling.tildaNodes as TN
import polliPipe.simpleNodes as SN
from polliPipe.node import Node
import Service.ProgramConfigs as progConfigs

from polliPipe.pipeline import Pipeline

def TrsPipe(scanPars):
    start = Node()

    pipe = Pipeline(start)

    pipe.pipeData.update(scanPars)
    pipe.pipeData.update(curVoltInd=0)
    pipe.pipeData.update(nOfTotalSteps=0)
    pipe.pipeData.update(progConfigs=progConfigs.programs)

    walk = start.attach(TN.NSplit32bData())
    walk = walk.attach(TN.NSumBunchesTRS(pipe.pipeData))
    walk = walk.attach(SN.NPrint())

    return pipe

