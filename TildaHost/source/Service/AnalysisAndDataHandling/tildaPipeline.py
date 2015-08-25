"""

Created on '20.05.2015'

@author:'simkaufm'

"""

import Service.AnalysisAndDataHandling.tildaNodes as TN
import polliPipe.simpleNodes as SN
from polliPipe.node import Node
import Service.ProgramConfigs as progConfigs
import Service.draftScanParameters as draftPars
import Service.FolderAndFileHandling as FaFH
# import PyQtGraphPlotter
import matplotlib.pyplot as plt

from polliPipe.pipeline import Pipeline

def TrsPipe(initialScanPars):
    """
    Pipeline for the dataflow and analysis of one Isotope using the time resolved sequencer.
    Mutliple Tracks are supported.
    """
    start = Node()

    pipe = Pipeline(start)

    pipe.pipeData = initPipeData(initialScanPars)

    walk = start.attach(TN.NSaveRawData())
    walk = walk.attach(TN.NSaveRawData())
    # walk = walk.attach(TN.NSplit32bData())
    # walk = walk.attach(TN.NSumBunchesTRS(pipe.pipeData))
    # walk = walk.attach(TN.NSaveSum())
    walk = walk.attach(SN.NPrint())

    return pipe

def CsPipe(initialScanPars=None):
    """
    Pipeline for the dataflow and analysis of one Isotope using the continous sequencer.
    Mutliple Tracks are supported.
    """
    start = Node()

    pipe = Pipeline(start)

    plt.ion()

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    pipe.pipeData = initPipeData(initialScanPars)

    # walk = start.attach(TN.NSaveRawData())
    walk = start.attach(TN.NSplit32bData())
    # walk = walk.attach(SN.NPrint())
    walk = walk.attach(TN.NAcquireOneScanCS(pipe.pipeData))
    # walk = walk.attach(SN.NPrint())
    # walk = walk.attach(SN.NPrint())
    # branch = walk.attach(SN.NPrint())
    # branch = walk.attach(TN.NSumCS(pipe.pipeData))
    # branch = branch.attach(TN.NMPlLivePlot(pipe.pipeData, ax1))

    # branch not supported currently

    walk = walk.attach(TN.NSumCS(pipe.pipeData))
    walk = walk.attach(TN.NMPlLivePlot(pipe.pipeData, ax2))
    # walk = walk.attach(SN.NPrint())
    walk = walk.attach(TN.NCheckIfTrackComplete())
    walk = walk.attach(TN.NMPlLivePlot(pipe.pipeData, ax3))
# # # walk = walk.attach(TN.NSaveSumCS())
    # walk = walk.attach(TN.NPlotSum(pipe.pipeData))
    # # walk = walk.attach(TN.NPlotSum(pipe.pipeData))
    # walk = walk.attach(SN.NPrint())
    #
    return pipe

def initPipeData(initialScanPars):
    """
    initialize the pipeData used for the analysis Pipeline
    :return: dict, {'isotopeData', 'progConfigs', 'activeTrackPar', 'pipeInternals'}
    """
    pipeData = initialScanPars

    pipeData['activeTrackPar']['nOfCompletedSteps'] = 0
    pipeData['pipeInternals']['curVoltInd'] = 0
    pipeData['pipeInternals']['activeTrackNumber'] = 0
    pipeData['pipeInternals']['activeXmlFilePath'] = FaFH.createXmlFileOneIsotope(pipeData)
    return pipeData