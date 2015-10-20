"""

Created on '20.05.2015'

@author:'simkaufm'

"""

import Service.AnalysisAndDataHandling.tildaNodes as TN
import polliPipe.simpleNodes as SN
from polliPipe.node import Node
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
    """
    start = Node()

    pipe = Pipeline(start)

    fig, axes = plt.subplots(6, sharex=True)

    pipe.pipeData = initPipeData(initialScanPars)

    # walk = start.attach(TN.NSaveRawData())
    walk = start.attach(TN.NSplit32bData())
    walk = walk.attach(TN.NSortRawDatatoArray())

    branch = walk.attach(TN.NAccumulateSingleScan())
    branch1 = branch.attach(TN.NArithmetricScaler([0]))
    branch1 = branch1.attach(TN.NMPlLivePlot(axes[0], 'scaler 0'))

    branch2 = branch.attach(TN.NArithmetricScaler([1]))
    branch2 = branch2.attach(TN.NMPlLivePlot(axes[1], 'scaler 1'))

    branch3 = branch.attach(TN.NArithmetricScaler([0, 1]))
    branch3 = branch3.attach(TN.NMPlLivePlot(axes[2], 'scaler 0+1'))

    walk = walk.attach(TN.NRemoveTrackCompleteFlag())
    walk = walk.attach(TN.NSumCS())

    walk = walk.attach(TN.NMPlLivePlot(axes[3], 'live sum'))

    branch4 = walk.attach(TN.NArithmetricScaler([0, 1]))
    branch4 = branch4.attach(TN.NMPlLivePlot(axes[4], 'scaler 0+1'))

    walk = walk.attach(TN.NCheckIfTrackComplete())
    walk = walk.attach(TN.NMPlLivePlot(axes[5], 'final sum'))
    # walk = walk.attach(TN.NSaveSumCS())
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