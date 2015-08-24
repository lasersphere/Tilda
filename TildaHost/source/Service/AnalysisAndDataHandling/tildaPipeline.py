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
import PyQtGraphPlotter


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

    pipe.pipeData = initPipeData(initialScanPars)

    plots = []

    proc, rpg, win = PyQtGraphPlotter.init()
    pipe.pipeData['pipeInternals']['activeGraphicsWindow'] = win

    # walk = start.attach(TN.NSaveRawData())
    # walk = start.attach(SN.NPrint())
    walk = start.attach(TN.NSplit32bData())
    walk = walk.attach(TN.NSortRawDatatoArray(pipe.pipeData))
    # branch = walk.attach(SN.NPrint())
    branch = walk.attach(TN.NAccumulateSingleScan(pipe.pipeData))
    branch = branch.attach(TN.NLivePlot(pipe.pipeData, 'SingleScan'))
    # plots.append(branch)  # necessary to prevent garbage collection from clean-up
    # walk = walk.attach(SN.NPrint())
    walk = walk.attach(TN.NSumCS(pipe.pipeData))
    walk = walk.attach(TN.NLivePlot(pipe.pipeData, 'Sum'))
    # plots.append(walk)  # necessary to prevent garbage collection from clean-up
    walk = walk.attach(TN.NCheckIfTrackComplete())
    # # walk = walk.attach(TN.NSaveSumCS())
    # walk = walk.attach(TN.NLivePlot(pipe.pipeData, 'finalSum'))
    # # walk = walk.attach(TN.NPlotSum(pipe.pipeData))
    # walk = walk.attach(SN.NPrint())
    #
    return pipe, plots

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