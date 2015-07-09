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

from polliPipe.pipeline import Pipeline

def TrsPipe(initialTrackPars):
    """
    Pipeline for the dataflow and analysis of one Isotope using the time resolved sequencer.
    Mutliple Tracks are supported.
    """
    start = Node()

    pipe = Pipeline(start)

    pipe.pipeData = initPipeData(initialTrackPars)

    walk = start.attach(TN.NSaveRawData())
    # walk = walk.attach(TN.NSaveRawData())
    walk = walk.attach(TN.NSplit32bData())
    # walk = walk.attach(TN.NSumBunchesTRS(pipe.pipeData))
    # walk = walk.attach(TN.NSaveSum())
    walk = walk.attach(SN.NPrint())

    return pipe

def CsPipe(initialTrackPars):
    """
    Pipeline for the dataflow and analysis of one Isotope using the continous sequencer.
    Mutliple Tracks are supported.
    """
    start = Node()

    pipe = Pipeline(start)

    pipe.pipeData = initPipeData(initialTrackPars)

    walk = start.attach(TN.NSaveRawData())
    walk = walk.attach(TN.NSplit32bData())
    walk = walk.attach(TN.NAcquireOneLoopCS(pipe.pipeData))
    walk = walk.attach(TN.NSumCS(pipe.pipeData))
    walk = walk.attach(TN.NSaveSumCS())

    return pipe

def initPipeData(initialTrackPars):
    """
    initialize the pipeData used for the analysis Pipeline
    :return: dict, {'isotopeData', 'progConfigs', 'activeTrackPar', 'pipeInternals'}
    """
    pipeData = draftPars.draftScanDict
    pipeData.update(activeTrackPar=initialTrackPars, nOfCompletedSteps=0)
    pipeData['pipeInternals']['curVoltInd'] = 0
    pipeData['pipeInternals']['activeTrackNumber'] = 0
    pipeData['pipeInternals']['activeXmlFilePath'] = FaFH.createXmlFileOneIsotope(pipeData)
    return pipeData