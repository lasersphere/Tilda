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


def find_pipe_by_seq_type(scan_dict):
    seq_type = scan_dict['isotopeData']['type']
    if seq_type == 'cs' or 'csdummy':
        return CsPipe(scan_dict)
    elif seq_type == 'trs':
        return TrsPipe(scan_dict)
    elif seq_type == 'kepco':
        return kepco_scan_pipe(scan_dict)
    else:
        return None



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

    walk = start.attach(TN.NSaveRawData())
    walk = start.attach(TN.NSplit32bData())

    walk = walk.attach(TN.NSortRawDatatoArray())

    branch = walk.attach(TN.NAccumulateSingleScan())
    branch = branch.attach(SN.NPrint())
    branch = branch.attach(TN.NSingleArrayToSpecData())

    branch1 = branch.attach(TN.NSingleSpecFromSpecData([0]))
    branch1 = branch1.attach(TN.NMPlLivePlot(axes[0], 'scaler 0', ['b-']))

    branch2 = branch.attach(TN.NSingleSpecFromSpecData([1]))
    branch2 = branch2.attach(TN.NMPlLivePlot(axes[1], 'scaler 1', ['g-']))

    branch3 = branch.attach(TN.NSingleSpecFromSpecData([0, 1]))
    branch3 = branch3.attach(TN.NMPlLivePlot(axes[2], 'scaler 0+1', ['r-']))

    walk = walk.attach(TN.NRemoveTrackCompleteFlag())
    walk = walk.attach(TN.NSumCS())

    sum = walk.attach(TN.NSingleArrayToSpecData())
    sum0 = sum.attach(TN.NMultiSpecFromSpecData([[0], [1]]))
    sum0 = sum0.attach(TN.NMPlLivePlot(axes[3], 'live sum', ['b-', 'g-']))

    sum01 = sum.attach(TN.NSingleSpecFromSpecData([0, 1]))
    sum01 = sum01.attach(TN.NMPlLivePlot(axes[4], 'scaler 0+1', ['r-']))

    walk = walk.attach(TN.NCheckIfTrackComplete())
    finalsum = walk.attach(TN.NSingleArrayToSpecData())
    finalsum = finalsum.attach(TN.NMultiSpecFromSpecData([[0], [1]]))
    finalsum = finalsum.attach(TN.NMPlLivePlot(axes[5], 'final sum', ['b-', 'g-']))

    walk = walk.attach(TN.NSaveSumCS())
    return pipe

def kepco_scan_pipe(initial_scan_pars):
    """
    pipeline for the measurement and analysis of a kepco scan
    :param initial_scan_pars: full sacn dictionary which will be used as starting point for this pipeline
    """
    start = Node()

    pipe = Pipeline(start)

    walk = start.attach(TN.NSaveRawData())
    walk = walk.attach(SN.NPrint())
    # more has to be included...
    return pipe

def initPipeData(initialScanPars):
    """
    initialize the pipeData used for the analysis Pipeline
    :return: dict, {'isotopeData', 'progConfigs', 'activeTrackPar', 'pipeInternals'}
    """
    pipeData = initialScanPars

    pipeData['pipeInternals']['curVoltInd'] = 0
    pipeData['pipeInternals']['activeXmlFilePath'] = FaFH.createXmlFileOneIsotope(pipeData)
    return pipeData


def simple_counter_pipe():
    start = Node()

    fig, axes = plt.subplots(2, sharex=True)

    plotpoints = 600
    sample_rate = 1 / 0.02  # values per second
    pipe = Pipeline(start)

    walk = start.attach(TN.NSplit32bData())
    walk = walk.attach(TN.NSortByPmt(sample_rate))
    walk = walk.attach(TN.NMovingAverage())
    # walk = walk.attach(SN.NPrint())

    walk = walk.attach(TN.NAddxAxis())
    pmt0 = walk.attach(TN.NOnlyOnePmt(0))
    pmt0 = pmt0.attach(TN.NMPlLivePlot(axes[0], 'mov. avg', ['b-']))

    pmt1 = walk.attach(TN.NOnlyOnePmt(1))
    pmt1 = pmt1.attach(TN.NMPlLivePlot(axes[1], 'mov. avg', ['b-']))

    # walk = walk.attach(SN.NPrint())

    return pipe

