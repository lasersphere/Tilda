"""

Created on '20.05.2015'

@author:'simkaufm'

"""

import matplotlib.pyplot as plt
import logging

import Service.AnalysisAndDataHandling.tildaNodes as TN
import polliPipe.simpleNodes as SN
from polliPipe.node import Node
import Service.FolderAndFileHandling as FaFH
# import PyQtGraphPlotter

from polliPipe.pipeline import Pipeline


def find_pipe_by_seq_type(scan_dict, callback_sig):
    seq_type = scan_dict['isotopeData']['type']
    if seq_type == 'cs' or seq_type == 'csdummy':
        logging.debug('starting pipeline of type: cs')
        return CsPipe(scan_dict, callback_sig)
    elif seq_type == 'trs' or seq_type == 'trsdummy':
        logging.debug('starting pipeline of type: trs')
        return TrsPipe(scan_dict, callback_sig)
    elif seq_type == 'kepco':
        logging.debug('starting pipeline of type: kepco')
        return kepco_scan_pipe(scan_dict, callback_sig)
    else:
        return None


def TrsPipe(initialScanPars=None, callback_sig=None):
    """
    Pipeline for the dataflow and analysis of one Isotope using the time resolved sequencer.
    Mutliple Tracks are supported.
    """
    start = Node()

    pipe = Pipeline(start)

    pipe.pipeData = initPipeData(initialScanPars)
    # walk = start.attach(SN.NPrint())
    # walk = start.attach(TN.NSaveRawData())
    walk = start.attach(TN.NSplit32bData())
    # walk = walk.attach(TN.NSplit32bData())
    walk = walk.attach(TN.NCSSortRawDatatoArray())
    walk = walk.attach(TN.NSendnOfCompletedStepsViaQtSignal(callback_sig))
    walk = walk.attach(TN.NRemoveTrackCompleteFlag())
    walk = walk.attach(TN.NCSSum())

    pl_branch_2d = walk.attach(TN.NMPLImagePLot(0))
    pl_branch_2d = pl_branch_2d.attach(TN.NMPlDrawPlot())

    compl_tr_br = walk.attach(TN.NCheckIfTrackComplete())
    compl_tr_br = compl_tr_br.attach(TN.NAddWorkingTime(True))

    # meas_compl_br = walk.attach(TN.NCheckIfMeasurementComplete())
    walk = walk.attach(TN.NSaveAllTracks())
    #
    walk = walk.attach(TN.NTRSProjectize())
    walk = walk.attach(TN.NSaveProjection())
    # walk = walk.attach(SN.NPrint())

    return pipe


def CsPipe(initialScanPars=None, callback_sig=None):
    """
    Pipeline for the dataflow and analysis of one Isotope using the continous sequencer.
    """
    start = Node()

    pipe = Pipeline(start)

    fig, axes = plt.subplots(6, sharex=True)
    fig.canvas.set_window_title(initialScanPars.get('isotopeData', {'isotope': ''}).get('isotope'))
    pipe.pipeData = initPipeData(initialScanPars)

    walk = start.attach(TN.NSaveRawData())
    walk = start.attach(TN.NSplit32bData())
    #
    walk = walk.attach(TN.NCSSortRawDatatoArray())
    walk = walk.attach(TN.NSendnOfCompletedStepsViaQtSignal(callback_sig))

    #
    branch = walk.attach(TN.NAccumulateSingleScan())
    # branch = branch.attach(SN.NPrint())
    branch = branch.attach(TN.NSingleArrayToSpecData())

    branch1 = branch.attach(TN.NSingleSpecFromSpecData([0]))
    # branch1 = branch1.attach(TN.NPlotUpdater(fig, axes[0], 'scaler 0', ['blue']))
    branch1 = branch1.attach(TN.NMPlLivePlot(axes[0], 'scaler 0', ['blue']))

    branch2 = branch.attach(TN.NSingleSpecFromSpecData([1]))
    branch2 = branch2.attach(TN.NMPlLivePlot(axes[1], 'scaler 1', ['green']))

    branch3 = branch.attach(TN.NSingleSpecFromSpecData([0, 1]))
    branch3 = branch3.attach(TN.NMPlLivePlot(axes[2], 'scaler 0+1', ['red']))

    walk = walk.attach(TN.NRemoveTrackCompleteFlag())
    walk = walk.attach(TN.NCSSum())

    summe = walk.attach(TN.NSingleArrayToSpecData())
    sum0 = summe.attach(TN.NMultiSpecFromSpecData([[0], [1]]))
    sum0 = sum0.attach(TN.NMPlLivePlot(axes[3], 'live sum', ['blue', 'green']))

    sum01 = summe.attach(TN.NSingleSpecFromSpecData([0, 1]))
    sum01 = sum01.attach(TN.NMPlLivePlot(axes[4], 'scaler 0+1', ['red']))
    sum01 = sum01.attach(TN.NMPlDrawPlot())

    compl_tr_br = walk.attach(TN.NCheckIfTrackComplete())
    compl_tr_br = compl_tr_br.attach(TN.NAddWorkingTime(True))

    finalsum = walk.attach(TN.NSingleArrayToSpecData())
    finalsum = finalsum.attach(TN.NMultiSpecFromSpecData([[0], [1]]))
    finalsum = finalsum.attach(TN.NMPlLivePlot(axes[5], 'final sum', ['blue', 'green']))
    finalsum = finalsum.attach(TN.NMPlDrawPlot())

    # walk = walk.attach(TN.NSaveIncomDataForActiveTrack())
    # walk = walk.attach(TN.NCheckIfMeasurementComplete())
    # walk = walk.attach(TN.NSendnOfCompletedStepsViaQtSignal(callback_sig))
    walk = walk.attach(TN.NSaveAllTracks())

    return pipe


def kepco_scan_pipe(initial_scan_pars, callback_sig=None):
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
    :return: dict, {'isotopeData', 'progConfigs', 'track0', 'pipeInternals'}
    """
    pipeData = initialScanPars

    pipeData['pipeInternals']['curVoltInd'] = 0
    xml_file_name = FaFH.createXmlFileOneIsotope(pipeData)
    pipeData['pipeInternals']['activeXmlFilePath'] = xml_file_name
    if pipeData['isotopeData']['type'] in ['trs', 'trsdummy']:
        pipeData['pipeInternals']['activeXmlProjFilePath'] = FaFH.createXmlFileOneIsotope(
            pipeData, 'trs_proj', xml_file_name.replace('.xml', '_proj.xml'))
    return pipeData


def simple_counter_pipe(qt_sig):
    start = Node()

    fig, axes = plt.subplots(2, sharex=True)
    fig.canvas.set_window_title('Simple Counter')

    sample_rate = 1 / 0.02  # values per second, fpga samples at 20ms currently
    pipe = Pipeline(start)

    walk = start.attach(TN.NSplit32bData())
    walk = walk.attach(TN.NSPSortByPmt(sample_rate))
    walk = walk.attach(TN.NSumListsInData())
    walk = walk.attach(TN.NSendDataViaQtSignal(qt_sig))
    walk = walk.attach(TN.NMPLCloseFigOnClear(fig))
    # walk = walk.attach(SN.NPrint())

    walk = walk.attach(TN.NSPAddxAxis())
    pmt0 = walk.attach(TN.NOnlyOnePmt(0))
    pmt0 = pmt0.attach(TN.NMPlLivePlot(axes[0], 'mov. avg', ['blue']))

    pmt1 = walk.attach(TN.NOnlyOnePmt(1))
    pmt1 = pmt1.attach(TN.NMPlLivePlot(axes[1], 'mov. avg', ['green']))
    pmt1 = pmt1.attach(TN.NMPlDrawPlot())

    # walk = walk.attach(SN.NPrint())

    return pipe

