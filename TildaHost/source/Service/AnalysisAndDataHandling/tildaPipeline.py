"""

Created on '20.05.2015'

@author:'simkaufm'

"""

import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt

import Service.AnalysisAndDataHandling.tildaNodes as TN
import Service.FileOperations.FolderAndFileHandling as FaFH
import polliPipe.simpleNodes as SN
from polliPipe.node import Node

# import PyQtGraphPlotter

from polliPipe.pipeline import Pipeline


def find_pipe_by_seq_type(scan_dict, callback_sig, live_plot_callback_tuples):
    seq_type = scan_dict['isotopeData']['type']
    if seq_type == 'cs' or seq_type == 'csdummy':
        logging.debug('starting pipeline of type: cs')
        return CsPipe(scan_dict, callback_sig)
    elif seq_type == 'trs' or seq_type == 'trsdummy':
        logging.debug('starting pipeline of type: trs')
        return TrsPipe(scan_dict, callback_sig, live_plot_callbacks=live_plot_callback_tuples)
    elif seq_type == 'kepco':
        logging.debug('starting pipeline of type: kepco')
        return kepco_scan_pipe(scan_dict, callback_sig)
    else:
        return None


def TrsPipe(initialScanPars=None, callback_sig=None, x_as_voltage=True, live_plot_callbacks=None):
    """
    Pipeline for the dataflow and analysis of one Isotope using the time resolved sequencer.
    Mutliple Tracks are supported.
    always feed raw data.
    """
    if live_plot_callbacks is None:
        live_plot_callbacks = (None, None, None)
    start = Node()
    maintenance = start.attach(TN.NMPLCloseFigOnClear())
    maintenance = maintenance.attach(TN.NAddWorkingTimeOnClear(True))

    pipe = Pipeline(start)

    pipe.pipeData = initPipeData(initialScanPars)
    # # walk = start.attach(SN.NPrint())
    # walk = start.attach(TN.NFilterDMMDicts())
    # walk = walk.attach(TN.NSaveRawData())
    #
    # # walk = walk.attach(TN.NSplit32bData())
    # # walk = walk.attach(TN.NSplit32bData())
    # walk = walk.attach(TN.NCSSortRawDatatoArray())
    # walk = walk.attach(TN.NSendnOfCompletedStepsViaQtSignal(callback_sig))
    #
    # walk = walk.attach(TN.NSortedTrsArraysToSpecData(x_as_voltage))
    #
    # walk = walk.attach(TN.NMPLImagePlotAndSaveSpecData(0, None, None, None))  # *live_plot_callbacks))
    #
    # compl_tr_br = walk.attach(TN.NCheckIfTrackComplete())
    # compl_tr_br = compl_tr_br.attach(TN.NAddWorkingTime(True))

    # alternative pipeline:
    fast = start.attach(TN.NFilterDMMDicts())
    fast = fast.attach(TN.NSaveRawData())
    fast = fast.attach(TN.NTRSSortRawDatatoArrayFast())
    fast = fast.attach(TN.NSendnOfCompletedStepsViaQtSignal(callback_sig))
    # fast = fast.attach(SN.NPrint())
    fast = fast.attach(TN.NTRSSumFastArrays())
    # fast = fast.attach(SN.NPrint())

    fast_spec = fast.attach(TN.NSortedZeroFreeTRSDat2SpecData())
    fast_spec = fast_spec.attach(TN.NSpecDataZeroFreeProjection())
    fast_spec = fast_spec.attach(TN.NMPLImagePlotAndSaveSpecData(0, *live_plot_callbacks))

    # fast_spec = fast_spec.attach(TN.NSaveSpecData())

    return pipe


def CsPipe(initialScanPars=None, callback_sig=None):
    """
    Pipeline for the dataflow and analysis of one Isotope using the continous sequencer.
    always feed raw data.
    """
    start = Node()

    pipe = Pipeline(start)
    # start = start.attach(SN.NPrint())
    start = start.attach(TN.NFilterDMMDicts())

    maintenance = start.attach(TN.NMPLCloseFigOnInit())

    fig, axes = plt.subplots(5, sharex=True)
    pipe.pipeData = initPipeData(initialScanPars)

    filen = os.path.split(pipe.pipeData['pipeInternals']['activeXmlFilePath'])[1]
    window_title = 'plot ' + filen
    fig.canvas.set_window_title(window_title)

    walk = start.attach(TN.NSaveRawData())
    # walk = start.attach(TN.NSplit32bData())
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
    compl_tr_br = compl_tr_br.attach(SN.NPrint())

    # walk = walk.attach(TN.NSaveIncomDataForActiveTrack())
    # walk = walk.attach(TN.NCheckIfMeasurementComplete())
    # walk = walk.attach(TN.NSendnOfCompletedStepsViaQtSignal(callback_sig))
    walk = walk.attach(TN.NSaveAllTracks())

    return pipe


def kepco_scan_pipe(initial_scan_pars, callback_sig=None, as_voltage=False):
    """
    pipeline for the measurement and analysis of a kepco scan
    raw data and readback from dmm are fed into the pipeline.
    :param initial_scan_pars: full scan dictionary which will be used as starting point for this pipeline
    always feed raw data.
    """
    start = Node()

    maintenance = start.attach(TN.NMPLCloseFigOnInit())
    maintenance = maintenance.attach(TN.NAddWorkingTimeOnClear(True))

    pipe = Pipeline(start)
    pipe.pipeData = initPipeData(initial_scan_pars)
    dmm_names = sorted(list(pipe.pipeData['measureVoltPars']['dmms'].keys()))

    fig, axes = plt.subplots(len(dmm_names), sharex=True)
    if len(dmm_names) == 1:
        axes = [axes]
    filen = os.path.split(pipe.pipeData['pipeInternals']['activeXmlFilePath'])[1]
    window_title = 'plot ' + filen
    fig.canvas.set_window_title(window_title)
    #
    walk = start.attach(TN.NSaveRawData())
    # walk = start.attach(SN.NPrint())
    specdata_path = start.attach(TN.NStartNodeKepcoScan(as_voltage, dmm_names))
    specdata_path = specdata_path.attach(TN.NSendnOfCompletedStepsViaQtSignal(callback_sig))
    #
    plot_dict = {}
    for ind, dmm in enumerate(dmm_names):
        plot_dict[dmm] = specdata_path.attach(TN.NSingleSpecFromSpecData([ind]))
        plot_dict[dmm] = plot_dict[dmm].attach(TN.NMPlLivePlot(axes[ind], '%s [v]' % dmm, ['blue']))
    axes[-1].set_xlabel('DAC line[V]' if as_voltage else 'DAC Register')
    draw = plot_dict[dmm_names[-1]].attach(TN.NMPlDrawPlot())

    specdata_path = specdata_path.attach(TN.NSaveSpecData())

    specdata_path = specdata_path.attach(TN.NStraightKepcoFitOnClear(axes, dmm_names))
    # # specdata_path = specdata_path.attach(TN.NSaveIncomDataForActiveTrack())
    # # more has to be included...
    return pipe


def initPipeData(initialScanPars):
    """
    initialize the pipeData used for the analysis Pipeline
    -> store the initialScanPars in the pipe.pipeData and create an .xml file
    :return: dict, {'isotopeData', 'progConfigs', 'track0', 'pipeInternals'}
    always feed raw data.
    """
    pipeData = initialScanPars

    # trackDict = pipeData['track0']
    # trackDict.update(dacStartVoltage=get_voltage_from_18bit(trackDict['dacStartRegister18Bit']))
    # trackDict.update(dacStopVoltage=get_voltage_from_18bit(
    #     VCon.calc_dac_stop_18bit(trackDict['dacStartRegister18Bit'],
    #                              trackDict['dacStepSize18Bit'],
    #                              trackDict['nOfSteps'])))

    pipeData['pipeInternals']['curVoltInd'] = 0
    pipeData['isotopeData']['isotopeStartTime'] = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    xml_file_name = FaFH.createXmlFileOneIsotope(pipeData)
    pipeData['pipeInternals']['activeXmlFilePath'] = xml_file_name

    # in the past an extra projection file was created. but for now,
    #  the projection should stay within the .xml file for this scan.
    # if pipeData['isotopeData']['type'] in ['trs', 'trsdummy']:
    #     pipeData['pipeInternals']['activeXmlProjFilePath'] = FaFH.createXmlFileOneIsotope(
    #         pipeData, 'trs_proj', xml_file_name.replace('.xml', '_proj.xml'))
    return pipeData


def simple_counter_pipe(qt_sig, act_pmt_list):
    """
    pipeline for analysis and displaying while the simple counter is running.
    always feed raw data.
    """
    start = Node()

    fig, axes = plt.subplots(len(act_pmt_list), sharex=True)
    fig.canvas.set_window_title('Simple Counter')

    sample_rate = 1 / 0.2  # values per second, fpga samples at 200ms currently
    pipe = Pipeline(start)

    start = start.attach(TN.NFilterDMMDicts())

    walk = start.attach(TN.NSplit32bData())
    walk = walk.attach(TN.NSPSortByPmt(sample_rate))
    walk = walk.attach(TN.NSumListsInData())
    walk = walk.attach(TN.NSendDataViaQtSignal(qt_sig))
    walk = walk.attach(TN.NMPLCloseFigOnClear(fig))
    # walk = walk.attach(SN.NPrint())

    walk = walk.attach(TN.NSPAddxAxis())
    pmt_dict = {}
    for pmt_ind, pmt_num in enumerate(act_pmt_list):
        pmt_dict['pmt' + str(pmt_num)] = walk.attach(TN.NOnlyOnePmt(pmt_num))
        pmt_dict['pmt' + str(pmt_num)] = pmt_dict['pmt' + str(pmt_num)].attach(
            TN.NMPlLivePlot(axes[pmt_ind], 'mov. avg. Ch %s' % pmt_num, ['blue']))

    draw = pmt_dict['pmt' + str(act_pmt_list[-1])].attach(TN.NMPlDrawPlot())

    return pipe


def tilda_passive_pipe(initial_scan_pars, raw_callback, steps_scans_callback):
    """
    pipeline for running the tilda passive mode.
    always feed raw data.
    """
    start = Node()

    pipe = Pipeline(start)
    initial_scan_pars['track0']['nOfSteps'] = None
    initial_scan_pars['track0']['nOfScans'] = 0
    pipe.pipeData = initPipeData(initial_scan_pars)

    start = start.attach(TN.NFilterDMMDicts())

    maintenance = start.attach(TN.NMPLCloseFigOnInit())
    maintenance = maintenance.attach(TN.NAddWorkingTimeOnClear(True))

    # walk = start.attach(SN.NPrint())
    walk = start.attach(TN.NSendDataViaQtSignal(raw_callback))
    walk = walk.attach(TN.NSaveRawData())
    walk = walk.attach(TN.NTiPaAccRawUntil2ndScan(steps_scans_callback))
    # walk = walk.attach(SN.NPrint())

    # walk = walk.attach(TN.NSplit32bData())
    walk = walk.attach(TN.NCSSortRawDatatoArray())
    walk = walk.attach(TN.NSendnOfCompletedStepsAndScansViaQtSignal(steps_scans_callback))
    walk = walk.attach(TN.NRemoveTrackCompleteFlag())
    walk = walk.attach(TN.NCSSum())
    # walk = walk.attach(SN.NPrint())

    pl_branch_2d = walk.attach(TN.NMPLImagePLot(1, False))
    pl_branch_2d = pl_branch_2d.attach(TN.NMPlDrawPlot())
    #
    # compl_tr_br = walk.attach(TN.NCheckIfTrackComplete())
    #
    # # meas_compl_br = walk.attach(TN.NCheckIfMeasurementComplete())
    walk = walk.attach(TN.NSaveAllTracks())
    # #
    walk = walk.attach(TN.NTRSProjectize(False))
    walk = walk.attach(TN.NSaveProjection())

    return pipe


def time_resolved_display(filepath, liveplot_callbacks):
    """
    pipeline for displaying a time resolved spectra.
    feed only time resolved specdata to it!
    """
    start = Node()

    pipe = Pipeline(start)

    pipe.pipeData['pipeInternals'] = {}

    pipe.pipeData['pipeInternals']['activeXmlFilePath'] = filepath
    pipe.pipeData['pipeInternals']['activeTrackNumber'] = (0, 'track0')
    # path of file is used mainly for the window title.
    walk = start.attach(SN.NPrint())
    # walk = walk.attach(TN.NMPLImagePlotSpecData(0, dataPath))
    walk = walk.attach(TN.NMPLImagePlotAndSaveSpecData(0, *liveplot_callbacks))
    # walk = walk.attach(TN.NMPlDrawPlot())

    return pipe
