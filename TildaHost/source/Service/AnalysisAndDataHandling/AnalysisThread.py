"""
Created on 15.11.2016

@author: simkaufm

Module Description:  Module that will hold the main Analysis Thread for each scan.
"""
import sys
import time
import logging
from copy import deepcopy
from datetime import datetime

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, QObject, QMutex
from PyQt5.QtWidgets import QApplication

from Service.AnalysisAndDataHandling.tildaPipeline import find_pipe_by_seq_type
import TildaTools as TiTs


class AnalysisThread(QThread):
    # analysis_done = pyqtSignal()

    def __init__(self, scan_dict, callback_sig, live_plot_callback_tuples, fit_res_callback_dict,
                 stop_request_signal, prep_track_in_pipe_signal, new_data_signal, scan_complete_callback,
                 dac_new_volt_set_callback):
        # TODO comment all parameters here!!!!!
        super(AnalysisThread, self).__init__()
        self.pipeline = find_pipe_by_seq_type(scan_dict, callback_sig,
                                              live_plot_callback_tuples, fit_res_callback_dict,
                                              scan_complete_callback, dac_new_volt_set_callback)
        self.stop_analysis_bool = False
        self.clear_after_finish = False  # boolean that will tell the pipeline to clear(->save)
        #  or not after analysis completion
        self.num_of_analysed_elements_total = 0
        self.raw_data_storage = np.zeros(0, dtype=np.int32)
        self.dmm_dict_list = []
        self.dmm_dict_merge = {} # replaces dmm_dict_list in order to avoid building a huge list
        self.triton_dict_merge = {} # same for triton dicts
        self.mutex = QMutex()
        # using mute expression in order to solve race cond, between append from outside loop
        # and read(/shrink) from inside loop
        stop_request_signal.connect(self.stop_analysis)
        prep_track_in_pipe_signal.connect(self.prepare_track_in_pipe)
        new_data_signal.connect(self.new_data)

    def run(self):
        logging.info('analysis thread running now')
        while not self.stop_analysis_bool or len(self.raw_data_storage) or any(self.dmm_dict_list) \
                or any(self.dmm_dict_merge) or any(self.triton_dict_merge):
            if len(self.raw_data_storage):
                self.mutex.lock()

                # print('analysing data now')
                data = deepcopy(self.raw_data_storage)
                self.num_of_analysed_elements_total += len(data)
                self.raw_data_storage = np.zeros(0, dtype=np.int32)
                self.mutex.unlock()
                st_feed = datetime.now()
                logging.info('Analyzing now!')
                self.pipeline.feed(data)  # takes a while
                done_feed = datetime.now()
                elapsed_feed_ms = (done_feed - st_feed).total_seconds() * 1000
                logging.debug('Analyzing %d data points took %.1f ms'
                              % (self.num_of_analysed_elements_total, elapsed_feed_ms))
                # self.sleep(1)  # simulate feed
                # print('number of total analysed data: %s ' % self.num_of_analysed_elements_total)
            if any(self.dmm_dict_list) or any(self.dmm_dict_merge) or any(self.triton_dict_merge):
                self.mutex.lock()
                self.dmm_dict_list.append(self.dmm_dict_merge)
                self.dmm_dict_list.append(self.triton_dict_merge)
                to_feed = deepcopy(self.dmm_dict_list)
                self.dmm_dict_list = []
                self.dmm_dict_merge = {}
                self.triton_dict_merge = {}
                self.mutex.unlock()
                for dmm_dict in to_feed:
                    # print('feeding dmm dict: %s ' % dmm_dict)
                    self.pipeline.feed(dmm_dict)
                # self.sleep(1)
            self.msleep(50)  # not sure if necessary
        if self.stop_analysis_bool:
            logging.info('stopping pipeline now!')
            self.pipeline.stop()
        if self.clear_after_finish:
            # this means saving! -> finish analysis of all stored elements,
            # before clearing the pipe!
            logging.info('will save now!')
            self.pipeline.save()
            # self.sleep(5)  # simulate saving
        # print('done with analysis')
        self.stop_analysis_bool = False
        self.clear_after_finish = False
        self.quit()  # stop the thread from running

    def stop_analysis(self, clear_also):
        logging.info('ScanMain received stop pipeline command')
        # also finish analysis of all stored elements first before stopping all analysis.
        # when aborted or halted no new data will be fed anyhow!
        self.stop_analysis_bool = True  # anyhow stop the thread from running
        self.clear_after_finish = clear_also

    def prepare_track_in_pipe(self, track_num, track_index):
        track_name = 'track' + str(track_num)
        self.pipeline.pipeData['pipeInternals']['activeTrackNumber'] = (track_index, track_name)
        logging.info('starting pipeline: ' + str(self.pipeline))
        self.pipeline.start()

    def new_data(self, data, dmm_dict):
        """
        new data coming in either from the raw datastream of the fpga or
         a dictionary from the digital mulitmeter's / triton listener.
         The data is emitted from the TritonListener._receive() / ScanMain.read_multimeter()
        :param data: np.array, holds 32b raw events from the fpga data stream
        :param dmm_dict: dict, either triton or dmm dict.
            triton dict: {'track0':
                            {'triton':
                             {'duringScan': {'dev_name':
                                                {'ch0': {'data': [ ... ], 'required': -1, 'acquired': 20},
                                                 'ch1': {'data': ...}}}}}}
                 --> NOTE: the triton_log is always sending ALL data, so the same data is emitted multiple times.

             dmm dict: {'dev_name0': array([ 1.,  1.,  1.,  1.,  1.]),
                        'dev_name1': array([ 1.,  1.,  1.,  1.,  1.])}
                --> NOTE: data is only emitted ONCE -> can be stored here or is lost.
        :return:
        """
        self.mutex.lock()
        self.raw_data_storage = np.append(self.raw_data_storage, data)
        if any(dmm_dict):
            # self.dmm_dict_list.append(dmm_dict)
            # current_track_no = self.pipeline.pipeData['pipeInternals']['activeTrackNumber']
            current_track = self.pipeline.pipeData['pipeInternals']['activeTrackNumber'][1]
            if current_track in dmm_dict:
                # it is a triton dict
                logging.debug('merging triton dict %s into %s.' %(dmm_dict, self.triton_dict_merge))
                TiTs.merge_extend_dicts(self.triton_dict_merge, dmm_dict)  # this will take care of the multiple emits.
                logging.debug('merge result is: %s ' % self.triton_dict_merge)
            else:
                # it must be a dmm dict
                logging.debug('merging dmm dict %s into %s.' % (dmm_dict, self.dmm_dict_merge))
                TiTs.deepupdate(self.dmm_dict_merge, dmm_dict)
                logging.debug('merge result is: %s ' % self.dmm_dict_merge)
        self.mutex.unlock()


class TestEmitter(QObject):
    stop_req = pyqtSignal(bool)
    prep_sig = pyqtSignal(int, int)
    new_data_sig = pyqtSignal(np.ndarray, dict)

    def stop(self, eins):
        self.stop_req.emit(eins)

    def prep(self, tpl):
        self.prep_sig.emit(tpl[0], tpl[1])

    def emit_test_data(self, num, dicti):
        self.new_data_sig.emit(np.random.random_sample(num), dicti)


class TestReceiver(QObject):

    def __init__(self, sig):
        super(TestReceiver, self).__init__()
        sig.connect(self.display)

    def display(self, event):
        print(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    te = TestEmitter()
    from Service.Scan.draftScanParameters import draftScanDict
    scan_d = draftScanDict
    scan_d['pipeInternals']['workingDirectory'] = 'E:\Workspace\deleteMe'
    an_th = AnalysisThread(draftScanDict, None, (None, None, None), None,
                           te.stop_req, te.prep_sig, te.new_data_sig)
    an_th.start()

    # an_th.finished.connect(app.quit)  # will not quit if still running...
    last_emit = datetime.now()
    for j in range(1000):
        time.sleep(0.005)
        te.emit_test_data(10, {'demo': j})
        # te.emit_test_data(0, {'demo': 0})
        # te.emit_test_data(10, {})
        now = datetime.now()
        print('time since last emit: %s' % (now - last_emit))
        last_emit = now
        # app.processEvents()  # with qtimer instead of time.sleep this is hopefully not necessary

    te.stop(False)
    print('is running: %s' % an_th.isRunning())
    print('first round done, will start again')
    while an_th.isRunning():
        pass
    an_th.start()
    print('is running: %s' % an_th.isRunning())
    an_th.finished.connect(app.quit)
    time.sleep(0.005)
    te.stop(False)

    app.exec_()
