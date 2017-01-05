"""
Created on 15.11.2016

@author: simkaufm

Module Description:  Module that will hold the main Analysis Thread for each scan.
"""
import sys
import time
from copy import deepcopy
from datetime import datetime

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, QObject, QMutex
from PyQt5.QtWidgets import QApplication

from Service.AnalysisAndDataHandling.tildaPipeline import find_pipe_by_seq_type


class AnalysisThread(QThread):
    # analysis_done = pyqtSignal()

    def __init__(self, scan_dict, callback_sig, live_plot_callback_tuples, fit_res_callback_dict,
                 stop_request_signal, prep_track_in_pipe_signal, new_data_signal):
        super(AnalysisThread, self).__init__()
        self.pipeline = find_pipe_by_seq_type(scan_dict, callback_sig,
                                              live_plot_callback_tuples, fit_res_callback_dict)
        self.stop_analysis_bool = False
        self.clear_after_finish = False  # boolean that will tell the pipeline to clear(->save)
        #  or not after analysis completion
        self.num_of_analysed_elements_total = 0
        self.raw_data_storage = np.ndarray(0, dtype=np.int32)
        self.dmm_dict_list = []
        self.mutex = QMutex()
        # using mute expression in order to solve race cond, between append from outside loop
        # and read(/shrink) from inside loop
        stop_request_signal.connect(self.stop_analysis)
        prep_track_in_pipe_signal.connect(self.prepare_track_in_pipe)
        new_data_signal.connect(self.new_data)

    def run(self):
        print('thread running')
        while not self.stop_analysis_bool or len(self.raw_data_storage) or any(self.dmm_dict_list):
            if len(self.raw_data_storage):
                self.mutex.lock()

                # print('analysing data now')
                data = deepcopy(self.raw_data_storage)
                self.num_of_analysed_elements_total += len(data)
                self.raw_data_storage = np.ndarray(0, dtype=np.int32)
                self.mutex.unlock()
                self.pipeline.feed(data)
                # self.sleep(1)  # simulate feed
                # print('number of total analysed data: %s ' % self.num_of_analysed_elements_total)
            if any(self.dmm_dict_list):
                self.mutex.lock()
                to_feed = deepcopy(self.dmm_dict_list)
                self.dmm_dict_list = []
                self.mutex.unlock()
                for dmm_dict in to_feed:
                    # print('feeding dmm dict: %s ' % dmm_dict)
                    self.pipeline.feed(dmm_dict)
                # self.sleep(1)
            self.msleep(50)  # not sure if necessary
        if self.clear_after_finish:
            # this means saving! -> finish analysis of all stored elements,
            # before clearing the pipe!
            print('will save now!')
            self.pipeline.clear()
            # self.sleep(5)  # simulate saving
        # print('done with analysis')
        self.stop_analysis_bool = False
        self.clear_after_finish = False

    def stop_analysis(self, clear_also):
        self.pipeline.stop()
        # also finish analysis of all stored elements first before stopping all analysis.
        # when aborted or halted no new data will be fed anyhow!
        self.stop_analysis_bool = True  # anyhow stop the thread from running
        self.clear_after_finish = clear_also

    def prepare_track_in_pipe(self, track_num, track_index):
        track_name = 'track' + str(track_num)
        self.pipeline.pipeData['pipeInternals']['activeTrackNumber'] = (track_index, track_name)
        print('starting pipeline: ', self.pipeline)
        self.pipeline.start()

    def new_data(self, data, dmm_dict):
        self.mutex.lock()
        self.raw_data_storage = np.append(self.raw_data_storage, data)
        if any(dmm_dict):
            self.dmm_dict_list.append(dmm_dict)
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
