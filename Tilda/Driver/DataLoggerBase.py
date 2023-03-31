"""
Created on 2023-03

@author: Tim Lellinger

Module Description: Template class to use for database-like data logging (SQL, Influx, maybe Triton in the future...)
"""

import time
from datetime import datetime
import logging
from copy import deepcopy
import numpy as np
from functools import wraps
from threading import Thread
from PyQt5.QtCore import QObject, pyqtSignal

import Tilda.Application.Config as Cfg

class DataLoggerBase(QObject):
    # signal to emit dmm values for live plotting during the pre-/postScan.
    pre_post_meas_data_dict_callback = pyqtSignal(dict)
    # signal to emit data to the pipeLine
    data_to_pipe_sig = pyqtSignal(np.ndarray, dict)

    def __init__(self, name):#the name is also the tag under which the data is saved
        super().__init__()
        self.name = name
        self.logger = logging.getLogger(name+"logger")
        self.run_id = -1

        self._thread = None
        self._interval = 0
        self.adjust_interval = False

        self.track_name = ''
        self.pre_dur_post_str = ''
        self.logging_enabled = False
        self.logging_complete = True
        self.log = {}
        self.back_up_log = deepcopy(self.log)
        self.ch_time = {}#dictionary of string:int with the timestamps (unix time) when the channel was aquired last time

        self.paused = False

        self.interval = 0.5
        self.last_emit_to_analysis_pipeline_datetime = datetime.now()



    def get_channels_from_db(self):
        """
        To be implemented by the child class. return a list of all available channels. Names have to be compatible with xml tag
        """
        raise NotImplementedError

    def aquireChannel(self, chname, lastaqu):
        """
        To be implemented by the child class, called in _aquire for each subscribed channel.
        Gets the name of a single channel and the last time it was aquired (in unixtime) as input.
        Returns an array of float measurements.
        """
        raise NotImplementedError

    def write_run_to_db(self, unix_time, xml_file):
        raise NotImplementedError

    def update_run_status_in_db(self, status):
        raise NotImplementedError

    def conclude_run_in_db(self, unix_time_stop, status):
        raise NotImplementedError

    def setupChtimes(self, pre_dur_post_str):
        """called in setup_log and on unpause() in _aquire to set up the dictionary with the information on when the channel was aquired last time. By default
        it pretends to have read the channels when the log was set up. Can be overridden by child, but not necessary"""
        self.ch_time = {ch: time.time() for ch in self.log.keys()}

    """ Implementation of the periodic readout. """
    def _run(self):
        """ The periodic logic """
        while self._interval > 0:
            t0 = time.time()
            self._acquire()  # Process new data.
            dt = time.time() - t0
            if dt < self._interval:
                time.sleep(self._interval - dt)
            elif dt > self._interval > 0:
                self.logger.debug('Influx processing time is bigger than interval!')
                if self.adjust_interval:
                    self.set_interval(dt)
                    self.logger.debug("setting interval to {}".format(dt))

    def _acquire(self):
        """ Called in self._run(), acquires and processes the data from the db. """
        if self.logging_enabled:
            self.new_data_flag = False
            if Cfg._main_instance.scan_main.sequencer.pause_bool:
                self.paused = True
            else:
                if self.paused:
                    self.paused = False
                    self.setupChtimes()
                for ch in self.log.keys():
                    acq_on_log_start = self.back_up_log[ch]['acquired']
                    lastaquired = self.ch_time[ch]
                    data = self.aquireChannel(ch,lastaquired)

                    if len(data)>0:
                        self.new_data_flag = True
                        self.ch_time[ch] = data[-1][0]
                        data = [d[1] for d in data if d is not None]

                        self.log[ch]['data'] += data
                        self.log[ch]['acquired'] += len(data)

                        if self.log[ch]['required'] > 0:
                            dif = self.log[ch]['acquired'] - acq_on_log_start - self.log[ch]['required']
                            if dif > 0:
                                self.log[ch]['data'] = self.log[ch]['data'][:-dif]
                                self.log[ch]['acquired'] = self.log[ch]['required'] + acq_on_log_start

            if self.new_data_flag:  # Only update if data comes in.
                live_data_dict = {self.track_name: {self.name : {self.pre_dur_post_str: self.log}}}
                if self.pre_dur_post_str == 'duringScan':
                    to_send = deepcopy(live_data_dict)
                    self.data_to_pipe_sig.emit(np.ndarray(0, dtype=np.int32), to_send)
                else:  # in pre and postScan emit received value to callback for live data plotting
                    self.pre_post_meas_data_dict_callback.emit(deepcopy(live_data_dict))
                self.last_emit_to_analysis_pipeline_datetime = datetime.now()
            self.check_log_complete()

    def set_interval(self, t):
        """ Set the interval. Start or stop periodic thread as necessary. """

        if self._thread is not None and self._interval > 0:
            self._interval = t
        elif t > 0:
            self._interval = t
            self._thread = Thread(target=self._run)
            self._thread.start()

    """ Configure Tilda signals. """

    # noinspection PyProtectedMember
    def get_existing_callbacks_from_main(self):
        """ check wether existing callbacks are still around in the main and then connect to those. """
        if Cfg._main_instance is not None:
            logging.info('SQLStream is connecting to existing callbacks in main.')
            callbacks = Cfg._main_instance.gui_live_plot_subscribe()
            self.pre_post_meas_data_dict_callback = callbacks[6]
            self.data_to_pipe_sig = Cfg._main_instance.scan_main.data_to_pipe_sig

    def setup_log(self, log, pre_dur_post_str, track_name):
        """
        set up the log. The log is a dict containing the channels as a dict with the number of required values:
         {'ch1': {'required': 2, 'data': [], 'acquired': 0}, ...}
        """
        # connect to the callback for live data plotting so the log can be emitted as well
        self.get_existing_callbacks_from_main()
        # note down the track and which part of the scan is done
        self.track_name = track_name
        self.pre_dur_post_str = pre_dur_post_str

        if log is None or log == {}:
            self.log = {}

            self.ch_time = {}
            self.logging_complete = True  # do not log
            self.back_up_log = deepcopy(self.log)  # store backup log, because work is done in self.log
        else:
            self.log = log

            self.setupChtimes(pre_dur_post_str)

            self.logging_complete = False
            self.back_up_log = deepcopy(self.log)  # store backup log, because work is done in self.log
            if self.log:
                self.logging_complete = False
            else:
                # could not resolve names etc. do not log!
                self.logging_complete = True
                self.log = {}

                self.ch_time = {}
                self.back_up_log = deepcopy(self.log)  # store backup log, because work is done in self.log

    def check_log_complete(self):
        """ calls self.stop_log() if all values have been acquired """
        check_sum = 0
        for ch, val in self.log.items():
            # for periodic data taking 'required' will be negative. Checksum must not be 0.
            if val['required'] == -1:
                if self.pre_dur_post_str == 'duringScan':
                    check_sum += 1
                else:  # continuous acquisition only makes sense duringScan
                    self.logger.error('channel %s: Continuous acquisition not allowed in %s.'
                                      % (ch, self.pre_dur_post_str))
            else:
                acq_on_log_start = self.back_up_log[ch]['acquired']
                check_sum += max(0, val['required'] - val['acquired'] + acq_on_log_start)
        if check_sum == 0:
            self.logging_complete = True
            self.stop_log()
            return True
        else:
            return False

    def start_log(self):
        """ start logging of the desired channels. Be sure to set up the log beforehand with self.setup_log """
        self.logging_complete = self.log == {}
        self.last_emit_to_analysis_pipeline_datetime = datetime.now()

        self.set_interval(self.interval)
        self.logging_enabled = True

    def stop_log(self):
        """ stop logging, by setting self.logging_enabled to False """
        self.logging_enabled = False
        self.set_interval(0)

    """ Static threading methods. """

    @staticmethod
    def locked(func):
        """ This is a decorator for simplified usage of the threadlock """

        @wraps(func)
        def wrap_lock(self, *args, **kwargs):
            if self.lock.acquire(timeout=self.locktimeout):
                ret = func(self, *args, **kwargs)
                self.lock.release()
                return ret
            else:
                self.send('err', 'Could not acquire lock in ' + func.__name__ + '!')

        return wrap_lock

    @staticmethod
    def wait_for_initialization(func):
        """ This is a decorator for simplified usage of the threadlock """

        @wraps(func)
        def wrap_waiting(self, *args, **kwargs):
            if self.initialized is True:
                func(self, *args, **kwargs)
            else:
                pass

        return wrap_waiting