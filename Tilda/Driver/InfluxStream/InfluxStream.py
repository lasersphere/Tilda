"""
Created on 2023-03

@author: Tim Lellinger

Module Description: Receive data from an InfluxDB database
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
from Tilda.Application.Importer import InfluxConfig


from influxdb import InfluxDBClient
import requests
requests.packages.urllib3.disable_warnings()#this disables the unverified SSL warning


class InfluxStream(QObject):
    # signal to emit dmm values for live plotting during the pre-/postScan.
    # is also used for sql values in SQLStream.
    pre_post_meas_data_dict_callback = pyqtSignal(dict)
    # signal to emit data to the pipeLine
    data_to_pipe_sig = pyqtSignal(np.ndarray, dict)

    def __init__(self, infl_cfg=InfluxConfig.Influx_CFG):
        super().__init__()

        self.logger = logging.getLogger('InfluxLogger')

        self.infl_cfg = infl_cfg
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


        self.ch_time = {}
        self.paused = False

        self.interval = 0.5
        self.last_emit_to_analysis_pipeline_datetime = datetime.now()

    """ Encapsulate db functionalities to handle connectivity problems and allow to operate without a db. """

    def db_query(self, strrequest):
        dbClient = InfluxDBClient(host=self.infl_cfg['host'],
                                  port=self.infl_cfg['port'],
                                  username=self.infl_cfg['username'],
                                  password=self.infl_cfg['password'],
                                  database=self.infl_cfg['database'],
                                  ssl=True,
                                  # verify_ssl=False,
                                  )

        result = dbClient.query(strrequest,epoch="ns")
        dbClient.close()
        return result


    def get_channels_from_db(self):
        """
        return a list of all available channels.
        :return: dict, {dev: ['ch1', 'ch2' ...]}
        """
        if self.infl_cfg['useinflux']:
            result = self.db_query('SHOW FIELD KEYS ON "'+self.infl_cfg["database"]+'"')
            channelsafields = []
            for series in result.raw["series"]:
                chname = series["name"]
                fields = series["values"]
                for field in fields:
                    if field[1] == 'float':
                        channelsafields.append(self.makestringXMLcompatible(chname + ":" + field[0]))
            return channelsafields

        else:
            return [self.makestringXMLcompatible("nodbdev:value")]

    """ Implementation of the periodic readout. """

    def makestringXMLcompatible(self,inp):

        return inp.replace("/","...").replace(":",".")


    def undoXMLcompatible(self,inp):
        return inp.replace("...","/").replace(".",":")

    def _run(self):
        """ The periodic logic """
        self.ch_time = {ch: int(time.time_ns()) for ch in self.log.keys()}  # Reset time before the first data acquisition.
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
                    self.ch_time = {ch: time.time_ns() for ch in self.ch_time.keys()}
                for ch in self.log.keys():
                    acq_on_log_start = self.back_up_log[ch]['acquired']
                    lastaquired = self.ch_time[ch]

                    data = []
                    if self.infl_cfg["useinflux"]:
                        measname, field = self.undoXMLcompatible(ch).split(":")
                        query = 'SELECT "'+field+'" FROM "'+measname+'" WHERE time > '+str(int(lastaquired))
                        result = self.db_query(query)
                        if len(result.raw["series"]) > 0:
                            data = result.raw["series"][0]["values"]
                    else:
                        data = [[time.time_ns(), 0.12345]] #dummy mode

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
                sql_live_data_dict = {self.track_name: {'sql': {self.pre_dur_post_str: self.log}}}
                if self.pre_dur_post_str == 'duringScan':
                    to_send = deepcopy(sql_live_data_dict)
                    self.data_to_pipe_sig.emit(np.ndarray(0, dtype=np.int32), to_send)
                else:  # in pre and postScan emit received value to callback for live data plotting
                    self.pre_post_meas_data_dict_callback.emit(deepcopy(sql_live_data_dict))
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

            if pre_dur_post_str == "duringScan":
                self.ch_time = {ch: time.time_ns() for ch in self.log.keys()}
            else:#for pre/post scan it is allowed to specify a time tolerance
                self.ch_time = {ch: int(time.time_ns()-self.infl_cfg['notolderthan_s']*1E9) for ch in self.log.keys()}

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