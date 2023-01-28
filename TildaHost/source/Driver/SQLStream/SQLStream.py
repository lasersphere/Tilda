"""
Created on 2022-12-12

@author: Patrick Mueller

Module Description: Receive/send data to/from a MySQL or MariaDB database.
Specify server in SQLConfig.py.
"""

import time
from datetime import datetime, timedelta
import logging
from copy import deepcopy
import numpy as np
import mysql.connector as sql
from functools import wraps
from threading import Thread
from PyQt5.QtCore import QObject, pyqtSignal

import Application.Config as Cfg
from Driver.SQLStream.SQLConfig import SQL_CFG, EXCLUDE_CHANNELS


class SQLStream(QObject):
    # signal to emit dmm values for live plotting during the pre/post scans.
    # is also used for sql values in SQLStream.
    pre_post_meas_data_dict_callback = pyqtSignal(dict)
    # signal to emit data to the pipeLine
    data_to_pipe_sig = pyqtSignal(np.ndarray, dict)

    def __init__(self, sql_cfg=SQL_CFG):
        super().__init__()
        
        self.logger = logging.getLogger('SQLLogger')

        self.sql_cfg = sql_cfg
        self.db = None  # can be set to 'local' or '' or {} for testing without any database, see below
        self.db_cur = None
        self.db_connect()
        
        self._thread = None
        self._interval = 0
        self.adjust_interval = True

        self.track_name = ''
        self.pre_dur_post_str = ''
        self.logging_enabled = False
        self.logging_complete = True
        self.log = {}
        self.back_up_log = deepcopy(self.log)
        self.ch_id = {}
        self.ch_time = {}

        self.last_emit_to_analysis_pipeline_datetime = datetime.now()
        self.time_between_emits_to_pipeline = timedelta(milliseconds=500)
        self.last_received_times = [datetime.now(), datetime.now()]
        self.rcvd_time_deltas_total_s = []
        self.mean_rcvd_time_delta = timedelta(seconds=60).total_seconds()

    """ Encapsulate db functionalities to handle connectivity problems and allow to operate without a db. """

    def db_connect(self):
        if isinstance(self.sql_cfg, dict):
            try:
                self.db = sql.connect(**self.sql_cfg)
                self.db_cur = self.db.cursor()
            except Exception as e:
                self.logger.error('could not connect to database %s, error is: %s' % (
                    self.sql_cfg.get('database', 'unknown'), e))
                self.db = 'local'
                self.db_cur = None
        elif isinstance(self.sql_cfg, str) or self.sql_cfg == {}:
            # if the sql_cfg is a string or an empty dict, it will be assumed that no db connection is wanted
            # -> self.db is set to 'local' and all following db calls will be ignored.
            self.db = 'local'  # for testing without db
            self.db_cur = None

    def db_execute(self, var1, var2):
        if self.db != 'local':
            try:
                self.db_cur.execute(var1, var2)
            except Exception as e:
                self.logger.error('could not execute sql command to database %s, error is: %s\nTrying to reconnect.' % (
                    self.sql_cfg.get('database', 'unknown'), e))
                self.db_connect()
                self.db_cur.execute(var1, var2)

    def db_commit(self):
        if self.db != 'local':
            try:
                self.db.commit()
            except Exception as e:
                self.logger.error('could not commit changes to database %s, error is: %s\nTrying to reconnect.' % (
                    self.sql_cfg.get('database', 'unknown'), e))
                self.db_connect()
                self.db.commit()

    def db_fetchall(self, local_ret_val=None):
        if self.db != 'local':
            try:
                var = self.db_cur.fetchall()
                return var
            except Exception as e:
                self.logger.error('could not fetchall from database %s, error is: %s\nTrying to reconnect.' % (
                    self.sql_cfg.get('database', 'unknown'), e))
                self.db_connect()
                return self.db.fetchall()
        else:
            if local_ret_val is None:
                local_ret_val = [(None,)]
            return local_ret_val

    def db_fetchone(self, local_ret_val=None):
        """
        fetch only one result from previous query result
        :param local_ret_val: value, default None can be used for testing without db
        :return:
        """
        if self.db != 'local':
            try:
                var = self.db_cur.fetchone()
                return var
            except Exception as e:
                self.logger.error('could not fetchone from database %s, error is: %s\nTrying to reconnect.' % (
                    self.sql_cfg.get('database', 'unknown'), e))
                self.db_connect()
                return self.db.fetchone()
        else:
            return local_ret_val

    def db_close(self):
        if self.db != 'local':
            try:
                self.db.close()
            except Exception as e:
                self.logger.error('could not close database %s, error is: %s\nTrying to reconnect.' % (
                    self.sql_cfg.get('database', 'unknown'), e))
                self.db_connect()
                self.db.close()

    def get_channels_from_db(self):
        """
        return a list of all available channels.
        :return: dict, {dev: ['ch1', 'ch2' ...]}
        """
        channels = []
        if self.db != 'local':
            self.db_execute('SHOW TABLES', None)
            # self.db_execute('SELECT deviceName FROM devices WHERE uri IS NOT NULL', None)
            tables = self.db_fetchall()
            for t in tables:
                self.db_execute('SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE'
                                ' TABLE_SCHEMA = %s AND TABLE_NAME = %s', (self.sql_cfg['database'], t[0]))
                sub_channels = ['{}.{}'.format(t[0], c[0]) for c in self.db_fetchall() if c[0] not in EXCLUDE_CHANNELS]
                channels += sub_channels
        else:
            logging.warning('No db connection.')
        return channels
    
    """ Implementation of the periodic readout. """

    def _run(self):
        """ The periodic logic """
        self.ch_id = {ch: -1 for ch in self.log.keys()}  # Reset ID before the first data acquisition.
        self.ch_time = {ch: time.time() for ch in self.log.keys()}  # Reset time before the first data acquisition.
        while self._interval > 0:
            t0 = time.time()
            self._acquire()  # Process new data.
            dt = time.time() - t0
            if dt < self._interval:
                time.sleep(self._interval - dt)
            elif dt > self._interval > 0 and self.adjust_interval:
                self.logger.debug('processing time is bigger than interval! Setting interval to {}.'.format(dt))
                self.set_interval(dt)

    def _acquire(self):
        """ Called in self._run(), acquires and processes the data from the db. """
        if self.logging_enabled:
            for ch in self.log.keys():
                acq_on_log_start = self.back_up_log[ch]['acquired']
                if self.log[ch]['required'] + acq_on_log_start > self.log[ch]['acquired'] \
                        or (self.log[ch]['required'] == -1 and self.pre_dur_post_str == 'duringScan'):

                    # --- fetch ---
                    # data = [np.random.random()]
                    t, c = ch.split('.')
                    self.db_execute('SELECT ID, unix_time, {} FROM {} WHERE ID > {} AND unix_time > {}'
                                    .format(c, t, self.ch_id[ch], self.ch_time[ch]), None)
                    data = self.db_fetchall()
                    self.db_commit()
                    if data:
                        self.ch_id[ch] = data[-1][0]
                        self.ch_time[ch] = data[-1][1]
                        data = [d[2] for d in data if d is not None]
                        # --- fetch ---

                        self.log[ch]['data'] += data
                        self.log[ch]['acquired'] += len(data)

                        if self.log[ch]['required'] > 0:
                            dif = self.log[ch]['acquired'] - acq_on_log_start - self.log[ch]['required']
                            if dif > 0:
                                self.log[ch]['data'] = self.log[ch]['data'][:-dif]
                                self.log[ch]['acquired'] = self.log[ch]['required'] + acq_on_log_start

            sql_live_data_dict = {self.track_name: {'sql': {self.pre_dur_post_str: self.log}}}
            # now update the 'acquired' number for each channel and dev
            # for dev, chs in self.triton_live_data_dict[
            #     self.track_name]['triton'][self.pre_dur_post_str].items():
            #     for ch_name, ch_data in chs.items():
            #         ch_data['acquired'] = len(ch_data['data'])
            if self.pre_dur_post_str == 'duringScan':
                self.last_received_times[0] = deepcopy(self.last_received_times[1])
                self.last_received_times[1] = datetime.now()
                if self.last_received_times[0] is not None:
                    self.rcvd_time_deltas_total_s.append(
                        (self.last_received_times[1] - self.last_received_times[0]).total_seconds())
                    self.mean_rcvd_time_delta = np.mean(self.rcvd_time_deltas_total_s)
                timedelta_since_last_send = datetime.now() - self.last_emit_to_analysis_pipeline_datetime
                if timedelta_since_last_send >= self.time_between_emits_to_pipeline:
                    # in duringScan emit the received values to the pipe!
                    to_send = deepcopy(sql_live_data_dict)
                    self.data_to_pipe_sig.emit(np.ndarray(0, dtype=np.int32), to_send)
                    self.last_emit_to_analysis_pipeline_datetime = datetime.now()

            else:  # in pre and postScan emit received value to callback for live data plotting
                self.pre_post_meas_data_dict_callback.emit(deepcopy(sql_live_data_dict))
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
            self.ch_id = {}
            self.ch_time = {}
            self.logging_complete = True  # do not log
            self.back_up_log = deepcopy(self.log)  # store backup log, because work is done in self.log
        else:
            self.log = log
            self.ch_id = {ch: -1 for ch in self.log.keys()}
            self.ch_time = {ch: time.time() for ch in self.log.keys()}
            self.logging_complete = False
            self.back_up_log = deepcopy(self.log)  # store backup log, because work is done in self.log
            if self.log:
                self.logging_complete = False
            else:
                # could not resolve names etc. do not log!
                self.logging_complete = True
                self.log = {}
                self.ch_id = {}
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
        self.last_received_times = [None, None]
        self.rcvd_time_deltas_total_s = []
        self.mean_rcvd_time_delta = timedelta(seconds=60)
        self.set_interval(0.5)
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


if __name__ == '__main__':
    sql_stream = SQLStream()
    log = {'preScan': {'hv_34401a:hv_readout': {'required': 5, 'acquired': 0, 'data': []},
                       'wm:wm_readout': {'required': 16, 'acquired': 0, 'data': []}},
           'duringScan': {'hv_34401a:hv_readout': {'required': -1, 'acquired': 0, 'data': []},
                          'wm:wm_readout': {'required': 10, 'acquired': 0, 'data': []}},
           'postScan': {}}
    sql_stream.setup_log(log['duringScan'], 'duringScan', 'track0')
    sql_stream.logging_enabled = True
    print(sql_stream.log)
    sql_stream._acquire()
    print(sql_stream.log)
