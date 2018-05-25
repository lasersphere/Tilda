"""
Created on 

@author: simkaufm

Module Description:
    The idea of this module is to be able to connect to a running Triton device somewhere within the network
    and listen to this device. The module will listen to the device prescan, (during) and after scan.

"""
import ast
import logging
import socket
import sys
from copy import deepcopy
from datetime import datetime
from datetime import timedelta

import Pyro4
import mysql.connector as Sql
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

import Application.Config as Cfg
from Driver.TritonListener.DummyTritonDevice import DummyTritonDevice
from Driver.TritonListener.TritonObject import TritonObject
import TildaTools as TiTs


class TritonListener(TritonObject):
    # signal to emit dmm values for live plotting during the pre/post scans.
    # is also used for triton values in TritonListener
    pre_post_meas_data_dict_callback = pyqtSignal(dict)
    # signal to emit data to the pipeLine
    data_to_pipe_sig = pyqtSignal(np.ndarray, dict)

    def __init__(self, name='TritonListener'):
        """
        :parameter name: str, name of this class
        :parameter sql_cfg: dict, dictionary
        """
        try:
            from Driver.TritonListener.TritonConfig import sqlCfg, hmacKey
        except ImportError as e:
            from Driver.TritonListener.TritonDraftConfig import sqlCfg, hmacKey
            logging.error('error, while loading Triton config from Driver.TritonListener.TritonConfig : %s'
                          '\n will use default (Driver.TritonListener.TritonDraftConfig) and dummy mode now!' % e,
                          exc_info=True)
        self.setup_pyro(hmacKey)
        super(TritonListener, self).__init__(name, sqlCfg)

        self.dummy_dev = None

        self.log = {}
        self.back_up_log = {}
        self.logging = False
        self.logged_data = {}
        self.logging_complete = False

        self.last_emit_to_analysis_pipeline_datetime = datetime.now()
        self.time_between_emits_to_pipeline = timedelta(milliseconds=500)
        # limit this to 500 ms in order nto to flush the pipeline with emitted signals

        # variables to store the actual track and scan strings for emitting the live data dict
        self.pre_dur_post_str = 'preScan'
        self.track_name = 'track0'

    def create_dummy_dev(self, name='dummyDev'):
        self.dummy_dev = DummyTritonDevice(name)
        # self.subscribe(str(self.dummy_dev.uri))
        self.dummy_dev.setInterval(1)

    def get_channels_of_dev(self, dev):
        """
        returns the available channels of the dev as a list
        :param dev: str, name of dev
        :return: list, ['ch1', 'ch2' ...]
        """
        ret = ''
        channels = ['calls', 'random']
        logging.debug('getting channels of dev %s' % str(dev))

        try:
            db = Sql.connect(**self.sql_config)
            dbCur = db.cursor()
            logging.info('connecting to db: %s at ip: %s' % (self.sql_config.get('database', 'unknown'),
                                                             self.sql_config.get('host', 'unknown')))
        except Exception as e:
            logging.error('error, TritonObject Could not connect to db, error is: %s' % e)
            db = None
            dbCur = None

        if dbCur is not None:
            dbCur.execute(
                '''SELECT devicetypes.channels FROM devices JOIN devicetypes ON
                    devicetypes.deviceType = devices.deviceType WHERE devices.deviceName = %s''',
                (str(dev),))
            try:
                ret = dbCur.fetchone()
                if ret is None:
                    return ['None']
                channels = ast.literal_eval(ret[0])
            except Exception as e:
                logging.error(
                    'error in converting list of channels %s from dev %s, error message is: %s' % (ret, dev, e),
                    exc_info=True)
            db.close()
        logging.debug('available channels of dev %s are: %s ' % (str(dev), str(channels)))
        return channels

    def get_devs_from_db(self):
        """
        return a dict with all channels and their channels as a list.
        if no db is available return dummy dev
        :return: dict, {dev: ['ch1', 'ch2' ...]}
        """
        devs = {}
        try:
            db = Sql.connect(**self.sql_config)
            dbCur = db.cursor()
            logging.info('connecting to db: %s at ip: %s' % (self.sql_config.get('database', 'unknown'),
                                                             self.sql_config.get('host', 'unknown')))
        except Exception as e:
            logging.error('error, TritonObject Could not connect to db, error is: %s' % e)
            db = None
            dbCur = None
        if dbCur is not None:
            dbCur.execute('''SELECT deviceName FROM devices WHERE uri IS NOT NULL''')
            res = dbCur.fetchall()
            for dev in res:
                devs[dev[0]] = self.get_channels_of_dev(dev[0])
            db.close()
        else:
            logging.warning('no db connection, returning local dummyDev!')
            devs['dummyDev'] = self.get_channels_of_dev('dummyDev')
        return devs

    def get_existing_callbacks_from_main(self):
        """ check wether existing callbacks are still around in the main and then connect to those. """
        if Cfg._main_instance is not None:
            logging.info('TritonListener is connecting to existing callbacks in main')
            callbacks = Cfg._main_instance.gui_live_plot_subscribe()
            self.pre_post_meas_data_dict_callback = callbacks[6]
            self.data_to_pipe_sig = Cfg._main_instance.scan_main.data_to_pipe_sig  # TODO: this works, but is it nice?

    def setup_log(self, log, pre_dur_post_str, track_name):
        """
        setup the log and subscribe to all required devices
        the log is a dict containing the devs which should be logged and
        the channels as a dict with the number of required values:

            {'dummyDev': {'ch1': {'required': 2, 'data': [], 'acquired': 0}, ...}}

        """
        # connect to the callback for live data plotting so the log can be emitted as well
        self.get_existing_callbacks_from_main()
        # note down the track and which part of the scan is done
        self.track_name = track_name
        self.pre_dur_post_str = pre_dur_post_str

        if log is None or log == {}:
            self.log = {}
            self.logging_complete = True  # do not log
            self.back_up_log = deepcopy(self.log)  # store backup log, because i will work on self.log
        else:
            self.log = log
            self.logging_complete = False
            self.back_up_log = deepcopy(self.log)  # store backup log, because i will work on self.log
            self.subscribe_to_devs_in_log()
            if len(list(self._recFrom.keys())):
                self.logging_complete = False
            else:
                # could not resolve names etc. do not log!
                self.logging_complete = True
                self.log = {}
                self.back_up_log = deepcopy(self.log)  # store backup log, because i will work on self.log

    def subscribe_to_devs_in_log(self):
        """ subscribe to all devs in the log if not already subscribed to """
        existing = list(self._recFrom.keys())
        for dev in self.log.keys():
            if dev not in existing:
                if dev != 'dummyDev':
                    self.subscribe(dev)
                else:  # dummyDev is wanted!
                    if self.dummy_dev is None:
                        self.create_dummy_dev()
                    # logging.debug('subscribing to uri: %s' % str(self.dummy_dev.uri))
                    self.subscribe('dummyDev', str(self.dummy_dev.uri))
        existing2 = list(self._recFrom.keys())
        for subscribed_dev in existing2:  # unsubscribe from all devs which are not in the log
            if subscribed_dev not in self.log.keys():
                # logging.debug('unsubscribing: %s' % str(subscribed_dev))
                self.unsubscribe(subscribed_dev)
        logging.info('subscribed triton devices after setup: ' + str(list(self._recFrom.keys())))

    def _receive(self, dev, t, ch, val):
        """
        overwrites the _receive class of the TritonObject.
        Is called by all subscribed devices, when they send a value over pyro.
        :param dev: str, name of dev that is sending
        :param t: str, timestamp of event
        :param ch: str, name of channel that was sending
        :param val: anything, value that was send from the device
        :return:
        """
        if self.logging:
            if dev in self.log.keys():
                if ch in self.log[dev].keys():
                    acq_on_log_start = self.back_up_log[dev][ch]['acquired']
                    if self.log[dev][ch]['required'] + acq_on_log_start > self.log[dev][ch]['acquired'] \
                            or self.log[dev][ch]['required'] < 1 and self.pre_dur_post_str == 'duringScan':
                        # not enough data on this channel yet or continuous acquisition (only allowed during scan)
                        # all data is always stored in the .log until the logging is stopped.
                        self.log[dev][ch]['data'].append(val)
                        self.log[dev][ch]['acquired'] += 1
                        # store data in the existing dict: DOOh not necessary because data is stored in log!
                        triton_live_data_dict = {self.track_name: {'triton': {self.pre_dur_post_str: self.log}}}
                        # now update the 'acquired' number for each channel and dev
                        # for dev, chs in self.triton_live_data_dict[
                        #     self.track_name]['triton'][self.pre_dur_post_str].items():
                        #     for ch_name, ch_data in chs.items():
                        #         ch_data['acquired'] = len(ch_data['data'])
                        if self.pre_dur_post_str == 'duringScan':
                            timedelta_since_laste_send = datetime.now() - self.last_emit_to_analysis_pipeline_datetime
                            if timedelta_since_laste_send >= self.time_between_emits_to_pipeline:
                                # in duringScan emit the received values to the pipe!
                                to_send = deepcopy(triton_live_data_dict)
                                # self.triton_live_data_dict = {}  # reset storage after emit
                                self.data_to_pipe_sig.emit(np.ndarray(0, dtype=np.int32), to_send)
                                self.last_emit_to_analysis_pipeline_datetime = datetime.now()
                        else:  # in pre and postScan emit received value to callback for live data plotting
                            self.pre_post_meas_data_dict_callback.emit(deepcopy(triton_live_data_dict))
                            # self.triton_live_data_dict = {}  # reset storage after emit
            self.check_log_complete()

    def check_log_complete(self):
        """ calls self.stop_log() if all values have ben acquired """
        check_sum = 0
        for dev, dev_log in self.log.items():
            for ch, val in dev_log.items():
                # for periodic data taking 'required' will be negative. Checksum must not be 0
                if val['required'] < 1:
                    if self.pre_dur_post_str == 'duringScan':
                        check_sum += 1
                    else:  # continuous acquisition only makes sense duringScan
                        logging.error('Triton device %s, channel %s: continuous acquisition not allowed in %s.'
                                      % (dev, ch, self.pre_dur_post_str))
                else:
                    acq_on_log_start = self.back_up_log[dev][ch]['acquired']
                    check_sum += max(0, val['required'] - val['acquired'] + acq_on_log_start)
        if check_sum == 0:
            logging.info('TritonListener: logging complete')
            self.logging_complete = True
            # for dev, dev_log in self.log.items():
            #     for ch, val in dev_log.items():
            #         val['acquired'] = len(val['data'])
            logging.debug('TritonListener self.log after completion: %s' % str(self.log))
            self.stop_log()
            return True
        else:
            return False

    def start_log(self):
        """ start logging of the desired channels and devs.
         Be sure to setup the log before hand with self.setup_log """
        self.logging_complete = self.log == {}
        # for dev, dev_dict in self.log.items():
        #     for ch, ch_dict in dev_dict.items():
        #         ch_dict['acquired'] = 0
        logging.debug('log before start: %s' % str(self.log))
        self.logging = True

    def stop_log(self):
        """ stop logging, by setting self.logging to False """
        self.logging = False

    def off(self, stop_dummy_dev=True):
        """ unsubscribe from all devs and stop the dummy device if this was started. """
        self.stop_log()
        self._stop()
        # If there is a dummy_dev stop it, except if we only want to reset the pipeline.
        if self.dummy_dev is not None and stop_dummy_dev:
            self.dummy_dev._stop()
            self.dummy_dev = None

    def setup_pyro(self, hmackey):
        """
          Set Pyro variables
        :param hmackey: bytes, hmackkey, e.g. b'6\x19\n\xad\x909\xda\xea\xb5\xc5]\xbc\xa1m\x863'
        :return:
        """
        Pyro4.config.SERIALIZER = "serpent"
        Pyro4.config.HMAC_KEY = hmackey
        Pyro4.config.HOST = socket.gethostbyname(socket.gethostname())
        # Pyro4.config.SERVERTYPE = 'multiplex'
        Pyro4.config.SERVERTYPE = 'thread'
        sys.excepthook = Pyro4.util.excepthook
        # Pyro4.config.DETAILED_TRACEBACK = True

    def get_receivers(self):
        return list(sorted(self._recFrom.keys()))


if __name__ == '__main__':
    app_log = logging.getLogger()
    # app_log.setLevel(getattr(logging, args.log_level))
    app_log.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    # ch.setFormatter(log_formatter)
    app_log.addHandler(ch)

    app_log.info('****************************** starting ******************************')
    app_log.info('Log level set to DEBUG')

    trit_lis = TritonListener()
    # trit_lis.create_dummy_dev()
    trit_lis.setup_log({'DummyPS': {'current': {'required': 50, 'data': []}}}, 'track0')
    # trit_lis.setup_log({})
    trit_lis.start_log()
    # input('anything to stop')
    # trit_lis.start_log()
    input('anything to stop')
    print(trit_lis.log)
    trit_lis.off()
