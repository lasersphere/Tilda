"""
Created on 01.03.2019

@author: simkaufm

Module Description:
        This is a Triton device which will be able to control Scan Devices.

        Details see in class below
"""

import ast
import time
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
from Driver.TritonListener.DummyTritonScanDevice import DummyScanDevice
from Driver.TritonListener.TritonDeviceBase import DeviceBase
import TildaTools as TiTs


class TritonScanDevControl(DeviceBase, QObject):
    """
    This is a Triton device which will be able to control Scan Devices.

    It will be able to send commands to a designated scan device in order to tell this device
    the scan parameters and when to set the next step.
    In return, the scan device will tell the TritonScanDevControl, what scan parameters can be set
    and when it has set a new step.
    To control the scan device, the TritonScanDevControl will subscribe to the scan device
    and store the pyro-remote method in self.scan_dev.
    Therefore functions of the scan_dev can be accessed from the TritonScanDevControl directly,
    like "self.scan_dev.set_next_step()".
    The scan_dev, can therefore retunr values on these calls directly,
    but mostly it will use tha asynchrone self.send(ch, val) call,
    which will fire the self.receive in the TritonScanDevControl and there actions can be taken.
    This is a well established mechanism from within Triton at the moment.
    -> maybe as own device?! There is no need that one device can do both!?
    """
    # whenever the currently connected scan device is sending new values, this will be emitted.
    # dict, see in self.receive() -> 'devPars'
    scan_dev_sends_new_settings_pyqtsig = pyqtSignal(dict)
    # when the scan device has set the scan parameters it will emit a dictionary with the corresponding values
    # dict, see self.receive() -> 'scanParsSet'
    scan_dev_has_setup_these_pars_pyqtsig = pyqtSignal(dict)
    # when the scan device has set a new step, it will emit a dictionary
    # dict, see self.receive() -> 'scanProgress'
    scan_dev_has_set_a_new_step_pyqtsig = pyqtSignal(dict)
    # signal to emit data to the pipeLine
    # will be overwritten if main exists in self.get_existing_callbacks_from_main()
    data_to_pipe_sig = pyqtSignal(np.ndarray, dict)

    ''' Device standard functions, do not delete! on, off, periodic, receive, load, emit'''
    '''Called when added'''

    def on(self, cfg):
        '''Setting necessary attributes, interval defaults to 0:'''

        self.type = 'TritonScanDevControl'

        self.dummy_scan_dev = None  # will be initialized if db is set to local
        self.dummy_scan_dev_name = 'DummyScanDev'
        if self.db == 'local':
            self.create_dummy_scan_dev()

        self.scan_dev = None
        self.scan_dev_name = ''

        self.scan_dev_settings = {}
        self.scan_pars_set_from_dev = {}

        self._request_next_step_from_scan_dev = False

        '''Resolving and subscribing to other devices:'''
        # self.dev = self.resolveName('Name')
        # self.subscribe(dev)

    '''Called when removed'''

    def off(self):
        """ called on _stop() """
        self.unsubscribe_from_scan_device()
        if self.dummy_scan_dev is not None:
            self.dummy_scan_dev._stop()
            self.dummy_scan_dev = None

    '''Called regularly, running in separate thread'''

    def periodic(self):
        if self._request_next_step_from_scan_dev:
            if self.scan_dev is not None:
                answ = self.scan_dev.setup_next_step()
                if answ:
                    # step is about to be set by the scan dev
                    self._request_next_step_from_scan_dev = False

    '''Called by subscriptors'''

    def receive(self, dev, t, ch, val):
        """
        called from subscriptors. Default calls from scanning devices:

        cH: scanParsSet val: dict, {
            'unit': self.start_step_units.name,
            'start': self.sc_start,
            'stop': self.sc_stop,
            'stepSize': self.sc_stepsize,
            'stepNums': self.sc_num_of_steps,
            'valsArrrayOneScan': self.sc_one_scan_vals}
        ch: scanProgress, val: dict, {
            'curStep': self.sc_l_cur_step,
            'curScan': self.sc_l_cur_scan,
            'curStepVal': self.sc_one_scan_vals[self.sc_l_cur_step],
            'scanStatus': self.scan_status}
        ch: devPars, val: dict, {
            'step_size_min_max': (self.dev_min_step_size, self.dev_max_step_size),
            'set_val_min_max': (self.dev_min_val, self.dev_max_val),
            'unit': self.start_step_units}

        :param dev: str, dev that has send the following
        :param t: str, timestamp
        :param ch: str, channel of the device
        :param val: anything that is serialisable by Pyro4, e.g. dict, list, etc.
        :return:
        """
        logging.info('%s rcvd: %s' % (self.name, str((dev, t, ch, val))))
        if dev == self.scan_dev_name:  # usually anyhow only subscirbed to scan dev
            if ch == 'devPars':
                self.store_dev_pars(val)
            elif ch == 'scanProgress':
                self.scan_dev_has_set_next_step(val)
            elif ch == 'scanParsSet':
                self.store_scan_pars(val)

    '''Called when settings are loaded, vals contains setting dictionary'''

    def load(self, vals):
        pass

    '''Send current status on this command'''

    def emit(self):
        pass

    ''' comunication with scan dev '''
    ''' find available devs '''

    def get_devs_from_db(self):
        """
        return a list with the names of all currently available devices in the db.
        if db is set to local return the name of the DummyScanDev
        """
        devs = []

        if self.db != 'local':
            self.dbCur_execute('''SELECT deviceName FROM devices WHERE uri IS NOT NULL''', None)
            res = self.dbCur_fetchall()
            if res is not None:
                devs = res[0]
        else:
            logging.warning('no db connection, returning local DummyScanDev!')
            devs = [self.dummy_scan_dev_name]
        return devs

    def create_dummy_scan_dev(self):
        """ cretae a dummy scan device for testing """
        self.dummy_scan_dev = DummyScanDevice(self.dummy_scan_dev_name, self.sql_conf)

    def subscribe_to_scan_dev(self, scan_dev_name):
        """ subscribe to a scan dev by it's name in the Triton database """
        if self.db != 'local':
            self.scan_dev = self.subscribe(scan_dev_name)
        else:
            self.scan_dev = self.subscribe(self.dummy_scan_dev_name, self.dummy_scan_dev.uri)
        self.scan_dev_name = self.scan_dev.getName()

    def unsubscribe_from_scan_device(self):
        """ unsubscribe from the current scan device """
        self.unsubscribe(self.scan_dev_name)
        self.scan_dev = None
        self.scan_dev_name = ''

    def get_scan_dev_settings(self):
        """
        request the settings of the scan dev. This will emit its values and here it
        will be handled in self.receive()
        :return:
        """
        self.scan_dev.emit_device_parameters()

    def store_dev_pars(self, dev_pars_dict):
        """
        in receive, the scan deveice has published new settings
        -> send them upstream to gui etc.
        :param dev_pars_dict: dict, see in self.receive() -> 'devPars'
        :return:
        """
        self.scan_dev_settings = dev_pars_dict
        self.scan_dev_sends_new_settings_pyqtsig.emit(self.scan_dev_settings)

    def setup_scan_in_scan_dev(self, start, stepsize, num_of_steps, num_of_scans, invert_in_odd_scans):
        """
        set these values in the scan device.
        Once this has completed setting this up, it will emit the scan pars
        and this can be handled her in the self.receive()
        :param start: float, starting point in units of self.start_step_units
        :param stepsize: float, step size in units of self.start_step_units
        :param num_of_steps: int, number of steps per scan
        :param num_of_scans: int, number of scans
        :param invert_in_odd_scans: bool, True -> invert after step complete
        :return: None
        """
        self._request_next_step_from_scan_dev = False
        self.scan_dev.setup_scan(start, stepsize, num_of_steps, num_of_scans, invert_in_odd_scans)

    def store_scan_pars(self, set_scan_pars_from_dev):
        """
        Once the device has set the scan settings it will return the coerced once.
        This will be send from here to upstream via a pyqtsignal!
        :param set_scan_pars_from_dev: dict, see self.receive() -> 'scanParsSet'
        """
        self.scan_pars_set_from_dev = set_scan_pars_from_dev
        self.scan_dev_has_setup_these_pars_pyqtsig = self.scan_pars_set_from_dev

    def scan_dev_has_set_next_step(self, scan_progress_dict):
        """
        wrapper for pyqtsignal and to store the scan progress
        :param scan_progress_dict: dict, see self.receive() -> 'scanProgress'
        """
        self.scan_prog_dict = scan_progress_dict
        self.scan_dev_has_set_a_new_step_pyqtsig.emit(self.scan_prog_dict)
        dict_to_pipe = {'scanDev': {'name': self.scan_dev_name, 'scanProgress': self.scan_prog_dict}}
        self.data_to_pipe_sig.emit(np.zeros(0, dtype=np.int32), dict_to_pipe)

    def request_next_step(self):
        """
        request the next step from the scan device.
        Even if the device is currently busy it will keep requesting until device answers.
        If meanwhile the function is called yet again, this call will get lost
            -> next step is called too often from upstream.
            -> here a warning is emitted but one has to make sure not to let this happen.
            -> always only call for the next step when the last one has been ensured to be set

        Might be called from pipeline by pyqtsignal or so.
        This will also start the scan after it has ben setup properly.

        start the periodic call and if the scan device is busy,
        keep requesting it to set the next step until it is free.
        :return: bool, True for success, False for failure.
        """
        if self._request_next_step_from_scan_dev:
            # will happen if the scan device still is setting the step and repeated calls for the step are coming in.
            # this must be avoided!
            logging.warning('Warning, %s was told to set the next step but'
                            ' it is still busy setting the last step, this call is ignored!' % self.name)
            return False
        else:
            self._request_next_step_from_scan_dev = True
            if self._interval <= 1.0:
                self.setInterval(0.1)
            return True

    def abort_scan(self):
        """
        abort the scan.
        """
        if self.scan_dev is not None:
            self.scan_dev.abort_scan()


    ''' pyqtsignal related '''
    def get_existing_callbacks_from_main(self):
        """ check wether existing callbacks are still around in the main and then connect to those. """
        if Cfg._main_instance is not None:
            logging.info('%s is connecting to existing callbacks in main' % self.name)
            #TODO still adapt this:
            # callbacks = Cfg._main_instance.gui_live_plot_subscribe()
            # self.pre_post_meas_data_dict_callback = callbacks[6]
            self.data_to_pipe_sig = Cfg._main_instance.scan_main.data_to_pipe_sig  # use this signal!


# Testing:
if __name__ == '__main__':
    app_log = logging.getLogger()
    # app_log.setLevel(getattr(logging, args.log_level))
    app_log.setLevel(logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    # ch.setFormatter(log_formatter)
    app_log.addHandler(ch)

    app_log.info('****************************** starting ******************************')
    app_log.info('Log level set to DEBUG')

    from Driver.TritonListener.TritonDraftConfig import hmacKey

    # Set Pyro variables
    Pyro4.config.SERIALIZER = "serpent"
    Pyro4.config.HMAC_KEY = hmacKey
    Pyro4.config.HOST = socket.gethostbyname(socket.gethostname())
    # Pyro4.config.SERVERTYPE = 'multiplex'
    Pyro4.config.SERVERTYPE = 'thread'
    sys.excepthook = Pyro4.util.excepthook


    class recever_class_dummy(QObject):
        def __init__(self, sig):
            super(recever_class_dummy, self).__init__()
            sig.connect(self.printer)

        def printer(self, nparr, dictio):
            print(nparr, dictio)
            logging.info(str(dictio))

    sc_ctrl = TritonScanDevControl('ScanControlTestUnit')
    rcver = recever_class_dummy(sc_ctrl.scan_dev_has_set_a_new_step_pyqtsig)

    devs_db = sc_ctrl.get_devs_from_db()
    print('available devs in db: ', devs_db)
    sc_ctrl.subscribe_to_scan_dev('')
    sc_ctrl.setup_scan_in_scan_dev(0, 10, 11, 50, False)
    # calling next step too often will cause failures:
    # normal and must be avoid upstream by only requesting a
    # new step when the last one was confirmed to be set
    i = 0
    while i < 21:
        i += 1
        sc_ctrl.request_next_step()
        time.sleep(2.5)
    input('anything to stop')
    sc_ctrl._stop()
