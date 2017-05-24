"""
Created on 

@author: simkaufm

Module Description:
    The idea of this module is to be able to connect to a running Triton device somewhere within the network
    and listen to this device. The module will listen to the device prescan, (during) and after scan.

"""
import Pyro4
import socket
import sys
import ast
from copy import deepcopy

from Driver.TritonListener.TritonObject import TritonObject
from Driver.TritonListener.DummyTritonDevice import DummyTritonDevice


class TritonListener(TritonObject):
    def __init__(self, name='TritonListener'):
        """
        :parameter name: str, name of this class
        :parameter sql_cfg: dict, dictionary
        """
        try:
            from Driver.TritonListener.TritonConfig import sqlCfg, hmacKey
        except ImportError as e:
            from Driver.TritonListener.TritonDraftConfig import sqlCfg, hmacKey
            print('error, while loading Triton config from Driver.TritonListener.TritonConfig : %s' % e)
            print('will use default (Driver.TritonListener.TritonDraftConfig) and dummy mode now!')
        self.setup_pyro(hmacKey)
        super(TritonListener, self).__init__(name, sqlCfg)

        self.dummy_dev = None

        self.log = {}
        self.back_up_log = {}
        self.logging = False
        self.logged_data = {}
        self.logging_complete = False

        # TODO remove after debugging is done!
        self.create_dummy_dev()

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

        if self.dbCur is not None:
            self.dbCur.execute(
                '''SELECT devicetypes.channels FROM devices JOIN devicetypes ON
                    devicetypes.deviceType = devices.deviceType WHERE devices.deviceName = %s''',
                (str(dev),))
            try:
                ret = self.dbCur.fetchone()
                if ret is None:
                    return ['None']
                channels = ast.literal_eval(ret[0])
            except Exception as e:
                print('error in converting list of channels %s from dev %s, error message is: %s' % (ret, dev, e))
        return channels

    def get_devs_from_db(self):
        """
        return a dict with all channels and their channels as a list.
        :return: dict, {dev: ['ch1', 'ch2' ...]}
        """
        devs = {}
        if self.dbCur is not None:
            self.dbCur.execute('''SELECT deviceName FROM devices WHERE uri IS NOT NULL''')
            res = self.dbCur.fetchall()
            for dev in res:
                devs[dev] = self.get_channels_of_dev(dev)
        else:
            devs['dummyDev'] = self.get_channels_of_dev('dummyDev')
        return devs

    def setup_log(self, log):
        """
        setup the log and subscribe to all required devices
        the log is a dict containing the devs which should be logged and
        the channels as a dict with the number of required values:

            {'dummyDev': {'ch1': {'required': 2, 'data': [], 'acquired': 0}, ...}}

        """
        self.log = log
        self.logging_complete = False
        self.back_up_log = deepcopy(self.log)  # store backup log, because i will work on self.log
        self.subscribe_to_devs_in_log()

    def subscribe_to_devs_in_log(self):
        """ subscirbe to all devs in the log if not already subscribed to """
        existing = list(self._recFrom.keys())
        for dev in self.log.keys():
            if dev not in existing:
                if dev != 'dummyDev':
                    self.subscribe(dev)
                else:  # dummyDev is wanted!
                    if self.dummy_dev is None:
                        self.create_dummy_dev()
                    # print('subscribing to uri:',  self.dummy_dev.uri)
                    self.subscribe('dummyDev', str(self.dummy_dev.uri))
        existing2 = list(self._recFrom.keys())
        for subscribed_dev in existing2:  # unsubscribe from all devs which are not in the log
            if subscribed_dev not in self.log.keys():
                # print('unsubscribing: ', subscribed_dev)
                self.unsubscribe(subscribed_dev)
        print('subscribed triton devices after setup: ', list(self._recFrom.keys()))

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
                    if self.log[dev][ch]['required'] > self.log[dev][ch]['acquired']:
                        # not enough data on this channel yet
                        self.log[dev][ch]['data'].append(val)
                        self.log[dev][ch]['acquired'] += 1
                        # print(dev, t, ch, val)
            self.check_log_complete()

    def check_log_complete(self):
        """ return True if all values have ben acquired """
        check_sum = 0
        for dev, dev_log in self.log.items():
            for ch, val in dev_log.items():
                check_sum += max(0, val['required'] - val['acquired'])
        if check_sum == 0:
            print('logging complete')
            self.logging_complete = True
            for dev, dev_log in self.log.items():
                for ch, val in dev_log.items():
                    val['acquired'] = len(val['data'])
            print(self.log)
            self.stop_log()

    def start_log(self):
        """ start logging of the desired channels and devs.
         Be sure to setup the log before hand with self.setup_log """
        self.logging_complete = self.log == {}
        for dev, dev_dict in self.log.items():
            for ch, ch_dict in dev_dict.items():
                ch_dict['acquired'] = 0
        print('log before start:', self.log)
        self.logging = True

    def stop_log(self):
        """ stop logging, by setting self.logging to False """
        # print(self.log)
        self.logging = False

    def off(self):
        """ unsubscribe from all devs and stop the dummy device if this was started. """
        self.stop_log()
        self._stop()
        if self.dummy_dev is not None:
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

if __name__=='__main__':
    trit_lis = TritonListener()
    # trit_lis.create_dummy_dev()
    # trit_lis.setup_log({'dummyDev': {'calls': {'required': 2, 'data': []}, 'random': {'required': 5, 'data': []}}})
    trit_lis.setup_log({})
    trit_lis.start_log()
    input('anything to stop')
    trit_lis.start_log()
    input('anything to stop')
    trit_lis.off()
