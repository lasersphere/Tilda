"""
Created on 

@author: simkaufm

Module Description:
    Dummy Triton device useful for testing.

    copied from Triton Base/DraftDeviceTest.py on 01.03.19 git rev. number: 74e28f9804a8d2a27f2f53aa8c0671cd6dc804e4

    If changes are made within Triton maybe a copy is needed again.
    Required modifications for Tilda are marked with a comment  # changed!
    look for them before overwriting again!
"""

# python imports here
import logging
import random
import sys
import Pyro4
import socket
# other modules here

# own imports here

from Driver.TritonListener.TritonDeviceBase import DeviceBase  # changed!


# rename class here
class DummyTritonDevice(DeviceBase):  # changed!
    '''
    draft device with minimal functionality
    '''

    '''Called when added'''

    def on(self, cfg):
        '''Setting necessary attributes, interval defaults to 0:'''
        # self.setup_pyro()  # changed!

        self.type = 'DummyTritonDevice'
        # self.addCfg(['par'])
        # self.addStg(['par'])

        self.per_calls = 0
        # self.setInterval(1)
        '''Resolving and subscribing to other devices:'''
        # self.dev = self.resolveName('Name')
        # self.subscribe(dev)

    '''Called when removed'''

    def off(self):
        pass

    '''Called regularly, running in separate thread'''

    def periodic(self):
        self.per_calls += 1
        self.send('calls', self.per_calls)
        self.send('random', random.random())
        # self.send('out', self.per_calls)
        # print('periodicCalls', self.per_calls)
        # self.send('', value)
        pass

    '''Called by subscriptors'''

    def receive(self, dev, t, ch, val):
        logging.info('%s rcvd: %s' % (self.name, str((dev, t, ch, val))))

    '''Called when settings are loaded, vals contains setting dictionary'''

    def load(self, vals):
        pass

    '''Send current status on this command'''

    def emit(self):
        pass


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
    # Pyro4.config.HMAC_KEY = hmacKey
    Pyro4.config.HOST = socket.gethostbyname(socket.gethostname())
    # Pyro4.config.SERVERTYPE = 'multiplex'
    Pyro4.config.SERVERTYPE = 'thread'
    sys.excepthook = Pyro4.util.excepthook
    # Pyro4.config.DETAILED_TRACEBACK = True
    dev1 = DummyTritonDevice('dev1', 'local')
    dev2 = DummyTritonDevice('dev2', 'local')
    dev1.setInterval(1)
    dev2.setInterval(1)

    dev1.subscribe(dev2.name, dev2.uri)
    input('anything to stop')
    dev1._stop()
    dev2._stop()