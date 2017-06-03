"""
Created on 

@author: simkaufm

Module Description:
    Dummy Triton device which can be setup to send values via Pyro4
    Therefore no database is required. One can connenct to it via its Pyro4 - uri
"""
import logging
import random
import time
from datetime import datetime
from functools import wraps
from threading import Thread, Event, Lock

import Pyro4

from Driver.TritonListener.TritonObject import TritonObject


class DummyTritonDevice(TritonObject):
    """
    Base Device class for the Triton control system.
    """

    def __init__(self, name):
        '''
        Set up the device and calls self.on() for device specific construction
        '''
        super(DummyTritonDevice, self).__init__()

        self.name = name
        self._thread = None
        self._timer = Event()
        self._interval = 0
        self.i = 0

        self._cfg = ['_interval']
        self._stg = ['_interval']

        self._sendTo = {}

        self._commitUri(str(self.uri))

        self.lock = Lock()
        self.locktimeout = 5


    def _stop(self):
        """
        Call self.off() for device specific destruction and afterwards deinitializes the base object
        """

        # self._commitUri(None)

        if self._thread is not None:
            self.setInterval(0)
        self.send('out', 'Deleted')

        TritonObject._stop(self)

    """Publishing"""

    def _receive(self, dev, t, ch, val):
        """Called first on receiving, blank wrapper"""
        if ch == 'out' and val == 'Deleted':
            logging.debug('Deleting from ' + str(dev))
            del self._recFrom[dev]
            self.connectionLost()
        self.receive(dev, t, ch, val)

    def receive(self, dev, t, ch, val):
        print(self.name, dev, t, ch, val)
        # pass

    def _emit(self):
        """Called on new subscriber. Emit the concurrent values."""
        self.send('interval', self._interval)
        # self.emit()

    def _addSub(self, uri, ndev):
        """Add device with uri and name to Subscribers"""
        dev = Pyro4.Proxy(uri)
        self._sendTo[ndev] = dev
        self.send('out', ndev + ' subscribed.')
        self._emit()
        logging.debug(self.name + ': Emitting done in _addSub')

    def _remSub(self, ndev):
        """Remove device with name from subscribers"""
        if ndev in self._sendTo:
            del self._sendTo[ndev]
        self.send('out', ndev + ' unsubscribed.')

    def checkSub(self, sub):
        """Check whether sub is subscribed"""
        return self in self._sendTo

    def send(self, ch, val):
        """Send value on channel, add timestamp and copy to console"""
        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if ch == 'err':
            logging.error(t + ' ' + self.name + ' ' + ch + ": \t" + str(val))
        elif ch == 'out':
            logging.info(t + ' ' + self.name + ' ' + ch + ": \t" + str(val))
        # else:
        #     logging.debug(t + ' ' + self.name + ' ' + ch + ": \t" + str(val))

        for sub in self._sendTo.values():
            sub._receive(self.name, t, ch, val)

    def connectionLost(self):
        pass

    """Methods for periodic thread"""

    def _run(self):
        """The periodic logic"""
        while self._interval > 0.0:
            startTime = time.time()
            self.periodic()
            diff = round(time.time() - startTime, 1)
            # logging.debug('processing time: ' + str(diff))
            if diff > self._interval and self._interval != 0:
                self.send('err', 'processing time is bigger than interval! Setting interval to ' + str(diff))
                self.setInterval(diff)
            if self._timer.wait(abs(self._interval - diff)):
                self._timer.clear()
        self._thread = None

    def periodic(self):
        # print('periodic called')
        self.send('calls', self.i)
        self.i += 1
        self.send('random', random.random())

    def setInterval(self, t):
        """Set the interval. Start or stop periodic thread as necessary"""
        self._interval = t
        if self._thread is not None:
            self._timer.set()
        elif self._interval > 0:
            self._thread = Thread(target=self._run)
            self._thread.start()
        self.send("interval", self._interval)

    """Other Stuff"""

    def _commitUri(self, uri):
        """Write the uri into the device table"""
        print(uri)

    @staticmethod
    def locked(func):
        """This is a decorator for simplified usage of the threadlock"""

        @wraps(func)
        def wrap_lock(self, *args, **kwargs):
            if self.lock.acquire(timeout=self.locktimeout):
                ret = func(self, *args, **kwargs)
                self.lock.release()
                return ret
            else:
                self.send('err', 'Could not acquire lock in ' + func.__name__ + '!')

        return wrap_lock
