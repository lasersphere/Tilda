"""
Created on 

@author: simkaufm

Module Description:
    copied from Triton 27.02.19 Git Revision number: bd4d7fceb11d6338a6ecd9cdd4ec02069b8353c1
    If changes are made within Triton maybe a copy is needed again.
    Required modifications for Tilda are marked with a comment  # changed!
"""

from functools import wraps
from threading import Thread, Event, Lock
from datetime import datetime
import logging, time

import Pyro4

from Driver.TritonListener.TritonObject import TritonObject  # changed!


class DeviceBase(TritonObject):
    """
    Base Device class for the Triton control system.
    """

    def __init__(self, name):
        '''
        Set up the device and calls self.on() for device specific construction
        '''
        super(DeviceBase, self).__init__()

        self.name = name
        self._thread = None
        self._timer = Event()
        self._interval = 0

        self._cfg = ['_interval']
        self._stg = ['_interval']

        self._sendTo = {}

        self.dbCur_execute("SELECT deviceType, uri, config FROM devices WHERE deviceName=%s", (self.name,))

        db = self.dbCur_fetchone()

        if db[1] != None:
            logging.warning(self.name + ' already exists! Overwriting.')

        self._commitUri(str(self.uri))

        if db[2] is not None:
            cfg = eval(db[2])
        else:
            cfg = None

        self.lock = Lock()
        self.locktimeout = 5

        try:
            self.on(cfg)
        except Exception:
            self.errsend()

    def _stop(self):
        '''
        Call self.off() for device specific destruction and afterwards deinitializes the base object
        '''
        try:
            self.off()
        except Exception:
            self.errsend()

        self._commitUri(None)

        if self._thread is not None:
            self.setInterval(0)
        self.send('out', 'Deleted')

        TritonObject._stop(self)

    """Publishing"""

    def _receive(self, dev, t, ch, val):
        '''Called first on receiving, blank wrapper'''
        if ch == 'out' and val == 'Deleted':
            logging.debug('Deleting from ' + str(dev))
            del self._recFrom[dev]
            self.connectionLost()
        self.receive(dev, t, ch, val)

    def _emit(self):
        '''Called on new subscriber. Emit the concurrent values.'''
        self.send('interval', self._interval)
        self.emit()

    def _addSub(self, uri, ndev):
        '''Add device with uri and name to Subscribers'''
        dev = Pyro4.Proxy(uri)
        self._sendTo[ndev] = dev
        self.send('out', ndev + ' subscribed.')
        self._emit()
        logging.debug(self.name + ': Emitting done in _addSub')

    def _remSub(self, ndev):
        '''Remove device with name from subscribers'''
        if ndev in self._sendTo:
            del self._sendTo[ndev]
        self.send('out', ndev + ' unsubscribed.')

    def checkSub(self, sub):
        '''Check whether sub is subscribed'''
        return self in self._sendTo

    def send(self, ch, val):
        '''Send value on channel, add timestamp and copy to console'''
        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if ch == 'err':
            logging.error(t + ' ' + self.name + ' ' + ch + ": \t" + str(val))
        elif ch == 'out':
            logging.info(t + ' ' + self.name + ' ' + ch + ": \t" + str(val))
        else:
            logging.debug(t + ' ' + self.name + ' ' + ch + ": \t" + str(val))

        for sub in self._sendTo.values():
            sub._receive(self.name, t, ch, val)

    def connectionLost(self):
        pass

    """Saving settings"""

    def saveStg(self, comment=''):
        '''Write settings in settings table'''
        stg = repr({val: getattr(self, val) for val in self._stg})
        # logging.debug('saving settings start:')
        self.dbCur_execute("INSERT INTO settings (device, date, comment, settings) VALUES (%s, %s, %s, %s)",
                           (self.name, datetime.now(), comment, stg))
        # logging.debug('execute happend')
        self.db_commit()
        # logging.debug('commit happend')
        self.send('out', "Saved Setting " + comment + ": " + stg)

    def loadStg(self, stgID):
        '''Load settings with ID from settings table'''
        # logging.debug('saving settings start:')
        self.dbCur_execute("SELECT settings FROM settings WHERE ID=%s", (stgID,))
        # logging.debug('execute happend')
        stg = eval(self.dbCur_fetchone()[0])
        # logging.debug('fetchone happend')

        self.setInterval(stg['_interval'])
        self.send('out', "Loading setting " + str(stgID) + ": " + str(stg))
        self.load(stg)

    def addStg(self, stg):
        '''Used for setting configuration'''
        self._stg.extend(stg)

    """Methods for periodic thread"""

    def _run(self):
        '''The periodic logic'''
        while self._interval > 0.0:
            startTime = time.time()
            self.periodic()
            diff = round(time.time() - startTime, 1)
            logging.debug('processing time: ' + str(diff))
            if diff > self._interval and self._interval != 0:
                self.send('err', 'processing time is bigger than interval! Setting interval to ' + str(diff))
                self.setInterval(diff)
            if self._timer.wait(abs(self._interval - diff)):
                self._timer.clear()
        self._thread = None

    def setInterval(self, t):
        '''Set the interval. Start or stop periodic thread as necessary'''
        self._interval = t
        if self._thread is not None:
            self._timer.set()
        elif self._interval > 0:
            self._thread = Thread(target=self._run)
            self._thread.start()
        self.send("interval", self._interval)

    """Other Stuff"""

    def _commitUri(self, uri):
        '''Write the uri into the device table'''
        # logging.debug('comitting URI to dab')
        self.dbCur_execute("UPDATE devices SET uri=%s WHERE deviceName=%s", (uri, self.name))
        # logging.debug('execute happend')
        self.db_commit()
        # logging.debug('commit happend')

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