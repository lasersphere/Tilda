"""
Created on 

@author: simkaufm

Module Description:
    copied from Triton 26.03.2020 for new backend

    If changes are made within Triton maybe a copy is needed again.
    Required modifications for Tilda are marked with a comment  # changed!
"""
from functools import wraps
from threading import Thread, Event, Lock
from datetime import datetime
import time

from Driver.TritonListener.TritonObject import TritonObject
from Driver.TritonListener.TritonConfig import sqlCfg as sqlConf


class DeviceBase(TritonObject):
    '''
    Base Device class for the Triton control system.
    '''

    def __init__(self, name, sql_conf=sqlConf):
        '''
        Set up the device and calls self.on() for device specific construction
        '''
        super(DeviceBase, self).__init__(sql_conf)
        self.adjustinterval = True #if this is set to false, the set interval will not change if the device is too slow
        self.name = name
        self._thread = None
        self._timer = Event()
        self._interval = 0

        self._cfg = ['_interval']
        self._stg = ['_interval']

        self.dbCur_execute("SELECT deviceType, uri, config FROM devices WHERE deviceName=%s", (self.name,))

        db = self.dbCur_fetchone(local_ret_val=(self.type, None, str(self._cfg)))

        if db[1] != None:
            self.logger.warning(self.name + ' already exists! Overwriting.')

        self._commitUri(str(self.uri))

        if db[2] is not None:
            cfg = eval(db[2])
            print('cfg: {}'.format(cfg))
            devices = eval(db[2]).get('devs', None)
            print('devices: {}'.format(devices))
        else:
            cfg = None
            devices = None

        self.lock = Lock()
        self.locktimeout = 5
        self.initialized = False

        if devices != None and db[0]!= 'TritonMain':
            for dev in devices:
                self.start_and_subscribe(dev)

        try:
            self.on(cfg)
        except Exception as exc:
            self.logger.error('Error starting on func in dev: {} error: {}'.format(self.name,str(exc)))
            raise


    def _stop(self):
        '''
        Call self.off() for device specific destruction and afterwards deinitializes the base object
        '''
        if self._thread is not None:
            self.setInterval(0)
            self._thread.join()
            self.logger.debug("device thread shut down!")
        self.send('out', 'Deleted')
        try:
            self.off()
        except Exception as exc:
            self.logger.error('Error in off func in dev: {} error: {}'.format(self.name,str(exc)))

        self._commitUri(None)

        TritonObject._stop(self)

    """Publishing"""

    def _receive(self, dev, t, ch, val):
        '''Called first on receiving, blank wrapper'''
        if ch == 'out' and val == 'Deleted':
            self.logger.debug('Deleting from ' + str(dev))
            del self._recFrom[dev]
            self.connectionLost()
        self.receive(dev, t, ch, val)

    def _emit(self):
        '''Called on new subscriber. Emit the concurrent values.'''
        self.send('interval', self._interval)
        self.emit()

    def send(self, ch, val):
        '''Send value on channel, add timestamp and copy to console'''
        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if ch == 'err':
            self.logger.error(t + ' ' + self.name + ' ' + ch + ": \t" + str(val))
        elif ch == 'out':
            self.logger.info(t + ' ' + self.name + ' ' + ch + ": \t" + str(val))
        else:
            self.logger.debug(t + ' ' + self.name + ' ' + ch + ": \t" + str(val))

        self.server_backend.send(self.name, t, ch, val)
        # for sub in self._sendTo.values():
        #    sub._receive(self.name, t, ch, val)

    def connectionLost(self):
        pass

    def return_name(self):
        return self.name

    """Saving settings"""

    def saveStg(self, comment=''):
        '''Write settings in settings table'''
        stg = repr({val: getattr(self, val) for val in self._stg})
        self.dbCur_execute("INSERT INTO settings (device, date, comment, settings) VALUES (%s, %s, %s, %s)",
                           (self.name, datetime.now(), comment, stg))
        self.db_commit()
        self.send('out', "Saved Setting " + comment + ": " + stg)

    def loadStg(self, stgID):
        '''Load settings with ID from settings table'''
        # self.logger.debug('saving settings start:')
        self.dbCur_execute("SELECT settings FROM settings WHERE ID=%s", (stgID,))
        # self.logger.debug('execute happend')
        stg = eval(self.dbCur_fetchone()[0])
        # self.logger.debug('fetchone happend')

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
            self._periodic()
            diff = time.time() - startTime
            self.logger.debug('processing time: ' + str(diff))
            if diff > self._interval and self._interval != 0 and self.adjustinterval:
                self.send('err', 'processing time is bigger than interval! Setting interval to ' + str(diff))
                self.setInterval(diff)
            # if self._timer.wait(abs(self._interval - diff)):#???
            #    self._timer.clear()
            # print(diff)
            if diff < self._interval:
                time.sleep(self._interval - diff)
        # self._thread = None

    def _periodic(self):
        """ wrapper for periodic since here maybe default operations
         before periodic execution will be implemented, see ScanDeviceBase.py """
        self.periodic()

    def setInterval(self, t):
        '''Set the interval. Start or stop periodic thread as necessary'''
        try:
            tn=float(t)
        except Exception as e:
            self.logger.debug('error in setInterval in Device Base: {} \t devicetype: {}'.format(e, self.type))
            return()

        if self._thread is not None and self._interval>0: #set interval was changed so the pericodic can be restarted
            self._interval = tn
            self._timer.set()
        elif tn > 0:
            self._interval = tn
            self._thread = Thread(target=self._run)
            self._thread.start()
        self.send("interval", self._interval)

    """Other Stuff"""

    def _commitUri(self, uri):
        '''Write the uri into the device table'''
        # self.logger.debug('comitting URI to dab')
        self.dbCur_execute("UPDATE devices SET uri=%s WHERE deviceName=%s", (uri, self.name))
        # self.logger.debug('execute happend')
        self.db_commit()
        # self.logger.debug('commit happend')

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

    @staticmethod
    def wait_for_initialization(func):
        """This is a decorator for simplified usage of the threadlock"""
        @wraps(func)
        def wrap_waiting(self, *args, **kwargs):
            if self.initialized is True:
                func(self, *args, **kwargs)
            else:
                pass
        return wrap_waiting