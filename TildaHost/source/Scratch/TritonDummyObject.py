'''

author: Simon Kaufmann

create a local Device (-> no db connection!) and have it send stuff periodically

'''

'''
Created on 01.10.2013

@author: hammen
'''

import Pyro4


class TritonObject(object):
    '''
    Basic TritonObject with fundamental abilities: Pyro receiving, DB connections, subscribing
    '''

    def __init__(self):
        '''
        Constructor
        '''
        super(TritonObject, self).__init__()

        self.name = None
        self.type = 'TritonObject'

        self._recFrom = {}
        self.db = ''
        self.dbCur = None

        self._serve()

    def _stop(self):
        '''Unsubscribe from all and stop pyro daemon'''
        logging.debug('Unsubscribing from ' + str(self._recFrom))
        for dev in self._recFrom.copy().keys():
            self.unsubscribe(dev)

        self._daemon.shutdown()
        self._daemonT.join()

    def _serve(self):
        '''Start pyro daemon'''
        self._daemon = Pyro4.Daemon()
        self.uri = self._daemon.register(self)
        self._daemonT = Thread(target=self._daemon.requestLoop)
        self._daemonT.start()

    def getName(self):
        return self.name

    def getType(self):
        return self.type

    """Methods for subscribing"""

    def subscribe(self, ndev):
        """Subscribe to an object using its name"""
        dev = self.resolveName(ndev)
        if dev != None:
            self.send('out', 'Subscribing to ' + ndev)
            self._recFrom[ndev] = dev
            dev._addSub(self.uri, self.name)
            self.send('out', 'Added')
            dev._pyroRelease()
            self.send('out', 'Done with subscribe')
        else:
            self.send('err', 'Could not resolve ' + ndev)
        return dev

    def unsubscribe(self, ndev):
        """Unsubscribe from an object"""
        self.send('out', 'Unsusbcribing from ' + ndev)
        if ndev in self._recFrom:
            try:
                self._recFrom[ndev]._remSub(self.name)
                del self._recFrom[ndev]
            except:
                self.send('err', 'Could not unsubscribe from ' + str(ndev))

    def send(self, ch, val):
        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if ch == 'err':
            logging.error(t + ' ' + self.name + ' ' + ch + ": \t" + str(val))
        elif ch == 'out':
            logging.info(t + ' ' + self.name + ' ' + ch + ": \t" + str(val))
        else:
            logging.debug(t + ' ' + self.name + ' ' + ch + ": \t" + str(val))

    def errsend(self):
        self.send('err', "".join(Pyro4.util.getPyroTraceback()))

    def _receive(self, dev, t, ch, val):
        print(self.name, t, dev, ch, val)

    def resolveName(self, uri):
        """Resolve a device name to a Proxy using the uri from the database. Return None if not started"""
        dev = Pyro4.Proxy(uri)
        return dev


from functools import wraps
from threading import Thread, Event, Lock
from datetime import datetime
import logging, time


class DeviceBase(TritonObject):
    '''
    Base Device class for the Triton control system.
    '''

    def __init__(self, name):
        '''
        Set up the device and calls self.on() for device specific construction
        '''
        super(DeviceBase, self).__init__()

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
        '''
        Call self.off() for device specific destruction and afterwards deinitializes the base object
        '''

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

    def receive(self, dev, t, ch, val):
        print(self.name, dev, t, ch, val)

    def _emit(self):
        '''Called on new subscriber. Emit the concurrent values.'''
        self.send('interval', self._interval)
        # self.emit()

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

    def periodic(self):
        # print('periodic called')
        self.send('calls', self.i)
        self.i += 1

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


class DraftDevice(DeviceBase):
    '''
    Put a short description of the device here
    '''

    '''Called when added'''

    def on(self, cfg):
        '''Setting necessary attributes, interval defaults to 0:'''
        self.type = 'draftDev'
        # self.addCfg(['par'])
        # self.addStg(['par'])

        self.setInterval(1)

        '''Resolving and subscribing to other devices:'''
        # self.dev = self.resolveName('Name')
        # self.subscribe(dev)

    '''Called when removed'''

    def off(self):
        pass

    '''Called regularly, running in separate thread'''

    def periodic(self):
        # self.send('', value)
        pass

    '''Called by subscriptors'''

    def receive(self, dev, t, ch, val):
        pass

    '''Called when settings are loaded, vals contains setting dictionary'''

    def load(self, vals):
        pass

    '''Send current status on this command'''

    def emit(self):
        pass

if __name__=='__main__':
    sender = DeviceBase('sender')  # get this from db later in tilda
    receiver = DeviceBase('receiver')  # create this
    sender._addSub(receiver.uri, 'receiver')  # subscribe to triton dev, done
    sender.send('a', 10)
    sender.setInterval(1)
    input('enter anything to stop')
    # receiver._stop()
    sender._stop()
