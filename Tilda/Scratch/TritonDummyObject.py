'''

author: Simon Kaufmann

create a local Device (-> no db connection!) and have it send stuff periodically

'''

# TODO: Just copy pasted some stuff from TritonObject with new Triton. Not tested in any way. Maybe need to write new...

from datetime import datetime
import mysql.connector as Sql
from Tilda.Driver.TritonListener.TritonConfig import sqlCfg as sqlConf
import Tilda.Driver.TritonListener.Backend.udp_server
import Tilda.Driver.TritonListener.Backend.tcp_server
import logging
import Tilda.Driver.TritonListener.Backend.hybrid_server
import Tilda.Driver.TritonListener.Backend.server_conf


class TritonObject(object):
    """
    Basic TritonObject with fundamental abilities: Pyro receiving, DB connections, subscribing
    """

    def __init__(self, sql_conf='local'):
        """
        Constructor
        """
        super(TritonObject, self).__init__()
        self.name = None
        self.type = 'TritonObject'

        self._recFrom = {}
        self.sql_conf = 'local'
        self.db = 'local'  # can be set to 'local' or '' or {} for testing without any database, see below
        self.dbCur = None
        self.db_connect()
        self.logger = logging.getLogger('TritonLogger')

        # start the appropriate server_backend depending on the selection in server_conf
        if Tilda.Driver.TritonListener.Backend.server_conf.SERVER_CONF.TRANS_MODE == "UDP":
            self.server_backend = Tilda.Driver.TritonListener.Backend.udp_server.TritonServerUDP(self.type, self)
            self.logger.debug("Backend: TritonServer started in UDP mode!")
        elif Tilda.Driver.TritonListener.Backend.server_conf.SERVER_CONF.TRANS_MODE == "HYB":
            self.server_backend = Tilda.Driver.TritonListener.Backend.hybrid_server.TritonServerHybrid(self.type, self)
            self.logger.debug("Backend: TritonServer started in HYBRID mode!")
        else:
            self.server_backend = Tilda.Driver.TritonListener.Backend.tcp_server.TritonServerTCP(self.type, self)
            self.logger.debug("Backend: TritonServer started in TCP mode!")

        self.uri = self.server_backend.uri

    def db_connect(self):
        pass

    def db_close(self):
        pass

    def _stop(self):
        """ Unsubscribe from all and stop server object """
        self.logger.debug('Unsubscribing from ' + str(self._recFrom))
        for dev in self._recFrom.copy().keys():
            self.unsubscribe(dev)

        self.server_backend.shutdown()
        self.db_close()
        self.logger.debug('Stopped device: {}'.format(self.name))

    def getName(self):
        return self.name

    def getType(self):
        return self.type

    """Methods for subscribing"""
    def subscribe(self, ndev, known_uri=''):
        """Subscribe to an object using its name and returning at a TritonRemoteObject"""
        remuri = self.resolveName(ndev, known_uri)
        remoteobj = None
        if remuri != None:
            self.send('out', 'Subscribing to ' + ndev)
            remoteobj = self.server_backend.subscribeToUri(remuri)
            if remoteobj != None:
                self._recFrom[ndev] = remoteobj
                self.send('out', 'Added')
                self.send('out', 'Done with subscribe')
            else:
                logging.error("Subscribing to device "+ndev+" failed!")
                self.send('err', 'Could not connect to ' + ndev)
        else:
            self.send('err', 'Could not resolve ' + ndev)
        return remoteobj


    def send(self, ch, val): # important note: this will be overwritten in DeviceBase! It is just used for the UIs to log the info
        """ send ch and val to logging.error (ch='err'), logging.info (ch='out') or logging.debug
         Note: this is not send to any device or so, this is only done in the DeviceBase.py
         """
        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if ch == 'err':
            self.logger.error(t + ' ' + self.name + ' ' + ch + ": \t" + str(val))
        elif ch == 'out':
            self.logger.info(t + ' ' + self.name + ' ' + ch + ": \t" + str(val))
        else:
            self.logger.debug(t + ' ' + self.name + ' ' + ch + ": \t" + str(val))

    def _receive(self, dev, t, ch, val): # important note: this will be overwritten in DeviceBase! just ignore it
        """ just a print here, will be overwritten in DeviceBase.py """
        print(t, dev, ch, val)

    def resolveName(self, name, known_uri=''):
        """
        Resolve a device name to a URI using the uri from the database. Return None if not started
        :param name: str, name of the device which pyro4 uri should be found in the db
        :param known_uri: str, uri can be provided if no database is present.
        """
        self.logger.debug('resolve name {} to database'.format(name))
        self.db_commit()
        # self.logger.debug('commit happend')
        self.dbCur_execute("SELECT uri FROM devices WHERE deviceName=%s", (name,))
        # self.logger.debug('execute happend')
        result = self.dbCur_fetchall(local_ret_val=[(known_uri,)])
        return result[0][0]


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
        dev = None # Pyro4.Proxy(uri)
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
            if diff > self._interval and self._interval != 0:
                logging.debug('processing time: ' + str(diff))
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
