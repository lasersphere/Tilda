"""
Created on 11.05.2017

@author: simkaufm

Module Description:
    Triton Object copied from TRITON in order to connect to TritonObjects(devices) in Triton.
    If this ever changes within Triton, changes here might be necessary
"""

import logging
from datetime import datetime
from threading import Thread

import Pyro4
import mysql.connector as Sql


class TritonObject(object):
    """
    Basic TritonObject with fundamental abilities: Pyro receiving, DB connections, subscribing
    """

    def __init__(self, name='TildaListener', sqlCfg={}):
        """
        Constructor
        :parameter name: str, name of the object one is creating
        :parameter sqlCfg: dict
        """
        super(TritonObject, self).__init__()

        self.name = name
        self.type = 'TritonObject'
        self.sql_config = sqlCfg

        self._recFrom = {}

        self._serve()

    def _stop(self):
        """
        Unsubscribe from all and stop pyro daemon
        """
        logging.debug('Unsubscribing from ' + str(self._recFrom))
        for dev in self._recFrom.copy().keys():
            self.unsubscribe(dev)

        self._daemon.shutdown()
        self._daemonT.join()

    def _serve(self):
        """
        Start pyro daemon
        """
        self._daemon = Pyro4.Daemon()
        self.uri = self._daemon.register(self)
        self._daemonT = Thread(target=self._daemon.requestLoop)
        self._daemonT.start()

    def getName(self):
        return self.name

    def getType(self):
        return self.type

    """Methods for subscribing"""

    def subscribe(self, ndev, uri=''):
        """Subscribe to an object using its name"""
        dev = self.resolveName(ndev, uri)
        logging.debug('resolved name of %s, dev is: %s, self.uri is: %s' % (ndev, str(dev), str(self.uri)))
        if dev is not None:
            self.send('out', 'Subscribing to ' + ndev)
            self._recFrom[ndev] = dev
            logging.info('%s with uri %s is subscribing to: %s' % (self.name, self.uri, str(dev)))
            dev._addSub(self.uri, self.name)
            self.send('out', 'Added')
            dev._pyroRelease()
            self.send('out', 'Done with subscribe')
        else:
            self.send('err', 'Could not resolve ' + ndev)
        return dev

    def unsubscribe(self, ndev):
        """Unsubscribe from an object"""
        self.send('out', 'Unsubscribing from ' + ndev)
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
        logging.debug('received: %s\t%s\t%s\t%s' % (str(t), str(dev), str(ch), str(val)))

    def resolveName(self, name, uri=''):
        """Resolve a device name to a Proxy using the uri from the database. Return None if not started"""
        logging.info('TritonObject resolving name: %s' % name)
        dev = None
        try:
            db = Sql.connect(**self.sql_config)
            dbCur = db.cursor()
            logging.info('connecting to db: %s at ip: %s' % (self.sql_config.get('database', 'unknown'),
                                                             self.sql_config.get('host', 'unknown')))
        except Exception as e:
            logging.error('error, TritonObject Could not connect to db, error is: %s' % e)
            db = None
            dbCur = None
        if db is not None:
            db.commit()
            dbCur.execute('''SELECT uri FROM devices WHERE deviceName=%s''', (name,))
            result = dbCur.fetchall()
            if len(result):
                logging.info('TritonObject resolve name result: %s' % str(result[0][0]))
                if result[0][0] is not None:
                    dev = Pyro4.Proxy(result[0][0])
        else:  # name should be str(uri) then
            dev = Pyro4.Proxy(uri)
        if db is not None:
            db.close()
        return dev


if __name__=='__main__':
    import socket
    import sys

    hmacKey = b'6\x19\n\xad\x909\xda\xea\xb5\xc5]\xbc\xa1m\x863'
    # Set Pyro variables
    Pyro4.config.SERIALIZER = "serpent"
    Pyro4.config.HMAC_KEY = hmacKey
    Pyro4.config.HOST = socket.gethostbyname(socket.gethostname())
    # Pyro4.config.SERVERTYPE = 'multiplex'
    Pyro4.config.SERVERTYPE = 'thread'
    sys.excepthook = Pyro4.util.excepthook
    Pyro4.config.DETAILED_TRACEBACK = True

    triton_obj = TritonObject()
    from Driver.TritonListener.DummyTritonDevice import DummyTritonDevice
    dummy_dev = DummyTritonDevice('dummyDev')
    triton_obj.subscribe(str(dummy_dev.uri))
    dummy_dev.setInterval(1)
    input('anykey to stop')
    triton_obj._stop()
    dummy_dev._stop()
