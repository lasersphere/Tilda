"""
Created on 11.05.2017

@author: simkaufm

Module Description:
    Triton Object copied from TRITON in order to connect to TritonObjects(devices) in Triton.
    If this ever changes within Triton, changes here might be necessary

    Last copied on 01.03.19 trito git rev num: 74e28f9804a8d2a27f2f53aa8c0671cd6dc804e4

    If changes are made within Triton maybe a copy is needed again.
    Required modifications for Tilda are marked with a comment  # changed!
"""

from threading import Thread
from datetime import datetime
import logging

import Pyro4
import mysql.connector as Sql

from Driver.TritonListener.TritonDraftConfig import sqlCfg as sqlConf  # changed!


class TritonObject(object):
    '''
    Basic TritonObject with fundamental abilities: Pyro receiving, DB connections, subscribing
    '''

    def __init__(self, sql_conf=sqlConf):
        '''
        Constructor
        '''
        super(TritonObject, self).__init__()

        self.name = None
        self.type = 'TritonObject'

        self._recFrom = {}
        self.sql_conf = sql_conf
        self.db = None  # can be set to 'local' or '' or {} for testing without any database, see below
        self.dbCur = None
        self.db_connect()

        self._serve()

    """ encapsule db functionalities to handle connectivity problems and allow to operate without a db """

    def db_connect(self):
        if isinstance(self.sql_conf, dict):
            try:
                self.db = Sql.connect(**self.sql_conf)
                self.dbCur = self.db.cursor()
            except Exception as e:
                logging.error('could not connect to database %s, error is: %s' % (
                    self.sql_conf.get('database', 'unknown'), e))
                self.db = None
                self.dbCur = None
        elif isinstance(self.sql_conf, str) or self.sql_conf == {}:
            # if the sql_conf is a string or an empty dict, it will be assumed taht no db connection is wanted
            # -> self.db is set to 'local' and all following db calls will be ignored.
            # helpful for developing Triton devs without a db but passing them the uri of another device directly.
            self.db = 'local'  # for testing without db
            self.dbCur = None

    def dbCur_execute(self, var1, var2):
        if self.db != 'local':
            try:
                self.dbCur.execute(var1, var2)
            except:
                self.db_connect()
                self.dbCur.execute(var1, var2)

    def db_commit(self):
        if self.db != 'local':
            try:
                self.db.commit()
            except:
                self.db_connect()
                self.db.commit()

    def dbCur_fetchall(self, local_ret_val=None):
        """
        fetch all result from previous query result
        :param local_ret_val: list of tuples, default [(None)] can be used for testing without db
        :return:
        """
        if self.db != 'local':
            try:
                var = self.dbCur.fetchall()
                return var
            except:
                self.db_connect()
                return self.db.fetchall()
        else:
            if local_ret_val is None:
                local_ret_val = [(None,)]
            return local_ret_val

    def dbCur_fetchone(self, local_ret_val=None):
        """
        fetch only one result from previous query result
        :param local_ret_val: value, default None can be used for testing without db
        :return:
        """
        if self.db != 'local':
            try:
                var = self.dbCur.fetchone()
                return var
            except:
                self.db_connect()
                return self.db.fetchone()
        else:
            return local_ret_val

    def db_close(self):
        if self.db != 'local':
            try:
                self.db.close()
            except:
                self.db_connect()
                self.db.close()

    """ pyro4 thread """

    def _stop(self):
        '''Unsubscribe from all and stop pyro daemon'''
        logging.debug('Unsubscribing from ' + str(self._recFrom))
        for dev in self._recFrom.copy().keys():
            self.unsubscribe(dev)

        self._daemon.shutdown()
        self._daemonT.join()
        self.db_close()

    def _serve(self):
        '''Start pyro daemon'''
        self._daemon = Pyro4.Daemon()
        self.uri = self._daemon.register(self)
        self._daemonT = Thread(target=self._daemon.requestLoop)
        self._daemonT.start()

    """ who am I ? """

    def getName(self):
        return self.name

    def getType(self):
        return self.type

    """Methods for subscribing"""

    def subscribe(self, ndev, known_uri=''):
        """Subscribe to an object using its name"""
        dev = self.resolveName(ndev, known_uri)
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
        """ send ch and val to logging.error (ch='err'), logging.info (ch='out') or logging.debug
         Note: this is not send to any device or so, this is only done in the DeviceBase.py
         """
        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if ch == 'err':
            logging.error(t + ' ' + self.name + ' ' + ch + ": \t" + str(val))
        elif ch == 'out':
            logging.info(t + ' ' + self.name + ' ' + ch + ": \t" + str(val))
        else:
            logging.debug(t + ' ' + self.name + ' ' + ch + ": \t" + str(val))

    def errsend(self):
        """ send last pyro4 traceback error to self.send -> logging.error(...)"""
        self.send('err', "".join(Pyro4.util.getPyroTraceback()))

    def _receive(self, dev, t, ch, val):
        """ just a print here, will be overwritten in DeviceBase.py """
        print(t, dev, ch, val)

    def resolveName(self, name, known_uri=''):
        """
        Resolve a device name to a Proxy using the uri from the database. Return None if not started
        :param name: str, name of the device which pyro4 uri should be found in the db
        :param known_uri: str, uri can be provided if no database is present.
        """
        logging.debug('resolve name to database')
        self.db_commit()
        # logging.debug('commit happend')
        self.dbCur_execute("SELECT uri FROM devices WHERE deviceName=%s", (name,))
        # logging.debug('execute happend')
        result = self.dbCur_fetchall(local_ret_val=[(known_uri,)])
        # logging.debug('fetchall happend')
        dev = Pyro4.Proxy(result[0][0])
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

    triton_obj = TritonObject('local')
    from Driver.TritonListener.DummyTritonDevice import DummyTritonDevice
    dummy_dev = DummyTritonDevice('dummyDev', 'local')
    print('dummy_dev.uri', dummy_dev.uri)
    triton_obj.subscribe('dummyDev', str(dummy_dev.uri))
    dummy_dev.setInterval(1)
    input('anykey to stop')
    triton_obj._stop()
    dummy_dev._stop()
