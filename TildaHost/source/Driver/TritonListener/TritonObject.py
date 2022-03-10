"""
Created on 11.05.2017

@author: simkaufm

Module Description:
    Triton Object copied from TRITON in order to connect to TritonObjects(devices) in Triton.
    If this ever changes within Triton, changes here might be necessary

    Last copied on 26.03.2020 to fit new backend

    If changes are made within Triton maybe a copy is needed again.
    Required modifications for Tilda are marked with a comment  # changed!

    (ann. Tim Lellinger: ONLY change dependenies, noting else!)

"""

from datetime import datetime
import mysql.connector as Sql
from Driver.TritonListener.TritonConfig import sqlCfg as sqlConf
import Driver.TritonListener.Backend.udp_server
import Driver.TritonListener.Backend.tcp_server
import logging
import Driver.TritonListener.Backend.hybrid_server
import Driver.TritonListener.Backend.server_conf

class TritonObject(object):
    '''
    Basic TritonObject with fundamental abilities: receiving messages, DB connections, subscribing
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
        self.logger = logging.getLogger('TritonLogger')

        #start the appropriate server_backend depending on the selection in server_conf
        if Driver.TritonListener.Backend.server_conf.SERVER_CONF.TRANS_MODE == "UDP":
            self.server_backend = Driver.TritonListener.Backend.udp_server.TritonServerUDP(self.type, self)
            self.logger.debug("Backend: TritonServer started in UDP mode!")
        elif Driver.TritonListener.Backend.server_conf.SERVER_CONF.TRANS_MODE == "HYB":
            self.server_backend = Driver.TritonListener.Backend.hybrid_server.TritonServerHybrid(self.type, self)
            self.logger.debug("Backend: TritonServer started in HYBRID mode!")
        else:
            self.server_backend = Driver.TritonListener.Backend.tcp_server.TritonServerTCP(self.type, self)
            self.logger.debug("Backend: TritonServer started in TCP mode!")

        self.uri = self.server_backend.uri

    """ encapsule db functionalities to handle connectivity problems and allow to operate without a db """

    def db_connect(self):
        if isinstance(self.sql_conf, dict):
            try:
                self.db = Sql.connect(**self.sql_conf)
                self.dbCur = self.db.cursor()
            except Exception as e:
                self.logger.error('could not connect to database %s, error is: %s' % (
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

    def _stop(self):
        '''Unsubscribe from all and stop server object'''
        self.logger.debug('Unsubscribing from ' + str(self._recFrom))
        for dev in self._recFrom.copy().keys():
            self.unsubscribe(dev)

        self.server_backend.shutdown()
        self.db_close()
        self.logger.debug('Stopped device: {}'.format(self.name))

    """ who am I ? """
    def getName(self):
        return self.name

    def getType(self):
        return self.type

    # def get_local_py_obj(self): #evtl eine option für Picoscope
    #     return(self)

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

    def start_and_subscribe(self, ndev):
        """subscribe to an object using its name, and return a TritonRemoteObject. If the device is not active,
        start it (on the right TritonMain if specified)"""
        self.dbCur_execute("SELECT actualMain, run_on FROM devices WHERE deviceName=%s", (ndev,)) # gets the actual main the device is alrady running on and a main-name if teh device has to be run on a certain device
        result_db = self.dbCur_fetchall()
        res_acmain=result_db[0][0]
        res_runon=result_db[0][1]
        print('res_acmain is: {} ; res_runon is: {}'.format(res_acmain,res_runon))
        if res_acmain is None: #this means the device is not running already
            self.logger.debug('s_a_s: {} device not running, starting now!'.format(ndev))
            self.dbCur_execute("SELECT actualMain FROM devices WHERE deviceName=%s", (self.name,)) #gets the local main
            mainName = self.dbCur_fetchall()[0][0]
            print('Main name ist: {}'.format(mainName))
            if res_runon is None or res_runon in mainName: # this means the device can be started on any main, so it will be started locally or the local main is already the correct main
                print('device kann unabhängig oder in diesem main gestartet werden: {}'.format(res_runon))
                pass
            else:
                print('Devicemuss auf anderer main gestartet werden')
                self.dbCur_execute("SELECT deviceName FROM devices WHERE actualMain='running' AND deviceName LIKE %s", (res_runon+'%',)) #in this case one running correct main will be looked up
                mainName = self.dbCur_fetchall()[0][0]
            if mainName != None:
                devicemain = self.subscribe(mainName)
                devicemain.startDev(ndev)
                self.unsubscribe(mainName)
            else:
                self.logger.warning("Required main {} is not available to start the device. Please start such a main!".format(res_runon))
        return self.subscribe(ndev)

    def unsubscribe(self, ndev):
        """Unsubscribe from an object"""
        self.send('out', 'Unsubscribing from ' + ndev)
        if ndev in self._recFrom:
            try:
                self.server_backend.unsubscribeFromRemoteObject(self._recFrom[ndev])
                del self._recFrom[ndev]
            except:
                self.send('err', 'Could not unsubscribe from ' + str(ndev))

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
