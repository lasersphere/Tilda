import ast
import logging
from datetime import datetime

import mysql.connector as Sql

sqlCfg = {
    'user': 'root',
    'password': 'Alive2015!',
    'host': '192.168.30.39',
    'database': 'tl_db',
}

name_of_dev_to_subscribe = 'Picoamperemeter'

try:
    db = Sql.connect(**sqlCfg)
except Sql.Error:
    raise

dbCur = db.cursor()

dbCur.execute('''SELECT deviceName FROM devices WHERE uri IS NOT NULL''')
devs = []
res = dbCur.fetchall()
for r in res:
    devs.append(r[0])

print('active devices are: ', devs)


def get_channels_of_dev(dev):
    dbCur.execute(
        '''SELECT devicetypes.channels FROM devices JOIN devicetypes ON
            devicetypes.deviceType = devices.deviceType WHERE devices.deviceName = %s''',
        (str(dev),))
    res = ast.literal_eval(dbCur.fetchone()[0])
    return res

for dev in devs:
    print(dev)
    channels = get_channels_of_dev(dev)
    print('dev %s has channels: ', channels)

# self must have uri!


class TritonListener:
    def _serve(self):
        '''Start pyro daemon'''
        pass
        # TODO: update or remove
        # self._daemon = Pyro4.Daemon()
        # self.uri = self._daemon.register(self)
        # self._daemonT = Thread(target=self._daemon.requestLoop)
        # self._daemonT.start()

    def connect_to_dev(self, uri, name):
        pass
        # TODO: Update or remove
        # dev = Pyro4.Proxy(self.uri)  # ??
        # dev._addSub(uri, name)
        # dev._pyroRelease()


class TritonObject(object):
    '''
    Basic TritonObject with fundamental abilities: Pyro receiving, DB connections, subscribing
    '''

    def __init__(self, name):
        '''
        Constructor
        '''
        super(TritonObject, self).__init__()

        self.name = name
        self.type = 'TritonObject'

        self._recFrom = {}
        self.db = Sql.connect(**sqlCfg)
        self.dbCur = self.db.cursor()

        self._serve()

    def _stop(self):
        '''Unsubscribe from all and stop pyro daemon'''
        # TODO: update or remove
        pass
        # logging.debug('Unsubscribing from ' + str(self._recFrom))
        # for dev in self._recFrom.copy().keys():
        #     self.unsubscribe(dev)
        #
        # self._daemon.shutdown()
        # self._daemonT.join()
        # self.db.close()

    def _serve(self):
        '''Start pyro daemon'''
        # TODO: update or remove
        pass
        # self._daemon = Pyro4.Daemon()
        # self.uri = self._daemon.register(self)
        # self._daemonT = Thread(target=self._daemon.requestLoop)
        # self._daemonT.start()

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
        # TODO: update or remove
        pass
        # self.send('err', "".join(Pyro4.util.getPyroTraceback()))

    def _receive(self, dev, t, ch, val):
        print(t, dev, ch, val)

    def resolveName(self, name):
        """Resolve a device name to a Proxy using the uri from the database. Return None if not started"""
        print('resolving name: ', name)
        self.db.commit()
        self.dbCur.execute('''SELECT uri FROM devices WHERE deviceName=%s''', (name,))
        result = self.dbCur.fetchall()
        print('result: ', result[0][0])
        # TODO: update or remove
        # dev = Pyro4.Proxy(result[0][0])
        dev = None
        return dev


if __name__=='__main__':
    listener = TritonObject('TildaListener')
    listener.subscribe(name_of_dev_to_subscribe)
    input('press anything to stop')
