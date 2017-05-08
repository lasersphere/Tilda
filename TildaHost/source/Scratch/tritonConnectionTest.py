import ast
import socket
import sys
from threading import Thread

import Pyro4
import mysql.connector as Sql

sqlCfg = {
    'user': 'userName',
    'password': '123456',
    'host': 'hostAdress',
    'database': 'DataBaseName',
}


hmacKey = b'6\x19\n\fko\x909\loa\poa\xb5\xc5]\xbc\xa1m\x863'

#Set Pyro variables
Pyro4.config.SERIALIZER = "serpent"
Pyro4.config.HMAC_KEY = hmacKey
Pyro4.config.HOST = socket.gethostbyname(socket.gethostname())
#Pyro4.config.SERVERTYPE = 'multiplex'
Pyro4.config.SERVERTYPE = 'thread'
sys.excepthook = Pyro4.util.excepthook
#Pyro4.config.DETAILED_TRACEBACK = True

try:
    db = Sql.connect(**sqlCfg)
except Sql.Error:
    raise

dbCur = db.cursor()

dbCur.execute('''SELECT deviceName FROM devices WHERE uri IS NOT NULL''')
devs = []
res = dbCur.fetchall()
for r in res:
    devs.append(r)

print('active devices are: ', devs)


def get_channels_of_dev(dev):
    dbCur.execute(
        '''SELECT devicetypes.channels FROM devices JOIN devicetypes ON
            devicetypes.deviceType = devices.deviceType WHERE devices.deviceName = ?''',
        (str(dev),))
    res = ast.literal_eval(dbCur.fetchone()[0])
    return res

for dev in devs:
    channels = get_channels_of_dev(dev)
    print('dev %s has channels: ', channels)

# self must have uri!


class TritonListener:
    def _serve(self):
        '''Start pyro daemon'''
        self._daemon = Pyro4.Daemon()
        self.uri = self._daemon.register(self)
        self._daemonT = Thread(target=self._daemon.requestLoop)
        self._daemonT.start()

    def connect_to_dev(self, uri, name):
        dev = Pyro4.Proxy(self.uri)  # ??
        dev._addSub(uri, name)
        dev._pyroRelease()

