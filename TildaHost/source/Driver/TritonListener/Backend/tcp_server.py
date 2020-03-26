import datetime
import socket
import threading
import socketserver
import time
from Driver.TritonListener.Backend.server_conf import SERVER_CONF
import Driver.TritonListener.Backend.triton_trans as tt
import Driver.TritonListener.Backend.tritonremoteobject
import logging
import uuid
import Driver.TritonListener.TritonConfig

if Driver.TritonListener.Backend.server_conf.SERVER_CONF.ENCRYPTION: #if encryption is not enabled, the package is not required to be installed.
    from cryptography.fernet import Fernet

'''This class implements a server using the TCP protocol, and is the base with which every Triton object can
communicate. When tested with the DummyPA device, it has a minimum interval time of about 5 ms and a max data rate of 
70 MB/s (tested with an array of random numbers). This puts it ahead of the udp implementation, which has a similar
response time but is much slower when transmitting large amount of data.

Please note that this is a potential security risk, since pickle can also serialize program code. For example, a method 
call could return a manipulated class over the network, which executes something like os.system('shutdown -s') whenever
 somebody tries to access an attribute. This problem can be somewhat defused by enabling the AES256 encryption in the
 server_conf file (make sure that every host computer also has AESKey set in Base.Config). However, enabling this 
 currently hammers performance for large amounts of data (~1/5 the speed on an old PC)
'''

class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):

    def handle(self):
        msg_len = int.from_bytes(self.request.recv(8), byteorder='big', signed=False)

        data = bytearray(self.request.recv(SERVER_CONF.TCP_BUFFER_SIZE))
        while len(data) < msg_len:
            data.extend(bytearray(self.request.recv(SERVER_CONF.TCP_BUFFER_SIZE)))

        if Driver.TritonListener.Backend.server_conf.SERVER_CONF.ENCRYPTION:
            f = Fernet(Driver.TritonListener.Config.AESKey)
            data = f.decrypt(bytes(data))

        if data[0] == SERVER_CONF.MSG_DATA or data[0] == SERVER_CONF.MSG_REPLY:# these two will return no reply immediatly, the others will first execute, then reply
            self.request.sendall((0).to_bytes(8, byteorder='big', signed=False))
            self.server.evaluate_request(data)
        else:
            response = bytearray(self.server.evaluate_request(data))
            if Driver.TritonListener.Backend.server_conf.SERVER_CONF.ENCRYPTION:
                response = bytearray(f.encrypt(bytes(response)))
            rsp_len = len(response).to_bytes(8, byteorder='big', signed=False)
            self.request.sendall((rsp_len + response))


class TritonServerTCP(socketserver.ThreadingMixIn, socketserver.TCPServer):
    def __init__(self, type, parentobj=None):
        self.type = type  # device name
        if parentobj is None:
            self.parentobj = self #this is just used for remote function calling.
        else:
            self.parentobj = parentobj
            self.receive = parentobj._receive # the dummy receive method gets overwritten by the actual receive method of the parent TritonObject

        addr = socket.gethostbyname(socket.gethostname())
        socketserver.TCPServer.__init__(self, (addr, 0),
                                        ThreadedTCPRequestHandler)  # Port 0 means to select an arbitrary unused port
        socketserver.ThreadingMixIn.__init__(self)

        server_thread = threading.Thread(target=self.serve_forever)

        self.send_to_list = []  # the list of subscribers. These [IP,PORT] will receive messages from send()
        self.listlock = threading.RLock()  # object to circumvent errors due to multiple threads writing/reading to send_to_list
        # Exit the server thread when the main thread terminates
        server_thread.daemon = True
        server_thread.start()
        self.uri = "TritonObject_" + str(uuid.uuid4()) + "@" + str(addr) + ":" + str(self.server_address[1])
        self.logger = logging.getLogger('TritonLogger')


    def update_remote_lists(self): #not used yet, could be used to complete Pyro functionality
        self.__remote_methods = []
        self.__remote_vars = []

        for attr in dir(self.parentobject):
            if not attr.startswith('__'): #everything which starts with more than two underscores will be ignored
                if callable(getattr(self.parentobj, attr)):
                    self.__remote_methods.extend([attr])
                else:
                    self.__remote_vars.extend([attr])

    def _emit(self):
        pass #placeholder
    def evaluate_request(self, data):

        msg_type = data[0]
        dd = tt.TritonTransmission.unserialize(data[1:])
        if msg_type == SERVER_CONF.MSG_DATA:
            self.receive(dd[0], dd[1], dd[2], dd[3])
        elif msg_type == SERVER_CONF.REMOTE_CALL:
            try:
                callfunc = getattr(self.parentobj, dd[0])
                reply = callfunc(*dd[1])
                return tt.TritonTransmission.serialize(reply)
            except Exception as err:
                self.logger.error("method " + dd[0] + " could not be called or does not exist: " + str(err))
        elif msg_type == SERVER_CONF.SUB_REQUEST:
            self.listlock.acquire()
            if dd not in self.send_to_list:
                self.send_to_list += [dd]
                self.logger.debug("Backend: Subscription request from " + str(dd))
            else:
                self.logger.debug("Backend: " + str(dd) + " is already subscribed, why request it again?")
            self.listlock.release()
            self.parentobj._emit()#calls the emit function of the corresponding TritonObject (important e.g. for TritonMain)
        elif msg_type == SERVER_CONF.SUB_CANCEL:
            self.listlock.acquire()
            index = None
            for i in range(len(self.send_to_list)):
                if self.send_to_list[i][0] == dd[0] and self.send_to_list[i][1] == dd[1]:
                    index = i
                    break
            if index is not None:
                del self.send_to_list[index]
                self.logger.debug(str(dd) + " has unsubscribed.")
            else:
                self.logger.debug(str(dd) + " has tried to unsubscribe without being subscribed first...?")
            self.listlock.release()
        return tt.TritonTransmission.serialize(SERVER_CONF.MSG_ACK)

    def receive(self, dev, t, ch, val):
        print(self.type + ": " + str(t) + ": received a message from '" + str(dev) + "'on channel '" + str(
            ch) + "' with the length: " + str(len(val))) #this is just a dummy function that is not called unless the server exists without a parent TritonObject (see initializer)

    def subscribeToUri(self, uri):
        spl = uri.split('@')
        spll = spl[1].split(':')
        ip = str(spll[0])
        port = int(spll[1])
        return self.subscribeTo(ip, port)

    def subscribeTo(self, ip, port):
        msg_data = tt.TritonTransmission.serialize(self.server_address)
        try:
            self.sendrequest(ip, port, msg_bytes=msg_data, msg_type=SERVER_CONF.SUB_REQUEST)
            return Driver.TritonListener.Backend.tritonremoteobject.TritonRemoteObject(self, ip, port)
        except:
            self.logger.error("TCP NETWORK ERROR while subscribing: Host "+str(ip)+":"+str(port)+" could not be reached!")
            return None

    def unsubscribeFrom(self, ip, port):
        msg_data = tt.TritonTransmission.serialize(self.server_address)
        self.sendrequest(ip, port, msg_bytes=msg_data, msg_type=SERVER_CONF.SUB_CANCEL)
        pass

    def unsubscribeFromRemoteObject(self, remobj):
        self.unsubscribeFrom(remobj.getip(), remobj.getport())
        pass

    def send(self, name, t, ch, val):
        data = Driver.TritonListener.Backend.triton_trans.TritonTransmission.serialize_sub(name, t, ch, val)
        self.listlock.acquire()
        for sub in self.send_to_list:
            self.sendrequest(sub[0], sub[1], data)
        self.listlock.release()

    def sendrequest(self, ip, port, msg_bytes, msg_type=SERVER_CONF.MSG_DATA):
        message = msg_type.to_bytes(1, byteorder='big', signed=False) + msg_bytes

        if Driver.TritonListener.Backend.server_conf.SERVER_CONF.ENCRYPTION:
            f = Fernet(Driver.TritonListener.Config.AESKey)
            message = f.encrypt(message)

        msg_len_bytes = len(message).to_bytes(8, byteorder='big', signed=False)
        response = None
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(SERVER_CONF.TCP_TIMEOUT)
            sock.connect((ip, port))


            sock.sendall(msg_len_bytes)
            sock.sendall(message)
            resp_len = int.from_bytes(sock.recv(8), byteorder='big', signed=False)
            if resp_len > 0:
                data = bytearray(sock.recv(SERVER_CONF.TCP_BUFFER_SIZE))
                # for i in range(0, int(resp_len / SERVER_CONF.TCP_BUFFER_SIZE)+1):
                #     data.extend(bytearray(sock.recv(SERVER_CONF.TCP_BUFFER_SIZE)))
                while len(data)<resp_len:
                    data.extend(bytearray(sock.recv(SERVER_CONF.TCP_BUFFER_SIZE)))
                if Driver.TritonListener.Backend.server_conf.SERVER_CONF.ENCRYPTION:
                    data = f.decrypt(bytes(data))
                response = tt.TritonTransmission.unserialize(data)
        return response
    def shutdown(self):
        socketserver.TCPServer.shutdown(self)
        self.server_close()
