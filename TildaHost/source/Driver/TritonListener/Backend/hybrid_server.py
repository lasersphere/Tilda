import Driver.TritonListener.Backend.tcp_server
import Driver.TritonListener.Backend.udp_server
import concurrent.futures
import threading
import socket
from Driver.TritonListener.Backend.server_conf import SERVER_CONF
import Driver.TritonListener.Backend.triton_trans as tt

if Driver.TritonListener.Backend.server_conf.SERVER_CONF.ENCRYPTION:#if encryption is not enabled, the package is not required to be installed.
    from cryptography.fernet import Fernet

import uuid
import time
import logging

'''
This class implements a server using a hybrid mode. It derives from TritonServerTCP, and as such uses the TCP
protocol to send out subsription requests, call remote function or transmit large amounts of data. However, in the case
of very small (< UDP_BUFFER_SIZE) packets that are send from the send() function, it switches to udp for higher
transmission frequencies and lower latency.

It therefore should reach both transmission speeds in the order of 70 MB/s, as well as frequencies of around 1 kHz
(depending on the network connection of course)
'''

class TritonServerHybrid(Driver.TritonListener.Backend.tcp_server.TritonServerTCP):
    def __init__(self, type, parentobj=None):
        super().__init__(type, parentobj)
        self.running = True
        self.UDP_PORT_REC = Driver.TritonListener.Backend.udp_server.TritonServerUDP.getfreeport()  # port used to receive messages
        self.UDP_PORT_CON = Driver.TritonListener.Backend.udp_server.TritonServerUDP.getfreeport()  # port used to send messages
        self.UDP_PORT_ACK = Driver.TritonListener.Backend.udp_server.TritonServerUDP.getfreeport()

        self.sender_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # creates and binds the socket used to send messages
        self.sender_sock.bind((self.server_address[0], self.UDP_PORT_CON))
        self.sender_sock.setblocking(False)

        self.sentmessages = {}
        self.recmsgids = {}

        self.ack_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # creates and binds the socket used to send acks
        self.ack_sock.bind((self.server_address[0], self.UDP_PORT_ACK))
        self.ack_sock.setblocking(False)
        self.threadpoolexecutor = concurrent.futures.ThreadPoolExecutor(max_workers=4)  # used to multitask receive()
        self.recdaemon = threading.Thread(target=self.udp_server_module, daemon=True)  # creates the thread that listens to incoming messages
        self.recdaemon.start()
        self.logger = logging.getLogger('TritonLogger')
    def udp_server_module(self):

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind((self.server_address[0], self.UDP_PORT_REC))
        s.settimeout(SERVER_CONF.UDP_TIMEOUT)
        if Driver.TritonListener.Backend.server_conf.SERVER_CONF.ENCRYPTION:
            fern = Fernet(Driver.TritonListener.Backend.Config.AESKey)
        while self.running:
            try:
                data, addr = s.recvfrom(SERVER_CONF.UDP_BUFFER_SIZE)
            except:
                pass
            else:
                if Driver.TritonListener.Backend.server_conf.SERVER_CONF.ENCRYPTION:
                    data = fern.decrypt(bytes(data))
                msg_type = int.from_bytes([data[0]], byteorder='big', signed=False)
                if msg_type == SERVER_CONF.MSG_ACK:
                    msgid = uuid.UUID(bytes=data[1:17])
                    try:
                        self.sentmessages.pop(msgid)
                    except KeyError:
                        pass  # this means that the ACK packet arrived already, and the key is no longer in the dictionary
                else:
                    msg_port = int.from_bytes(data[1:3], byteorder='big', signed=False)
                    msgid = uuid.UUID(bytes=data[3:19])
                    ackmsg = (SERVER_CONF.MSG_ACK).to_bytes(1, byteorder='big', signed=False) + data[3:19]
                    if Driver.TritonListener.Backend.server_conf.SERVER_CONF.ENCRYPTION:
                        ackmsg = fern.encrypt(bytes(ackmsg))
                    self.ack_sock.sendto(ackmsg, (addr[0], msg_port))

                    if msgid not in self.recmsgids:  # checks for duplicate messages
                        self.recmsgids[msgid] = time.time()
                        if msg_type == SERVER_CONF.MSG_DATA:  #
                            tritonmsg = Driver.TritonListener.Backend.triton_trans.TritonTransmission.unserialize(data[19:])
                            self.threadpoolexecutor.submit(self.receive, tritonmsg[0], tritonmsg[1], tritonmsg[2], tritonmsg[3])

            for msgid in list(self.sentmessages):
                if time.time() - self.sentmessages[msgid][0] > SERVER_CONF.UDP_TIMEOUT_ACK:
                    self.resendmessage(msgid)
            oldmsgs = []
            for msgid in self.recmsgids:
                if time.time() - self.recmsgids[msgid] > SERVER_CONF.UDP_DUPLIC_TIMEOUT:
                    oldmsgs += [msgid]
            for msgid in oldmsgs:
                self.recmsgids.pop(msgid)

        s.close()
        self.logger.debug("TritonServer thread shut down!")

    def __del__(self):
        self.sender_sock.close()
        self.ack_sock.close()
    def shutdown(self):
        super().shutdown()
        self.running = False
        self.recdaemon.join()
        self.threadpoolexecutor.shutdown(True)
    def subscribeTo(self, ip, port):
        msg_data = tt.TritonTransmission.serialize([self.server_address[0], self.server_address[1], self.UDP_PORT_REC])
        try:
            self.sendrequest(ip, port, msg_bytes=msg_data, msg_type=SERVER_CONF.SUB_REQUEST)
            return Driver.TritonListener.Backend.tritonremoteobject.TritonRemoteObject(self, ip, port)
        except:
            self.logger.error("TCP NETWORK ERROR while subscribing: Host "+str(ip)+":"+str(port)+" could not be reached!")
            return None
    def send(self, name, t, ch, val):
        data = Driver.TritonListener.Backend.triton_trans.TritonTransmission.serialize_sub(name, t, ch, val)
        if len(data) > SERVER_CONF.UDP_BUFFER_SIZE:
            self.listlock.acquire()
            for sub in self.send_to_list:
                self.sendrequest(sub[0], sub[1], data)
            self.listlock.release()
        else:
            self.listlock.acquire()
            for sub in self.send_to_list:
                self.udp_send_data(sub[0], sub[2], data)
            self.listlock.release()
    def udp_send_data(self, ip, port, msg_bytes=[]):
        msg_port_bytes = self.UDP_PORT_REC.to_bytes(2, byteorder='big', signed=False)  # the port to which the ACK is sent
        msgid = uuid.uuid4()
        msgid_bytes = msgid.bytes
        msg_type_bytes = SERVER_CONF.MSG_DATA.to_bytes(1, byteorder='big', signed=False)
        complmessage = msg_type_bytes + msg_port_bytes + msgid_bytes + msg_bytes

        if Driver.TritonListener.Backend.server_conf.SERVER_CONF.ENCRYPTION:
            f = Fernet(Driver.TritonListener.Backend.Config.AESKey)
            complmessage = f.encrypt(bytes(complmessage))

        self.sender_sock.sendto(complmessage, (ip, port))
        self.sentmessages[msgid] = [time.time(), (ip, port), complmessage]
        pass

    def resendmessage(self, msgid):
        trans = self.sentmessages[msgid]
        self.sender_sock.sendto(trans[2], trans[1])
        self.sentmessages[msgid][0] = time.time()
        pass