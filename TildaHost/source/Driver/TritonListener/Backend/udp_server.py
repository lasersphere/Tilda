import concurrent.futures
import socket
import uuid
import time
import logging
import threading
import math
import Driver.TritonListener.Backend.triton_trans
import Driver.TritonListener.Backend.tritonremoteobject
from Driver.TritonListener.Backend.server_conf import SERVER_CONF
'''This class implements a server using the UDP protocol, and is the base with which every Triton object can
communicate (although its currently not in use, because tcp_sever performs significantly better for high amounts
of data. Even when low latencies are required, the available hyprid mode is probably the better choice. Because this
class is legacy code, it does not support encryption.
'''


class TritonServerUDP:

    def __init__(self, type, parentobj = None):
        self.logger = logging.getLogger('TritonLogger')
        self.logger.warning("starting a pure UDP server. THIS IS LEGACY CODE, USE ONLY FOR DEBUGGING PURPOSES. For a more stable, equally performant option, select 'HYB' mode in 'server_conf.py'")
        if Driver.TritonListener.Backend.server_conf.SERVER_CONF.ENCRYPTION:
            self.logger.warning("ENCRYPTION IS NOT SUPPORTED IN PURE UDP MODE. MESSAGES WILL BE SEND IN CLEAR TEXT!!!")
        self.type = type  # device name
        if parentobj is None:
            self.parentobj = self
        else:
            self.parentobj = parentobj
            self.receive = parentobj._receive
        self.running = True
        self.UDP_PORT_REC = self.getfreeport()  # port used to receive messages
        self.UDP_PORT_CON = self.getfreeport()  # port used to send messages
        self.UDP_PORT_ACK = self.getfreeport()
        self.myip = socket.gethostbyname(socket.gethostname())  # own ip address
        self.uri = "TritonObject_"+str(uuid.uuid4())+"@"+str(self.myip)+":"+str(self.UDP_PORT_REC)

        self.logger.debug("starting device '" + type + "' on " + str(self.myip) + ":" + str(self.UDP_PORT_REC))

        self.sentmessages = {}
        self.recmsgids = {}
        self.replies = {}

        self.sender_sock = socket.socket(socket.AF_INET,
                                         socket.SOCK_DGRAM)  # creates and binds the socket used to send messages
        self.sender_sock.bind((self.myip, self.UDP_PORT_CON))
        self.sender_sock.setblocking(False)

        self.ack_sock = socket.socket(socket.AF_INET,
                                      socket.SOCK_DGRAM)  # creates and binds the socket used to send acks
        self.ack_sock.bind((self.myip, self.UDP_PORT_ACK))
        self.ack_sock.setblocking(False)

        self.send_to_list = []  # the list of subscribers. These [IP,PORT] will receive messages from send()
        self.listlock = threading.RLock()  # object to circumvent errors due to multiple threads writing/reading to send_to_list
        self.threadpoolexecutor = concurrent.futures.ThreadPoolExecutor(max_workers=4)  # used to multitask receive()

        self.recdaemon = threading.Thread(target=self.server_module,
                                          daemon=True)  # creates the thread that listens to incoming messages
        self.recdaemon.start()

        pass

    @staticmethod
    def getfreeport():
        s = socket.socket()
        s.bind(('', 0))
        port = s.getsockname()[1]
        s.close()  # slight race condition
        return port

    def __del__(self):
        self.sender_sock.close()
        self.ack_sock.close()

    def shutdown(self):
        self.running = False
        self.recdaemon.join()
        self.threadpoolexecutor.shutdown(True)

    def _remote_call(self,tritoncall, addr, msg_port, msgid):
        reply = None
        try:
            callfunc = getattr(self.parentobj, tritoncall[0])
            reply = callfunc(*tritoncall[1])
        except Exception as err:
            self.logger.error("method " + tritoncall[0] + " could not be called or does not exist: " + str(err))
        repbytes = Driver.TritonListener.Backend.triton_trans.TritonTransmission.serialize_udp_reply(msgid, reply)
        self.send_msg_to(addr[0], msg_port, repbytes, SERVER_CONF.MSG_REPLY)
        pass
    def _emit(self):
        pass #placeholder
    def server_module(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind((self.myip, self.UDP_PORT_REC))
        s.settimeout(SERVER_CONF.UDP_TIMEOUT)
        bigtransms = {}

        while self.running:
            try:
                data, addr = s.recvfrom(SERVER_CONF.UDP_BUFFER_SIZE)
            except:
                data = None
            if data is not None:
                msg_type = int.from_bytes([data[0]], byteorder='big', signed=False)
                msg_port = int.from_bytes(data[1:3], byteorder='big', signed=False)

                if msg_type == SERVER_CONF.MSG_ACK:
                    msgid = uuid.UUID(bytes=data[1:17])
                    try:
                        self.sentmessages.pop(msgid)
                    except KeyError:
                        pass  # this means that the ACK packet arrived already, and the key is no longer in the dictionary

                    #print(self.type + ": successful transmission was confirmed! " + str(len(self.sentmessages)) + " remain unconfirmed!")

                else:
                    msgid = uuid.UUID(bytes=data[3:19])

                    ackmsg = (SERVER_CONF.MSG_ACK).to_bytes(1, byteorder='big', signed=False) + data[3:19]
                    self.ack_sock.sendto(ackmsg, (addr[0], msg_port))

                    if msgid not in self.recmsgids:  # checks for duplicate messages
                        self.recmsgids[msgid] = time.time()
                        if msg_type == SERVER_CONF.MSG_DATA:  #
                            tritonmsg = Driver.TritonListener.Backend.triton_trans.TritonTransmission.unserialize(data[19:])
                            self.threadpoolexecutor.submit(self.receive, tritonmsg[0], tritonmsg[1], tritonmsg[2],
                                                           tritonmsg[3])
                        elif msg_type == SERVER_CONF.MSG_BIG:
                            bigid = Driver.TritonListener.Backend.triton_trans.BigUDPTransmission.get_big_id_from_bytes(data)

                            if bigid in bigtransms:
                                bigtransms[bigid].add_data(data)
                            else:
                                bigtransms[bigid] = Driver.TritonListener.Backend.triton_trans.BigUDPTransmission(data)

                            if bigtransms[bigid].is_complete():
                                complbigtrans = bigtransms[bigid]
                                assembled_trans = complbigtrans.assemble_transmission()
                                tritonmsg = Driver.TritonListener.Backend.triton_trans.TritonTransmission.unserialize(assembled_trans)
                                if complbigtrans.type == SERVER_CONF.MSG_DATA:
                                    self.threadpoolexecutor.submit(self.receive, tritonmsg[0], tritonmsg[1],tritonmsg[2], tritonmsg[3])
                                elif complbigtrans.type == SERVER_CONF.REMOTE_CALL:
                                    self._remote_call(tritonmsg, addr, msg_port, msgid)
                                elif complbigtrans.type == SERVER_CONF.MSG_REPLY:
                                    self.replies[tritonmsg[0]] = tritonmsg[1]
                                del bigtransms[bigid]

                        elif msg_type == SERVER_CONF.REMOTE_CALL:
                            tritoncall = Driver.TritonListener.Backend.triton_trans.TritonTransmission.unserialize(data[19:])
                            self._remote_call(tritoncall,addr,msg_port, msgid)
                        elif msg_type == SERVER_CONF.MSG_REPLY:
                            reply = Driver.TritonListener.Backend.triton_trans.TritonTransmission.unserialize(data[19:])
                            self.replies[reply[0]] = reply[1]
                        elif msg_type == SERVER_CONF.SUB_REQUEST:  #
                            sub_ip = socket.inet_ntoa(data[19:23])
                            sub_port = int.from_bytes(data[23:25], byteorder='big', signed=False)
                            self.listlock.acquire()
                            try:
                                self.send_to_list.index([sub_ip, sub_port])
                                self.logger.debug(str(sub_ip) + ":" + str(
                                    sub_port) + " is already subscribed, why would you request it again...?")
                            except:
                                self.send_to_list += [[sub_ip, sub_port]]
                                self.logger.debug("Backend: Subscription request from " + str(sub_ip) + ":" + str(sub_port))
                            self.listlock.release()
                            self.parentobj._emit()
                        elif msg_type == SERVER_CONF.SUB_CANCEL:  #
                            sub_ip = socket.inet_ntoa(data[19:23])
                            sub_port = int.from_bytes(data[23:25], byteorder='big', signed=False)

                            self.listlock.acquire()
                            try:
                                index = self.send_to_list.index([sub_ip, sub_port])
                                del self.send_to_list[index]
                                self.logger.debug(str(sub_ip) + ":" + str(sub_port) + "has unsubscribed.")
                            except:
                                self.logger.error(
                                    str(sub_ip) + ":" + str(
                                        sub_port) + "has tried to unsubscribe without being subscribed first...?")
                            self.listlock.release()

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

    def receive(self, dev, t, ch, val):
        print(self.type + ": " + str(t) + ": received a message from '" + str(dev) + "'on channel '" + str(ch) + "' that reads: "+str(len(val)))
        #dummy function again, gets overwritten unless no TritonObject is specified as parent
        pass

    def send(self, name, t, ch, val):
        self.listlock.acquire()
        sublist = self.send_to_list.copy()
        self.listlock.release()
        data = Driver.TritonListener.Backend.triton_trans.TritonTransmission.serialize_sub(name, t, ch, val)
        for sub in sublist:
            self.send_msg_to(sub[0], sub[1], data)
        pass

    def subscribeTo(self, ip, port):
        msg_data = socket.inet_aton(self.myip) + int(self.UDP_PORT_REC).to_bytes(2, byteorder='big', signed=False)
        self.send_msg_and_wait_for_ack(ip, port, msg_bytes=msg_data, msg_type=SERVER_CONF.SUB_REQUEST)
        return Driver.TritonListener.Backend.tritonremoteobject.TritonRemoteObject(self, ip, port)

    def subscribeToUri(self, uri):
        spl = uri.split('@')
        spll = spl[1].split(':')
        ip = str(spll[0])
        port = int(spll[1])
        return self.subscribeTo(ip, port)

    def unsubscribeFrom(self, ip, port):
        msg_data = socket.inet_aton(self.myip) + int(self.UDP_PORT_REC).to_bytes(2, byteorder='big', signed=False)
        self.send_msg_and_wait_for_ack(ip, port, msg_bytes=msg_data, msg_type=SERVER_CONF.SUB_CANCEL)
        pass

    def unsubscribeFromRemoteObject(self, remobj):
        self.unsubscribeFrom(remobj.getip(), remobj.getport())
        pass

    def send_msg_to(self, ip, port, msg_bytes=[], msg_type=SERVER_CONF.MSG_DATA, msgid = None):
        overhead = 19#19 bytes are overhead (type, port, id)

        msg_type_bytes = msg_type.to_bytes(1, byteorder='big', signed=False)
        msg_port_byte = self.UDP_PORT_REC.to_bytes(2, byteorder='big', signed=False)  # the port to which the ACK is sent
        if msgid is None:
            msgid = uuid.uuid4()
        msgid_bytes = msgid.bytes


        if len(msg_bytes)+overhead > SERVER_CONF.UDP_BUFFER_SIZE:
            msg_type_big_bytes = SERVER_CONF.MSG_BIG.to_bytes(1, byteorder='big', signed=False)
            overhead_big = overhead+1+8+16# 25 bytes additional overhead for type, packet number, total number of packets, and uuid
            bigmessageuuid = uuid.uuid4()
            bigmessageuuidbytes = bigmessageuuid.bytes #for identification of the overall transmission (used to reassemble the message)
            totalmessages = math.ceil(len(msg_bytes) / (SERVER_CONF.UDP_BUFFER_SIZE - overhead_big))
            #print("splitting message into "+str(totalmessages)+" fragments")
            totalmessages_bytes = totalmessages.to_bytes(4, byteorder='big', signed=False)
            for msg_num in range(0, totalmessages):
                msg_num_bytes = msg_num.to_bytes(4, byteorder='big', signed=False)

                splitmsgbytes = msg_bytes[msg_num*(SERVER_CONF.UDP_BUFFER_SIZE - overhead_big): (msg_num + 1) * (SERVER_CONF.UDP_BUFFER_SIZE - overhead_big)]
                complmessage = msg_type_big_bytes+msg_port_byte+msgid_bytes+msg_type_bytes+msg_num_bytes+totalmessages_bytes+bigmessageuuidbytes+splitmsgbytes
                self.sender_sock.sendto(complmessage, (ip, port))
                self.sentmessages[msgid] = [time.time(), (ip, port), complmessage]
                msgid = uuid.uuid4()
                msgid_bytes = msgid.bytes

        else:
            complmessage = msg_type_bytes + msg_port_byte + msgid_bytes + msg_bytes

            self.sender_sock.sendto(complmessage, (ip, port))
            self.sentmessages[msgid] = [time.time(), (ip, port), complmessage]
        pass

    def send_msg_and_wait_for_reply(self, ip, port, msg_bytes=[], msg_type=0, timeout = 5):
        msgid = uuid.uuid4()
        self.send_msg_to(ip,port,msg_bytes,msg_type,msgid)
        stt = time.time()
        while msgid not in self.replies:
            time.sleep(0.0001)
            if time.time() - stt > timeout:
                self.logger.error("No reply was recieved within the specified timeout of" + str(timeout) + " seconds")
                return None
        rep =  self.replies[msgid]
        del self.replies[msgid]
        #print("f2")
        return rep

    def send_msg_and_wait_for_ack(self, ip, port, msg_bytes=[], msg_type=0, timeout = 1): #this is intended for small
        #messages that need to be confirmed before the program can continue, e.g. subscriptions. It will NOT work with
        #messages that carry more data than BUFFER_SIZE
        msgid = uuid.uuid4()
        self.send_msg_to(ip, port, msg_bytes, msg_type, msgid)
        stt = time.time()
        while msgid in self.sentmessages:
            time.sleep(0.0001)
            if time.time()-stt > timeout:
                self.logger.warning("message was not confirmed within the specified timeout of "+str(timeout)+" seconds")
                break
        #print("f1")
        return

    def resendmessage(self, msgid):
        trans = self.sentmessages[msgid]
        self.sender_sock.sendto(trans[2], trans[1])
        self.sentmessages[msgid][0] = time.time()
        pass
