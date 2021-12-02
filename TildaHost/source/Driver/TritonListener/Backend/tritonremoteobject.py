import Driver.TritonListener.Backend.triton_trans
import Driver.TritonListener.Backend.udp_server
import Driver.TritonListener.Backend.tcp_server
import functools
import logging
import inspect
from Driver.TritonListener.Backend.server_conf import SERVER_CONF

'''this class is given as a dummy object for a remote TritonObject. The class intercepts any method calls, and relays
them over the network. Up to this point, other attributes like variables cannot be accessed, because the implementation
would require changes that impact performance.'''


class TritonRemoteObject:
    def __init__(self, server, ip, port):
        self.__server = server

        self.logger = logging.getLogger('TritonLogger')
        if isinstance(server, Driver.TritonListener.Backend.tcp_server.TritonServerTCP):
            self.__send_msg_to_server_and_wait_for_reply = server.sendrequest
        elif isinstance(server, Driver.TritonListener.Backend.udp_server.TritonServerUDP):
            self.__send_msg_to_server_and_wait_for_reply = server.send_msg_and_wait_for_reply
        else:
            self.logger.error("server is of no known type!")

        self.__remoteport = port
        self.__remoteip = ip
        pass

    def __getattr__(self, item):
        return functools.partial(self._call_remote_, item=item) # this method returns the _call_remote_ method (without calling it), but predefines the item argument.
    def getip(self):
        return self.__remoteip
    def getport(self):
        return self.__remoteport
    def _call_remote_(self, *args, item):
        bits = Driver.TritonListener.Backend.triton_trans.TritonTransmission.serialize_remote_call(item, args)
        return self.__send_msg_to_server_and_wait_for_reply(self.__remoteip, self.__remoteport, msg_bytes=bits, msg_type=SERVER_CONF.REMOTE_CALL)
    # def __setattr__(self, key, value):
    #   print("set attr "+key+" to value "+str(value))
