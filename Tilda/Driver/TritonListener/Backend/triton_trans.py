import pickle
import uuid

'''
These methods are used to convert a Triton message containing python objects into a list of bytes, and back again.
This is implemented using pickle, so the same constrains apply. In theory, even complex classes containing methods can 
be serialized using pickle, at the cost of much higher network traffic. Please note that this is a potential security
risk, since pickle can also serialize program code. For example, a method call could return a manipulated class over
the network, and then executing code on the targeted machine. If securitiy is a concern, activate AES256 encryption in 
server_conf.py
'''

class TritonTransmission:
    @staticmethod
    def serialize(val):
        return pickle.dumps(val, protocol=pickle.HIGHEST_PROTOCOL, fix_imports=False)
    @staticmethod
    def serialize_sub(dev, t, ch, val):
        return pickle.dumps([dev, t, ch, val],protocol=pickle.HIGHEST_PROTOCOL,fix_imports=False)
    @staticmethod
    def unserialize(bytes):
        return pickle.loads(bytes, fix_imports=False)

    @staticmethod
    def serialize_remote_call(funcname, vars):
        return pickle.dumps([funcname, vars], protocol=pickle.HIGHEST_PROTOCOL, fix_imports=False)
    @staticmethod
    def serialize_udp_reply(msgid, val):
        return pickle.dumps([msgid, val], protocol=pickle.HIGHEST_PROTOCOL, fix_imports=False)

'''class which collects and assembles chunks of an UDP transmission that had to be split into multiple packages.'''
class BigUDPTransmission:
    def __init__(self, firstchunk):
        self.datachunks = {}
        self.type = int.from_bytes([firstchunk[19]], byteorder='big', signed=False)
        self.totalnumber = int.from_bytes(firstchunk[24:28],byteorder='big',signed=False)
        self.add_data(firstchunk)
        pass
    def add_data(self, datachunk):
        num = int.from_bytes(datachunk[20:24],byteorder='big',signed=False)
        data = datachunk[44:]
        self.datachunks[num] = data
    def is_complete(self):
        return len(self.datachunks) == self.totalnumber
    def assemble_transmission(self):
        data = []
        for i in range(0,self.totalnumber):
            data += self.datachunks[i]

        return bytearray(data)
    @staticmethod
    def get_big_id_from_bytes(transmission):
        return uuid.UUID(bytes=transmission[28:28+16])