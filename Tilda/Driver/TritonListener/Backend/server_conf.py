'''this class encapsulates some parameters of the server implementations.'''


class SERVER_CONF:
    TRANS_MODE = "HYB"  # select either TCP, UDP or HYB (for a hybrid mode)

    UDP_BUFFER_SIZE = 4096  # max buffer size (in bytes) before message is split into multiple packets for udp mode
    TCP_BUFFER_SIZE = 1000000  # max size (in bytes) before message is split for tcp mode. This is OS limited, and seems to be arount 1 MB
    TCP_TIMEOUT_CONNECT = 2 # after this time, the connect() method in sendrequest() will raise a timeout exception
    TCP_TIMEOUT_RESPONSE = 30 #after this time, the sendrequest function will no longer wait for a response from the server
    UDP_TIMEOUT = 0.2  # after this time, the socket will timeout, go through the rest of the loop, and come back
    UDP_TIMEOUT_ACK = 0.2  # timeout in seconds to confirm a message. If no ACK is received before it expires, the message is resent.
    UDP_DUPLIC_TIMEOUT = 60  # this is the time interval in which the system accounts for duplicates.

    # optional AES256 encryption mode (only implemented for tcp and hybrid mode at this time)
    ENCRYPTION = False  # this requires an additional entry (AESKey) in Config.py (the existing hmacKey isn't long
    # enough and has the wrong format. Example for a working key:
    # AESKey = b'4viYT-5TLJRS82fRRDtmOP3Vl-I3GRFEL30tbFIky4o=', generated with Fernet.generate_key()

    # the following bytes are codes to signal which type of message is transmitted
    MSG_DATA = 0  # used to send out to subscribers
    MSG_ACK = 1  # used by udp to confirm messages
    MSG_BIG = 2  # only used for udp mode when a message is split up
    SUB_REQUEST = 3  # request a subsription on a server
    SUB_CANCEL = 4  # cancel a subsription
    REMOTE_CALL = 5  # call a method on a server remotely
    MSG_REPLY = 6  # only used for udp mode, tcp replies directly
