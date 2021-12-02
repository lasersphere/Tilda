"""
Created on 2021-07-01
@author: fsommer
Module Description:  Minimal example of a ZeroMqClient. Basis of this file is a stolen from Wolgang Geitner @ GSI who
basically copied the 0MQ official guide examples
Requires zmq package (pip install pyzmq OR  conda install -c anaconda pyzmq )
"""

#
#   Hello World client in Python
#   Connects REQ socket to tcp://localhost:5555
#   Sends "Hello" to server, expects "World" back
#

import zmq

context = zmq.Context()

#  Socket to talk to server
print("Connecting to hello world server...")
socket = context.socket(zmq.REQ)
socket.connect("tcp://192.168.0.150:5555")

#  Do 10 requests, waiting each time for a response
for request in range(10):
    print("Sending request {} ...".format(request,))
    socket.send_string("Hello")

    #  Get the reply.
    message = socket.recv()
    print("Received reply {} [ {} ]".format(request, message))

# import pickle
# import socket
# from time import sleep
#
# import zmq
# import json
#
#
# class ZeroMqClient:
#
#     def __init__(self, client_type: int, connect_address: str = 'tcp://192.168.0.150:5555', send_timeout_ms: int = 100, receive_timeout_ms: int = 100):
#         zmq_context = zmq.Context()
#         zmq_context.setsockopt(zmq.SNDTIMEO, send_timeout_ms)
#         #zmq_context.setsockopt(zmq.RCVTIMEO, receive_timeout_ms)
#
#         self._sender_name = 'sender:' + socket.gethostname()
#         self._socket = zmq_context.socket(client_type)
#         if client_type == zmq.PUB:
#             print('Connecting PUB socket to address:', connect_address)
#             self._socket.connect(connect_address)
#         elif client_type == zmq.SUB:
#             print('Binding SUB socket to address:', connect_address)
#             self._socket.bind(connect_address)
#
#             self._socket.subscribe('')
#         elif client_type == zmq.PAIR:
#             print('Connecting PAIR socket to address:', connect_address)
#             self._socket.connect(connect_address)
#
#         sleep(0.05)
#
#     def get_socket(self) -> zmq.Socket:
#         return self._socket
#
#     def send_json(self, data: dict, zmq_topic: str):
#         self._socket.send_multipart((zmq_topic.encode(), self._sender_name.encode(), json.dumps(data).encode()))
#
#     def send_raw(self, payload, zmq_topic: str):
#         self._socket.send_multipart((zmq_topic.encode(), self._sender_name.encode(), payload.encode()))
