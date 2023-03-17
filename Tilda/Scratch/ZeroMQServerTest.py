"""
Created on 2021-07-01
@author: fsommer
Module Description:  Minimal example of a ZeroMqClient. Basis of this file is a stolen from Wolgang Geitner @ GSI who
basically copied the 0MQ official guide examples
Requires zmq package (pip install pyzmq OR  conda install -c anaconda pyzmq )

You may also have to allow python TCP in the firewall rules at server side, if no messages from the client are received.
"""

#
#   Hello World server in Python
#   Binds REP socket to tcp://*:5555
#   Expects b"Hello" from client, replies with b"World"
#

import time
import zmq

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

# print("Opened Server on: {}".format(address))

while True:
    #  Wait for next request from client
    message = socket.recv()
    print("Received request: {}".format(message))

    #  Do some 'work'
    time.sleep(1)

    #  Send reply back to client
    socket.send_string("World")


# import zmq
# import time
#
#
# class ZeroMqServer:
#     def __init__(self):
#         context = zmq.Context()
#         self._socket = context.socket(zmq.REP)
#         self._socket.bind("tcp://*:5555")
#         self._run = True
#
#     def run(self):
#         while self._run:
#             #  Wait for next request from client
#             message = self._socket.recv()
#             print("Received request: {}".format(message))
#             #  Do some 'work'
#             # time.sleep(0.01)
#             #  Send reply back to client
#             self._socket.send_string("World")
#
#     def stop(self):
#         self._run = False
#
#
# if __name__ == '__main__':
#
#     server = ZeroMqServer()
#     server.run()
