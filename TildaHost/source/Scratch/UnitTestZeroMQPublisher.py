"""
Created on 2021-07-01
@author: fsommer
Module Description:  Minimal example of a ZeroMqClient. Basis of this file is a stolen from Wolgang Geitner @ GSI who
basically copied the 0MQ official guide examples
Requires zmq package (pip install pyzmq OR  conda install -c anaconda pyzmq )
"""

import os
import socket
import timeit
import unittest
import numpy
import zmq

from Scratch.ZeroMQClientTest import ZeroMqClient
from matplotlib import pyplot as plt

connection_ip = '192.168.0.150'
connection_port = 5555
performance_loops = 10

host_name = socket.gethostname()
local_ip = socket.gethostbyname(host_name)

class ZeroMqClientTester(unittest.TestCase):

    def test_send_message(self):
        connection_uri = connection_ip + ':' + str(connection_port)
        receiver_uri = local_ip + ':' + str(connection_port + 1)
        client = ZeroMqClient(zmq.PUB, 'tcp://' + connection_uri, receive_timeout_ms=1000, send_timeout_ms=1000)
        client.send_raw('zmq://localhost/myZeroMQPerformanceTestProxy_UpdateTest2_Msg|1|123', 'decpc008')
        receiverClient = ZeroMqClient(zmq.SUB, 'tcp://' + receiver_uri, receive_timeout_ms=1000,
                                                    send_timeout_ms=1000)
        print('waiting for acknowledge...')
        result = receiverClient.get_socket().recv()
        receiverClient.get_socket().close()
        print('Server reply:', result)
        self.assertEqual(result, b'202')

    def test_send_json(self):
        payload = dict()

        data1 = dict()
        data1["parameter"] = "name1"
        data1["type"] = "double"
        data1['length'] = 1
        data1['value'] = [0.00123]
        data1['url'] = "zmq://abc"

        data2 = dict()
        data2["parameter"] = "name2"
        data2["type"] = "int"
        data2['length'] = 4
        data2['value'] = [1,2,3,5]
        data2['url'] = "zmq://def"

        payload["data"] = [data1, data2]
        connection_uri = connection_ip + ':' + str(connection_port)
        client = ZeroMqClient(zmq.PUB, 'tcp://' + connection_uri, receive_timeout_ms=1000, send_timeout_ms=1000)
        client.send_json(payload, 'decpc008')
        receiver_uri = local_ip + ':' + str(connection_port + 1)
        receiverClient = ZeroMqClient(zmq.SUB, 'tcp://' + receiver_uri, receive_timeout_ms=1000,
                                                    send_timeout_ms=1000)
        print('waiting for acknowledge...')
        result = receiverClient.get_socket().recv()
        receiverClient.get_socket().close()
        print('Server reply:', result)
        self.assertEqual(result, b'202')

    def test_performance(self):
        connection_uri = connection_ip + ':' + str(connection_port)
        print(connection_uri)
        client = ZeroMqClient(zmq.PUB, 'tcp://' + connection_uri, receive_timeout_ms=1000, send_timeout_ms=1000)
        receiver_uri = connection_ip + ':' + str(connection_port + 1)
        durations = numpy.ndarray([performance_loops])
        receiverClient = ZeroMqClient(zmq.SUB, 'tcp://' + receiver_uri, receive_timeout_ms=1000,
                                                    send_timeout_ms=1000)

        for i in range(performance_loops):
            start_time = timeit.default_timer()
            theString = 'perf' + str(i)
            client.send_raw(b'test', 'decpc008')
            print('waiting for acknowledge...')
            result = receiverClient.get_socket().recv()
            print('Server reply:', result)
            self.assertEqual(result, b'202')
            duration = timeit.default_timer() - start_time
            durations[i] = duration

        client.get_socket().disconnect('tcp://' + connection_uri)
        print('Mittelwert:', durations.mean())
        print('Max:', durations.max(0, initial=0.0))
        print('Variance:', durations.var(0))

        plt.plot(durations)
        plt.show()
