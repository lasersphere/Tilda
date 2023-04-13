"""
This is just a script to test some of the backend functionality without starting Triton
"""

import Tilda.Driver.TritonListener.Backend.udp_server
import time
import Tilda.Driver.TritonListener.Backend.tcp_server
import logging
import numpy
import sys


def udp_test():
    server1 = Tilda.Driver.TritonListener.Backend.udp_server.TritonServerUDP("device1")
    server1.test = testfunction

    server2 = Tilda.Driver.TritonListener.Backend.udp_server.TritonServerUDP("device2")
    server1remote = server2.subscribeTo(server1.myip, server1.UDP_PORT_REC)
    time.sleep(1)
    print(server1remote.test("Dieser hässliche Spagetticode funktioniert auch noch..."))
    arr = numpy.random.rand(1000000)
    server1.send("dev1",time.time(),"ch_big",arr)
    time.sleep(10)


def tcp_test():
    logging.basicConfig(level='DEBUG', format='%(message)s', stream=sys.stdout)

    server1 = Tilda.Driver.TritonListener.Backend.tcp_server.TritonServerTCP("testdev1")
    server2 = Tilda.Driver.TritonListener.Backend.tcp_server.TritonServerTCP("testdev2")

    ip, port = server1.server_address

    server1remote = server2.subscribeTo(ip, port)

    arr = numpy.random.rand(10000000)
    t = time.time()
    server1.send("testdev1TR",time.time(), "ch1", arr)
    print(time.time()-t)
    server2.unsubscribeFromRemoteObject(server1remote)
    time.sleep(2)

    server1.shutdown()

    server2.shutdown()


def testfunction(x):
    print(x)
    return 5


if __name__ == '__main__':
    tcp_test()
    udp_test()
