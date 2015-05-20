"""
Created on 22.03.2015

@author: dropy
"""

from polliPipe.node import Node


class NAccumulate(Node):
    """
    classdocs
    """

    def __init__(self):
        """
        Constructor
        """
        super(NAccumulate, self).__init__()
        self.type = "Accumulate"

        self.buf = 0

    def processData(self, data, pipeData):
        """
        Add data and
        """
        self.buf += data
        return self.buf
    
    def clear(self):
        self.buf = 0


class NSubtract(Node):
    """
    classdocs
    """

    def __init__(self):
        """
        Constructor
        """
        super(NSubtract, self).__init__()
        self.type = "Subtract"

        self.buf = 0

    def processData(self, data, pipeData):
        """
        Add data and
        """
        self.buf -= data
        return self.buf

    def clear(self):
        self.buf = 0


class NPrint(Node):
    """
    classdocs
    """


    def __init__(self):
        """
        Constructor
        """
        super(NPrint, self).__init__()
        self.type="Output_Print"

        
        
        
    def processData(self, data, pipeData):
        """
        Add data and
        """
        print("PrintNode " + str(self.id) + " item content: "+ str(data))
        return None