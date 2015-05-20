"""

Created on '20.05.2015'

@author:'simkaufm'

"""

import numpy as np


from polliPipe.node import Node
from Service.Formating import Formatter


class NrawFormatToReadable(Node):
    def __init__(self):
        """
        Constructor
        """
        super(NrawFormatToReadable, self).__init__()
        self.type = "rawFormatToReadable"

        self.buf = np.zeros(500, dtype=np.uint32)

    def processData(self, data, pipeData):
        """
        convert rawData to a readable form
        can data be a list??
        """
        if len(data) > 500:
            self.buf = np.resize(self.buf, len(data))
        for i in data:
            self.buf[i] = Formatter.integerSplitHeaderInfo(data[i])
        return self.buf

    def clear(self):
        self.buf.fill(0)