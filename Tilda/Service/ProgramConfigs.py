"""

Created on '27.05.2015'

@author:'simkaufm'

Enum of all available FPGA Programs, beginning with 0, maximum is 255.
For more details see OneNote
"""


from enum import Enum, unique


@unique
class Programs(Enum):
    errorHandler = 0
    simpleCounter = 1
    continuousSequencer = 2
    dac = 3
    infoHandler = 4
