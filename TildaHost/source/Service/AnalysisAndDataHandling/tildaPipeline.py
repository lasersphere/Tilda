"""

Created on '20.05.2015'

@author:'simkaufm'

"""

import Service.AnalysisAndDataHandling.tildaNodes as TN
import polliPipe.simpleNodes as SN
from polliPipe.node import Node

from polliPipe.pipeline import Pipeline

def tildapipe():
    start = Node()

    walk = start.attach(TN.NrawFormatToReadable())
    walk = walk.attach(SN.NPrint())

    pipe = Pipeline(start)
    return pipe

