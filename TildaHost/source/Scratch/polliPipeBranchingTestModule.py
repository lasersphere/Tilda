"""

Created on '27.08.2015'

@author:'simkaufm'


This module was created to test the Pipeline towards branching with plots.
Currently there is a problem with the references to the figure
"""

import Service.AnalysisAndDataHandling.tildaPipeline as TP



def TestPipe():

    start = TP.Node()

    pipe = TP.Pipeline(start)

    TP.initPipeData()

