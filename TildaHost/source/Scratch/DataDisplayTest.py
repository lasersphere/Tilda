"""
Created on 19/04/2016

@author: sikaufma

Module Description:

Testmodule for displaying a simple Set of Data

"""

from Measurement.XMLImporter import XMLImporter
from polliPipe.node import Node
import polliPipe.simpleNodes as SN
from polliPipe.pipeline import Pipeline


import Service.AnalysisAndDataHandling.tildaNodes as TN

dataPath = '.\\exampleData\\sums\\Ni_tipa_032.xml'

spec = XMLImporter(dataPath, False)
print(spec.get_scaler_step_and_bin_num(0))


def test_pipe(filepath):
    start = Node()

    pipe = Pipeline(start)

    pipe.pipeData['pipeInternals'] = {}

    pipe.pipeData['pipeInternals']['activeXmlFilePath'] = filepath
    walk = start.attach(SN.NPrint())
    # walk = walk.attach(TN.NMPLImagePlotSpecData(0, dataPath))
    walk = walk.attach(TN.NMPLImagePlotSpecData(0))
    walk = walk.attach(TN.NMPlDrawPlot())

    return pipe

if spec.seq_type in ['trs', 'tipa']:
    pipe = test_pipe(dataPath)
    pipe.start()
    print('hello')
    pipe.feed(spec)
    pipe.clear()


