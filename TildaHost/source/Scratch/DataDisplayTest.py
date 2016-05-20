"""
Created on 19/04/2016

@author: sikaufma

Module Description:

Testmodule for displaying a simple Set of Data

"""

import numpy as np

import Service.AnalysisAndDataHandling.tildaNodes as TN
import polliPipe.simpleNodes as SN
from Measurement.XMLImporter import XMLImporter
from polliPipe.node import Node
from polliPipe.pipeline import Pipeline

dataPath = '.\\exampleData\\sums\\Ni_tipa_032.xml'
dataPath = 'C:\Workspace\PyCharm\Tilda\PolliFit\\test\Project\Data\\testTildaTRS.xml'

spec = XMLImporter(dataPath, True)
spec.time_res[0] = spec.time_res[0].astype(np.double)
print(spec.time_res[0][0][0])
spec.time_res[0][ spec.time_res[0]==0 ] = np.nan
print(spec.time_res[0][0][0])
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

if spec.seq_type in ['trs', 'tipa', 'trsdummy']:
    pipe = test_pipe(dataPath)
    pipe.start()
    print('hello')
    pipe.feed(spec)
    pipe.clear()


