"""

Created on '27.08.2015'

@author:'simkaufm'


This module was created to test the Pipeline towards branching with plots.
Currently there is a problem with the references to the figure
"""

import Service.AnalysisAndDataHandling.tildaPipeline as TP
import Service.FolderAndFileHandling as fileHandl
import matplotlib.pyplot as plt

import os

# set paths for test data
rawfiles = [os.path.join('exampleData\\raw', file) for file in os.listdir('exampleData\\raw') if file.endswith('.raw')]
file = 'exampleData\\sums\\cs_sum_Ca_40_000.xml'
workdir = 'C:\\TestData'
# configure scan dictionary
scnd = fileHandl.loadPickle('exampleData\\raw\\cs_Ca_40_track0_000.pipedat')
scnd['pipeInternals']['activeXmlFilePath'] = None
scnd['pipeInternals']['filePath'] = workdir


def TestPipe():

    start = TP.Node()

    pipe = TP.Pipeline(start)
    pipe.pipeData = scnd
    # print(pipe.pipeData)
    # pipe.start()
    # print(pipe.pipeData)
    fig, axes = plt.subplots(3, sharex=True)

    walk = start.attach(TP.TN.NSplit32bData())
    walk = walk.attach(TP.TN.NSortRawDatatoArray())

    # branching works for now by using shallow copy, if the NotImplementedError occures
    # this is realized by a try loop in processItem in node.py
    # if this is removed and deepcopy is used, the pipeline will fail
    branch = walk.attach(TP.TN.NAccumulateSingleScan())

    branch1 = branch.attach(TP.TN.NArithmetricScaler([0]))
    branch1 = branch1.attach(TP.TN.NMPlLivePlot(axes[0], 'single Scan scaler 0'))

    walk = walk.attach(TP.TN.NRemoveTrackCompleteFlag())
    walk = walk.attach(TP.TN.NSumCS())

    walk = walk.attach(TP.TN.NMPlLivePlot(axes[2], 'live sum'))

    return pipe


pipe = TestPipe()
pipe.start()

for f in rawfiles:
    pipe.feed(fileHandl.loadPickle(f))

pipe.clear()