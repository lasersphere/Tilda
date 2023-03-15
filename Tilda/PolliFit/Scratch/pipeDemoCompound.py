"""
Created on 29.03.2015

@author: dropy
"""

from Tilda.PolliFit.polliPipe.pipeline import Pipeline
from Tilda.PolliFit.polliPipe.node import Node
from Tilda.PolliFit.polliPipe.simpleNodes import NAccumulate
from Tilda.PolliFit.polliPipe.simpleNodes import NPrint
from Tilda.PolliFit.polliPipe import specialNodes

if __name__ == '__main__':
    start = Node()

    walk = NAccumulate()
    walk.attach(NPrint())
    compound = specialNodes.CompoundNode(walk)

    start.attach(compound)
    branch2 = start.attach(NPrint())
    
    pipe = Pipeline(start)
    pipe.start()
    
    #for i in range(100):
    #    pipe.feed(i)
        
    pipe.clear()
    
    for i in range(10):
        pipe.feed(i)