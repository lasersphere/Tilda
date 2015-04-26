"""
Created on 29.03.2015

@author: dropy
"""

from polliPipe.pipeline import Pipeline
from polliPipe.node import Node
from polliPipe.simpleNodes import NAccumulate
from polliPipe.simpleNodes import NPrint
from polliPipe import specialNodes

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