"""
Created on 29.03.2015

@author: dropy
"""

from Tilda.PolliFit.polliPipe.pipeline import Pipeline
from Tilda.PolliFit.polliPipe.node import Node
from Tilda.PolliFit.polliPipe.simpleNodes import NAccumulate
from Tilda.PolliFit.polliPipe.simpleNodes import NPrint

if __name__ == '__main__':
    start = Node()
    
    walk = start.attach(NAccumulate())
    walk = walk.attach(NPrint())
    branch2 = start.attach(NPrint())
    
    pipe = Pipeline(start)
    pipe.start()
    
    # for i in range(100):
    #    pipe.feed(i)
        
    # pipe.clear()

    for i in range(1000):
        pipe.feed(i)