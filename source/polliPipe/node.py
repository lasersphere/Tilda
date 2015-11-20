"""
Created on 16.03.2015

@author: dropy
"""
from copy import copy, deepcopy
import logging

import logging

class Node(object):
    """
    A Node performing transformations on items in the Pipeline defined in the processData function
    """


    def __init__(self):
        """
        Create basic attributes and bind node to Pipeline
        """
        self.active = True
        self.id = None
        self.Pipeline = None
        self.next = []

    def processData(self, data, pipeData):
        """
        The action of the node
        """
        return data

    def processItem(self, item):
        """
        rewrapping processData to hide the internals of the transport mechanism
        """
        newjobs = []
        if item.type == "start":
            self.id = item.data['id']
            self.Pipeline = item.data['pipe']
            item.data['id'] += 1
            self.start()
            newjobs = self.createJobs(item, False)
        elif item.type == "stop":
            newjobs = self.createJobs(item, False)
        elif item.type == "clear":
            self.clear()
            newjobs = self.createJobs(item, False)
        elif item.type == "data":
            newData = self.processData(item.data, self.Pipeline.pipeData)
            if newData is not None:
                item.data = newData
                newjobs = self.createJobs(item)
        
        return newjobs

    def createJobs(self, item, docopy=True, inactive=False):
        item.previousId = self.id
        result = []
        for i, n in enumerate(self.next):
            #Process if node is active or inactive nodes are requested as well
            if n.active or inactive:
                if i == 0:
                    result.append((n, item))
                elif not docopy:
                    result.append((n, copy(item)))
                else:
                    result.append((n, deepcopy(item)))
        
        return result

    def clear(self):
        """
        Clear the internal memory. Does nothing for generic node, overwrite!
        """
        pass

    def start(self):
        """
        setup the memories in the nodes. Does nothing for generic node, overwrite!
        """
        pass

    def activate(self):
        """
        activate the node
        """
        self.active = True
        
    def deactivate(self):
        """
        deactivate the Node
        """
        self.active = False
    
    def attach(self, _next):
        """
        attach _next as follower to this node
        """
        self.next.append(_next)
        return _next
