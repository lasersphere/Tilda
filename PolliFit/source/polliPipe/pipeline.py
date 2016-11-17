"""
Created on 16.03.2015

@author: dropy
"""

import collections


class Pipeline(object):
    """
    A Pipeline managing the nodes and the items as well as their process through the line
    """

    def __init__(self, first):
        """
        Create attributes with default values
        """
        self.first = first
        self.nextItemId = 0
        self.pipeData = {}

    def start(self):
        """
        prepare the Pipeline for work
        """
        starter = Item(self.nextItemId, "start")
        starter.data = {'id': 0, 'pipe': self}
        self.nextItemId += 1
        self.processItem(starter)

    def stop(self):
        """
        halt the Pipeline
        """
        stopper = Item(self.nextItemId, "stop")
        self.nextItemId += 1
        self.processItem(stopper)

    def feed(self, data):
        """
        Have inp processed by the Pipeline
        """
        item = Item(self.nextItemId, "data")
        self.nextItemId += 1
        item.data = data
        self.processItem(item)

    def clear(self):
        """
        Iterate over nodes and tell them to flush, e.g. clear their internal memory
        """
        item = Item(self.nextItemId, "clear")
        self.nextItemId += 1
        # item.data = pipeData
        self.processItem(item)

    def processItem(self, item):
        """
        Push the item through the Pipeline
        """
        jobs = collections.deque()
        jobs.append((self.first, item))

        while jobs:
            node, item = jobs.popleft()
            newjobs = node.processItem(item)
            jobs.extend(newjobs)

    def getNodeByPos(self, position):
        """
        return the node at position
        """
        pass

    def getNodeById(self, _id):
        """
        Return the node with Id
        """
        pass

    def getId(self, node):
        """
        Return the ID of node
        """
        pass


class Item(object):
    """
    This represents an item going through the data analysis Pipeline. Actual items do not need
    to inherit from this, as they are dynamically bound
    """

    def __init__(self, _id, nType):
        """
        Constructor
        """
        self.id = _id
        self.type = nType
        self.data = None
        self.previous = None
    