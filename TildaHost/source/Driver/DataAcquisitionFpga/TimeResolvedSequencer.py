"""

Created on '07.05.2015'

@author:'simkaufm'

"""



"""
Module in  charge for loading and accessing the TimeResolvedSequencer
Access Via the NiFpgaUniversalInterfaceDll.dll
"""

from Driver.DataAcquisitionFpga.FPGAInterfaceHandling import FPGAInterfaceHandling
import Driver.DataAcquisitionFpga.TimeResolvedSequencerConfig as TrsCfg
import time


class TimeResolvedSequencer(FPGAInterfaceHandling):
    def __init__(self):
        self.TrsCfg = TrsCfg.TRSConfig()
        session = super(TimeResolvedSequencer, self).__init__(self.TrsCfg.bitfilePath, self.TrsCfg.bitfileSignature, self.TrsCfg.fpgaResource)
        return session
        
    def getMCSState(self):
        self.ReadWrite(self.TrsCfg.MCSstate)
        return self.TrsCfg.MCSstate['val']

    def cmdByHost(self, cmd):
        self.TrsCfg.cmdByHost['val'].value = cmd
        self.ReadWrite(self.TrsCfg.cmdByHost)
        if self.TrsCfg.cmdByHost['val'].value == cmd and self.status == self.statusSuccess:
            return True
        else:
            print('could not send command to Fpga: '+ str(cmd) 
                  + 'status is: ' + str(self.status))
            return False

    def getSeqState(self):
        self.ReadWrite(self.TrsCfg.seqState)
        return self.TrsCfg.seqState['val'].value
    
    def changeSeqState(self, cmd, tries=-1, requestedState=-1):
        maxTries = 10
        waitForNextTry = 0.001
        if tries > 0:
            time.sleep(waitForNextTry)
        curState = self.getSeqState()       
        if curState == cmd:
            return True
        elif tries == maxTries:
            print('could not Change State within ' + str(maxTries)
             + ' tries, Current State is: ' + str(curState))
            return False         
        elif curState in [self.TrsCfg.seqState['measureOffset'], self.TrsCfg.seqState['measureTrack'],
                           self.TrsCfg.seqState['init']]:
            '''cannot change state while measuring or initializing, try again'''
            print('cannot change state while measuring or initializing, trying again...')
            return self.changeSeqState(cmd, tries+1, requestedState)
        elif curState in [self.TrsCfg.seqState['idle'], self.TrsCfg.seqState['measComplete'],
                          self.TrsCfg.seqState['error']]:
            '''send command to change state'''
            if requestedState < 0:
                '''only request to change state once'''
                self.cmdByHost(cmd)
                return self.changeSeqState(cmd, tries+1, cmd)
            else:
                return self.changeSeqState(cmd, tries+1, requestedState)
              
    


# instanciate that bitch:
blub2 = TimeResolvedSequencer()
print('init: ' + str(blub2.__init__()))
print(blub2.getSeqState())
time.sleep(0.1)
print(blub2.getSeqState())
print(blub2.changeSeqState(2))
print(blub2.changeSeqState(4))
print(blub2.getSeqState())


# print(blub2.cmdByHost(5))
