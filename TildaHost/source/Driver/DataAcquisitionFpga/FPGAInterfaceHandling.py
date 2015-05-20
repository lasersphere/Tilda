"""

Created on '11.05.2015'

@author:'simkaufm'

"""

"""
Module to Wrap all the Handling of universal FPGA interactions, like Start, run etc.

"""
import ctypes
import sys
import os


class FPGAInterfaceHandling():
    def __init__(self, bitfilePath, bitfileSignature, resource, reset=True, run=True):
        """
        Initiates the Fpga
        :param bitfilePath: String, to the Bitfile created by Labview and C Api generator
        :param bitfileSignature: String, Signature of the Bitfile found in the corresponding header file
        :param resource: String, location of the fpga, like Rio0, Rio1, etc.
        :param reset: Boolean, to chose if you want to reset the fpga on startup, default is True
        :param run: Boolean, to chose if you want to run the fpga on startup, default is True
        :return: None
        """
        self.dmaReadTimeout = 1  # timeout to read from Dma Queue in ms
        self.NiFpgaUniversalInterfaceDll = ctypes.CDLL(
            'D:\\Workspace\\PyCharm\\Tilda\\TildaHost\\binary\\NiFpgaUniversalInterfaceDll.dll')
        self.session = ctypes.c_ulong()
        self.statusSuccess = 0
        self.status = 0
        self.InitFpga(bitfilePath, bitfileSignature, resource, reset, run)


    '''Initializing/Deinitilaizing'''
    def InitFpga(self, bitfilePath, bitfileSignature, resource, reset=True, run=True):
        """
        Initialize the FPGA with all necessary functions to start an FPGA Session.
        These are: NiFpga_Initialize, NiFpga_Open.
        Optional: NiFpga_Reset, NiFpga_Run
        :param bitfilePath: String Path to the Bitfile
        :param bitfileSignature: String Signature generated when running C Api
        :param resource: Ni-Resource, e.g. Rio0
        :param reset: bool, if True, Reset the Vi, default is True
        :param run: bool, if True, Run the FPGA, default is True
        :return: int,  session which is the number of the session
        """

        bitfilePath = ctypes.create_string_buffer(bitfilePath.encode('utf-8'))
        bitfileSignature = ctypes.create_string_buffer(bitfileSignature.encode('utf-8'))
        resource = ctypes.create_string_buffer(resource.encode('utf-8'))
        self.StatusHandling(self.NiFpgaUniversalInterfaceDll.NiFpga_Initialize())
        self.StatusHandling(self.NiFpgaUniversalInterfaceDll.NiFpga_Open(
            bitfilePath, bitfileSignature, resource, 1, ctypes.byref(self.session)
        ))
        if reset:
            self.StatusHandling(self.NiFpgaUniversalInterfaceDll.NiFpga_Reset(self.session))
        if run:
            self.StatusHandling(self.NiFpgaUniversalInterfaceDll.NiFpga_Run(self.session, 0))
        if self.status < self.statusSuccess:
            sys.exit('Initialization of the Bitfile' + str(os.path.split(bitfilePath.value)[1]) +
                     ' on Fpga with ' + str(resource.value) + ' failed, status is: ' + str(self.status))
        else:
            print('Fpga Initialised on ' + str(resource.value) + '. The Session is ' + str(self.session)
                  + '. Status is: ' + str(self.status) + '.')
        return self.session

    def DeInitFpga(self):
        """
        Deinitilize the Fpga with all necessary functions.
        :return: status
        """
        self.StatusHandling(self.NiFpgaUniversalInterfaceDll.NiFpga_Close(self.session, 0))
        self.StatusHandling(self.NiFpgaUniversalInterfaceDll.NiFpga_Finalize())
        return self.status

    '''FPGA Status Handling'''
    def StatusHandling(self, newstatus):
        """
         Conditionally sets the status to a new value. The previous status is
         preserved unless the new status is more of an error, which means that
         warnings and errors overwrite successes, and errors overwrite warnings. New
         errors do not overwrite older errors, and new warnings do not overwrite
         older warnings.
        :param newstatus: new status value that may be set
        :return: the resulting status
        """
        if self.status >= self.statusSuccess and (
                        self.status == self.statusSuccess or int(newstatus) < self.statusSuccess):
            self.status = int(newstatus)
        return self.status

    def checkFpgaStatus(self):
        """
        can be used if you only want to execute something when status is ok.
        :return: bool, True if everything is fine or warning
        """
        if self.status == self.statusSuccess:
            return True
        elif self.status < self.statusSuccess:
            print('Fpga Status Yields Error, Code is: ' + str(self.status))
            return False
        elif self.status > self.statusSuccess:
            print('There is a WarningCode ' + str(self.status) +
                  ' on Session ' + str(self.session.value) + ' .')
            return True

    '''Indicator/Control Operations:'''
    def ReadWrite(self, controlOrIndicatorDictionary, valInput=None):
        """
        Function to encapsule the reading and writing of Controls or Indicators in the Fpga Interface.
        The type of val determines which function must be called.
        :param controlOrIndicatorDictionary: dictionary containing: 'ref' the C Api reference number
         of the indicator/control (usually hex number), 'val' the ctypes instance of the Control/Indicator
         and 'ctr' which is True for Controls and False for Indicators 
        :return: val, which is the value passed to the control or the value gained from an indicator
        
        """
        ref = controlOrIndicatorDictionary['ref']
        if valInput == None:
            val = controlOrIndicatorDictionary['val']
        if valInput != None:
            '''use the dictionary to determine the type of the variable'''
            controlOrIndicatorDictionary['val'].value = valInput
            val = controlOrIndicatorDictionary['val']
        write = controlOrIndicatorDictionary['ctr']
        if write:
            if type(val) == ctypes.c_ubyte:
                self.StatusHandling(self.NiFpgaUniversalInterfaceDll.NiFpga_WriteU8(self.session, ref, val))
            elif type(val) == ctypes.c_byte:
                self.StatusHandling(self.NiFpgaUniversalInterfaceDll.NiFpga_WriteI8(self.session, ref, val))
            elif type(val) == ctypes.c_ushort:
                self.StatusHandling(self.NiFpgaUniversalInterfaceDll.NiFpga_WriteU16(self.session, ref, val))
            elif type(val) == ctypes.c_short:
                self.StatusHandling(self.NiFpgaUniversalInterfaceDll.NiFpga_WriteI16(self.session, ref, val))
            elif type(val) == ctypes.c_ulong:
                self.StatusHandling(self.NiFpgaUniversalInterfaceDll.NiFpga_WriteU32(self.session, ref, val))
            elif type(val) == ctypes.c_long:
                self.StatusHandling(self.NiFpgaUniversalInterfaceDll.NiFpga_WriteI32(self.session, ref, val))
            elif type(val) == ctypes.c_bool:
                self.StatusHandling(self.NiFpgaUniversalInterfaceDll.NiFpga_WriteBool(self.session, ref, val))
        elif not write:
            if type(val) == ctypes.c_ubyte:
                self.StatusHandling(
                    self.NiFpgaUniversalInterfaceDll.NiFpga_ReadU8(self.session, ref, ctypes.byref(val)))
            elif type(val) == ctypes.c_byte:
                self.StatusHandling(
                    self.NiFpgaUniversalInterfaceDll.NiFpga_ReadI8(self.session, ref, ctypes.byref(val)))
            elif type(val) == ctypes.c_ushort:
                self.StatusHandling(
                    self.NiFpgaUniversalInterfaceDll.NiFpga_ReadU16(self.session, ref, ctypes.byref(val)))
            elif type(val) == ctypes.c_short:
                self.StatusHandling(
                    self.NiFpgaUniversalInterfaceDll.NiFpga_ReadI16(self.session, ref, ctypes.byref(val)))
            elif type(val) == ctypes.c_ulong:
                self.StatusHandling(
                    self.NiFpgaUniversalInterfaceDll.NiFpga_ReadU32(self.session, ref, ctypes.byref(val)))
            elif type(val) == ctypes.c_long:
                self.StatusHandling(
                    self.NiFpgaUniversalInterfaceDll.NiFpga_ReadI32(self.session, ref, ctypes.byref(val)))
            elif type(val) == ctypes.c_bool:
                self.StatusHandling(
                    self.NiFpgaUniversalInterfaceDll.NiFpga_ReadBool(self.session, ref, ctypes.byref(val)))
        return val

    '''FIFO / DMA Queue Operations '''
    def ReadU32Fifo(self, fifoRef, nOfEle=-1):
        """
        Reading the Host Buffer which is connected to a Target-to-Host Fifo on the FPGA
        Will read all elements inside the Buffer if nOfEle < 0, default.
        Will read desired number of elements from Fifo, if nOfEle > 0 or timeout if to less data is in buffer.
        Will read no Data, but how many Elements are inside the Queue if nOfEle = 0.

        :param fifoRef: int, Reference number of the Target-to-Host Fifo as found in hex in the C-Api generated file.
        :param nOfEle: int,
        If nOfEle < 0, everything will be read.
        If nOfEle = 0, no Data will be read. Use to get how many elements are in fifo
        If nOfEle > 0, desired number of element will be read or timeout.
        :return: nOfEle = int, number of Read Elements, newData = integer Python Array containing all data that was read
               elemRemainInFifo = int, number of Elements still in FifoBuffer
        """
        elemRemainInFifo = ctypes.c_long()
        if nOfEle < 0:
            # check how many Elements are in Fifo and than read all of them
            return self.ReadU32Fifo(fifoRef, self.ReadU32Fifo(fifoRef, 0)['elemRemainInFifo'])
        newDataCType = (ctypes.c_ulong * nOfEle)()
        self.StatusHandling(self.NiFpgaUniversalInterfaceDll.NiFpga_ReadFifoU32(
            self.session, fifoRef, ctypes.byref(newDataCType), nOfEle, self.dmaReadTimeout,
            ctypes.byref(elemRemainInFifo)
        ))
        return {'nOfEle': nOfEle, 'newData': newDataCType, 'elemRemainInFifo': elemRemainInFifo.value}

    def ConfigureU32FifoHostBuffer(self, fifoRef, nOfReqEle):
        """
        Function to configure the Size of the Host sided Buffer.
        If Timeouts occur during writing to the Target-to-Host Fifo increasing this size might help.
        If this function is not called, standard is 10000 Elements.
        NI recommends that you increase this buffer to a size multiple of 4,096 elements
        if you run into overflow or underflow errors.
        :param fifoRef: Reference number of the Target-to-Host Fifo as found in hex in the C-Api generated file.
        :param nOfReqEle: int, number of requested elements.
        :return:bool, True if Status is ok
        """
        self.StatusHandling(self.NiFpgaUniversalInterfaceDll.NiFpga_ConfigureFifo(self.session, fifoRef, nOfReqEle))
        return self.checkFpgaStatus()

    def ClearU32FifoHostBuffer(self, fifoRef, nOfEle=-1):
        """
        Releases previously acquired FIFO elements.

        The FPGA target cannot read elements acquired by the host. Therefore, the
        host must release elements after acquiring them. Always release all acquired
        elements before closing the session. Do not attempt to access FIFO elements
        after the elements are released or the session is closed.
        :param fifoRef: Reference number of the Target-to-Host Fifo as found in hex in the C-Api generated file.
        :param nOfEle: ctypes.c_long instance,
        If nOfEle < 0, everything will be released.
        If nOfEle >= 0, desired number of element will be released.
        :return:bool, True if Status ok
        """
        if nOfEle < 0:
            #check for number of elements in fifo and than release all of them.
            return self.ClearU32FifoHostBuffer(fifoRef, self.ReadU32Fifo(fifoRef, 0)['elemRemainInFifo'])
        if nOfEle > 0:
            self.StatusHandling(self.NiFpgaUniversalInterfaceDll.NiFpga_ReleaseFifoElements(
                self.session, fifoRef, nOfEle
            ))
        return self.checkFpgaStatus()

    def freeMemory(self, ptr):
        self.NiFpgaUniversalInterfaceDll.freeMemory(ptr)