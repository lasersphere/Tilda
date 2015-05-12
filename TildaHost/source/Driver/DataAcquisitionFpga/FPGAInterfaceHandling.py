"""

Created on '11.05.2015'

@author:'simkaufm'

"""

"""
Module to Wrap all the Handling of universal FPGA interactions, like Start, run etc.

"""
import ctypes

class FPGAInterfaceHandling():
    def __init__(self, bitfilePath, bitfileSignature, resource, reset=True, run=True):
        self.dmaReadTimeout = 1 #timeout to read from Dma Queue in ms
        self.NiFpgaUniversalInterfaceDll = ctypes.CDLL('D:\\Workspace\\Eclipse\\Tilda\\TildaHost\\binary\\NiFpgaUniversalInterfaceDll.dll')
        self.session = ctypes.c_ulong()
        self.statusSuccess = 0
        self.status = 0
        self.InitFpga(bitfilePath, bitfileSignature, resource, reset, run)

    def InitFpga(self, bitfilePath, bitfileSignature, resource, reset=True, run=True):
        """
        Initialize the FPGA with all necessary functions to start an FPGA Session.
        These are: NiFpga_Initialize, NiFpga_Open.
        Optional: NiFpga_Reset, NiFpga_Run
        :param bitfilePath: String Path to the Bitfile
        :param bitfileSignature: String Signature generated when running C Api
        :param resource: Ni-Resource, e.g. Rio0
        :param reset: Bool, if True, Reset the Vi, default is True
        :param run: Bool, if True, Run the FPGA, default is True
        :return: session if no error, else status
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
            return self.status
        return self.session

    def DeInitFpga(self):
        """
        Deinitilize the Fpga with all necessary functions.
        :return: status
        """
        self.StatusHandling(self.NiFpgaUniversalInterfaceDll.NiFpga_Close(self.session, 0))
        self.StatusHandling(self.NiFpgaUniversalInterfaceDll.NiFpga_Finalize())
        return self.status


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
        if self.status >= self.statusSuccess and (self.status == self.statusSuccess or int(newstatus) < self.statusSuccess):
            self.status = int(newstatus)
        return self.status

    def ReadWrite(self, controlOrIndicatorDictionary):
        """
        Function to encapsule the reading and writing of Controls or Indicators in the Fpga Interface.
        The type of val determines which function must be called.
        :param controlOrIndicatorDictionary: dictionary containing: 'ref' the C Api reference number
         of the indicator/control (usually hex number), 'val' the ctypes instance of the Control/Indicator
         and 'ctr' which is True for Controls and False for Indicators 
        :return: val, which is the value passed to the control or the value gained from an indicator
        
        """
        ref = controlOrIndicatorDictionary['ref']
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
                self.StatusHandling(self.NiFpgaUniversalInterfaceDll.NiFpga_ReadU8(self.session, ref, ctypes.byref(val)))
            elif type(val) == ctypes.c_byte:
                self.StatusHandling(self.NiFpgaUniversalInterfaceDll.NiFpga_ReadI8(self.session, ref, ctypes.byref(val)))
            elif type(val) == ctypes.c_ushort:
                self.StatusHandling(self.NiFpgaUniversalInterfaceDll.NiFpga_ReadU16(self.session, ref, ctypes.byref(val)))
            elif type(val) == ctypes.c_short:
                self.StatusHandling(self.NiFpgaUniversalInterfaceDll.NiFpga_ReadI16(self.session, ref, ctypes.byref(val)))
            elif type(val) == ctypes.c_ulong:
                self.StatusHandling(self.NiFpgaUniversalInterfaceDll.NiFpga_ReadU32(self.session, ref, ctypes.byref(val)))
            elif type(val) == ctypes.c_long:
                self.StatusHandling(self.NiFpgaUniversalInterfaceDll.NiFpga_ReadI32(self.session, ref, ctypes.byref(val)))
            elif type(val) == ctypes.c_bool:
                self.StatusHandling(self.NiFpgaUniversalInterfaceDll.NiFpga_ReadBool(self.session, ref, ctypes.byref(val)))
        return val

    def ReadU32Fifo(self, fifoRef, data, nOfEle= -1):
        """
        Reading the Host Buffer which is connected to a Target-to-Host Fifo on the FPGA
        Will read all elements inside the Buffer if nOfEle < 0, default.
        Will read desired number of elements from Fifo, if nOfEle > 0 or timeout if to less data is in buffer.
        Will read no Data, but how many Elements are inside the Queue if nOfEle = 0.

        :param fifoRef: Reference number of the Target-to-Host Fifo as found in hex in the C-Api generated file.
        :param data: ctypes.c_ulong instance, Data from Host will be written in there.
        :param nOfEle: ctypes.c_long instance,
        If nOfEle < 0, everything will be read.
        If nOfEle = 0, no Data will be read.
        If nOfEle > 0, desired number of element will be read or timeout.
        :return:nOfEle = number of Read Elements, data = First Element in data, use data[i] to get the i-th element.
        elemRemainInFifo = number of Elements still in FifoBuffer
        """
        elemRemainInFifo = ctypes.c_long()
        if nOfEle < 0:
            self.StatusHandling(self.NiFpgaUniversalInterfaceDll.NiFpga_ReadFifoU32(
                self.session, fifoRef, ctypes.byref(data), 0, self.dmaReadTimeout, ctypes.byref(elemRemainInFifo)
            ))
            elemRemainInFifo = nOfEle
        self.StatusHandling(self.NiFpgaUniversalInterfaceDll.NiFpga_ReadFifoU32(
            self.session, fifoRef, ctypes.byref(data), nOfEle, self.dmaReadTimeout, ctypes.byref(elemRemainInFifo)
        ))
        return {'nOfEle':nOfEle,'data':data, 'elemRemainInFifo':elemRemainInFifo}


    def ConfigureU32FifoHostBuffer(self, fifoRef, nOfReqEle):
        """
        Function to configure the Size of the Host sided Buffer.
        If Timeouts occur during writing to the Target-to-Host Fifo increasing this size might help.
        If this function is not called, standard is 10000 Elements.
        NI recommends that you increase this buffer to a size multiple of 4,096 elements
        if you run into overflow or underflow errors.
        :param fifoRef: Reference number of the Target-to-Host Fifo as found in hex in the C-Api generated file.
        :param nOfReqEle: ctypes.c_long instance of the number of requested elements.
        :return:elementsAcquired
        """
        elementsDummy = ctypes.byref(ctypes.c_ulong())
        elementsAcquired = ctypes.c_size_t()
        elementsRemainingDummy = ctypes.c_size_t()
        self.StatusHandling(self.NiFpgaUniversalInterfaceDll.NiFpga_AcquireFifoReadElementsU32(
            self.session, fifoRef, elementsDummy, nOfReqEle,
            self.dmaReadTimeout, ctypes.byref(elementsAcquired), ctypes.byref(elementsRemainingDummy)
        ))
        if self.status < self.statusSuccess:
            return self.status
        return elementsAcquired

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
        :return: status
        """
        if nOfEle < 0:
            self.ClearU32FifoHostBuffer(fifoRef, self.ReadU32Fifo(fifoRef, ctypes.c_ulong(), nOfEle)['elemRemainInFifo'])
        self.StatusHandling(self.NiFpgaUniversalInterfaceDll.NiFpga_ReleaseFifoElements(
         self.session, fifoRef, nOfEle
        ))
        return self.status







