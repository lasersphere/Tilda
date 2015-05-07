"""

Created on '07.05.2015'

@author:'simkaufm'

"""



"""
Module in  charge for loading and accessing the TRS.dll

"""


from Application.General.GeneralPurpose import GeneralPurpose


GenPurp = GeneralPurpose()
BitFile = GenPurp.ResolveBitfileLocation('SimpleCounter', 'wrapper.c')
print(BitFile)


# dllTRS = ctypes.CDLL()
