'''
Created on 21.01.2015

@author: skaufmann
'''


def headunfold(data):
    data32b = format(data, '032b')
    preHd4b = int(data32b[0:4], 2)
    postHd4b = int(data32b[4:8], 2)
    data24b = int(data32b[8:], 2)
    return [preHd4b, postHd4b, data24b]

