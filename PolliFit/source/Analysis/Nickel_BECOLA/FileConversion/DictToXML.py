from lxml import etree as ET
import numpy as np
import functools
import os

def xmlFindOrCreateSubElement(parentEle, tagString, value=''):
    """
    TAKEN FROM TILDA
    finds or creates a Subelement with the tag tagString and text=value to the parent Element.
    Try not to use colons in the tagstring!
    :return: returns the SubElement
    """
    if ':' in tagString:  # this will otherwise cause problems with namespace in xml!
        tagString = tagString.replace(':', '.')
    subEle = parentEle.find(tagString)
    if subEle == None:
        ET.SubElement(parentEle, tagString)
        return xmlFindOrCreateSubElement(parentEle, tagString, value)
    # print(dt.now(), ' string conversion started ', tagString, type(value))
    if isinstance(value, np.ndarray):
        # print('numpy conversion started', value.dtype)
        val_str = ''
        if value.dtype == [('sc', '<u2'), ('step', '<u4'), ('time', '<u4'), ('cts', '<u4')]:
            np.savetxt('temp.out', value, fmt=['%d', '%d', '%d', '%d'])
            with open('temp.out', 'r') as f:
                ret = f.readlines()
            for each in ret:
                each = each.replace(' ', ', ').replace('\n', '')
                each = '(' + each + ') '
                val_str += each
            val_str = '[' + val_str + ']'
        elif value.dtype == np.int32:
            np.savetxt('temp.out', value, fmt='%d')
            with open('temp.out', 'r') as f:
                ret = f.readlines()
            for each in ret:
                val_str += '[' + each + ']'
            val_str = '[' + val_str + ']'
        else:
            val_str = str(value)
    else:
        # print('normal str conv')
        val_str = str(value)
    # print(dt.now(), ' string conversion done ', tagString)
    if val_str:
        subEle.text = val_str
        # print(dt.now(), ' subelement text is set.', tagString)
    return subEle


def xmlWriteDict(parentEle, dictionary, exclude=[]):
    """
    TAKEN FROM TILDA
    finds or creates a Subelement with the tag tagString and text=value to the
    parent Element for each key and value pair in the dictionary.
    :return: returns the modified parent Element.
    """
    for key, val in sorted(dictionary.items(), key=str):
        if key not in exclude:
            if isinstance(val, dict):
                xmlWriteDict(xmlFindOrCreateSubElement(parentEle, key), val)
            else:
                xmlFindOrCreateSubElement(parentEle, key, val)
    return parentEle


def writeXMLfromDict(dictionary, filename, tree_name_str):
    """
    filename must be in form name.xml
    """
    root = ET.Element(tree_name_str)
    xmlWriteDict(root, dictionary)
    xml = ET.ElementTree(root)
    xml.write(filename)

def load_xml(filename):
    """
    loads an .xml file and returns it as an lxml.etree.Element
    :return:lxml.etree.Element, Element of loaded File
    """
    parser = ET.XMLParser(huge_tree=True)
    tree = ET.parse(filename, parser)
    elem = tree.getroot()
    return elem

def xml_get_dict_from_ele(element, convert_to_float=False):
    """
    Converts an lxml Element into a python dictionary
    """
    if convert_to_float:
        try:
            # Try to convert to float. If possible return float not string
            element_return = float(element.text)
        except:
            # If conversion fails, return the string
            element_return = element.text
    else:
        element_return = element.text

    return element.tag, dict(map(functools.partial(xml_get_dict_from_ele, convert_to_float=convert_to_float), element)) or element_return
    #return element.tag, dict(map(xml_get_dict_from_ele, element)) or element_return

def readDictFromXML(filename, convert_to_float=False):
    """
    generates an lxml.etree.Element from the file
    and converts it to a numpy array
    :param filename: str, file to be loaded
    :return: numpy array
    """
    loaded = load_xml(filename) #convert to lxml.etree.Element
    dict_from_xml = xml_get_dict_from_ele(loaded, convert_to_float) #convert to python dictionary
    return dict_from_xml

test_dict = {'Z37_Rubidium':
                 {'T_5S0.5_to_5P1.5_D2':
                     {'isotopes':
                         {'Rb87':
                               {'iso_data':
                                    {'iso_I':1,
                                     'iso_mu':2,
                                     'iso_Q':3,
                                     'iso_name':'Rb87'
                                     },
                                'upper_lvl':
                                    {'A':1,
                                     'A_err':0.1,
                                     'B':2,
                                     'B_err':0.2
                                     },
                                'lower_lvl':
                                    {'A':1,
                                     'A_err':0.1,
                                     'B':2,
                                     'B_err':0.2
                                     },
                                'hfs_transitions':{}
                                },
                           'Rb85':
                               {'iso_data':
                                    {'iso_I':1,
                                     'iso_mu':2,
                                     'iso_Q':3,
                                     'iso_name':'Rb85'
                                     },
                                'upper_lvl':
                                    {'A':1,
                                     'A_err':0.1,
                                     'B':2,
                                     'B_err':0.2
                                     },
                                'lower_lvl':
                                    {'A':1,
                                     'A_err':0.1,
                                     'B':2,
                                     'B_err':0.2
                                     },
                                'hfs_transitions':{},
                                'is_reference':'True'
                                }
                        }}}}

#writeXMLfromDict(test_dict, "HFS_Database.xml", "Database")


