"""

Created on '23.09.2015'

@author:'simkaufm'



File for global Configs of Tilda
"""

version = '1.25'
versiondate = '02.12.2021'
branch = 'not found'  # will be filled dynamically
commit = 'not found'  # will be filled dynamically
approved_by = 'Felix Sommer fsommer@ikp.tu-darmstadt.de, Laura Renth lrenth@ikp.tu-darmstadt.de'
config_dir = ''
_main_instance = None
# _main_instance is a global variable to store the main instance and make it accessible from everywhere.
# https://docs.python.org/3.4/faq/programming.html#how-do-i-share-global-variables-across-modules
# should be best practice to store the main instance in an accessible .py
