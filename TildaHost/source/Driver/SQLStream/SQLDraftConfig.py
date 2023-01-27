"""
Created on 2022-12-12

@author: Patrick Mueller

Module Description: Draft config for SQL streams.
Copy and rename to SQLConfig.py in this folder to get complete SQL stream support.
"""

SQL_CFG = 'local'  # use this for testing without a db!

""" Local config """
SQL_CFG = {
    'user': 'root',
    'password': 'root',
    'host': 'localhost',
    'database': 'laspec_data',
}

""" ANL config """
# SQL_CFG = {
#     'user': 'root',
#     'password': 'CLSpec=22!',
#     'host': '146.139.144.78',  # 192.168.108.1
#     'database': 'laspec_data',
# }

EXCLUDE_CHANNELS = {'ID', 'unix_time'}
