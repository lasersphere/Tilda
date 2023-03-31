"""
Created on '31.03.2023'

@author: Patrick Mueller

The config directory is set at the very beginning, since it is already needed when importing some modules.
The log-level argument will only be processed during the main routine in TildaStart.py.
"""

import argparse

import Tilda.Application.Config as Cfg
from Tilda.Service.FileOperations.FolderAndFileHandling \
    import get_default_config_dir, make_config_dir, check_config_dir


parser = argparse.ArgumentParser(description='Start Tilda')
parser.add_argument('--config_dir', '-d', type=check_config_dir)
parser.add_argument('--log_level', '-l', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')
args = parser.parse_args()
Cfg.config_dir = args.config_dir if args.config_dir else get_default_config_dir()
make_config_dir(Cfg.config_dir)


# Delete all local variables
del argparse, Cfg, get_default_config_dir, make_config_dir, check_config_dir, parser, args
