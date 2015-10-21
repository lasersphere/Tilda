"""

Created on '07.05.2015'

@author:'simkaufm'

"""
import logging
import argparse
import sys

from Application.Main.Main import Main

def main():
    """
    main loop of tilda
    """
    # Parser argument
    parser = argparse.ArgumentParser(description='Start Tilda')
    parser.add_argument('--log_level', '-l', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')
    args = parser.parse_args()

    # setup logging
    logging.basicConfig(level=getattr(logging, args.log_level), format='%(message)s', stream=sys.stdout)
    logging.info('Log level set to ' + args.log_level)

    # starting the main loop
    Main()



if __name__ == "__main__":
    main()
