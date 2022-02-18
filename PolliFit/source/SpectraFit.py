"""
Created on 18.02.2022

@author: Patrick Mueller
"""


class SpectraFit:
    def __init__(self, files, db, run, guess_offset=True, x_in_freq=True, save_ascii=False, fmt='k.', font_size=10):
        self.files = files
        self.db = db
        self.run = run
        self.guess_offset = guess_offset
        self.x_in_freq = x_in_freq
        self.save_ascii = save_ascii
        self.fmt = fmt
        self.font_size = font_size
