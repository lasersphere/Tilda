"""
Created on 

@author: simkaufm

Module Description:
"""

from PyQt5 import QtWidgets
import platform
import os
import random

from Interface.DialogsUi.Ui_ScanCompleteDial import Ui_ScanComplete


class ScanCompleteDialUi(QtWidgets.QDialog, Ui_ScanComplete):
    def __init__(self, main_gui):
        self.main_gui = main_gui
        super(ScanCompleteDialUi, self).__init__(main_gui)
        self.setupUi(self)
        self.show()

        self.checkBox.stateChanged.connect(self.cb_clicked)
        self.pushButton.clicked.connect(self.close)
        self.music_is_playing = False

        self.play_music()

    def cb_clicked(self, state):
        if state == 2:  # checked - > do not show in future
            self.main_gui.show_scan_finished_change(False)
        elif state == 0:
            self.main_gui.show_scan_finished_change(True)

    def play_music(self):
        if 'Win' in platform.system():
            import winsound

            if self.music_is_playing:
                winsound.PlaySound(None, 0)
            else:
                winsound.PlaySound(
                    os.path.join(os.path.dirname(__file__), os.pardir, 'Sounds', self.get_random_sound()),
                    winsound.SND_ASYNC)
            self.music_is_playing = not self.music_is_playing

    def get_random_sound(self):
        sound_path = os.path.join(os.path.dirname(__file__), os.pardir, 'Sounds')
        sounds = os.listdir(sound_path)
        return random.choice(sounds)

    def closeEvent(self, *args, **kwargs):
        """ overwrite the close event """
        self.play_music()
        if self.main_gui is not None:
            # tell main window that this window is closed.
            self.main_gui.close_scan_complete_win()






