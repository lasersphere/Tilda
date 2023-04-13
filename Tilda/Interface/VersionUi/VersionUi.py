"""

Created on '23.09.2015'

@author:'simkaufm'

"""
import os
import re
import logging
from PyQt5 import QtWidgets

from Tilda.Interface.VersionUi.Ui_Version import Ui_Version
import Tilda.Application.Config as tildaCfg


class VersionUi(QtWidgets.QDialog, Ui_Version):
    def __init__(self):
        QtWidgets.QDialog.__init__(self)

        self.setupUi(self)

        self.labelDate.setText(tildaCfg.versiondate)
        self.labelVersion.setText(tildaCfg.version)
        self.labelBranch.setText(tildaCfg.branch)
        self.labelCommit.setText(tildaCfg.commit)

        self.pushButton_whatsNew.clicked.connect(self.open_change_log)

        cur_dir = os.path.dirname(__file__)
        self.chg_log_path = os.path.normpath(
            os.path.join(cur_dir, os.path.pardir, os.path.pardir, 'Application/ReleaseChangeList.txt'))
        self.chg_log_qtext = None
        self.change_log_widg = None
        self.exec_()

    def open_change_log(self):
        if self.chg_log_qtext is None:
            self.chg_log_qtext = ''
            file_str = open(self.chg_log_path, 'r').read()
            # replace some html specific symbols:
            file_str = file_str.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

            self.chg_log_qtext = '<pre>' + file_str + '</pre>'
            # find links look for (http ... )
            link_start = 'http'  # start to look for '(' is not so good :(
            link_start_2 = 'www.'
            link_end = ')'  # end to look for in proximity to the start
            link_starts = [m.start() for m in re.finditer(link_start, self.chg_log_qtext)]
            link_starts += [m.start() for m in re.finditer(link_start_2, self.chg_log_qtext)]
            spaces_after_link_starts = [self.chg_log_qtext.find(' ', start_i + 1) for start_i in link_starts]
            # search until the next space after the link starts
            link_ends = [self.chg_log_qtext.find(link_end, start_i, spaces_after_link_starts[ind])
                         for ind, start_i in enumerate(link_starts)]
            while -1 in link_ends:
                link_ends.remove(-1)  # remove not found indices
            if len(link_starts) == len(link_ends):
                orig_links = [self.chg_log_qtext[l_start:link_ends[i]] for i, l_start in enumerate(link_starts)]
                orig_links = list(set(orig_links))  # might occur twice
                for orig_link in orig_links:
                    new_l = '<a href="%s">%s</a>' % (orig_link, orig_link)
                    self.chg_log_qtext = self.chg_log_qtext.replace(orig_link, new_l)
            else:
                logging.warning('Malformatted external links found in the release log file %s\n'
                                'link start indicator: \'http\' or \'www.\' occured %s times, '
                                'link stopp indicator: \')\' occured %s times'
                                '-> do not match -> no hyperref enabled'
                                % (self.chg_log_path, len(link_starts), len(link_ends)))

            self.change_log_widg = QtWidgets.QTextBrowser(self)
            self.change_log_widg.setOpenExternalLinks(True)
            self.change_log_widg.setText(self.chg_log_qtext)
            self.gridLayout_2.addWidget(self.change_log_widg)
            self.pushButton_whatsNew.setText('Hide news')
            self.resize(800, 800)
        else:
            self.gridLayout_2.removeWidget(self.change_log_widg)
            self.change_log_widg.setParent(None)
            self.change_log_widg = None
            self.chg_log_qtext = None
            self.resize(203, 96)
            self.pushButton_whatsNew.setText('What\'s new?')


if __name__=='__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ui = VersionUi()
    app.exec()
