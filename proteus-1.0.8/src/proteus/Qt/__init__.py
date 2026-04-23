"""
Pre-setup for Qt Interfaces

This file may _NEVER_ contain top-level pyqt imports!
_ANY_ pyqt imports, everywhere, may only happen _after_ setup_qt has been called at least once!
"""
from typing import List, Optional, Callable, cast
from time import sleep


class ThreadManager:
	"""
	Base class for the QtThreadManager

	The only purpose is to have something for syntax checkers to work with.
	It is here instead of in Utility so Utility can have module-level pyqt imports.
	"""

	def execute(self, function: Callable) -> Callable:
		"""
		Function wrapper used to move calls to the manager's thread

		Implementation is in the subclass.
		:param function: function (or other callable object) to be executed
		:return: return value of the executed function call
		"""
		raise NotImplementedError


_manager: Optional[ThreadManager] = None


def setup_qt(args: Optional[List[str]] = None) -> ThreadManager:
	"""
	Prepare the qt main thread and create a ThreadManager object to allow access to the thread
	:return: the manager object
	"""
	global _manager
	if _manager is not None:
		return _manager
	if args is None:
		args = []

	def setup_function(args: Optional[List[str]] = None):
		from PyQt5.QtWidgets import QApplication
		print(args)
		if QApplication.instance() is not None:
			raise RuntimeError('PyQt QApplication already started, this is not how the Proteus default Qt Ui works.')
		if args is None:
			app = QApplication(["Proteus default Qt Ui"])
		else:
			app = QApplication(["Proteus default Qt Ui", *args])
		# We sometimes got errors about Qt platform plugin not having been loaded
		# These lines seem to solve it:
		# https://stackoverflow.com/questions/67895528/how-can-i-fix-the-pyqt5-platfrom-plugin-error#comment120008057_67895528
		import os
		from PyQt5 import __file__ as PyQt5file
		dirname = os.path.dirname(PyQt5file)
		plugin_path = os.path.join(dirname, 'Qt5', 'plugins', 'platforms')
		os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_path
		from .Utility import QtThreadManager
		global _manager
		_manager = QtThreadManager()
		# TODO Event so one can wait for this
		# print('starting application', flush=True)
		app.exec()
		print('application done', flush=True)

	from threading import Thread
	Thread(target=setup_function, args=[args]).start()
	while _manager is None:
		sleep(0.1)
	return cast(ThreadManager, _manager)


def wait_until_windows_closed() -> None:
	from PyQt5.QtWidgets import QApplication
	while any([wnd.isVisible() for wnd in QApplication.topLevelWindows()]):
		sleep(1)
	application = QApplication.instance()
	if application is not None:
		application.exit(0)
