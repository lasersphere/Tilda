from PyQt5.QtCore import QObject, pyqtSlot, pyqtSignal, pyqtBoundSignal
from typing import Dict, Callable, Any, cast, Tuple
from threading import Lock, Event
from warnings import warn
from functools import wraps
from . import ThreadManager


class QtThreadManager(QObject, ThreadManager):
	# func_ex is short for function_execution
	func_ex_signal = pyqtSignal(int)
	func_ex_callables: Dict[int, Callable] = {}
	func_ex_args: Dict[int, Tuple] = {}
	func_ex_kwargs: Dict[int, Dict[str, Any]] = {}
	func_ex_finished: Dict[int, Event] = {}
	func_ex_return: Dict[int, Any] = {}
	func_ex_counter = 0
	func_ex_lock = Lock()

	@pyqtSlot(int)
	def func_ex_slot(self, index: int) -> None:
		with self.func_ex_lock:
			try:
				callable = self.func_ex_callables.pop(index)
			except KeyError:
				warn(
					"QtThreadManager execute_function encountered an index error. The function at index " + str(
						index) + " was not found.")
			else:
				args = self.func_ex_args.pop(index, [])
				kwargs = self.func_ex_kwargs.pop(index, {})
				try:
					self.func_ex_return[index] = callable(*args, **kwargs)
				except Exception as ex:
					warning_message = RuntimeWarning("Error in function executed by QtThreadManager: " + str(ex))
					warning_message.__cause__ = ex
					warn(warning_message)
			finally:
				self.func_ex_finished[index].set()

	def __init__(self):
		super().__init__()
		self.func_ex_signal.connect(self.func_ex_slot)
		# cast(self.func_ex_signal, pyqtBoundSignal).connect(self.func_ex_slot)

	def execute(self, function: Callable) -> Callable:
		"""
		This is a decorator that executes the function in the QThread of
		the QApplication. This is required for any manipulation of Qt Widgets.
		It waits for the result before continuing to allow utilizing the return value.
		:param function: function (or other callable object) to be executed
		:return: return value of the executed function call
		"""

		@wraps(function)
		def execute_function(*args, **kwargs) -> Any:
			with self.func_ex_lock:
				index = self.func_ex_counter
				self.func_ex_callables[index] = function
				self.func_ex_args[index] = args
				self.func_ex_kwargs[index] = kwargs
				self.func_ex_finished[index] = Event()
				self.func_ex_counter += 1
			self.func_ex_signal.emit(index)
			# cast(self.func_ex_signal, pyqtBoundSignal).emit(index)
			# The 'emit' is blocking if the call comes from the same thread, but non-blocking otherwise.
			# Waiting for the Event ensures consistent behavior and makes the return value accessible.
			self.func_ex_finished[index].wait()
			with self.func_ex_lock:
				self.func_ex_finished.pop(index)
				return self.func_ex_return.pop(index, None)

		return execute_function