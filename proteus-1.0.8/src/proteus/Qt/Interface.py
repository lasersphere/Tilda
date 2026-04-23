import functools
from threading import Lock
from typing import Dict, Tuple, Callable, Any, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget

from proteus.Connection import Connection
from proteus.Instance import Instance
from proteus.InstanceObject import InstanceObject


class QtInterface(InstanceObject, QWidget):

	def __init__(self, instance: Instance, **kwargs):
		super().__init__(instance, **kwargs)


class WrappedInterface(QtInterface):
	"""
	Extended base class for Interfaces that fixes threading problems.

	A problem with the use of Interfaces is that some reaction functions
	trigger changes in the Interface from a different thread (e.g. a
	Variable update reaction changes the value of a number shown). While
	this type of access is supported by PyQt, it is very slow and will cause
	errors if done with high frequency.

	This is solved by ensuring all changes to the Interface are done from
	the thread that handles the Interface. At the same time, this thread
	must not be used excessively as the user interface is non-responsive
	while any code is being executed in its thread.

	For this purpose, this class offers the wrap method wrapper that moves
	the function it wraps to the interface thread. Code in wrapped functions
	should exclusively contain interface access calls.
	"""

	_access_signal = pyqtSignal(int)
	_accesses: Dict[int, Tuple[Callable, Tuple[Any], Dict[str, Any], Lock]]
	_access_counter: int
	_access_lock: Lock
	_return_values: Dict[int, Any]

	def __init__(self, instance: Instance, **kwargs):
		"""
		This method is to be extended with the setup for subclasses. The call of
		super().__init__() has to happen before any wrapped methods are called.
		Calls to interface elements do not have to be wrapped in the constructor.

		:param inst: Local Instance to use for communication
		:param kwargs: Other keyword arguments are allowed to ensure
			consistency with child classes.
		"""
		super().__init__(instance=instance, **kwargs)
		if hasattr(self._access_signal, 'connect'):
			self._access_signal.connect(self._process_method)
		self._accesses = dict()
		self._access_counter = 0
		self._access_lock = Lock()
		self._return_values = dict()

	@staticmethod
	def wrap(this: Optional['WrappedInterface'] = None) -> Callable[[Callable], Callable]:
		"""
		Execute the wrapped function in the Interface thread.

		Uses the PyQt signal mechanism to execute the function in the thread
		that owns the Interface to avoid errors and execution slowdown when
		the function is called from a different Thread/QThread.
		When wrapping a method, use @wrap().
		When wrapping some other function, use @wrap(self).
		The second one can obviously only be done within a method.
		:param this: Interface that is changed by the wrapped function.
		:return: The actual wrapper.
		"""
		# TODO Type hint of this should be Optional[Self] when Python 3.11 is used.
		def _wrap(method):
			# copy the properties so the result appears as if it was the wrapped function.
			@functools.wraps(method)
			def wrapper(*args, **kwargs):
				# copy the variable to the local namespace, so it can be edited.
				self = this
				if self is None:
					self = args[0]
				# accessing the counter has its own lock to ensure no duplicate ids are used.
				with self._access_lock:
					access = self._access_counter
					self._access_counter += 1
				# This lock is responsible for waiting for the result
				# of the function call in the separate thread
				lock = Lock()
				lock.acquire(blocking=False)
				self._accesses[access] = (method, args, kwargs, lock)
				if hasattr(self._access_signal, 'emit'):  # TODO this is just so that PyCharm shuts up, replace with proper cast()
					# Trigger the processing of the method stored in the previous line.
					self._access_signal.emit(access)
				# Wait for lock release in _process_method
				lock.acquire(blocking=True)
				lock.release()
				return self._return_values.pop(access)
			return wrapper
		return _wrap

	def _process_method(self, access_reference: int) -> None:
		"""
		Executes a function call that was stored under the given id.

		This method is called by receiving the signal that the function call
		with the given id is to be executed. After the execution, the return
		value is stored to be read by the original caller.
		:param access_reference: Identification number of the call to execute.
		:return: None
		"""
		(method, args, kwargs, lock) = self._accesses.pop(access_reference)
		self._return_values[access_reference] = method(*args, **kwargs)
		lock.release()

	def connect(self, device_name: str, variable_name: str) -> Connection:
		# TODO check whether this implementation or the one in ActiveDevice works better and standardize
		#  Also, it is probably preferable to make this the only place that wraps functions.
		#  Maybe that can make the implementation cleaner? It saves on the whole wrapper thing.
		connection = super().connect(device_name, variable_name)
		connection.proc_var_update = self.wrap(self)(connection.proc_var_update)
		return connection
