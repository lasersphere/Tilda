from threading import Lock
from .Instance import Instance
from .Data import VariableReference
from typing import Type, Optional, Dict, Callable, Any

from proteus.Encoder import DatatypeEncoder
from .Utility import DeviceNotFound


class Connection(VariableReference):

	instance: Instance
	_variable_class: Optional[Type[DatatypeEncoder]]
	_data: Dict[str, Any]
	_variable_lock: Lock
	_use_update_callback: bool
	_update_callbacks: Dict[int, Callable]
	_update_lock: Lock

	def __init__(self, instance: Instance, device: str, name: str):
		super().__init__(device=device, name=name, type_id="")
		# type_id behavior is overwritten, so it doesn't really matter
		self.instance = instance
		self._variable_class = None
		self._data = {}
		self._variable_lock = Lock()
		self._use_update_callback = False
		self._update_callbacks = {}
		self._update_lock = Lock()
		try:
			query_reply = instance.send_command("PROPERTY", device, name)
		except DeviceNotFound:
			query_reply = None
		except RuntimeError as ex:
			query_reply = None
			print('error: ' + str(ex))
		self.instance.register_variable_change(self._device_name, self._variable_name, self._added, self._removed)
		try:
			self._added(query_reply[0]["type_id"])
		except TypeError:
			pass  # No reply/not a JSON object, so no variable to connect to.
		except KeyError:
			pass  # reply does not contain a type_id, nothing can be done to fix that.
			# TODO warn

	@property
	def is_connected(self):
		with self._variable_lock:
			return self._variable_class is not None

	@staticmethod
	def class_from_type_id(type_id: str) -> Type[DatatypeEncoder]:
		# This is not an ideal solution, but the alternative was somehow maintaining a list of type_ids and classes
		# that would require some metaclass shenanigans to create automatically and still some user interaction by
		# requiring the classes to be imported
		# Current best idea for an alternative: add a setting to the settings.json that tells proteus about some
		# functions (or similar) that tell about the relevant classes
		from pydoc import locate
		cls_obj = locate(type_id)
		if isinstance(cls_obj, type) and issubclass(cls_obj, DatatypeEncoder):
			return cls_obj
		else:
			raise ValueError("type_id is not a valid class path for a Variable object")

	@property
	def type_id(self) -> str:
		return self._variable_class.id()

	@property
	def value(self) -> Any:
		"""
		Value of the Variable
		:return: Lst known value of the Variable.
		"""
		return self.get()

	@value.setter
	def value(self, value: Any):
		self.set(value)

	def _proc_var_update(self, message):
		self.proc_var_update(message)

	def proc_var_update(self, message):
		with self._variable_lock:
			variable_class = self._variable_class
		old_value = variable_class.data_to_value(self._data)
		self._data = variable_class.message_to_data(message.content, message.data, self._data)
		new_value = variable_class.data_to_value(self._data)
		with self._update_lock:
			if self._use_update_callback:
				callbacks = list(self._update_callbacks.values())
			else:
				callbacks = []
		for callback in callbacks:
			callback(new_value, old_value, message)

	def update(self):
		# It is unclear if this can reasonably be supported going forward. It may require waiting for the periodic of an
		# ActiveDevice to finish, the feasibility of which is not clear.
		# raise NotImplementedError
		pass

	def get(self) -> Any:
		with self._variable_lock:
			try:
				return self._variable_class.data_to_value(self._data)
			except Exception:
				return None
		# TODO maybe a hard update when the variable is on the same instance? Only after local messages are out of the queue!!

	def set(self, value: Any):
		if self._variable_class is None:
			raise DeviceNotFound(f"Target Property {self.variable_name} on Device {self.device_name} not found")
		sent_data = self._variable_class.value_to_data(value, self._data)
		kwargs, data = self._variable_class.data_to_message_args(sent_data)
		reply = self.instance.send_command("WRITE", self._device_name, self._variable_name, data, **kwargs)
		# Maybe process reply? otherwise discard result
		return reply

	def trigger(self, value: Any):
		if self._variable_class is None:
			raise DeviceNotFound(f"Target Property {self.variable_name} on Device {self.device_name} not found")
		sent_data = self._variable_class.value_to_data(value, self._data)
		kwargs, data = self._variable_class.data_to_message_args(sent_data)
		reply = self.instance.send_command("TRIGGER", self._device_name, self._variable_name, data, **kwargs)
		# TODO mechanism for getting a result
		return reply

	def add_update_callback(self, callback: Optional[Callable] = None):
		with self._update_lock:
			if callback is not None:
				new_id = max(self._update_callbacks, default=0) + 1
				# Generate a new unique id. Might benefit from a better system (See general id generation issue)
				# e.g. does not guarantee the id has not been in use before, only that it is larger than all ids
				# currently in use, also not Thread-safe.
				# -> create a counter = itertools.count() and call next(counter). Relies on GIL for thread-safety
				self._update_callbacks[new_id] = callback
				self._use_update_callback = True
				return new_id
			elif len(self._update_callbacks) > 0:
				self._use_update_callback = True

	def remove_update_callback(self, callback_id: Optional[int] = None):
		with self._update_lock:
			if callback_id is not None:
				self._update_callbacks.pop(callback_id)
				if len(self._update_callbacks) == 0:
					self._use_update_callback = False
			else:
				self._use_update_callback = False

	def _added(self, type_id: str):
		if type_id == self._type_id:
			return
		try:
			variable_class = self.class_from_type_id(type_id)

			data = variable_class.setup_data_storage()
			try:
				self.instance.subscribe(self._device_name, self._variable_name, 'UPDATE', self._proc_var_update)
				# print(self.instance.send_command("read", self._device_name, self._variable_name))
				# self.instance.subscribe_variable(self._device_name, self._variable_name, self.proc_var_update)
			except KeyError:
				pass  # for ease of use, ignore the case where we add a registration that does already exist
				#       should have been caught at the beginning anyway
		except Exception as ex:
			# Connecting failed, consider this as not connected
			self.instance.warn(f"The connection at {self.instance.identification} failed to connect to {self._device_name}, {self._variable_name}: {repr(ex)}", ex)
			with self._variable_lock:
				self._variable_class = None
				self._data = {}
				self._type_id = ""
			return
		try:
			query_reply = self.instance.send_command("READ", self._device_name, self._variable_name)
		except RuntimeError:
			with self._variable_lock:
				self._variable_class = variable_class
				self._type_id = type_id
				self._data = data
		else:
			with self._variable_lock:
				self._variable_class = variable_class
				self._type_id = type_id
				self._data = self._variable_class.message_to_data(query_reply.content, query_reply.data, data)

	def _removed(self):
		with self._variable_lock:
			self._variable_class = None
			self._data = {}
		try:
			self.instance.unsubscribe_variable(self._device_name, self._variable_name, self._proc_var_update)
		except KeyError:
			pass  # for ease of use, ignore the case where we remove a registration that does not exist

	def exit(self):
		self._removed()
		self.instance.unregister_variable_change(self._device_name, self._variable_name)
