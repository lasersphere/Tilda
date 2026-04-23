"""
Contains classes for the development of Devices.

Devices organize Variables into groups that concern common physical
components. Such Variables often can not be handled by independent code
as their operations influence each other. In this case, it is
recommended to place the shared code in a suitable governing Device.

Device: Base class for Devices that implements communication functions.
ActiveDevice: Extension of Device that supports code extensions better.
"""
import types
from abc import abstractmethod

from queue import Queue, Empty
from threading import Thread, Lock, Event, current_thread
from typing import List, Union, Callable, Dict, Any, Tuple, Optional

from .Connection import Connection
from .Instance import Instance
from .MessageProcessor import MessageProcessor, StoredCommand
from .Utility import create_reply, RawMessage, ReplyFunction, ReactionFunction, ProteusError, ProteusUserCodeError
from .Data import DeviceReference, VariableReference
from .InstanceObject import InstanceObject


class Device(InstanceObject, MessageProcessor, DeviceReference):
	"""
	Base class for Device implementation.

	This class is the base class for the implementation of Devices. It
	handles all communication aspects of devices, but has no support for
	code that runs continuously. For this reason, it is only used as base
	for further extensions or very simple, purely reactive devices.

	Properties:
		Instance: The Instance the Device is managed by.
	None of the following properties can be edited (operations can be
	changed indirectly by methods).
		device_name: Name of the Device represented by this access point.
		variable_name: Variable name of the access point (always the
			empty string).
		type_id: Identifier for the type of device.
		operations: Valid operations for this access point.

	Methods:
		add_command: Inherited but should not be used.
		add_command_2: Add a command to the list of valid commands for the
			Device and provide the correct reaction.
		remove_command: Inherited but should not be used.
		remove_command_2: Remove a command from the list of valid commands
			for the Device.
		add_variable: Add a VariableReference to the ones of this Device.
		remove_variable: Remove a VariableReference from this Device.
		add_variable_command_2: Add a command with callback to a Variable.
		remove_variable_command_2: Remove a command from a Variable.
		publish: Send an update about the Device to all subscribers.
		publish_variable: Send an update about a Variable of the Device.

	A subclass needs to reimplement the following method:
		status: Get a user-comprehensible status message.
	If necessary, it can also reimplement the following methods:
		_setup: Initialization for new base classes.
		setup: Initialization for actually used classes.
	"""

	_setup_complete = False
	_registered_replies: dict[tuple[str, str], ReplyFunction]
	_registered_reactions: dict[tuple[str, str], ReactionFunction]
	_registered_functions: dict[tuple[str, str], tuple[Callable[[RawMessage], Tuple[RawMessage, Any]], Callable[[RawMessage, Any], Any]]]
	_status: str
	property_descriptor_list: dict[str, Callable] = {}

	@abstractmethod
	def __init__(self, instance: Instance, name: str, **kwargs):
		"""
		A Device requires an Instance that implements a communication protocol
		to work (e.g. a ZeroMQInstance). The name of the Device needs to be
		unique in the network.

		When subclassing the Device class, these parameters and their position
		as first and second arguments of the constructor need to be preserved.
		This is required for the creation of Devices from configuration files.

		:param instance: Complete BaseInstance subclass object that is to
			manage the Device.
		:param name: Unique identifier for the Device.
		"""
		super().__init__(instance, name=name, **kwargs)
		self._registered_functions = {}
		self._registered_replies = {}
		self._registered_reactions = {}
		self._status = ''
		try:
			self._instance.add_device(self)
		except KeyError as ex:
			self._instance.warn(f"Device {name!s} could not be added to the Instance at {self._instance.name}. Using the device is not recommended.", ex)
		else:
			self._instance.trigger_variable_added_listener(self.device_name, "", self.type_id)

		# To allow easy access to basic variables and triggers, they can be
		# created by a PropertyDescriptor that is replaced by the appropriate
		# Variable at this point. The descriptor is callable, returning the
		# Variable object.
		for descriptor_name, descriptor in self.get_property_descriptors().items():
			var = descriptor(self)
			if getattr(self, descriptor_name) is descriptor:
				setattr(self, descriptor_name, var)

	def add_command(self, name: str, variable: str = "", reply: ReplyFunction = None, reaction: Optional[ReactionFunction] = None) -> None:
		"""
		Add a command for a Variable belonging to the Device.

		Register a command for a Variable that belongs to the Device and a
		callback function for the command. When a message with the command that
		is addressed to the Variable arrives, the callback will be executed with
		the message as argument.
		The name is chosen for consistency with add_command_2.
		:param variable: Name of the Variable to register the command for.
		:param name: Identifier (usually name) of the command to register.
		:param reply: Non-optional callback function for the command.
		:return: True if the command was added successfully, otherwise False
		"""
		self._registered_functions[(variable, name)] = (reply, reaction)
		self._registered_replies[variable, name] = reply
		if reaction:
			self._registered_reactions[variable, name] = reaction

	def remove_command(self, name: str, variable: str = "") -> None:
		"""
		Remove a command for a variable belonging to the Device.

		Unregister a command of the specified Variable of the Device. Any
		associated callbacks will be removed.
		The name is chosen for consistency with add_command_2.
		:param variable: Name of the Variable to remove the command from.
		:param name: Identifier (usually name) of the command to remove.
		:return: True if the command was removed successfully, otherwise False.
		"""
		try:
			del self._registered_replies[variable, name]
		except KeyError:
			pass
		try:
			del self._registered_reactions[variable, name]
		except KeyError:
			pass
		del self._registered_functions[(variable, name)]
		# self.instance.remove_command(name, device=self._device_name, variable=variable)

	def publish(self, value: Union[str, List[str]], data: List[bytes] = None) -> None:
		"""
		Publish an update for the Device to all its subscribers.

		This method sends an update message concerning the Device to all
		subscribers of the Device. The interpretation of Device updates is not
		standardized.
		:param value: Content of the update
		:return: None
		"""
		if isinstance(value, str):
			value = [value]
		self._instance.publish_variable(self.device_name, "", data=data, value=value)

	def add_variable(self, variable: VariableReference) -> None:
		super().add_variable(variable)
		self._instance.trigger_variable_added_listener(self.device_name, variable.variable_name, variable.type_id)

	def publish_variable(self, variable: str, data: List[bytes] = None, **kwargs) -> None:
		"""
		Publish an update for a Variable that belongs to the Device.

		Send a message that contains a value update to all subscribers of the
		variable.
		:param variable: Name of the Variable the update concerns.
		:param value: JSON-compatible form of the updated value.
		:return: None
		"""
		self.instance.publish_variable(self.device_name, variable, data=data, **kwargs)

	def exit(self):
		self.instance.remove_device(self)

	@property
	def status(self):
		return self._status

	@status.setter
	def status(self, new_status):
		self._status = str(new_status)

	def process_command(self, command: StoredCommand):
		message = command.message
		try:
			property_name = message.content['property']
			command_name = message.content['command']
		except KeyError:
			reply = create_reply(message, error='malformed message')
		else:
			if property_name == "":
				# accessing device method
				reply = create_reply(message, error='no device commands yet')
			else:
				try:
					function = self._registered_replies[property_name, command_name]
				except KeyError:
					reply = create_reply(message, error='no reply function')
				else:
					reply = function(message)
		if command.address_info is None:
			return reply
		else:
			self.instance.send_reply(command, reply)

	def process_reaction(self, command: StoredCommand):
		message = command.message
		try:
			property_name = message.content['property']
			command = message.content['command']
		except KeyError:
			raise ProteusError(f'Malformed Command message: {message.content}')
		function = self._registered_reactions.get((property_name, command), None)
		if function:
			function(message, command.previous_message)

	@classmethod
	def get_property_descriptors(cls) -> dict[str, Any]:
		property_descriptors = dict()
		for subclass in cls.mro():
			if issubclass(subclass, Device):
				for descriptor_name, descriptor in subclass.property_descriptor_list.items():
					property_descriptors.setdefault(descriptor_name, descriptor)
		return property_descriptors

class ActiveDevice(Device):

	_is_on: bool
	_thread: Thread
	loop_time_s: float
	_loop_event: Event
	_exit: bool
	_command_lock: Lock
	_command_replies: Dict[Tuple[str, str], ReplyFunction]
	_command_reactions: Dict[Tuple[str, str], ReactionFunction]
	_commands_received: Dict[Event, Union[RawMessage, Exception]]
	_reactions_remaining: Queue[Tuple[RawMessage, ReactionFunction]]
	_jobs: Queue[Callable[[], None]]

	def __init__(self, instance: Instance, name: str, loop_time_s: float = 1.0):
		self._is_on = False
		self.loop_time_s = loop_time_s
		self._loop_event = Event()
		self._exit = False
		self._command_lock = Lock()
		self._command_replies = {}
		self._command_reactions = {}
		self._commands_received = {}
		self._reactions_remaining = Queue()
		self._jobs = Queue()
		super().__init__(instance, name)
		self._thread = Thread(target=self.loop, daemon=True)
		self._thread.start()

	def call_in_device_thread(self, fn: Callable, *args, wait: bool = True, **kwargs):
		if current_thread() is self._thread:
			return fn(*args, **kwargs)

		done = Event()
		result = {"value": None, "error": None}

		def job():
			try:
				result["value"] = fn(*args, **kwargs)
			except Exception as ex:
				result["error"] = ex
			finally:
				done.set()

		self._jobs.put(job)
		self._loop_event.set()

		if wait:
			done.wait()
			if result["error"] is not None:
				raise result["error"]
			return result["value"]

		return done, result

	def _process_jobs(self):
		while True:
			try:
				job = self._jobs.get_nowait()
			except Empty:
				break
			try:
				job()
			except Exception as ex:
				self._instance.warn(
					f"Exception during scheduled device job on {self._device_name}: {repr(ex)}",
					ex
				)

	def on(self):
		self._is_on = True

	def off(self):
		self._is_on = False

	def exit(self):
		if self._is_on:
			try:
				self.call_in_device_thread(self.off)
			except Exception:
				pass
		self._exit = True
		self._loop_event.set()
		try:
			self._thread.join(timeout=2 * self.loop_time_s if self.loop_time_s > 0 else 1.0)
		except Exception:
			pass
		super().exit()

	def loop(self):
		import time
		time.sleep(self.loop_time_s)
		while not self._exit:
			loop_end = time.time() + self.loop_time_s

			# preserve 1.0.7 priority: commands first
			self.process_commands()

			# then scheduled jobs
			self._process_jobs()

			if self._is_on:
				try:
					self.periodic()
				except Exception as ex:
					self._instance.warn(
						f"Exception during periodic function of Device {self._device_name}: {repr(ex)}",
						ex
					)

			remaining_time = loop_end - time.time()
			while remaining_time > 0 and self._loop_event.wait(remaining_time):
				self._loop_event.clear()

				self.process_commands()
				self._process_jobs()

				try:
					message, function = self._reactions_remaining.get_nowait()
				except Empty:
					message = None
					function = None

				if message is not None:
					try:
						function(message, None)
					except Exception as ex:
						self._instance.warn(
							f"Exception during command reaction {function}\nfor message {message!s}:\n{ex!s}",
							ex
						)

				remaining_time = loop_end - time.time()

	def add_command(self, name: str, variable: str = "", reply: ReplyFunction = None, reaction: Optional[ReactionFunction] = None) -> None:
		super().add_command(name, variable, self.command_callback)
		self._command_replies[(name, variable)] = reply
		self._command_reactions[(name, variable)] = reaction

	def command_callback(self, message: RawMessage) -> RawMessage:
		event = Event()
		with self._command_lock:
			self._commands_received[event] = message
		self._loop_event.set()

		variable = message.content['property']
		name = message.content['command']
		timeout = 2 * self.loop_time_s if self.loop_time_s > 0 else 1.0

		if event.wait(timeout):
			with self._command_lock:
				result = self._commands_received.pop(event)
			if isinstance(result, Exception):
				raise RuntimeError(
					f"Exception in reply function for Device {self._device_name}, Variable {variable}, command {name}: {message!s} with exception {result!r}"
				) from result
			return result
		else:
			with self._command_lock:
				self._commands_received.pop(event, None)
			raise RuntimeError(
				f"Timeout for reply function for Device {self._device_name}, Variable {variable}, command {name}."
			)

	def remove_command(self, name: str, variable: str = "") -> None:
		super().remove_command(name, variable=variable)
		del self._command_replies[(name, variable)]
		del self._command_reactions[(name, variable)]

	@abstractmethod
	def periodic(self):
		raise NotImplementedError

	def process_commands(self):
		while True:
			with self._command_lock:
				message = None
				data = None
				event_to_process = None

				for event in list(self._commands_received):
					if not event.is_set():
						event_to_process = event
						message, data = self._commands_received.pop(event)
						break

				if message is None:
					break

			try:
				key = (message["command"], message["property"])
			except KeyError as ex:
				self._instance.warn(
					f"Variable or command not included in command message {message!s}",
					ex
				)
				key = None
				reply = None
			else:
				reply_function = self._command_replies.get(key)
				if reply_function is not None:
					try:
						reply = reply_function(RawMessage(message, data))
					except Exception as ex:
						self._instance.warn(
							f"Error during reply function for message {message!s}",
							ex
						)
						reply = ex
				else:
					reply = None

			with self._command_lock:
				self._commands_received[event_to_process] = reply
				event_to_process.set()

			if key is not None:
				reaction_function = self._command_reactions.get(key)
				if reaction_function is not None:
					self._reactions_remaining.put((RawMessage(message, data), reaction_function))
					self._loop_event.set()

	def connect(self, device_name: str, variable_name: str) -> Connection:
		def proc_var_update(conn, message):
			with self._command_lock:
				self._reactions_remaining.put((message, conn._proc_var_update))
				self._loop_event.set()

		connection = super().connect(device_name, variable_name)
		connection._proc_var_update = connection.proc_var_update
		connection.proc_var_update = types.MethodType(proc_var_update, connection)
		return connection

class NewActiveDevice(Device):
	"""
	Variant of ActiveDevice using the same actor-model execution semantics,
	while preserving original command semantics.
	"""

	_is_on: bool
	_thread: Thread
	loop_time_s: float
	_loop_event: Event
	_exit: bool
	_command_lock: Lock
	_commands_received: Dict[Event, Union[Tuple[RawMessage, Callable], RawMessage, Exception]]
	_reactions_remaining: Queue[Tuple[RawMessage, RawMessage, ReactionFunction]]
	_jobs: Queue[Callable[[], None]]

	def __init__(self, instance: Instance, name: str, loop_time_s: float = 1.0):
		self._is_on = False
		self.loop_time_s = loop_time_s
		self._loop_event = Event()
		self._exit = False
		self._command_lock = Lock()
		self._commands_received = {}
		self._reactions_remaining = Queue()
		self._jobs = Queue()
		super().__init__(instance, name)
		self._thread = Thread(target=self.loop, daemon=True)
		self._thread.start()

	def call_in_device_thread(self, fn: Callable, *args, wait: bool = True, **kwargs):
		if current_thread() is self._thread:
			return fn(*args, **kwargs)

		done = Event()
		result = {"value": None, "error": None}

		def job():
			try:
				result["value"] = fn(*args, **kwargs)
			except Exception as ex:
				result["error"] = ex
			finally:
				done.set()

		self._jobs.put(job)
		self._loop_event.set()

		if wait:
			done.wait()
			if result["error"] is not None:
				raise result["error"]
			return result["value"]

		return done, result

	def _process_jobs(self):
		while True:
			try:
				job = self._jobs.get_nowait()
			except Empty:
				break
			try:
				job()
			except Exception as ex:
				self._instance.warn(
					f"Exception during scheduled device job on {self._device_name}: {repr(ex)}",
					ex
				)

	def on(self):
		self._is_on = True

	def off(self):
		self._is_on = False

	def exit(self):
		if self._is_on:
			try:
				self.call_in_device_thread(self.off)
			except Exception:
				pass
		self._exit = True
		self._loop_event.set()
		try:
			self._thread.join(timeout=2 * self.loop_time_s if self.loop_time_s > 0 else 1.0)
		except Exception:
			pass
		super().exit()

	def loop(self):
		import time
		time.sleep(self.loop_time_s)
		while not self._exit:
			loop_end = time.time() + self.loop_time_s

			# Commands first
			self.process_commands()

			# Then jobs
			self._process_jobs()

			# Then periodic
			if self._is_on:
				try:
					self.periodic()
				except Exception as ex:
					self._instance.warn(
						f"Exception during periodic function of Device {self._device_name}: {repr(ex)}",
						ex
					)

			remaining_time = loop_end - time.time()
			while remaining_time > 0 and self._loop_event.wait(remaining_time):
				self._loop_event.clear()

				self.process_commands()
				self._process_jobs()

				try:
					message, reply, function = self._reactions_remaining.get_nowait()
				except Empty:
					message = None
					reply = None
					function = None

				if message is not None:
					try:
						function(message, reply)
					except Exception as ex:
						self._instance.warn(
							f"Exception during command reaction {function}\nfor message {message!s}:\n{ex!s}",
							ex
						)

				remaining_time = loop_end - time.time()

	def process_command(self, command: StoredCommand):
		message = command.message
		try:
			property_name = message.content['property']
			command_name = message.content['command']
		except KeyError:
			reply = create_reply(message, error='No registered reply')
		else:
			if property_name == "":
				reply = create_reply(message, error='no device commands yet')
			else:
				try:
					function = self._registered_replies[property_name, command_name]
				except KeyError:
					reply = create_reply(message, error='No registered reply')
				else:
					event = Event()
					with self._command_lock:
						self._commands_received[event] = (message, function)
					self._loop_event.set()

					if event.wait(2 * self.loop_time_s if self.loop_time_s > 0 else 1.0):
						with self._command_lock:
							reply = self._commands_received.pop(event)
						if isinstance(reply, Exception):
							reply = create_reply(message, error=f'Exception in reply function: {reply!r}')
					else:
						with self._command_lock:
							self._commands_received.pop(event, None)
						reply = create_reply(message, error='Timeout in reply function')

		if command.address_info is None:
			return reply
		else:
			self.instance.send_reply(command, reply)

	def process_reaction(self, command: StoredCommand):
		message = command.message
		try:
			property_name = message.content['property']
			command_name = message.content['command']
		except KeyError:
			raise ProteusError(f'Malformed Command message: {message.content}')

		function = self._registered_reactions.get((property_name, command_name), None)
		if function:
			function(message, command.previous_message)

	def add_command(self, name: str, variable: str = "", reply: ReplyFunction = None, reaction: Optional[ReactionFunction] = None) -> None:
		super().add_command(name, variable, self.command_callback)
		self._registered_replies[(variable, name)] = reply
		if reaction is not None:
			self._registered_reactions[(variable, name)] = reaction

	def command_callback(self, message: RawMessage) -> RawMessage:
		event = Event()
		with self._command_lock:
			self._commands_received[event] = message
		self._loop_event.set()

		variable = message.content['property']
		name = message.content['command']

		if event.wait(2 * self.loop_time_s if self.loop_time_s > 0 else 1.0):
			with self._command_lock:
				result = self._commands_received.pop(event)
			if isinstance(result, Exception):
				raise RuntimeError(
					f"Exception in reply function for Device {self._device_name}, Variable {variable}, command {name}: {message!s} with exception {result!r}"
				) from result
			return result
		else:
			with self._command_lock:
				self._commands_received.pop(event, None)
			raise RuntimeError(
				f"Timeout for reply function for Device {self._device_name}, Variable {variable}, command {name}."
			)

	def remove_command(self, name: str, variable: str = "") -> None:
		super().remove_command(name, variable=variable)
		try:
			del self._registered_replies[(variable, name)]
		except KeyError:
			pass
		try:
			del self._registered_reactions[(variable, name)]
		except KeyError:
			pass

	@abstractmethod
	def periodic(self):
		raise NotImplementedError

	def process_commands(self):
		while True:
			with self._command_lock:
				event = None
				payload = None

				for candidate, stored in list(self._commands_received.items()):
					if not candidate.is_set():
						event = candidate
						payload = stored
						self._commands_received.pop(candidate, None)
						break

			if payload is None:
				break

			if isinstance(payload, tuple) and len(payload) == 2:
				message, function = payload
			else:
				message = payload
				function = None

			if function is None:
				try:
					key = (message["property"], message["command"])
				except KeyError as ex:
					self._instance.warn(
						f"Variable or command not included in command message {message!s}",
						ex
					)
					reply = None
					reaction_function = None
				else:
					reply_function = self._registered_replies.get(key)
					reaction_function = self._registered_reactions.get(key)

					if reply_function is not None:
						try:
							reply = reply_function(RawMessage(message, None))
						except Exception as ex:
							self._instance.warn(
								f"Error during reply function for message {message!s}",
								ex
							)
							reply = ex
					else:
						reply = None
			else:
				try:
					reply = function(message)
				except Exception as ex:
					reply = ex
				reaction_function = None

			with self._command_lock:
				self._commands_received[event] = reply
				event.set()

			if reaction_function is not None:
				self._reactions_remaining.put((message, reply, reaction_function))
				self._loop_event.set()

	def connect(self, device_name: str, variable_name: str) -> Connection:
		def proc_var_update(conn, message):
			with self._command_lock:
				self._reactions_remaining.put((message, None, conn._proc_var_update))
				self._loop_event.set()

		connection = super().connect(device_name, variable_name)
		connection._proc_var_update = connection.proc_var_update
		connection.proc_var_update = types.MethodType(proc_var_update, connection)
		return connection

