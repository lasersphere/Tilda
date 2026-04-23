"""
Base classes for all communication backend implementations.

This module contains the class that serves as a base class for
communication implementations. The required methods for the
implementation of a communication protocol are defined here.

Instance: Base class for communication backends.
"""
from queue import PriorityQueue, Empty
from threading import Thread, Lock
from typing import List, Callable, Dict, Any, Optional, Tuple, Union, cast
from abc import ABC, abstractmethod

from proteus.MessageProcessor import MessageProcessor, StoredCommand
from proteus.Utility import create_message, ProteusError, RawMessage, ReplyFunction, ReactionFunction, MessageCallback, DeviceNotFound
from proteus.Data import InstanceReference, DeviceReference, VariableReference
from warnings import warn


class Instance(InstanceReference, MessageProcessor, ABC):
	"""
	Base class for the creation of protocol implementations.

	The BaseInstance class defines the implementation of communication
	protocols for the use with the package. To implement a protocol for use
	with the package, one creates a subclass of BaseInstance that implements
	the required methods.
	The class also implements a processing loop that allows for a controlled
	execution of regular actions.

	Methods:
		send_variable_command: Send a command to a Variable-type access point.
		send_device_command: Send a command to a Device-type access point.
		send_instance_command: Send a command to another Instance.
		get_instance_address: Determine the instance a Device is managed by.
		add_device_variable_command: Register a command for a local variable.
		remove_device_variable_command: Remove a command from a local variable.
		publish_variable: Publish an update for a Variable.
		subscribe_variable: Subscribe updates for a Variable.
		add_to_loop: Add a PeriodicObject to the instances processing loop.
		add_command: Add a command as valid for the Instance.
		remove_command: Remove a command from the list of valid commands for
			the Instance.
		add_device: Add a Device to those managed by the Instance.
		remove_device: Remove a Device from those managed by the Instance.

	Properties:
		name: Optional user-friendly identifier for the instance.
		user: Optional information on the current user of the instance.
		minimum_loop_time: Time between processing loop iterations.
			For further details regarding the processing loop, see the
			PeriodicObjectLoop class in the Processing module.
	None of the following properties can be edited (operations can be
	changed indirectly by methods).
		device_name: Device name of the access point (always empty string).
		variable_name: Variable name of the access point (always empty string).
		type_id: Identifier for the type of access point.
		identification: Identifier of the specific instance(backend-dependent).
		backend_type: Identifier for the network backend used by the Instance.
		operations: Valid operations for this Interface.

	A subclass needs to reimplement the following methods:
		get_instance_address: Determine the instance a Device is managed by.
		_send_command: Send a JSON-compatible dict to another instance.
		_add_command_listener: Register a callback for a command.
		_remove_command_listener: Unregister a callback for a command.
		_publish: Publish an update to all subscribers.
		_subscribe: Set a callback as reaction for specific updates.
		More detailed descriptions are available in the docstrings of their
		dummy implementations in the BaseInstance class.
	"""
	_known_instances: Dict[str, InstanceReference]
	_registered_replies: Dict[str, ReplyFunction]
	_registered_reactions: Dict[str, ReactionFunction]
	_registered_messages: Dict[str, Dict[str, Dict[str, List[MessageCallback]]]]
	_registered_messages_lock: Lock
	_registered_update_listeners: Dict[Tuple[str, str], Tuple[Callable[[str], None], Callable[[], None]]]
	_processing_queue: PriorityQueue[StoredCommand]
	_is_running: bool
	_processing_thread: Thread

	def __init__(self, identification: str, backend_type: str = "undefined"):
		"""
		The specifics of instance creation depend on the implemented protocol.
		This method only provides handling of basic information and setup of the
		processing loop.
		:param identification: Identifying information of the instance. Format
			and meaning depend on the protocol used.
		:param backend_type: Identifier for the protocol used as a backend.
		"""
		super().__init__(identification, backend_type)
		self._known_instances = {self.identification: self}
		self._registered_replies: Dict[str, ReplyFunction] = {'JOIN': self._join_reply, 'STATUS': self._status_reply}
		self._registered_reactions: Dict[str, ReactionFunction] = {'JOIN': self._join_reaction}
		self._registered_messages: Dict[str, Dict[str, Dict[str, List[MessageCallback]]]] = {'': {'': {'UPDATE': [self._instance_message]}}}  # TODO unfinished, does not support connection's subscriptions!
		self._registered_messages_lock = Lock()
		self._registered_update_listeners = {}
		self._processing_queue = PriorityQueue()
		self._is_running = False
		self._processing_thread = Thread(target=self._processing_loop)

	def __enter__(self):
		self._is_running = True
		self._processing_thread.start()
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		self._is_running = False
		self._processing_thread.join()

	def _processing_loop(self):
		while self._is_running:
			try:
				stored_command = self._processing_queue.get(timeout=1)
				action_type = stored_command.action_type
				message = stored_command.message
			except Empty:
				continue
			if action_type == "message":
				self.process_message(stored_command)
				continue
			try:
				device_name = str(message.content['nomen'])
			except KeyError as ex:
				self.warn(f"message without device name: {message.content}", ex)
				continue
			if device_name == '':
				device = self
			else:
				try:
					device = self._devices[device_name]
				except KeyError as ex:
					self.warn(f'message sent to wrong instance {self.identification} and will be dropped: {message.content}', ex)
					continue
			if isinstance(device, MessageProcessor):
				if action_type == 'command':
					device.process_command(stored_command)
				elif action_type == 'reaction':
					device.process_reaction(stored_command)
				else:
					self.warn('invalid action_type in processing_queue', None)
			else:
				self.warn('invalid object in devices list', None)

	@abstractmethod
	def setup_message_processor(self, processor: MessageProcessor) -> None:
		raise NotImplementedError

	@abstractmethod
	def cleanup_message_processor(self, processor: MessageProcessor) -> None:
		raise NotImplementedError

	def send_reply(self, command: StoredCommand, reply: RawMessage):
		self._processing_queue.put(StoredCommand('reaction', reply, address_info=command.address_info, previous_message=command.message))

	def send_command(self, command: str, device: str, variable: str = "", data: Optional[List[bytes]] = None, **kwargs) -> RawMessage:
		if not device:
			raise ValueError("non-empty device name required")
		message = create_message(device, variable, command, data, **kwargs)
		try:
			device_object = self._devices[device]
		except KeyError:
			instance = self.get_instance_address(device)
			reply = self._send_command(instance, message)
			if reply.content['command'] == 'ERROR':
				error_string = reply.content.get('error', "unspecified error")
				raise ProteusError(f"Error at remote processing of command {command!r} for Device{device!r}, Variable {variable!r}: {error_string}")
		else:
			reply = cast(MessageProcessor, device_object).process_command(StoredCommand("command", message))
		return reply

	def send_command_to_instance(self, command: str, address: str, data: Optional[List[bytes]] = None, **kwargs) -> RawMessage:
		"""
		Create and send a command message to an Instance.

		This function creates a valid command message from its parameters and
		class the protocol-specific _send_command to send this message to the
		Instance specified.
		:param command: Identifier (usually name) of the command to be sent.
		:param address: Identifier of the target Instance
		:param kwargs: Other keyword arguments are added to the message.
		:return: The message received as reply, None if no reply was received.
		"""
		message = create_message("", "", command, data, **kwargs)
		# TODO shortcut local commands?
		reply = self._send_command(address, message)
		if reply.content['command'] == 'ERROR':
			error_string = reply.content.get('error', "unspecified error")
			raise ProteusError(f"Error at remote processing of command {command!r} for Instance {address}: {error_string}")
		return reply

	@abstractmethod
	def _send_command(self, target_address: str, message: RawMessage) -> RawMessage:
		"""
		Send a json-compatible dict to the target Instance.

		This method has to be reimplemented for any usable subclass. The
		message_json has to be transferred to the Instance at the target_address
		if such an Instance exists and that Instance has to call the callback
		method of the command, if the command has been added for the access
		point specified in the message.
		:param target_address: Identifier of the Instance the message is to be
			sent to.
		:param message: JSON-compatible dictionary containing the message
			to be sent. Is assumed to include all required parts as defined by the
			message standard.
		:return: The reply received. None if no reply was received.
		"""
		pass

	def get_instance_address(self, device: str) -> str:
		"""
		Find the identifier of the instance that manages a Device.

		This implementation does not retrieve any data from remote Instances and
		needs to be replaced in child classes.
		:param device: Name of the Device.
		:return: Instance identifier if found, otherwise None.
		"""
		for instance in self._known_instances:
			if device in self._known_instances[instance]:
				return instance
		raise DeviceNotFound(f"Device {device!r} was not found in the network")

	def add_command(self, name: str, reply: ReplyFunction = None, reaction: Optional[ReactionFunction] = None) -> None:
		"""
		Add a command to a Variable and register its callback.

		Registers a command as available for the access point specified. The
		provided callback will be called whenever a message with this command
		is received.
		:param name: Identifier (usually name) of the command to register.
		:param reply: Callback to pass incoming command message to. This
			argument is _not_ optional.
		:param reaction: Function to call at some point after the reply has
			been sent back. This is optional.
		:return: True if the command was registered successfully, otherwise
			False.
		"""
		if reply is None:
			# TODO create a default reply function to replace this error
			raise RuntimeError(f"It is not possible to add the command {name!s} to an Instance without a reply function.")
		super().add_command(name, "", "")

	def publish_variable(self, device: str, variable: str, data: List[bytes] = None, **kwargs) -> None:
		"""
		Make the value of a Variable available to all subscribers.

		The value is sent to all subscribers as an update for the Variable
		specified. Other properties of the update message are not accessible
		at this point. The method needs to be reimplemented in subclasses that
		use a filter system that is not based on the first part of the message.
		:param device: Name of the Device the Variable belongs to.
		:param variable: Name of the variable the update concerns.
		:param value: JSOn-compatible data of the updated value.
		:return: None
		"""
		message = create_message(device, variable, "UPDATE", data=data, **kwargs)
		self._publish(message.content, message.data)

	@abstractmethod
	def _publish(self, message: Dict[str, Any], data: Optional[List[bytes]] = None) -> None:
		"""
		Publish a message.

		This method is to be reimplemented in subclasses. The message should
		already include the relevant filter to ensure only intended recipients
		receive it. The message s to be sent to all connected instances as
		necessary to ensure all subscribers receive it.
		:param message: Message to be published.
		:return: None
		"""
		pass

	# protocol-specific implementation of handling incoming messages that fulfill the filter given by the filter_string
	@abstractmethod
	def _subscribe(self, device_name: str, variable_name: str, command: str) -> None:
		"""
		Subscribe to update messages that fit a specified filter.

		This method is to be reimplemented is subclasses. It is assumed all
		relevant messages for a single subscription will originate from a single
		instance. All messages published on this instance that fit the filter
		are to call the callback function with the value part of the message as
		argument.
		:param device_name: Device name of the access point the message will be
			received from
		:param variable_name: Variable name of the access point the message will
			be received from
		:param command: tbd
		:return: None
		"""
		pass

	@abstractmethod
	def _unsubscribe(self, device_name: str, variable_name: str, command: str) -> None:
		pass

	def publish_status_update(self):
		self.publish_variable("", "", **self._join_args())

	def _join_args(self) -> Dict[str, Any]:
		return {'value': list(self._known_instances), 'status': self._status_json()}

	def register_variable_change(self, device: str, variable: str, add_callback: Callable, remove_callback: Callable) -> None:
		self._registered_update_listeners[(device, variable)] = (add_callback, remove_callback)

	def unregister_variable_change(self, device: str, variable: str) -> None:
		del self._registered_update_listeners[(device, variable)]

	def trigger_variable_added_listener(self, device: str, variable: str, type_id: str) -> None:
		try:
			self._registered_update_listeners[(device, variable)][0](type_id)
		except KeyError:
			pass

		# Property was added to this Instance, the new info should also be published.
		# For devices (variable == ""), we *always* publish. For normal variables,
		# only publish if the variable still exists on that device.
		if variable == "":
			self.publish_status_update()
		else:
			if device in self and variable in self[device]:
				self.publish_status_update()


	def trigger_variable_removed_listener(self, device: str, variable: str) -> None:
		try:
			self._registered_update_listeners[(device, variable)][1]()
		except KeyError:
			pass

		# Property was removed from this Instance, the new info should also be published.
		# For devices (variable == ""), publish unconditionally, because the device
		# is *gone* from self._devices.
		if variable == "":
			self.publish_status_update()
		else:
			# For variables, only publish if the device still exists locally.
			if device in self:
				self.publish_status_update()


	def add_device(self, device: DeviceReference) -> None:
		super().add_device(device)
		self.trigger_variable_added_listener(device.device_name, "", device.type_id)

	def remove_device(self, device: Union[str, DeviceReference]) -> None:
		# Normalize to a device name string
		if isinstance(device, str):
			device_name = device
		else:
			device_name = device.device_name

		# Remove locally
		super().remove_device(device)

		# Notify listeners
		self.trigger_variable_removed_listener(device_name, "")

		# Publish updated status (without this device) to the network
		self.publish_status_update()

	def process_status_update(self, reference: InstanceReference, status_update: Dict[str, Dict[str, str]]):
		if isinstance(reference, Instance):
			raise TypeError("Status updates can not be applied to Instances, only to InstanceReference objects.")
		# TODO Need to switch removing the old and adding the new. This is important because it means Connections are
		#  disconnected before they are connected to a new address.

		# 1) Add / update devices & variables from the incoming status_update
		for device_name in status_update:
			try:
				device_ref = reference._devices[device_name]
			except KeyError:
				device_ref = DeviceReference(device_name)
				reference.add_device(device_ref)
				try:
					device_type = status_update[device_name].pop("")
					self._registered_update_listeners[device_name, ""][0](device_type)
				except KeyError:
					pass
			for variable_name in status_update[device_name]:
				if variable_name != "" and variable_name not in device_ref:
					variable_type = status_update[device_name][variable_name]
					device_ref.add_variable(VariableReference(device_name, variable_name, variable_type))
					try:
						self._registered_update_listeners[device_name, variable_name][0](variable_type)
					except KeyError:
						pass

		# 2) Remove / adjust outdated variables and devices
		#    IMPORTANT: iterate over copies to avoid "dict changed size during iteration"
		for device_name in list(reference._devices):  # <-- copy of keys
			if device_name in status_update:
				device_ref = reference._devices[device_name]
				for variable_name in list(device_ref):  # <-- copy of variable names
					if variable_name in status_update[device_name]:
						variable_ref = device_ref[variable_name]
						if variable_ref.type_id != status_update[device_name][variable_name]:
							device_ref.remove_variable(variable_ref)
							variable_type = status_update[device_name][variable_name]
							variable_ref = VariableReference(device_name, variable_name, variable_type)
							device_ref.add_variable(variable_ref)
							try:
								self._registered_update_listeners[device_name, variable_name][0](variable_type)
							except KeyError:
								pass
					else:
						device_ref.remove_variable(variable_name)
						try:
							self._registered_update_listeners[device_name, variable_name][1]()
						except KeyError:
							pass
			else:
				reference.remove_device(device_name)
				try:
					self._registered_update_listeners[device_name, ""][1]()
				except KeyError:
					pass
		# print(f"status update done at {self._identification}")


	# TODO error log implementation
	# This method is effectively static and thus thread-safe (until the error log is implemented, at least)
	def warn(self, text: str, cause: Optional[Exception] = None, **kwargs):
		warning_object = RuntimeWarning(f"{text}: {repr(cause)}")
		warning_object.__cause__ = cause
		warn(warning_object, stacklevel=2, **kwargs)

	def subscribe(self, device: str, variable: str, command: str, callback: Callable):
		with self._registered_messages_lock:
			list = self._registered_messages.setdefault(device, {}).setdefault(variable, {})
			new_subscription = not bool(list)
			list.setdefault(command, []).append(callback)
		if new_subscription:
			self._subscribe(device, variable, command)

	def unsubscribe(self, device: str, variable: str, command: str, callback: Callable):
		with self._registered_messages_lock:
			try:
				list = self._registered_messages[device][variable][command]
			except KeyError:
				return  # Attempt at removing a subscription that does not exist, skip the rest
			try:
				list.remove(callback)
			except ValueError:
				pass  # Attempt at removing a subscription that does not exist, but maybe the
				# implementation-specific subscription might need removal
			remove_subscription = bool(list)
		if remove_subscription:
			self._unsubscribe(device, variable, command)

	def process_command(self, stored_command: StoredCommand):
		raw_message = stored_command.message
		message = raw_message.content
		try:
			command = message['command']
		except KeyError:
			print(f"Malformed message: no 'command': {message}")
			return
		try:
			function = self._registered_replies[command]
		except KeyError:
			print(f"No reply function for '{command}'")
			return
		try:
			reply = function(raw_message)
		except Exception as ex:
			self.warn(f"error in instance reply function", ex)
			return
		if stored_command.address_info is None:
			return reply
		else:
			self.send_reply(stored_command, reply)

	def process_reaction(self, stored_command: StoredCommand):
		message = stored_command.previous_message
		try:
			command = message.content['command']
		except KeyError:
			print(f"Malformed message: no 'command': {message}")
			return
		try:
			function = self._registered_reactions[command]
		except KeyError:
			print(f"No reaction function for '{command}'")
			return
		try:
			function(message, stored_command.message)
		except Exception as ex:
			self.warn("error in instance reaction function", ex)

	def process_message(self, stored_command: StoredCommand):
		message = stored_command.message.content
		data = stored_command.message.data
		try:
			command, device, variable = message['command'], message['nomen'], message['property']
		except KeyError:
			print(f"Malformed message: {message}")
			return
		with self._registered_messages_lock:
			try:
				functions = self._registered_messages[device][variable][command].copy()
			except KeyError:
				# This is not an error at this point in development, but may be on ein the future
				# print(f"No message function for {device=}, {variable=}, {command=}")
				return
		for function in functions:
			try:
				function(RawMessage(message, data))  # TODO add data
			except Exception as ex:
				self.warn(f"error in instance message function {function} for {message=}: {repr(ex)}", ex)

	@abstractmethod
	def _join_reply(self, request: RawMessage) -> RawMessage:
		# TODO data
		raise NotImplementedError

	@abstractmethod
	def _status_reply(self, request: RawMessage) -> RawMessage:
		# TODO data
		raise  NotImplementedError

	@abstractmethod
	def _join_reaction(self, request: RawMessage, reply: RawMessage) -> None:
		# TODO data
		raise NotImplementedError

	@abstractmethod
	def _instance_message(self, message: RawMessage) -> None:
		# TODO data
		raise NotImplementedError
