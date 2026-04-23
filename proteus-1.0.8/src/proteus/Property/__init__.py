from abc import abstractmethod, ABC
from typing import Optional, Dict, Any, Type, Tuple
from ..Data import VariableReference
from ..Devices import Device
from ..Utility import create_reply, RawMessage, ReplyFunction, ReactionFunction


class Property(VariableReference, ABC):

	device: Optional[Device]
	_is_proxy: bool

	def __init__(self, name: str, device: Optional[Device], type_id: str):
		self._is_proxy = False
		if isinstance(device, Device):
			super().__init__(device=device.device_name, name=name, type_id=type_id)
			self.device = device
			device.add_variable(self)
			self.add_command("PROPERTY", self._property_callback)
		else:
			super().__init__(device="", name=name, type_id=type_id)
			self.device = None

	def add_command(self, name: str, reply: ReplyFunction = None, reaction: Optional[ReactionFunction] = None) -> None:
		"""
		Add a command and the appropriate reaction to the Variable.

		This method replaces add_command to preserve its signature while
		allowing adding a callback in a single step. When a message addressing
		the Variable with the specified command arrives, the callback function
		will be called with the message as argument.
		:param name: Identifier of the command to be added.
		:param reply: Function that calculates the reply to incoming messages
		:param reaction: Function called after the reply was sent back
		:return: None
		"""
		self.device.add_command(name, variable=self._variable_name, reply=reply, reaction=reaction)

	def remove_command(self, name: str) -> None:
		self.device.remove_command(name, variable=self._variable_name)

	def _property_callback(self, message: RawMessage):
		return create_reply(message, type_id=self.type_id, **self.additional_property_arguments())

	# TODO is this actually necessary? either integrate into _property_callback or remove altogether
	@abstractmethod
	def additional_property_arguments(self) -> Dict[str, Any]:
		pass


class PropertyDescriptor:

	def __init__(self, property_class: Type[Property], name: str = '', overwrite_object=None, *args, **kwargs):
		self._class: Type[Property] = property_class
		self._name: str = str(name)
		self._overwrite_object = overwrite_object
		self._args: Tuple = args
		self._kwargs: Dict[str, Any] = kwargs
		self._property_object: Optional[Property] = None

	def __set_name__(self, cls, name):
		self._name = self._name or name
		if issubclass(cls, Device):
			if 'property_descriptor_list' not in cls.__dict__:
				# If a class does not have its own property_descriptor_list,
				# copy the one from the parent class. Without this, properties
				# added to the child class are also added to the parent class
				cls.property_descriptor_list = {}
			cls.property_descriptor_list[self._name] = self
			# write into a list of things to add into the device
		if self._overwrite_object:
			setattr(cls, name, self._overwrite_object)

	def __call__(self, device: Device):
		if isinstance(device, Device):  # Exclude calls directly to the class
			try:
				self._property_object = self._class(self._name, device, *self._args, **self._kwargs)
				return self._property_object
			except Exception as ex:
				device.instance.warn("PropertyDescriptor created with invalid class", ex)
		return self

	def __getattr__(self, item: str):
		# Fallback in case the descriptor is not removed correctly
		if self._property_object is None:
			raise AttributeError
		else:
			return getattr(self._property_object, item)
