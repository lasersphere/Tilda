from typing import Dict, Any, Callable, Union, Tuple
from . import Property
from ..Utility import create_reply, RawMessage
from ..Devices import Device
from ..Encoder import Datatype

sentinel = object()


class Variable(Property):

	datatype: Datatype
	data: Any
	_value: Any
	registered_callbacks: Dict[int, Callable]
	auto_publish: bool
	allow_remote_set: bool

	def __init__(self, name: str, device: Device, datatype: Datatype, default_value=None, auto_publish: bool = True, allow_remote_set: bool = True):
		super().__init__(name=name, device=device, type_id=datatype.id())
		self.datatype = datatype
		self.auto_publish = auto_publish
		self.allow_remote_set = allow_remote_set
		self.registered_callbacks = {}
		if self.device is not None:
			self.data = datatype.setup_data_storage()
			if default_value is not None:
				self.data = datatype.value_to_data(default_value, self.data)
			self._value = datatype.data_to_value(self.data)
			self.add_command("READ", self._read_command_reply)
			self.add_command("WRITE", self._write_command_reply, self._write_command_reaction)

	def additional_property_arguments(self) -> Dict[str, Any]:
		return {'datatype': self.datatype.id()}

	def _read_command_reply(self, message: RawMessage) -> RawMessage:
		reply_args, reply_data = self.datatype.data_to_message_args(self.data)
		return create_reply(message, reply_data, **reply_args)

	def _write_command_reply(self, message: RawMessage) -> RawMessage:
		self.data = self.datatype.message_to_data(message.content, message.data, self.data)
		reply_args, reply_data = self.datatype.data_to_message_args(self.data)
		return create_reply(message, reply_data, **reply_args)

	def _write_command_reaction(self, message: RawMessage, reply: RawMessage):
		old_value = self._value
		new_value = self.datatype.data_to_value(self.data)
		if not self.datatype.compare_values(new_value, old_value):
			self._value = new_value
			if self.auto_publish:
				self._publish()
			self.call_registered_callbacks(new_value, old_value, message)

	def get(self, reset: bool = False) -> Any:
		if reset:
			self._value = self.datatype.data_to_value(self.data)
		return self._value

	def set(self, value: Any = sentinel, publish: bool = False):
		# TODO figure out the purpose of this sentinel and remove if unnecessary
		if value is not sentinel:
			self._value = value
		self.data = self.datatype.value_to_data(self._value, self.data)
		if publish or self.auto_publish:
			self._publish()

	def _publish(self):
		self.data = self.datatype.value_to_data(self._value, self.data)
		kwargs, data = self.datatype.data_to_message_args(self.data)
		self.device.publish_variable(variable=self.variable_name, data=data, **kwargs)

	@property
	def value(self) -> Any:
		return self.get()

	@value.setter
	def value(self, value: Any):
		self.set(value)

	def add_update_callback(self, callback: Callable[[Any, Any, RawMessage], Any]) -> int:
		new_callback_id = max(self.registered_callbacks, default=-1) + 1
		self.registered_callbacks[new_callback_id] = callback
		return new_callback_id

	def remove_update_callback(self, callback_or_id: Union[Callable, int]):
		if callback_or_id in self.registered_callbacks:
			self.registered_callbacks.pop(callback_or_id)
		else:
			for callback_id in self.registered_callbacks:
				if self.registered_callbacks[callback_id] is callback_or_id:
					self.registered_callbacks.pop(callback_id)
					return

	def call_registered_callbacks(self, new_value, old_value, message: RawMessage):
		for callback_id in self.registered_callbacks:
			print(f'calling registered callback {self.registered_callbacks[callback_id]}')
			self.registered_callbacks[callback_id](new_value, old_value, message)
			# TODO should pass arguments to callbacks