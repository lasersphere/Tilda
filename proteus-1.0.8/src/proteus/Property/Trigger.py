from collections.abc import Callable
from itertools import count
from typing import Tuple, Final, Any, Dict

from proteus.Encoder import Datatype
from proteus.Devices import Device
from proteus.Property import Property
from proteus.Utility import RawMessage, create_reply


class Trigger(Property):

	def __init__(self, name: str, device: Device, datatype: Datatype, triggered_function_generator: Callable[[Device], Callable]):
		super().__init__(name=name, device=device, type_id=datatype.id())
		self.datatype: Final[Datatype] = datatype
		self.value = datatype.setup_data_storage()
		self.triggered_function: Callable = triggered_function_generator(device)
		self.trigger_counter = count()
		if self.device is not None:
			self.add_command("TRIGGER", self.confirm, self.trigger)

	def confirm(self, message: RawMessage) -> RawMessage:
		event_id = next(self.trigger_counter)
		return create_reply(message, event_id=event_id)

	def trigger(self, message: RawMessage, reply: RawMessage):
		self.value = self.datatype.message_to_data(message.content, message.data, self.value)
		value = self.datatype.data_to_value(self.value)
		try:
			return_value = self.triggered_function(value)
		except Exception as ex:
			self.device.instance.warn("error in triggered function", ex)
		else:
			return
			event_id = reply.content['event_id']
			self.value = self.datatype.value_to_data(return_value, self.value)
			reply_args, reply_data = self.datatype.data_to_message_args(self.value)
			self.device.publish_variable(variable=self.variable_name, data=reply_data, **reply_args, event_id=event_id)

	def additional_property_arguments(self) -> Dict[str, Any]:
		return {'datatype': self.datatype.id()}
