from typing import Dict, Any, Callable, Union, Tuple, List, NamedTuple, Optional
from . import Property
from ..Utility import create_reply, RawMessage
from ..Devices import Device
from ..Encoder import Datatype
from datetime import datetime

sentinel = object()


class DataEvent(NamedTuple):
	id: int
	time: datetime
	value: Any
	data: Any
	datatype: Datatype
	message: Optional[RawMessage]

	@classmethod
	def from_message(cls, id: int, message: RawMessage, datatype: Datatype):  # -> Self when the package requirements include Python 3.11
		try:
			time = datetime.strptime(message.content['timestamp'], "%Y-%m-%dT%H:%M:%S.%fZ")
			data = datatype.setup_data_storage()
			data = datatype.message_to_data(message.content, message.data, data)
			value = datatype.data_to_value(data)
		except Exception as ex:
			print(f'Message could not be converted into DataEvent: {repr(ex)}')
		else:
			return cls(id=id, time=time, value=value, data=data, datatype=datatype, message=message)

	@classmethod
	def from_value(cls, id: int, value, datatype: Datatype):  # -> Self when the package requirements include Python 3.11
		time = datetime.now()
		try:
			data = datatype.setup_data_storage()
			data = datatype.value_to_data(value, data)
		except Exception as ex:
			print(f'Value could not be stored into DataEvent: {repr(ex)}')
		else:
			return cls(id=id, time=time, value=value, data=data, datatype=datatype, message=None)


class Datastream(Property):

	datatype: Datatype
	allow_remote_set: bool
	total_events: int
	stored_values: List[DataEvent]
	last_value: Optional[DataEvent]
	registered_callbacks: Dict[int, Callable]
	_length_limit: Optional[int]

	def __init__(self, name: str, device: Device, datatype: Datatype, allow_remote_set: bool = True, length_limit: int = None):
		super().__init__(name=name, device=device, type_id=datatype.id())
		self.datatype = datatype
		self.allow_remote_set = allow_remote_set
		self.registered_callbacks = {}
		self.total_events = 0
		self.stored_values = []
		self.last_value = None
		self.length_limit = length_limit
		if self.device is not None:
			pass  # setup

	@property
	def length_limit(self) -> int:
		return self._length_limit

	@length_limit.setter
	def length_limit(self, new_limit: Optional[int]):
		try:
			new_limit = int(new_limit)
			if new_limit > 0:
				self._length_limit = new_limit
				if new_limit < len(self.stored_values):
					self.stored_values = self.stored_values[-1*self.length_limit:]
			else:
				self._length_limit = None
		except TypeError:
			self._length_limit = None

	def set(self, value: Any, publish: bool = False):
		event = DataEvent.from_value(self.total_events, value, self.datatype)
		self.total_events += 1
		self.last_value = event
		self.stored_values.append(event)
		self.enforce_length_limit()

	def enforce_length_limit(self):
		if self.length_limit:
			while self.length_limit < len(self.stored_values):
				# Remove the oldest values (by timestamp) until the list is small enough
				# In case of equal timestamps, the items added first are removed first
				times = [event.time for event in self.stored_values]
				self.stored_values.pop(times.index(min(times)))

	def get(self) -> Any:
		if self.last_value:
			return self.last_value.value
		else:
			raise RuntimeError('No values in Datastream')

	def event_from_id(self, event_id: int) -> Optional[DataEvent]:
		try:
			return self.stored_values[[event.id for event in self.stored_values].index(event_id)]
		except ValueError:
			return None

	def additional_property_arguments(self) -> Dict[str, Any]:
		return {'datatype': self.datatype.id()}
