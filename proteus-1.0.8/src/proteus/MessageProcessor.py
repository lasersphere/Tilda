from abc import abstractmethod
from dataclasses import dataclass, field
from math import inf
from typing import Any, Optional

from proteus.Utility import RawMessage


@dataclass(order=True)
class StoredCommand:
	priority: int
	action_type: str = field(compare=False)
	message: RawMessage = field(compare=False)
	address_info: Any = field(compare=False)
	previous_message: Optional[RawMessage] = field(compare=False)
	action_priority = {
		'command': 0,
		'reaction': 1,
		'message': 1,
	}

	def __init__(self, action_type: str, message: RawMessage, address_info: Any = None, previous_message: RawMessage = None):
		self.priority = self.action_priority.get(action_type, inf)
		self.action_type = action_type
		self.message = message
		self.address_info = address_info
		if action_type == 'reaction' and not isinstance(previous_message, RawMessage):
			# TODO can probably removed in the future, only exists for testing purposes
			print(f'Stored reaction command without previous_message')
		self.previous_message = previous_message


class MessageProcessor:

	# Storage for any things required for running the MessageProcessor
	# that are set up by the Instance but are specific to this MessageProcessor
	local_objects: Any

	@abstractmethod
	def process_command(self,command: StoredCommand):
		pass

	@abstractmethod
	def process_reaction(self, command: StoredCommand):
		pass

	# TODO why is this not used?
	@abstractmethod
	def process_message(self, command: StoredCommand):
		pass


