"""
Helper functions and data structures that don't depend on other classes.

The data structures in this module are used to store data concerning the
objects defined by the package created by remote programs. They are also
used as base classes for those objects.

VariableReference: Contains the basic information for any remote-
	accessible program component.
DeviceReference: Contains the information for a group of connected
	variables.
InstanceReference: Contains the information for the network access of a
	remote program.
DelayedInitialization: Base class to declare an object to require
	explicit initialization at a later point.

The functions in this module define general data pattern conversions and
	other independent helper functions.

message_filter: Create a filter string to identify an access point.
create_message: Create a JSON-compatible message dictionary with all
	required information.
create_reply: Alter a message dictionary to create a reply.
get_ip: Find the IP of the computer the program runs on.
"""
from typing import Dict, Any, Tuple, Optional, List, NamedTuple, Callable, Union
from abc import ABC, abstractmethod
from datetime import datetime
import inspect


def get_ip() -> str:
	"""
	Determine the IP address of the computer the program runs on.

	:return: The IP address of the computer, written as a single string.
	"""
	import socket
	s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	try:
		s.connect(('10.255.255.255', 1))
		ip = s.getsockname()[0]
	except Exception:  # TODO find the correct exception(s) for this situation
		ip = '127.0.0.1'
	finally:
		s.close()
	return ip


def message_filter(device: str, variable: str):
	"""
	Create an unambiguous filter string for an access point.

	If the backend used requires a single string for message filtering
	(e.g. ZeroMQs PUB-SUB-Pattern), this function defines the correct
	filter to be used.

	:param device: device_name of the access point the message addresses.
	:param variable: variable name of the access point the message addresses.
	:return: Unambiguous filter string (with regard to prefix matching).
	"""
	return str(device) + ":" + str(variable)
	# TODO make actually unambiguous


class RawMessage(NamedTuple):
	content: Dict[str, Any]
	data: List[bytes]


def create_message(device: str, variable: str, command: str, data: Optional[List[bytes]] = None, **kwargs) -> RawMessage:
	"""
	Create a valid message from the arguments.

	This function standardizes the way to create a message, ensures all
	required parts are provided and supplies default values.

	:param device: Device name the message is sent to (for commands).
		Device name the message is sent by (for replies, updates).
	:param variable: Variable name the message is sent to.
		Variable name the message is sent by (for replies, updates).
	:param command: Command identifier for the operation to perform.
	:param kwargs: Other keyword arguments are added as additional parts of
		the message.
	:return: A dictionary that is a valid JSON-compatible message.
	"""
	message = dict()
	message['levelindex'] = 0
	for arg in kwargs:
		if kwargs[arg] is not None:
			message[arg] = kwargs[arg]
	message['nomen'] = str(device)
	message['property'] = str(variable)
	message['command'] = str(command)
	message['timestamp'] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
	message['timezone'] = "UTC"
	if not data:
		data = []
	else:
		data = list(data)
	return RawMessage(message, data)


def create_reply(message: RawMessage, reply_data: Optional[List[bytes]] = None, command: Optional[str] = None, error: Optional[str] = None, timestamp: Optional[str] = None, timezone: Optional[str] = None, **kwargs) -> RawMessage:
	"""
	Create a reply to a message.

	This function standardizes the way to create a reply to a message,
	ensures all required parts are provided and supplies default values.

	:param message: Message the reply is supposed to reply to.
	:param kwargs: other parts to be included in the reply.
	:return:  A dictionary that is a valid JSON-compatible message.
	"""
	reply = message.content.copy()
	if command is None:
		if error is None:
			command = str(message.content['command']) + "_REPLY"
		else:
			command = "ERROR"
	reply['command'] = command
	if error is not None:
		reply['error'] = str(error)
	if timestamp is not None:
		reply['timestamp'] = str(timestamp)
	else:
		reply['timestamp'] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
	if timezone is not None:
		reply['timezone'] = str(timezone)
	else:
		reply['timezone'] = "UTC"
	for arg in kwargs:
		if kwargs[arg] is not None:
			reply[arg] = kwargs[arg]
	if not reply_data:
		reply_data = []
	else:
		reply_data = list(reply_data)
	return RawMessage(reply, reply_data)


def get_bound_method_object_or_class() -> Union[object, type, None]:
	# Step one: get the frame in which the calling method was called
	# different approaches are used because the behavior is OS-dependent
	frame = inspect.currentframe()
	if frame is None:
		try:
			frame = inspect.stack()[2].frame
		except IndexError:
			return None
	else:
		try:
			frame = frame.f_back.f_back
		except AttributeError:
			return None
	# Step two: get the first argument
	# the first argument is used to determine if the frame determined
	# belongs to a bound method, class method or a class declaration
	names, args, kwargs, vars = inspect.getargvalues(frame)
	try:
		name = names[0]
	except IndexError:
		if args is not None:
			try:
				candidate, *_ = vars[args]
				return candidate
			except ValueError:
				pass
		try:
			vars = vars[kwargs]
		except KeyError:
			return None
		try:
			name = next(iter(vars))
		except StopIteration:
			return None
	try:
		candidate = vars[name]
	except KeyError:
		return None
	# Step three: check if the variable names fit the convention
	if isinstance(candidate, type) and name == 'cls':
		# class method
		return candidate
	elif name == 'self':
		# bound method
		return candidate
	else:
		# Doesn't fit the convention, is discarded (for now)
		return None


class DatatypeAccessor(ABC):
	"""
	Interface for telling a Connection how to handle data

	This class is an interface for classes that are used for the processing of messages. The interface does not expect
	to be created at any point, all methods are classmethods. It has been implemented this way because it allows an easy
	back-and-forth conversion between object that can be used to interpret messages and a string identifier for the same:
	__classpath__.__qualname__

	In practise, the interface is implemented by all Variable classes, so that the Variable's class can be used as type_id
	"""
	# this actually works slightly differently now

	def __init__(self, *args, **kwargs):
		if type(self).setup_data_storage == DatatypeAccessor.setup_data_storage:
			raise TypeError("DatatypeAccessor subclasses must overwrite the setup_data_storage method as a classmethod.")
		elif type(self).store_message_into_data == DatatypeAccessor.store_message_into_data:
			raise TypeError("DatatypeAccessor subclasses must overwrite the store_message_into_data method as a classmethod.")
		elif type(self).value_from_stored_data == DatatypeAccessor.value_from_stored_data:
			raise TypeError("DatatypeAccessor subclasses must overwrite the value_from_stored_data method as a classmethod.")
		elif type(self).value_to_message_value_and_arguments == DatatypeAccessor.value_to_message_value_and_arguments:
			raise TypeError("DatatypeAccessor subclasses must overwrite the value_to_message_value_and_arguments method as a classmethod.")
		super().__init__(*args, **kwargs)

	@classmethod
	@abstractmethod
	def setup_data_storage(cls) -> Dict[str, Any]:
		raise NotImplementedError

	@classmethod
	@abstractmethod
	def store_message_into_data(cls, message: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
		raise NotImplementedError

	@classmethod
	@abstractmethod
	def value_from_stored_data(cls, data: Dict[str, Any]) -> Any:
		raise NotImplementedError

	@classmethod
	@abstractmethod
	def value_to_message_value_and_arguments(cls, value: Any) -> Tuple[Any, Dict[str, Any]]:
		raise NotImplementedError


# proteus Error classes  TODO needs to be integrated
class ProteusUserCodeError(RuntimeWarning):
	"""
	Class for handling errors in user-created code.
	Proteus calls user-written code in multiple locations.
	When errors occur there, proteus usually ignores them.
	The main purpose of this class is to document these
	incidents so errors can be fixed or handling be added.
	"""
	pass


class ProteusError(RuntimeError):
	"""
	Class for internal proteus errors.
	This class is intended for internal errors that can not
	be handled automatically but require code or usage changes.
	It mostly tells the user that what happened comes from proteus
	and where in proteus it happened.
	"""
	pass


class DeviceNotFound(RuntimeWarning):
	"""
	Error to signify that a target Device is not in the network
	"""
	pass


ReplyFunction = Callable[[RawMessage], RawMessage]
# TODO change the return type to RawMessage and add the returned message of
#  the reply function to the reaction function arguments (relevant for
#  Trigger return values)
#  Needs adaption in every existent ReplyFunction and everywhere it is used
ReactionFunction = Callable[[RawMessage, RawMessage], Any]
MessageCallback = Callable[[RawMessage], Any]
