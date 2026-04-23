from typing import Optional, Any, cast
from collections.abc import Callable

from .Instance import Instance
from .Devices import ActiveDevice as Device
from .Encoder import JSONEncoder as JSON, NumpyArrayEncoder as npArray, Datatype
from .Utility import get_bound_method_object_or_class
from .Devices import Device
from .Property.Variable import Variable as Variable_class
from .Property import PropertyDescriptor
from .Property.Trigger import Trigger
from proteus.Qt.Window import Window


def create_variable_or_descriptor(datatype: Datatype, name: str = '', *args, **kwargs):
	device_candidate = get_bound_method_object_or_class()
	if isinstance(device_candidate, Device):
		return Variable_class(name, device_candidate, datatype=datatype, *args, **kwargs)
	elif isinstance(device_candidate, type) and issubclass(device_candidate, Device):
		raise NotImplementedError("Creation of Variables in classmethods is not implemented yet")
	else:
		return PropertyDescriptor(Variable_class, name, datatype=datatype, *args, **kwargs)


def create_trigger_or_descriptor(datatype: Datatype) -> Callable[[Callable], Any]:
	device_candidate = get_bound_method_object_or_class()
	if isinstance(device_candidate, Device):
		def create_trigger(triggered_function: Callable):
			def function_generator(device: Device):
				return triggered_function
			return Trigger(triggered_function.__name__, cast(device_candidate, Device), datatype, function_generator)
	elif isinstance(device_candidate, type) and issubclass(device_candidate, Device):
		raise NotImplementedError("Creation of Triggers in classmethods is not implemented yet")
	else:
		def create_trigger(triggered_function: Callable):
			if hasattr(triggered_function, '__get__'):
				def function_generator(device: Device):
					return triggered_function.__get__(device)
			else:
				def function_generator(device: Device):
					return triggered_function
			return PropertyDescriptor(Trigger, triggered_function.__name__, triggered_function, datatype, function_generator)
	return create_trigger


def get_default_instance(*, instance_type: str = 'zmq', protocol: str = 'tcp', address: Optional[str] = None, command_port: int = 6000, publish_port: int = 6001, port_range: int = 1000, port_attempts: int = 50, name = None, **kwargs) -> Instance:
	"""
	Create Instance using default protocol (ZeroMQ) and guessing parameters.

	This function implements the standard way to create an instance. It uses
	the ZeroMQ protocol and attempts to locate a suitable address and port
	without user input. For a stable central point of the network it can be
	useful to supply a more memorable address instead of the ip address and
	to clearly define the port by passing a specific port and setting
	attempts to 0. This function does not accept positional arguments.

	:param protocol: Name of the transport to be used by ZeroMQ.
		See http://api.zeromq.org/master:zmq-connect for acceptable values.
	:param address: Valid address to reach the computer the program runs on.
		The format of a valid address is defined by the transport.
	:param port: When using TCP as transport, a port is used as part of the
		address. This argument is used as the starting point for the search
		for a free port.
	:param attempts: Number of additional ports to try if the first one
		fails. Must be a non-negative value.
	:param kwargs: Other keyword arguments will be passed to the
		ZeroMQInstance class.
	:return: A ZeroMQInstance object if a valid open port has been found with
		the given parameters, otherwise None.
	"""
	# consider a .pyi file when there are multiple possible
	# instance_type values as the arguments will almost certainly
	# be different. Arguments for types besides the default type
	# should not be added to the function definition, but read from
	# kwargs. This is necessary for read_configuration to work.
	if instance_type == 'zmq':
		if command_port < 1024 or command_port > 65535:
			raise RuntimeError(f'Command port {command_port} is outside the allowed range for user/dynamic ports')
		if publish_port < 1024 or publish_port > 65535:
			raise RuntimeError(f'Command port {command_port} is outside the allowed range for user/dynamic ports')
		if address is None:
			from proteus.Utility import get_ip
			address = get_ip()
		import zmq
		test_context = zmq.Context()
		with test_context.socket(zmq.PUB) as test_socket, test_context.socket(zmq.PUB) as test_socket_2:
			try:
				test_socket.bind(f'{protocol}://{address}:{command_port}')
			except zmq.ZMQError:
				if port_range > 1:
					try:
						command_port = test_socket.bind_to_random_port(f'{protocol}://{address}', min_port=command_port + 1, max_port=command_port + port_range, max_tries=port_attempts)
					except zmq.ZMQBindError:
						raise RuntimeError(
							f'A valid command port could not be found (range: {command_port}-{command_port + port_range}, {port_attempts} tries)')
				else:
					raise RuntimeError(f'The desired command port {command_port} could not be bound')
			try:
				test_socket_2.bind(f'{protocol}://{address}:{publish_port}')
			except zmq.ZMQError:
				if port_range > 1:
					try:
						publish_port = test_socket.bind_to_random_port(f'{protocol}://{address}', min_port=publish_port + 1, max_port=publish_port + port_range, max_tries=port_attempts)
					except zmq.ZMQBindError:
						raise RuntimeError(
							f'A valid publish port could not be found (range: {publish_port}-{publish_port + port_range}, {port_attempts} tries)')
				else:
					raise RuntimeError(f'The desired publish port {publish_port} could not be bound')
		command_address = f'{protocol}://{address}:{command_port}'
		from .Instance.ZeroMQ import Instance as ZeroMQInstance
		instance = ZeroMQInstance(command_address, **kwargs)
		if name is None:
			import socket
			name = socket.gethostname()
		else:
			name = str(name)
		instance.name = name
		import getpass
		instance.user = getpass.getuser()
		print(instance.command_address)
		instance.publish_address = publish_port
		return instance
	# add new Instance types here
	else:
		raise RuntimeError(
			f'Instance type {instance_type} is not a PROTEUS default and can not be initialized by proteus.Instance')


# TODO readd device_directory, window_directory parameters so they are generated in the default config
#  TBD what should the default be for optimal Qt file selection window behavior?
def get_default_window(instance: Instance, /, *, window_type: str = 'qt', qt_args: Optional[list[str]] = None, **kwargs):
	if window_type == 'qt':
		from proteus.Qt import setup_qt
		manager = setup_qt(qt_args)

		@manager.execute
		def create_ui(inst: Instance, **wnd_args):
			from time import sleep
			wind = Window(inst, **wnd_args)
			sleep(0.05)
			wind.show()
			sleep(0.05)
			return wind

		window = create_ui(instance, **kwargs)
		return window
	# Add new Window types here
	else:
		raise RuntimeError(f'Window type {window_type} is not a PROTEUS default and can not be initialized by proteus.Window')


def read_configuration(config_file: Optional[str] = None, relevant_methods: tuple[Callable, ...] = (Instance, )) -> dict[str, Any]:
	if not config_file:
		from inspect import signature, Parameter
		default_config = {}
		for method in relevant_methods:
			name = method.__name__
			default_config[name] = {}
			parameters = signature(method).parameters
			for parameter_name in parameters :
				parameter = parameters[parameter_name]
				if parameter.kind is not Parameter.POSITIONAL_ONLY:
					# special condition to exclude the instance parameter of Window that has to be supplied at runtime
					default_value = parameter.default
					if default_value is not Parameter.empty and default_value is not None:
						default_config[name][parameter_name] = default_value
		return default_config
	import json
	try:
		with open(config_file, mode='r') as file:
			return json.load(file)
	except OSError:
		print("No configuration file found, creating default configuration file")
		config = read_configuration(relevant_methods=relevant_methods)
		with open(config_file, mode='x') as file:
			json.dump(config, file)
		return config
	except json.JSONDecodeError:
		print("Configuration file is invalid, using default configuration.")
		return read_configuration(relevant_methods=relevant_methods)