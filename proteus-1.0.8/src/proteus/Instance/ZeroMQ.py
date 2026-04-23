"""
Classes for the ZeroMQ implementation of the communication.

The ZeroMQ package contains all classes that are specific to the use of
ZeroMQ as communication protocol. While a new Instance is technically
sufficient for a new protocol, the ZeroMQ Instance uses multiple classes
to implement the correct threading behavior in the Qt threading
framework.
"""
from queue import PriorityQueue, SimpleQueue, Empty
from threading import Thread, Lock, current_thread
from typing import final, Final, Dict, Tuple, Callable, List, Any, Optional, Union
from time import sleep
import zmq
from zmq import ZMQError
from zmq.utils.jsonapi import loads

from proteus.Data import InstanceReference
from proteus.Instance import Instance
from proteus.MessageProcessor import StoredCommand, MessageProcessor
from proteus.Devices import Device
from proteus.Utility import create_reply, message_filter, create_message, get_ip, ProteusError, RawMessage, \
	DeviceNotFound


class ZeroMQInstanceReference(InstanceReference):
	"""
	Extension of InstanceReference to contain ZeroMQ-specific data.

	This class handles the specific data required for the ZeroMQ-
	Implementation of the Instance. This includes handling the
	formatting of the strings for the different forms of addresses required
	for pyzmq and verification of relevant properties like preventing of
	setting the addresses to different computers.

	Properties:
		name: Optional user-friendly identifier for the instance.
		user: Optional information on the current user of the instance.
	None of the following properties can be edited (operations can be
	changed indirectly by methods).
		device_name: Device name of the access point (always empty string).
		variable_name: Variable name of the access point (always empty string).
		type_id: Identifier for the type of access point.
		identification: Identifier of the specific instance(backend-dependent).
		backend_type: Identifier for the network backend used by the Instance.
		operations: Valid operations for this Interface.
		command_address: Address of the primary network interface.
		command_port: Port number of the primary network interface.
		publish_address: Address of the secondary network interface.

	Methods:
		add_command: Add a command as valid for the Instance.
		remove_command: Remove a command from the list of valid commands for
			the Instance.
		add_device: Add a Device to those managed by the Instance.
		remove_device: Remove a Device from those managed by the Instance.
		local_command_address: Alternate format of the command address.
		publish_port: Port number of the secondary network interface.
		local_command_address: Alternate format of the publish address.
	"""
	_protocol: str
	_address: str
	_command_port: int
	_publish_port: int

	def __init__(self, address: str, publish_address: Union[str, int, None] = None, **kwargs):
		"""
		An Instance in ZeroMQ-Implementation is identified by the address of its
		command port. For this reason, no further information is required at
		creation time.

		:param address: Address the Instance is to be available under, including
			port number
		:param kwargs: Accepted for consistency but not processed.
		"""
		super().__init__(address, backend_type="ZeroMQ", **kwargs)
		[protocol, address] = address.split("://", 1)
		[address, port_str] = address.rsplit(":", 1)
		self._protocol = str(protocol)
		self._address = str(address)
		self._command_port = int(port_str)
		self._publish_port = -1
		if publish_address is None:
			self._publish_port = -1
		else:
			if isinstance(publish_address, str):
				publish_address = publish_address.rsplit(":", 1)[-1]
			try:
				publish_address = int(publish_address)
				if publish_address > 0:
					self._publish_port = publish_address
				else:
					self._publish_port = -1
			except ValueError:
				self._publish_port = -1

	@property
	def command_address(self) -> str:
		"""
		Address of the primary network interface of the Instance.

		The command address is the primary address of an instance. All control
		messages are exchanged via this port, and it is considered the
		identifying property of the Instance. For this reason, it can not be
		changed at runtime.
		:return: Network address of the command port.
		"""
		return self._protocol + "://" + self._address + ":" + str(self._command_port)

	def local_command_address(self) -> str:
		"""
		Different format of the command address for socket creation.

		:return: Command address int the format required for the creation of a
			ZeroMQ socket.
		"""
		return self._protocol + "://*:" + str(self._command_port)

	@property
	def command_port(self) -> int:
		"""
		Port of the command address.

		:return: Port number of the command address.
		"""
		return self._command_port

	@property
	def publish_address(self) -> Optional[str]:
		"""
		Address of the secondary network interface of the Instance.

		The secondary port of a ZerMQInstance is used for value updates and
		similar messages without a specific recipient. It can only differ from
		the primary address by the port number as they have to be on the same
		computer. While it is not expected to change at runtime, it is not part
		of the identifying information of the instance and can be changed. This
		will require the user to manually repeat connecting all other instances
		for updates, so it is not recommended after setup.
		:return: Network address of the publish port.
		"""
		if self._publish_port > 0:
			return self._protocol + "://" + self._address + ":" + str(self._publish_port)
		else:
			return None

	@publish_address.setter
	def publish_address(self, publish_address: Union[str, int]):
		if isinstance(publish_address, str):
			publish_address = publish_address.rsplit(":", 1)[-1]
		try:
			publish_address = int(publish_address)
			if publish_address > 0:
				self._publish_port = publish_address
			else:
				self._publish_port = -1
		except ValueError:
			self._publish_port = -1

	# TODO make into a property as soon as AddressChecker is obsolete
	def publish_port(self) -> int:
		"""
		Port of the publish address.

		:return: Port number of the publish address.
		"""
		return self._publish_port

	def local_publish_address(self) -> str:
		"""
		Different format of the publish address for socket creation.

		:return: Publish address int the format required for the creation of a
			ZeroMQ socket.
		"""
		if self._publish_port > 0:
			return self._protocol + "://*:" + str(self._publish_port)
		else:
			return ""


@final
class Instance(ZeroMQInstanceReference, Instance):

	_zmq_context: Final[zmq.Context]
	_known_instances: Dict[str, ZeroMQInstanceReference]
	# This list also includes aliases, use known_instance_addresses for only distinct real addresses
	_local_inproc_address = "inproc://proteus_instance_internal"
	_local_new_subscription_address = "inproc://proteus_instance_subscription"

	_request_timeout: int
	_instance_timeout: int

	_command_thread: Thread
	_command_senders: SimpleQueue[zmq.Socket]
	_command_listener: zmq.Socket
	_registered_commands_lock: Lock

	_listener_thread: Thread
	_listener_sockets: Dict[str, zmq.Socket]
	_listener_sockets_lock: Lock
	_publish_socket: zmq.Socket
	_publish_lock: Lock

	_device_threads: Dict[Device, Thread]

	def __init__(self, address: str, pub_address: Union[str, int, None] = None, **kwargs):
		super().__init__(address, pub_address, **kwargs)
		address = self.command_address
		self._known_instances[address] = self
		self._is_running = False
		self._zmq_context = zmq.Context()
		self._zmq_context.linger = 500
		self._request_timeout = 500
		self._instance_timeout = 100

		self._network_thread = Thread(target=self._network_loop)

		self._command_senders = SimpleQueue()
		first_socket = self._zmq_context.socket(zmq.REQ)
		first_socket.sndtimeo = self._request_timeout
		first_socket.rcvtimeo = self._request_timeout
		self._command_senders.put(first_socket)

		self._registered_commands_lock = Lock()

		self._publish_socket = self._zmq_context.socket(zmq.PUB)
		self._publish_lock = Lock()

		self._listener_sockets = {}
		self._listener_sockets_lock = Lock()
		self._device_threads = {}
		if pub_address is not None:
			self.publish_address = pub_address

	# enter/exit methods for use as a context manager
	# make cleaner (threading-wise) by moving things like socket options and closing to the loops
	def __enter__(self):
		ret_val = super().__enter__()
		self._network_thread.start()
		for device in self._device_threads:
			self._device_threads[device].start()
		return ret_val

	def __exit__(self, exc_type, exc_val, exc_tb):
		super().__exit__(exc_type, exc_val, exc_tb)
		devices = []
		# create a flat copy, so the devices can be removed in a simple loop
		for device_name in self:
			devices.append(device_name)
		for device in devices:
			self[device].exit()
		self._network_thread.join()
		while True:
			try:
				self._command_senders.get_nowait().close()
			except Empty:
				break
		if hasattr(self._processing_thread, 'proteus_zmq_internal_reply_socket'):
			self._processing_thread.proteus_zmq_internal_reply_socket.close()
		self._publish_socket.close()
		for socket in self._listener_sockets.values():
			socket.close()
		self._zmq_context.term()

	# Methods that define the effects of the network management messages
	# Some of these contain ZeroMQ specific information, so they can not
	# be moved to BaseInstance without further consideration.
	def _join_reply(self, request: RawMessage) -> RawMessage:
		reply = create_reply(request, value=self.known_instance_addresses(), pub_port=self.publish_port(), status=self._status_json())
		return reply

	def _join_reaction(self, request: RawMessage, reply: Optional[RawMessage]) -> None:
		try:
			addresses = request.content['value']
			instance = addresses.pop(0)
		except KeyError as ex:
			self.warn(f"Invalid JOIN request received: {request!s}", ex)
			return
		status = request.content.get('status', None)
		pub_port = request.content.get('pub_port', None)
		self._join_processing(instance, instance, status, pub_port)
		self.add_instances(addresses)

	def _status_reply(self, request):
		return create_reply(
			request,
			value=self._status_json(),
			instances=self.known_instance_addresses(),
			pub_port=self.publish_port(),
		)
	
	# TODO move to variable & device, this makes the argument checking obsolete and causes a clearer error when trying
	#  to access a non-existent variable. Also, include the type_id somehow.
	def _variable_reply(self, request: RawMessage) -> Tuple[RawMessage, Any]:
		print('deprecated _variable_reply')
		try:
			device = request.content['value'][0]
			variable = request.content['value'][1]
			operations = self._devices[device][variable].operations
		except KeyError:
			operations = []
		return create_reply(request, value=operations), None

	def _instance_message(self, message: RawMessage) -> None:
		try:
			command = message.content['command']
		except KeyError as ex:
			self.warn(f"Incoming message without command: {message!s}", ex)
			return
		if command == "UPDATE":
			self._join_reaction(message, None)
		# elif command == "LEAVE":
		# 	self._leave_reaction(message)
		else:
			self.warn(f"Message with unknown command {command!s}")
	# End of network management message methods

	_new_listener_sockets = PriorityQueue()

	def _network_loop(self):
		# receives all zeromq messages and adds them to queues to be processed
		# sockets:
		#  one inproc SUB socket for the replies
		#  one command socket, tcp ROUTER
		#  multiple listener sockets, tcp SUB
		with self._zmq_context.socket(zmq.SUB) as reply_inproc, self._zmq_context.socket(zmq.ROUTER) as command_listener, self._zmq_context.socket(zmq.REP) as subscription_listener:
			reply_inproc.bind(self._local_inproc_address)
			reply_inproc.subscribe('')
			command_listener.bind(self.local_command_address())
			subscription_listener.bind(self._local_new_subscription_address)
			listener_sockets: Dict[str, zmq.Socket]
			candidates = [reply_inproc, subscription_listener, command_listener]
			if self.publish_address:
				local_listener_socket = self._zmq_context.socket(zmq.SUB)
				local_listener_socket.connect(self.publish_address)
				listener_sockets = {self.command_address: local_listener_socket}
				candidates.append(local_listener_socket)
			else:
				listener_sockets = {}
			try:
				while self._is_running:
					sockets, _, _ = zmq.select(candidates, [], [], self._instance_timeout/1000)
					if reply_inproc in sockets:
						# Reply that needs to be sent
						# correct message is created by processing, only pass it on.
						command_listener.send_multipart(reply_inproc.recv_multipart(flags=zmq.NOBLOCK))
					elif subscription_listener in sockets:
						action, new_address, pub_address_or_filter, *other = subscription_listener.recv_multipart(flags=zmq.NOBLOCK)
						subscription_listener.send_string('done')
						new_address = new_address.decode()
						if action == b'add':
							filter = message_filter('', '')
							pub_address = pub_address_or_filter.decode()
							try:
								old_socket = listener_sockets.pop(new_address)
								candidates.remove(old_socket)
							except KeyError:
								pass  # No old socket to close
							else:
								old_socket.close()
							socket = self._zmq_context.socket(zmq.SUB)
							socket.connect(pub_address)
							listener_sockets[new_address] = socket
							candidates.append(socket)
						elif action in (b'subscribe', b'unsubscribe'):
							filter = pub_address_or_filter.decode()
							socket = listener_sockets[new_address]
						else:
							print('unknown action in subscription_listener')
							continue
						if action in (b'add', b'subscribe'):
							socket.subscribe('')
						elif action == b'unsubscribe':
							socket.unsubscribe(filter)
					elif sockets:
						if command_listener in sockets:
							socket = command_listener
							msg_type = "command"
						else:
							socket = sockets[0]
							msg_type = "message"
						address_info, blank, message, *data = socket.recv_multipart(flags=zmq.NOBLOCK)
						message = loads(message)
						self._processing_queue.put(StoredCommand(msg_type, RawMessage(message, data), address_info))
			finally:
				for socket in listener_sockets.values():
					socket.close()

	def _subscribe(self, device_name: str, variable_name: str, command: str) -> None:
		with self._zmq_context.socket(zmq.REQ) as socket:
			with socket.connect(self._local_new_subscription_address):
				address = self.get_instance_address(device_name)
				socket.send(b'subscribe', flags=zmq.SNDMORE)
				socket.send_string(address, flags=zmq.SNDMORE)
				socket.send_string(message_filter(device_name, variable_name))
				if socket.recv_string() != 'done':
					print(f'subscribing to {device_name=}, {variable_name=} not successful')

	def _unsubscribe(self, device_name: str, variable_name: str, callback: Callable) -> None:
		with self._zmq_context.socket(zmq.REQ) as socket:
			with socket.connect(self._local_new_subscription_address):
				address = self.get_instance_address(device_name)
				socket.send(b'unsubscribe', flags=zmq.SNDMORE)
				socket.send_string(address, flags=zmq.SNDMORE)
				socket.send_string(message_filter(device_name, variable_name))
				if socket.recv_string() != 'done':
					print(f'subscribing to {device_name=}, {variable_name=} not successful')

	# TODO update to send_json, requires changing arguments of publish_variable
	def _publish(self, message: Dict[str, Any], data: Optional[List[bytes]] = None) -> None:
		with self._publish_lock:
			self._publish_socket.send_string(message_filter(message['nomen'], message['property']), flags=zmq.SNDMORE)
			self._publish_socket.send(b'', flags=zmq.SNDMORE)  # creates consistency between command and message formats
			if data:
				self._publish_socket.send_json(message, flags=zmq.SNDMORE)
				self._publish_socket.send_multipart(data)
			else:
				self._publish_socket.send_json(message)

	def _send_command(self, target_address: str, message: RawMessage) -> RawMessage:
		try:
			socket = self._command_senders.get_nowait()
		except Empty:
			socket = self._zmq_context.socket(zmq.REQ)
			socket.sndtimeo = self._request_timeout
			socket.rcvtimeo = self._request_timeout
		try:
			with socket.connect(target_address):
				try:
					if message.data:
						socket.send_json(message.content, flags=zmq.SNDMORE)
						socket.send_multipart(message.data)
					else:
						socket.send_json(message.content, flags=0)
				except zmq.ZMQError as ex:
					raise ProteusError(f"Unable to send message {message.content!s} to {target_address!s}") from ex
				try:
					replies = socket.recv_multipart(flags=0)
					reply = RawMessage(loads(replies[0]), replies[1:])
				except zmq.ZMQError as ex:
					raise ProteusError(f"Unable to receive a reply to {message.content!s} from {target_address!s}") from ex
				except IndexError as ex:
					raise ProteusError(f"Incomplete reply to {message.content!s} from {target_address!s}") from ex
			return reply
		except ZMQError as ex:
			# This happens if the address is nonsense
			raise DeviceNotFound from ex
		finally:
			if self._is_running:
				self._command_senders.put(socket)
			else:
				socket.close()

	def known_instance_addresses(self) -> List[str]:
		"""
		Create a list of all known Instances.

		:return: List of command addresses.
		"""
		return [alias for alias in self._known_instances if alias == self._known_instances[alias].command_address]

	def get_pub_address(self, instance_address: str) -> Optional[str]:
		"""
		Determine the publish address of an Instance.

		:param instance_address: Address of the Instance whose port to determine.
		:return: Publish address of the Instance.
		"""
		try:
			return self._known_instances[instance_address].publish_address
		except KeyError as ex:
			self.warn("Unknown instance address. Communication is only possible with known Instances", ex)
			return None

	def add_instance(self, new_address: str) -> None:
		self.add_instances([new_address])

	def add_instances(self, new_addresses: List[str]) -> None:
		while len(new_addresses) > 0:
			address = new_addresses.pop(0)
			if address not in self._known_instances:
				message = create_message("", "", "JOIN", value=self.known_instance_addresses(), pub_port=self.publish_port(), status=self._status_json())
				try:
					reply = self._send_command(address, message)
				except DeviceNotFound as ex:
					self.warn(f"Address {address} is invalid", ex)
					continue
				if reply is None:
					self.warn(f"No response to JOIN request st {address!s}")
					continue
				try:
					new_instance = reply.content['value'][0]
					new_addresses.extend(reply.content["value"][1:])
				except KeyError as ex:
					self.warn(f"Invalid response to JOIN request to {address!s}", ex)
					continue
				try:
					status_info = reply.content['status']
				except KeyError:
					status_info = self._send_command(new_instance, create_message("", "", "STATUS"))
				pub_port = reply.content.get('pub_port', -1)
				self._join_processing(new_instance, address, status_info, pub_port)

	def _join_processing(self, instance: str, alias: str, status: Optional[Dict[str, Dict[str, str]]], pub_port: Optional[int]) -> None:
		self._new_listener_sockets.put(instance)
		instance = self._known_instances.setdefault(instance, ZeroMQInstanceReference(instance, pub_port))
		if instance is self:
			return
		self._known_instances[alias] = instance
		pub_address = instance.publish_address
		if pub_address:
			with self._listener_sockets_lock,  self._zmq_context.socket(zmq.REQ) as socket:
				socket.connect(self._local_new_subscription_address)
				socket.send(b'add', flags=zmq.SNDMORE)
				socket.send_string(instance.identification, flags=zmq.SNDMORE)
				socket.send_string(pub_address)
				if socket.recv_string() != 'done':
					print('adding new listener socket not successful')
			if pub_address not in self._listener_sockets:
				socket = self._zmq_context.socket(zmq.SUB)
				socket.connect(pub_address)
				socket.subscribe(message_filter("", ""))
				self._listener_sockets[pub_address] = socket
		if status is not None:
			self.process_status_update(instance, status)

	@property
	def publish_address(self) -> Optional[str]:
		if self._publish_port > 0:
			return self._protocol + "://" + self._address + ":" + str(self._publish_port)
		else:
			return None

	@publish_address.setter
	def publish_address(self, publish_address: Union[str, int]):
		if isinstance(publish_address, str):
			publish_address = publish_address.rsplit(":", 1)[-1]
		try:
			publish_address = int(publish_address)
			if publish_address > 0:
				self._publish_port = publish_address
			else:
				self._publish_port = -1
		except ValueError:
			self._publish_port = -1
		try:
			print(self.local_publish_address())
			self._publish_socket.bind(self.local_publish_address())
		except zmq.ZMQError as ex:
			self._publish_port = -1
			raise ex
		self._listener_sockets[self.publish_address] = self._zmq_context.socket(zmq.SUB)
		self._listener_sockets[self.publish_address].connect(self.publish_address)

	def _join_args(self) -> Dict[str, Any]:
		return {'value': self.known_instance_addresses(), 'pub_port': self.publish_port(), 'status': self._status_json()}

	def setup_message_processor(self, processor: MessageProcessor) -> None:
		socket = self._zmq_context.socket(zmq.PUB)
		socket.connect(self._local_inproc_address)
		processor.local_objects = socket
		sleep(0.01)

	def cleanup_message_processor(self, processor: MessageProcessor) -> None:
		socket = processor.local_objects
		socket.close()
		del processor.local_objects

	def send_reply(self, command: StoredCommand, reply: RawMessage):
		thread = current_thread()
		try:
			socket = thread.proteus_zmq_internal_reply_socket
		except AttributeError:
			socket = self._zmq_context.socket(zmq.PUB)
			# TODO find a way to clean up these sockets
			# socket = cast(processor.local_objects, zmq.Socket)
			# Idea: thread subclass that does it
			socket.connect(self._local_inproc_address)
			thread.proteus_zmq_internal_reply_socket = socket
			# the connection is only set up properly after some time, so wait 10 ms (its inproc, it should be fast)
			sleep(0.01)
		socket.send(command.address_info, flags=zmq.SNDMORE)
		socket.send(b"", flags=zmq.SNDMORE)
		if reply.data:
			socket.send_json(reply.content, flags=zmq.SNDMORE)
			socket.send_multipart(reply.data)
		else:
			socket.send_json(reply.content)
		super().send_reply(command, reply)

	@classmethod
	def from_config(cls, protocol: str = 'tcp', address: Optional[str] = None, command_port: int = 6000, publish_port: int = 6001):
		if not address:
			address = get_ip()
		instance = cls(address=f'{protocol}://{address}:{command_port}')
		instance.publish_address = publish_port
		return instance
