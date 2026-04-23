from typing import List, Dict, Iterator, Union


class VariableReference:
	"""
	Data structure for basic information on an access point.

	Every object of the framework that can be accessed remotely (usually
	referenced as an access point) has its basic information stored in this
	format or extensions of it. The access points are organized in Devices.
	Access points within devices are called Variables.
	The combination of Device name and Variable name has to be unique within
	the network (with the exception of the case where both are the empty
	string, as used by Instances). This is not verified automatically.

	Properties:
		device_name: Name of the Device the access point belongs to.
		variable_name: Name of the access point (within its Device).
		type_id: Identifier for the type of access point to match the
			classes used in communication.
		operations: Valid operations for this access point.
	None of the properties can be edited (operations can be changed
	indirectly by the methods).

	Methods:
		add_command: Add a command as valid for the access point.
		remove_command: Remove a command from the list of valid commands for
			the access point.
	"""
	_device_name: str
	_variable_name: str
	_operations: List[str]
	_type_id: str

	def __init__(self, device: str, name: str, type_id: str, **kwargs):
		"""
		An access point is identified by the combination of device and name.
		For this reason, the combination should be unique in the network. The
		only exception is the combination where both are the empty string. This
		combination is reserved and will exist on every computer. This rule is
		neither enforced nor verified and has to be managed by the user. If it
		is violated, non-deterministic erroneous behavior can occur.
		Explanations for the meaning of the empty string can be found in the
		docstring of the DeviceReference and InstanceReference classes of this
		module. An empty variable name should not be utilized outside those
		cases.
		The type_id is used to identify the correct conversion between messages
		in JSON-Format and the data type used by the program.

		:param device: Name of the Device the access point belongs to
		:param name: Name of the access point within the Device
		:param type_id: Identifier for the type of access point.
		"""
		super().__init__(**kwargs)
		# TODO replace properties with attributes of final type?
		if device is None:
			raise AttributeError("Attempting to create a proteus access point without device name")
		self._device_name = str(device)
		if name is None:
			raise AttributeError("Attempting to create a proteus access point without variable name")
		self._variable_name = str(name)
		self._operations = []
		self._type_id = str(type_id)

	# The following properties are defining characteristics of an access point and as such should never change during
	# the lifetime of a variable.
	@property
	def device_name(self) -> str:
		"""
		Name of the device the access point belongs to.

		:return: Name of the Device the access point belongs to
		"""
		return self._device_name

	@property
	def variable_name(self) -> str:
		"""
		Identifier of the access point within its Device.

		:return: Variable name of the access point
		"""
		return self._variable_name

	@property
	def type_id(self) -> str:
		"""
		Identifier for the data type and conversion of the access point.

		:return: Identifier of the Variable type.
		"""
		return self._type_id

	@property
	def operations(self) -> List[str]:
		"""
		Valid operations to be sent to the access point.

		The operations can not be overwritten in bulk because this action is
		problematic in child classes. For this reason, the list of operations
		is read-only and can only be edited by adding or removing commands
		individually.

		:return: A list of valid commands to send to the access point.
		"""
		return self._operations.copy()

	# A VariableReference only stores the names of the relevant operations. They are added/removed the same way.
	# function arguments:
	#  name: the name of the command. This is the identifying string that defines the command.
	# If the key to be added is already in the list or the key to be removed is not in the list, a KeyError is raised
	def add_command(self, name: str) -> None:
		"""
		Add a command to the list of valid commands for this access point.

		:param name: The identifying string that defines the command.
		:return: True if the command was added successfully, False otherwise.
		"""
		name = str(name)
		if name in self.operations:
			raise KeyError(f"Command {name} is already present at the access point device: {self._device_name!s}, variable: {self._variable_name!s} and can not be added.")
		self.operations.append(name)

	def remove_command(self, name: str) -> None:
		"""
		Remove a command from the list of valid commands for this access point.

		:param name: The identifying string that defines the command.
		:return: True if the command was added successfully, False otherwise.
		"""
		name = str(name)
		if name not in self.operations:
			raise KeyError(f"Command {name} is not present at the access point device: {self._device_name!s}, variable: {self._variable_name!s} and can not be removed.")
		self.operations.remove(name)


class DeviceReference(VariableReference):
	"""
	Data concerning containers for access points and their shared code.

	Access points are organized in Devices. A Device is an access point
	whose variable name is the empty string. The other access points
	organized by a Device are Variables, with non-empty variable names.
	The device functions as iterable that iterates over all contained
	variable names. The VariableReferences can be accessed by indexing
	the DeviceReference with the name of the variable, like a
	dictionary.

	Properties:
		device_name: Name of the Device represented by this access point.
		variable_name: Variable name of the access point (always the
			empty string).
		type_id: Identifier for the type of device.
		operations: Valid operations for this access point.
	None of the properties can be edited (operations can be changed
	indirectly by the methods).

	Methods:
		add_command: Add a command as valid for the access point.
		remove_command: Remove a command from the list of valid commands for
			the access point.
		add_variable: Add a VariableReference to the ones of this Device.
		remove_variable: Remove a VariableReference from this Device.
	"""
	_variables: Dict[str, VariableReference]

	def __init__(self, name: str, type_id: str = "device", **kwargs):
		"""
		A device is identified by its name. This name has to be unique in the
		network. To give information regarding recurring properties between
		different devices, the type_id can be used to designate devices as
		identical in function.

		:param name: Identifier for the specific Device that is referenced.
		:param type_id: Identifier for the class of Devices this one belongs to.
		"""
		super().__init__(name, '', type_id, **kwargs)
		self._variables = {'': self}

	def __iter__(self) -> Iterator[str]:
		"""
		Get iterator over managed variable names.

		:return: A generator that yields the names of all variables contained.
		"""
		for variable in self._variables:
			if variable != '':
				yield variable

	def __getitem__(self, variable_name: str) -> VariableReference:
		"""
		Access a VariableReference contained in the DeviceReference.

		:param variable_name: Name of the Variable to access.
		:return: Reference with the requested variable name.
		"""
		return self._variables[str(variable_name)]

	def add_variable(self, variable: VariableReference) -> None:
		"""
		Add a reference to a variable to the DeviceReference.

		Information about Variables managed by a Device are stored within a
		DeviceReference as VariableReferences (with the correct device_name).
		This method can not add a Variable whose name is identical to one that
		is already included in the DeviceReference. To replace a reference,
		remove the old one with remove_variable() first.

		:param variable: Reference to the access point to add.
		:return: True if the variable was added successfully, otherwise False.
		"""
		if isinstance(variable, DeviceReference):
			raise TypeError("A device can not be added to another as a Variable.")
		# See if one can't also check for InstanceReference
		if variable.variable_name in self._variables:
			raise KeyError(f"The Variable {variable.variable_name} can not be added to the device {self._device_name} because a Variable with this name is already present.")
		if variable.device_name != self.device_name:
			raise KeyError(f"The Variable {variable.variable_name} can not be added to the device {self._device_name} because it has been created with device_name={variable.device_name}.")
		self._variables[variable.variable_name] = variable

	def remove_variable(self, variable: Union[str, VariableReference]) -> None:
		"""
		Remove a reference to a variable from the DeviceReference.

		:param variable: Reference to or name of the access point to remove.
		:return: True if the variable was removed successfully, otherwise False.
		"""
		if variable in self._variables.values():
			if variable is self:
				raise KeyError("A Device can not be removed from its own Variable list.")
			del self._variables[variable.variable_name]
		elif str(variable) in self._variables:
			if self._variables[variable] is self:
				raise KeyError("A Device can not be removed from its own Variable list.")
			del self._variables[str(variable)]
		else:
			raise KeyError(f"The Variable {variable!s} is not registered in the Device {self._device_name}.")

	def add_command(self, name: str, variable: str = "") -> None:
		if variable == "":
			super().add_command(name)
		elif variable in self._variables:
			VariableReference.add_command(self._variables[variable], name)
			# self._variables[variable].add_command(name)
			# Specify the used Implementation manually to allow use in Variable add_command
		else:
			raise KeyError(f"Device {self._device_name!s} can not add command {name!s} to Variable {variable!s} that is not registered to it.")

	def remove_command(self, name: str, variable: str = "") -> None:
		if variable == "":
			super().remove_command(name)
		elif variable in self._variables:
			VariableReference.remove_command(self._variables[variable], name)
			# self._variables[variable].remove_command(name)
			# Specify the used Implementation manually to allow use in Variable remove_command
		else:
			raise KeyError(f"Device {self._device_name!s} can not remove command {name!s} from Variable {variable!s} that is not registered to it.")


class InstanceReference(VariableReference):
	"""
	Reference to a connection-handling access point.

	An Instance is the object that handles network communication for the
	access points contained in all devices managed by it. As an access point
	itself it works different from normal access points in that all
	instances share the empty string as both device and variable name. This
	is the only case where using an identical device and variable name
	within the same network will not cause problems.
	While Interfaces are implemented as Device-like objects, using them to
	handle variables directly is deprecated.
	The names of all Devices managed by an Interface are accessible by using
	the InterfaceReference as iterable. The DeviceReferences can be accessed
	by indexing the InstanceReference with the name of the device.

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

	Methods:
		add_command: Add a command as valid for the Instance.
		remove_command: Remove a command from the list of valid commands for
			the Instance.
		add_device: Add a Device to those managed by the Instance.
		remove_device: Remove a Device from those managed by the Instance.
	"""
	_identification: str
	_backend_type: str
	name: str
	user: str
	_devices: Dict[str, DeviceReference]

	def __init__(self, identification: str, backend_type: str, **kwargs):
		"""
		An instance is identified by an identification that depends on the
		network backend used (Example: Address and port number of the command
		port for ZeroMQ). The backend is stored in the instance_type.
		Instances that use different backends are not supposed to interact.
		:param identification: Backend-specific Identifier for the Instance.
		:param backend_type: Identifier for the backend used by the Instance.
		:param kwargs:
		"""
		super().__init__('', '', backend_type + ' Instance', **kwargs)
		self._identification = str(identification)
		self._backend_type = str(backend_type)
		self.name = ''
		self.user = ''
		self._devices = dict()

	def __iter__(self) -> Iterator[str]:
		"""
		Get iterator over managed Device names.

		:return: A generator that yields the names of all devices managed by the
			Instance.
		"""
		for device in self._devices:
			if device != '':
				yield device

	def __getitem__(self, device_name: str) -> DeviceReference:
		"""
		Access a DeviceReference managed by the InstanceReference.

		:param device_name: Name of the Device to access.
		:return: Reference with the requested device name.
		"""
		device_name = str(device_name)
		if device_name == '':
			raise KeyError('Device with empty name can not exist.')
		else:
			return self._devices[device_name]

	# The following properties are defining properties of an Instance and as such should never change during the
	# existence of an instance.
	@property
	def identification(self) -> str:
		"""
		Network backend-specific identifier for an Instance object.

		:return: Network-unique identifier
		"""
		return self._identification

	@property
	def backend_type(self) -> str:
		"""
		Network backend identifier for the backend used by the Instance.

		:return: Name of the network backend.
		"""
		return self._backend_type

	def add_device(self, device: DeviceReference) -> None:
		"""
		Add a DeviceReference to the list of Devices managed by the Instance.

		Information about Devices managed by an Instance are stored within a
		InstanceReference as DeviceReferences.
		This method can not add a Device whose name is identical to one that
		is already included in the InstanceReference. To replace a reference,
		remove the old one with remove_device() first.

		:param device: DeviceReference to be added
		:return: True if the reference was added successfully, otherwise False.
		"""
		if device.device_name in self._devices:
			raise KeyError(f"Device {device.device_name} can not be added to Instance at {self._identification} as a Device with this name is already stored there.")
		self._devices[device.device_name] = device

	def remove_device(self, device: Union[str, DeviceReference]) -> None:
		"""
		Remove a DeviceReference from the InstanceReference.

		:param device: Name or Reference to remove.
		:return: True if the reference was removed successfully, otherwise False.
		"""
		if device in self._devices.values():
			del self._devices[device.device_name]
		elif str(device) in self._devices:
			del self._devices[str(device)]
		else:
			raise KeyError(f"The Device {device!s} is not registered in the Instance at {self._identification}.")

	def add_command(self, name: str, device: str = "", variable: str = "") -> None:
		if device == "" and variable == "":
			super().add_command(name)
		elif device in self._devices:
			DeviceReference.add_command(self._devices[device], name, variable=variable)
		else:
			raise KeyError(f"Instance at {self._identification} can not add a command to Device {device!s} that is not registered to it.")

	def remove_command(self, name: str, device: str = "", variable: str = "") -> None:
		if device == "" and variable == "":
			super().remove_command(name)
		elif device in self._devices:
			DeviceReference.remove_command(self._devices[device], name, variable=variable)
		else:
			raise KeyError(f"Instance at {self._identification} can not remove a command to Device {device!s} that is not registered to it.")

	def _status_json(self) -> Dict[str, Dict[str, str]]:
		# This code doesn't use the self[dev] notation to avoid an unnecessary __getitem__ call (for performance reasons)
		# The easy-to-read version is:
		# rep = dict()
		# for dev in self:
		# 	rep[dev] = {"": self[dev].type_id}
		# 	for var in self[dev]:
		# 		rep[dev][var] = self[dev][var].type_id
		# return rep
		# Device __getitem__ is used because it might be overloaded in some exotic Device type
		status_json = self._devices
		# status_json = {self._devices[dev].device_name: {self._devices[dev][var].variable_name: self._devices[dev][var].type_id for var in self._devices[dev]} | {"": self._devices[dev].type_id} for dev in self._devices}
		return {self._devices[dev].device_name: {self._devices[dev][var].variable_name: self._devices[dev][var].type_id for var in self._devices[dev]} | {"": self._devices[dev].type_id} for dev in self._devices}
