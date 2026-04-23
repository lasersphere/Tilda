from typing import Dict

from .Instance import Instance
from .Connection import Connection


class InstanceObject:

	_instance: Instance

	def __init__(self, instance: Instance, **kwargs):
		if not isinstance(instance, Instance):
			raise RuntimeError(f"Object {instance!s} is not a valid instance")
		super().__init__(**kwargs)
		self._instance = instance

	@property
	def instance(self) -> Instance:
		return self._instance

	@instance.setter
	def instance(self, instance: Instance):
		if isinstance(instance, Instance):
			self._instance = instance
			# TODO look if other dependencies need to be changed
		else:
			self._instance.warn(f"Object {instance!s} is not a valid instance")

	def connect(self, device_name: str, variable_name: str) -> Connection:
		return Connection(self._instance, device_name, variable_name)

	def known_devices(self) -> Dict[str, Dict[str, str]]:
		devs = {}
		for address in self._instance._known_instances:
			devs |= self._instance._known_instances[address]._status_json()
		return devs
