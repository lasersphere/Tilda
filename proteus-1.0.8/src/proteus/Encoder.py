from abc import abstractmethod, ABC
from typing import Any, Dict, Type, Tuple, List
from numpy import save, empty, ndarray, load, asarray
from io import BytesIO
from json import dumps


class DatatypeEncoder(ABC):

	@classmethod
	def id(cls) -> str:
		return f"{cls.__module__}.{cls.__qualname__}"

	@classmethod
	@abstractmethod
	def setup_data_storage(cls) -> Any:
		raise NotImplementedError

	@classmethod
	@abstractmethod
	def value_to_data(cls, value: Any, old_data: Any) -> Any:
		raise NotImplementedError

	@classmethod
	@abstractmethod
	def message_to_data(cls, message: Dict[str, Any], message_data: List[bytes], old_data: Any) -> Any:
		raise NotImplementedError

	@classmethod
	@abstractmethod
	def data_to_value(cls, data: Any) -> Any:
		raise NotImplementedError

	@classmethod
	@abstractmethod
	def data_to_message_args(cls, data: Any) -> Tuple[Dict[str, Any], List[bytes]]:
		raise NotImplementedError

	@classmethod
	def compare_values(cls, value_1: Any, value_2: Any) -> bool:
		return value_1 == value_2


Datatype = Type[DatatypeEncoder]


class JSONEncoder(DatatypeEncoder):

	@classmethod
	def setup_data_storage(cls) -> Any:
		return None

	@classmethod
	def value_to_data(cls, value: Any, old_data: Any) -> Any:
		try:
			dumps(value)
		except Exception as ex:
			raise TypeError(f"Datatype JSON can not process value {value}")
		else:
			return value

	@classmethod
	def message_to_data(cls, message: Dict[str, Any], message_data: List[bytes], old_data: Any) -> Any:
		return message['value']

	@classmethod
	def data_to_value(cls, data: Any) -> Any:
		return data

	@classmethod
	def data_to_message_args(cls, data: Any) -> Tuple[Dict[str, Any], List[bytes]]:
		return {'value': data}, []


class NumpyArrayEncoder(DatatypeEncoder):

	@classmethod
	def setup_data_storage(cls) -> Any:
		with BytesIO() as file:
			save(file, empty(0), allow_pickle=False)
			return file.getvalue()

	@classmethod
	def value_to_data(cls, value: Any, old_data: Any) -> Any:
		value = asarray(value)
		with BytesIO() as file:
			save(file, value, allow_pickle=False)
			return file.getvalue()

	@classmethod
	def message_to_data(cls, message: Dict[str, Any], message_data: List[bytes], old_data: Any) -> Any:
		try:
			# check if the data is valid numpy, so the content of data can always be interpreted
			cls.data_to_value(message_data[0])
		except Exception as ex:
			print(f"invalid ndarray data received: {message_data} {repr(ex)}")
		else:
			return message_data[0]

	@classmethod
	def data_to_value(cls, data: Any) -> ndarray:
		with BytesIO() as file:
			file.write(data)
			file.seek(0)
			# No pickling for security reasons (pickle executes arbitrary code)
			return load(file, allow_pickle=False)

	@classmethod
	def data_to_message_args(cls, data: Any) -> Tuple[Dict[str, Any], List[bytes]]:
		return {}, [data]

	@classmethod
	def compare_values(cls, value_1: ndarray, value_2: ndarray) -> bool:
		return (value_1.shape == value_2.shape) and all(value_1 == value_2)
