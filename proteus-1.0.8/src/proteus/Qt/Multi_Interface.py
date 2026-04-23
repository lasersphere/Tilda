from PyQt5.QtCore import Qt, QEvent, pyqtBoundSignal, pyqtSignal
from PyQt5.QtWidgets import QHBoxLayout, QTabWidget, QVBoxLayout, QFrame, QSizePolicy, QLabel, QScrollArea, QWidget, \
	QApplication, QAction
from proteus.Qt.Interface import QtInterface, WrappedInterface
from proteus.Instance import Instance
from typing import Dict, Any, Union, Type, List, Callable
from pydoc import locate


def connect_pyqt_signal(signal: pyqtSignal, function: Callable):
	if isinstance(signal, pyqtBoundSignal):
		signal.connect(function)
	else:
		print("unbound signal:", signal)


class AbstractMultiInterface(QtInterface):

	interfaces: Dict[str, QtInterface]
	fixed_source_class: Union[Type[QtInterface], None]

	def __init__(self, inst: Instance, sub_interfaces: Dict[str, Dict[str, Any]], fixed_source: Union[None, str] = None, **kwargs):
		super().__init__(inst, **kwargs)
		self.interfaces = dict()
		# TODO fixed source handling in _setup
		self.fixed_source_class = None
		if fixed_source is not None:
			fixed_source_class = locate(str(fixed_source))
			if isinstance(fixed_source_class, type) and issubclass(fixed_source_class, QtInterface):
				self.fixed_source_class = fixed_source_class
		self.setup(sub_interfaces, **kwargs)

	def setup(self, sub_interfaces: Dict[str, Dict[str, Any]], **kwargs):
		for name in sub_interfaces:
			new_interface = self.create_interface_from_config(sub_interfaces[name])
			if new_interface is not None:
				self.add_interface(name, new_interface)

	def create_interface_from_config(self, configuration: Dict[str, Any]) -> Union[QtInterface, None]:
		interface_source = configuration.pop('source', None)
		if self.fixed_source_class is None:
			interface_class = locate(interface_source)
			if callable(interface_class):
				new_interface = interface_class(instance=self._instance, **configuration)
				if isinstance(new_interface, QtInterface):
					return new_interface
			return None
		else:
			return self.fixed_source_class(instance=self._instance, **configuration)
		# TODO replace returning None with an error (that is handled when the method called) to be consistent with pydoc/json error behavior

	def add_interface(self, name: str, interface: QtInterface) -> bool:
		if name in self.interfaces:
			return False
		else:
			self.interfaces[name] = interface
			return True

	def remove_interface(self, name: str) -> Union[QtInterface, None]:
		return self.interfaces.pop(name, None)
		# TODO destroy interface


class TabMultiInterface(AbstractMultiInterface):

	tab_widget = QTabWidget

	def setup(self, sub_interfaces: Dict[str, Dict[str, Any]], **kwargs):
		layout = QHBoxLayout(self)
		self.setLayout(layout)
		self.tab_widget = QTabWidget(self)
		layout.addWidget(self.tab_widget)
		super().setup(sub_interfaces, **kwargs)

	def add_interface(self, name: str, interface: QtInterface) -> bool:
		if super().add_interface(name, interface):
			self.tab_widget.addTab(interface, name)
			return True
		else:
			return False

	def remove_interface(self, name: str) -> Union[QtInterface, None]:
		interface = super().remove_interface(name)
		if interface is not None:
			index = self.tab_widget.indexOf(interface)
			self.tab_widget.removeTab(index)
		return interface


class HBoxMultiInterface(AbstractMultiInterface):

	layout: QHBoxLayout

	def setup(self, sub_interfaces: Dict[str, Dict[str, Any]], **kwargs):
		self.layout = QHBoxLayout(self)
		self.setLayout(self.layout)
		super().setup(sub_interfaces, **kwargs)

	def add_interface(self, name: str, interface: QtInterface) -> bool:
		if super().add_interface(name, interface):
			self.layout.addWidget(interface)
			return True
		else:
			return False

	def remove_interface(self, name: str) -> Union[QtInterface, None]:
		interface = super().remove_interface(name)
		if interface is not None:
			self.layout.removeWidget(interface)
		return interface


class VBoxMultiInterface(AbstractMultiInterface):

	layout: QVBoxLayout

	def setup(self, sub_interfaces: Dict[str, Dict[str, Any]], **kwargs):
		self.layout = QVBoxLayout(self)
		self.setLayout(self.layout)
		super().setup(sub_interfaces, **kwargs)

	def add_interface(self, name: str, interface: QtInterface) -> bool:
		if super().add_interface(name, interface):
			self.layout.addWidget(interface)
			return True
		else:
			return False

	def remove_interface(self, name: str) -> Union[QtInterface, None]:
		interface = super().remove_interface(name)
		if interface is not None:
			self.layout.removeWidget(interface)
		return interface


class ScrollingTableSubInterface(WrappedInterface):

	name: str = None
	hover_update_function: Callable = None

	def set_hover_update_function(self, name: str, function: Callable):
		self.name = name
		self.hover_update_function = function
		self.setAttribute(Qt.WA_Hover)

	def enterEvent(self, event: QEvent) -> None:
		if callable(self.hover_update_function):
			self.hover_update_function(self.name)

	def leaveEvent(self, event: QEvent) -> None:
		if callable(self.hover_update_function):
			self.hover_update_function(None)

	@staticmethod
	def configuration_window(callback: Callable[[str, Dict[str, Any]], Any], default_values: Dict[str, Any] = None) -> QWidget:
		pass


class ScrollingTableInterface(AbstractMultiInterface):

	default_device: Union[str, None]
	table_layout: QHBoxLayout
	fixed_source_class: Union[Type[ScrollingTableSubInterface], None]
	hovered_interface: Union[None, str]
	remove_channel_action: QAction

	def setup(self, sub_interfaces: Dict[str, Dict[str, Any]], rows: List[str] = None, default_device: str = None, **kwargs):
		if rows is None:
			rows = [""]
		elif len(rows) < 1:
			rows = [""]
		if not issubclass(self.fixed_source_class, ScrollingTableSubInterface):
			print("invalid fixed subclass in configuration file")
		self.default_device = default_device

		main_layout = QHBoxLayout(self)
		label_column = QFrame(self)
		label_column.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred))
		main_layout.addWidget(label_column)
		label_layout = QVBoxLayout(label_column)
		label_layout.setContentsMargins(0, 0, 0, 22)
		for row in rows:
			label_layout.addWidget(QLabel(row))
		separator = QFrame(self)
		separator.setFrameShape(QFrame.VLine)
		separator.setFrameShadow(QFrame.Sunken)
		main_layout.addWidget(separator)
		scroll_area = QScrollArea(self)
		table_area = QWidget()
		scroll_area.setWidget(table_area)
		main_layout.addWidget(scroll_area)
		scroll_area.setFrameShape(QFrame.NoFrame)
		scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
		scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
		scroll_area.setWidgetResizable(True)
		scroll_area.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum))
		self.table_layout = QHBoxLayout(table_area)
		self.table_layout.setContentsMargins(0, 0, 0, 0)
		self.table_layout.addStretch()
		super().setup(sub_interfaces, **kwargs)
		self.add_context_menu()

	def add_interface(self, name: str, interface: ScrollingTableSubInterface) -> bool:
		if super().add_interface(name, interface):
			interface.set_hover_update_function(name, self.set_hovered_channel)
			self.table_layout.insertWidget(self.table_layout.count() - 1, interface)
			return True
		else:
			return False

	def remove_interface(self, name: str) -> QtInterface:
		interface = super().remove_interface(name)
		if interface is not None:
			self.table_layout.removeWidget(interface)
		return interface

	def set_hovered_channel(self, name: Union[None, str]):
		if QApplication.activePopupWidget() is None:
			self.hovered_interface = name
			self.remove_channel_action.setEnabled(name is not None)

	def add_context_menu(self):
		self.setContextMenuPolicy(Qt.ActionsContextMenu)
		new_column_action = QAction("add a new channel", self)

		def callback(name: str, config: Dict[str, Any]):
			config['name'] = name
			return self.add_interface(name, self.create_interface_from_config(config))
		if self.default_device is not None:
			window = self.fixed_source_class.configuration_window(callback, default_values={'device_name': self.default_device})
		else:
			window = self.fixed_source_class.configuration_window(callback)
		connect_pyqt_signal(new_column_action.triggered, window.show)
		self.addAction(new_column_action)

		self.remove_channel_action = QAction("remove this channel", self)
		self.hovered_interface = None
		self.remove_channel_action.setEnabled(False)

		def remove_hovered_channel():
			self.table_layout.removeWidget(self.interfaces.pop(self.hovered_interface))
			self.hovered_interface = None
			self.remove_channel_action.setEnabled(False)
		connect_pyqt_signal(self.remove_channel_action.triggered, remove_hovered_channel)
		self.addAction(self.remove_channel_action)

		save_channels_action = QAction("save channel setup and values", self)
		# TODO needs some thought: channel setup and values might be saved separately? If so, how is that organized?
		#  alternatively only save setup if the interface is used in fixed channels mode (ie setup is from config and can't be changed)?
