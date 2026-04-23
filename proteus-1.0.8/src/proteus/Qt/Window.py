from pathlib import Path
from threading import Event
from typing import List, Union, Tuple
from pydoc import locate
from json import load

from PyQt5.QtCore import Qt, pyqtSlot, QTimer
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QMainWindow, QTableWidget, QPushButton, \
	QTableWidgetItem, QScrollArea, QHBoxLayout, QCheckBox, QGroupBox, QGridLayout, QFileDialog, QStackedLayout, \
	QStackedWidget, QSizePolicy, QMenu, QStyle, QMessageBox

import proteus
from proteus.Instance import Instance
from proteus.Instance.ZeroMQ import Instance as ZeroMQInstance
from proteus.Devices import Device, ActiveDevice
from .Interface import QtInterface
from ..Utility import DeviceNotFound


class Window(QMainWindow):

	name_line: QLineEdit
	user_line: QLineEdit
	auto_start_check: QCheckBox
	sub_windows: List[QWidget]

	def __init__(self, inst: Instance, interface_source: str = "", device_source: str = "", update_timer_s: float = 0.5, **kwargs):
		super().__init__()
		# --- UI base font scaling for high-DPI screens ---
		try:
			screen = self.screen()
		except AttributeError:
			screen = None
		if screen is None:
			from PyQt5.QtWidgets import QApplication
			screen = QApplication.primaryScreen()
		if screen is not None:
			dpi = screen.logicalDotsPerInch()
		else:
			dpi = 96.0  # sensible default

		# Choose a base point size depending on DPI
		if dpi < 110:
			base_pt = 9
		elif dpi < 150:
			base_pt = 10
		elif dpi < 200:
			base_pt = 11
		else:
			base_pt = 12

		font = self.font()
		font.setPointSize(base_pt)
		self.setFont(font)
		# --- end UI base font scaling ---
		self.instance = inst
		self.interface_source = interface_source
		self.device_source = device_source
		self.sub_windows = []
		self.close_event = Event()

		self.setWindowTitle(f"Proteus v{proteus.__version__}")
		# TODO icon

		top_level_container = QWidget()
		top_level_layout = QHBoxLayout(top_level_container)

		left_widget = QWidget()
		left_layout = self.left_column()
		left_widget.setLayout(left_layout)
		left_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)

		top_level_layout.addWidget(left_widget)
		top_level_layout.addWidget(self.right_column(), stretch=1)

		scroll_area = QScrollArea()
		scroll_area.setWidgetResizable(True)
		scroll_area.setWidget(top_level_container)
		self.setCentralWidget(scroll_area)


		self.interface_dialog = QFileDialog(self)
		self.interface_dialog.setOption(QFileDialog.ReadOnly)
		self.interface_dialog.setAcceptMode(QFileDialog.AcceptOpen)
		self.interface_dialog.setFileMode(QFileDialog.ExistingFiles)
		self.interface_dialog.setNameFilter("Interface configurations (*.json)")
		self.interface_dialog.setDefaultSuffix("json")
		self.interface_dialog.filesSelected.connect(self.files_selected)

		self.device_dialog = QFileDialog(self)
		self.device_dialog.setOption(QFileDialog.ReadOnly)
		self.device_dialog.setAcceptMode(QFileDialog.AcceptOpen)
		self.device_dialog.setFileMode(QFileDialog.ExistingFiles)
		self.device_dialog.setNameFilter("Device configurations (*.json)")
		self.device_dialog.setDefaultSuffix("json")
		self.device_dialog.filesSelected.connect(self.files_selected)

		self.update_devices()
		self.update_timer = QTimer(self)
		self.update_timer.setInterval(int(update_timer_s * 1000))
		self.update_timer.setTimerType(Qt.CoarseTimer)
		self.update_timer.timeout.connect(self.update_devices)
		self.update_timer.start()

	def left_column(self) -> QVBoxLayout:
		layout = QVBoxLayout()

		instance_box = QGroupBox("Instance")
		instance_box_layout = QGridLayout(instance_box)
		instance_box_layout.addWidget(QLabel("address"), 0, 0)
		address_line = QLabel(self.instance.identification)
		address_line.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
		instance_box_layout.addWidget(address_line, 0, 1)
		instance_box_layout.addWidget(QLabel("name"), 1, 0)
		self.name_line = QLineEdit(self.instance.name)
		self.name_line.editingFinished.connect(self.set_name)
		instance_box_layout.addWidget(self.name_line, 1, 1)
		instance_box_layout.addWidget(QLabel("user"), 2, 0)
		self.user_line = QLineEdit(self.instance.user)
		self.user_line.editingFinished.connect(self.set_user)
		instance_box_layout.addWidget(self.user_line, 2, 1)
		layout.addWidget(instance_box)

		layout.addStretch()

		connect_box = QGroupBox('connect to Instance')
		connect_box_layout = QVBoxLayout(connect_box)
		self.connect_address = QLineEdit("tcp://")
		self.connect_address.returnPressed.connect(self.connect)
		connect_box_layout.addWidget(self.connect_address)
		layout.addWidget(connect_box)

		layout.addStretch()

		interface_button = QPushButton(self.style().standardIcon(QStyle.SP_DesktopIcon), "Open New &Interface")
		interface_button.clicked.connect(self.new_interface)
		layout.addWidget(interface_button)

		return layout

	def right_column(self) -> QWidget:
		widget = QGroupBox("devices")
		layout = QVBoxLayout(widget)
		table = QTableWidget()
		table.setColumnCount(4)
		table.setHorizontalHeaderLabels(["Instance", "Device", "Running", "Status"])

		table.verticalHeader().hide()
		table.setSizeAdjustPolicy(QTableWidget.AdjustToContents)
		table.setEditTriggers(QTableWidget.NoEditTriggers)
		table.setSelectionBehavior(QTableWidget.SelectRows)
		table.itemSelectionChanged.connect(self.device_selected)
		table.setContextMenuPolicy(Qt.CustomContextMenu)
		table.customContextMenuRequested.connect(self.table_context_menu)
		layout.addWidget(table, stretch=1)
		self.device_table = table

		self.table_context_menu = QMenu()
		self.table_context_menu.addAction(self.style().standardIcon(QStyle.SP_FileIcon), "Add Device").triggered.connect(self.add_device)
		self.table_context_menu.addAction(self.style().standardIcon(QStyle.SP_MediaPlay), "Start Device").triggered.connect(self.start_device)
		self.table_context_menu.addAction(self.style().standardIcon(QStyle.SP_MediaPause), "Stop Device").triggered.connect(self.stop_device)
		self.table_context_menu.addAction(self.style().standardIcon(QStyle.SP_DialogCloseButton), "Remove Device").triggered.connect(self.remove_device)

		button_row = QHBoxLayout()
		add_device_button = QPushButton(self.style().standardIcon(QStyle.SP_FileIcon), "Add")
		self.style().standardIcon(QStyle.SP_MediaPlay)
		add_device_button.clicked.connect(self.add_device)
		button_row.addWidget(add_device_button, stretch=1)
		self.start_stop_switch = QStackedWidget()
		self.start_device_button = QPushButton(self.style().standardIcon(QStyle.SP_MediaPlay), "Start", self.start_stop_switch)
		self.start_device_button.setDisabled(True)
		self.start_device_button.clicked.connect(self.start_device)
		self.start_stop_switch.addWidget(self.start_device_button)
		self.stop_device_button = QPushButton(self.style().standardIcon(QStyle.SP_MediaPause), "Stop", self.start_stop_switch)
		self.stop_device_button.setDisabled(True)
		self.stop_device_button.clicked.connect(self.stop_device)
		self.start_stop_switch.addWidget(self.stop_device_button)
		button_row.addWidget(self.start_stop_switch, stretch=1)
		self.remove_device_button = QPushButton(self.style().standardIcon(QStyle.SP_DialogCloseButton), "Remove")
		self.remove_device_button.setDisabled(True)
		self.remove_device_button.clicked.connect(self.remove_device)
		button_row.addWidget(self.remove_device_button, stretch=1)
		layout.addLayout(button_row, stretch=0)

		self.auto_start_check = QCheckBox("automatic start")
		self.auto_start_check.setChecked(True)
		layout.addWidget(self.auto_start_check, stretch=0)
		return widget

	@pyqtSlot()
	def connect(self):
		if not isinstance(self.instance, ZeroMQInstance):
			QMessageBox.warning(
				self,
				"Connect to instance",
				"Connecting to additional instances is only supported\n"
				"for ZeroMQ-based instances."
			)
			return
		address = self.connect_address.text().strip()
		if not address:
			QMessageBox.warning(
				self,
				"Connect to instance",
				"Please enter an instance address."
			)
			return

		try:
			self.instance.add_instance(address)
		except Exception as exc:
			QMessageBox.critical(
				self,
				"Connection failed",
				f"Could not connect to instance:\n\n{address}\n\n"
				f"Error: {exc}"
			)
		else:
			QMessageBox.information(
				self,
				"Connection successful",
				f"Connected to instance:\n\n{address}"
			)
			self.connect_address.setText("tcp://")


	@pyqtSlot()
	def new_interface(self):
		self.interface_dialog.setDirectory(self.interface_source)
		# can not deselect the files, as recognized on Qt forums:
		# https://forum.qt.io/topic/121235/qfiledialog-has-memory
		# self.device_dialog.selectFile("")
		# also other weird behavior?
		self.interface_dialog.show()

	@pyqtSlot()
	def add_device(self):
		self.device_dialog.setDirectory(self.device_source)
		# can not deselect the files, as recognized on Qt forums:
		# https://forum.qt.io/topic/121235/qfiledialog-has-memory
		# self.device_dialog.selectFile("")
		# also other weird behavior?
		self.device_dialog.show()

	@pyqtSlot()
	def start_device(self):
		row = self.device_table.selectedItems()[0].row()
		name_item = self.device_table.item(row, 1)  # column 1 = Device
		if not name_item:
			return
		name = name_item.text()
		device = self.instance[name]
		if isinstance(device, ActiveDevice):
			device.call_in_device_thread(device.on)


	@pyqtSlot()
	def stop_device(self):
		row = self.device_table.selectedItems()[0].row()
		name_item = self.device_table.item(row, 1)
		if not name_item:
			return
		name = name_item.text()
		device = self.instance[name]
		if isinstance(device, ActiveDevice):
			device.call_in_device_thread(device.off)


	@pyqtSlot()
	def remove_device(self):
		row = self.device_table.selectedItems()[0].row()
		name_item = self.device_table.item(row, 1)
		if not name_item:
			return
		name = name_item.text()
		device = self.instance[name]
		if isinstance(device, Device):
			device.exit()


	@pyqtSlot()
	def device_selected(self):
		self.update_buttons()

	def update_buttons(self):
		try:
			row = self.device_table.selectedItems()[0].row()
		except IndexError:
			self.remove_device_button.setDisabled(True)
			self.start_device_button.setDisabled(True)
			self.stop_device_button.setDisabled(True)
		else:
			running_item = self.device_table.item(row, 2)  # column 2 = Running
			running = running_item.text() if running_item else "-"
			self.remove_device_button.setDisabled(False)
			if running == "Yes":
				self.stop_device_button.setDisabled(False)
				self.start_stop_switch.setCurrentIndex(1)
			elif running == "No":
				self.start_device_button.setDisabled(False)
				self.start_stop_switch.setCurrentIndex(0)
			elif running == "-":
				self.start_device_button.setDisabled(True)
				self.stop_device_button.setDisabled(True)


	@pyqtSlot()
	def set_name(self):
		self.instance.name = self.name_line.text()

	@pyqtSlot()
	def set_user(self):
		self.instance.user = self.user_line.text()

	@pyqtSlot()
	def update_devices(self):
		"""
		Retrieve new status messages for the Devices in the Device overview.

		The Device overview shows status messages for all Devices known to this
		Instance. Devices are grouped by the Instance they belong to.
		"""
		if not self.name_line.hasFocus():
			self.name_line.setText(self.instance.name)
		if not self.user_line.hasFocus():
			self.user_line.setText(self.instance.user)

		table = self.device_table

		# Build list of existing (instance_label, device_name) keys so we can reuse rows
		existing_keys: List[Union[Tuple[str, str], None]] = []
		for i in range(table.rowCount()):
			inst_item = table.item(i, 0)
			dev_item = table.item(i, 1)
			if inst_item and dev_item:
				existing_keys.append((inst_item.text(), dev_item.text()))
			else:
				existing_keys.append(None)

		keys_in_table = existing_keys

		def upsert_row(instance_label: str, device_name: str, active: str, status: str) -> None:
			key = (instance_label, device_name)
			try:
				row = keys_in_table.index(key)
				keys_in_table[row] = None
			except ValueError:
				row = table.rowCount()
				table.insertRow(row)
				table.setItem(row, 0, QTableWidgetItem(instance_label, QTableWidgetItem.Type))
				table.setItem(row, 1, QTableWidgetItem(device_name, QTableWidgetItem.Type))
				table.setItem(row, 2, QTableWidgetItem(active, QTableWidgetItem.Type))
				table.setItem(row, 3, QTableWidgetItem(status, QTableWidgetItem.Type))
			else:
				table.item(row, 2).setText(active)
				table.item(row, 3).setText(status)

		# --- local devices on this instance ---
		for device_name in self.instance:
			if device_name == "":
				continue

			device_object = self.instance[device_name]

			# Running state
			if isinstance(device_object, ActiveDevice):
				active = "Yes" if device_object._is_on else "No"
			else:
				active = "-"

			# Status text
			if isinstance(device_object, Device):
				status = device_object.status
			else:
				status = "???"

			# Local instance label
			instance_label = self.instance.name or self.instance.identification

			upsert_row(instance_label, device_name, active, status)

				# --- remote devices on other instances ---
		if isinstance(self.instance, ZeroMQInstance):
			for address, ref in self.instance._known_instances.items():
				# skip the local instance if it is present in the map
				if ref is self.instance:
					continue

				# Human-readable label for the remote instance
				instance_label = getattr(ref, "name", None) or getattr(ref, "identification", None) or address

				# Devices known on this remote instance
				for device_name in ref:
					if device_name == "":
						continue

					device_ref = ref[device_name]

					# Try to infer "running" if the reference exposes something like _is_on
					if hasattr(device_ref, "_is_on"):
						active = "Yes" if getattr(device_ref, "_is_on") else "No"
					else:
						active = "-"

					# Try to use a status property if present
					if hasattr(device_ref, "status"):
						status = str(getattr(device_ref, "status"))
					else:
						status = "remote"

					upsert_row(instance_label, device_name, active, status)


        # Remove rows that no longer correspond to any device
		for row in reversed(range(len(keys_in_table))):
			if keys_in_table[row] is not None:
				table.removeRow(row)

		# Group by Instance (primary sort), then Device
		table.sortItems(0)
		self.update_buttons()


	@pyqtSlot("QStringList")
	def files_selected(self, paths):
		for path in paths:
			path = Path(path)
			with open(path) as file:
				config = load(file)
			print(config)
			name = path.stem
			new_class = locate(config.pop("source"))
			if not isinstance(new_class, type):
				print(f"invalid configuration {path!s} does not contain a valid source path")
				continue
			if issubclass(new_class, QtInterface):
				try:
					# noinspection PyCallingNonCallable
					new_interface = new_class(instance=self.instance, **config)
				except Exception as ex:
					# raise
					print(f"Error during Interface creation: {repr(ex)}")
					continue
				self.sub_windows.append(new_interface)
				new_interface.setWindowTitle(name)
				new_interface.show()
			elif issubclass(new_class, Device):
				try:
					instance_address = self.instance.get_instance_address(name)
				except DeviceNotFound:
					pass
				else:
					print(f"device already exists in the network at instance {instance_address}")
					continue
				try:
					# noinspection PyCallingNonCallable
					new_device = new_class(instance=self.instance, name=name, **config)
				except Exception as ex:
					print(f"Error during Device creation: {repr(ex)}")
					continue
				if self.auto_start_check.isChecked() and isinstance(new_device, ActiveDevice):
					new_device.call_in_device_thread(new_device.on)
			else:
				print(f"invalid configuration {path!s}")
				continue

	@pyqtSlot("QPoint")
	def table_context_menu(self, point):
		item = self.device_table.itemAt(point)
		actions = self.table_context_menu.actions()
		if item:
			# we don't care which column was clicked; get the row and read Running from col 2
			row = item.row()
			running_item = self.device_table.item(row, 2)  # column 2 = Running
			running = running_item.text() if running_item else "-"
			if running == "Yes":
				actions[1].setDisabled(True)
				actions[2].setDisabled(False)
			elif running == "No":
				actions[1].setDisabled(False)
				actions[2].setDisabled(True)
			elif running == "-":
				actions[1].setDisabled(True)
				actions[2].setDisabled(True)
			actions[3].setDisabled(False)
		else:
			actions[1].setDisabled(True)
			actions[2].setDisabled(True)
			actions[3].setDisabled(True)
		self.table_context_menu.popup(self.device_table.mapToGlobal(point))


	# close only if all interfaces are closed
	def closeEvent(self, event: QCloseEvent) -> None:
		if any([window.isVisible() for window in self.sub_windows]):
			event.ignore()
		else:
			event.accept()
			self.close_event.set()

	def wait_for_window(self):
		self.close_event.wait()
