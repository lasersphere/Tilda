"""
Proteus-backed external scan device controller for TILDA.

This controller owns its own local Proteus instance, discovers devices on a
configured list of remote Proteus instances, and drives a selected writable
Proteus variable step-by-step for scans.

Track UI input:
    Type box: tcp://host:7000::DeviceName
    Name box: VariableName
    Name box explicit: variable=set_val;readback=scan_var;ready=ready
"""

import logging
import math
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from Tilda.Driver.ProteusListener.ProteusImport import (
    ensure_proteus_on_path,
    ensure_remote_instance_connected,
)
from Tilda.Driver.ScanDevice.BaseTildaScanDeviceControl import BaseTildaScanDeviceControl
from Tilda.PolliFit.Measurement.SpecData import SpecDataXAxisUnits as Units

logger = logging.getLogger(__name__)
DEBUG_MODE = False


def _debug_exception(message: str, *args):
    """Log a short debug message unless full tracebacks are explicitly enabled."""
    if DEBUG_MODE:
        logger.debug(message, *args, exc_info=True)
        return
    logger.debug(
        message + "; set DEBUG_MODE=True in ProteusScanDevControl for full traceback",
        *args,
    )

ensure_proteus_on_path()

try:
    import proteus  # type: ignore
    from proteus.InstanceObject import InstanceObject  # type: ignore
    PROTEUS_AVAILABLE = True
except Exception as exc:
    logger.info("ProteusScanDevControl: proteus import failed (%s)", exc)
    proteus = None  # type: ignore
    InstanceObject = object  # type: ignore
    PROTEUS_AVAILABLE = False

try:
    from Tilda.Interface.PreScanConfigUi.PreScanConfigUi import PROTEUS_INSTANCE_CONFIG
except Exception:
    PROTEUS_INSTANCE_CONFIG = {
        "instance_type": "zmq",
        "protocol": "tcp",
        "command_port": 6000,
        "publish_port": 6001,
        "port_range": 1000,
        "port_attempts": 50,
    }


class DeferredInstanceObject(InstanceObject):
    """
    InstanceObject variant that can be constructed before a Proteus Instance
    exists. QObject's cooperative init path reaches this class with no
    arguments, so we must tolerate that case.
    """

    def __init__(self, instance=None, **kwargs):
        self._instance = instance
        super(InstanceObject, self).__init__(**kwargs)


class ProteusScanDevControl(BaseTildaScanDeviceControl, DeferredInstanceObject):
    """
    TILDA scan-device adapter for Proteus variables.

    The controller owns a local Proteus client instance and discovers remote
    devices/variables from a configured list of remote instance addresses.
    """

    # Edit this list to define which remote Proteus instances are probed for
    # scan-device discovery in the Track UI dropdown.
    DISCOVERY_INSTANCE_ADDRESSES = [
        "tcp://192.168.11.6:7000",
        "tcp://192.168.11.103:7000",
        #"tcp://192.168.14.251:7000"
    ]
    # Cache discovered ``instance::device`` entries for the Track UI.
    DISCOVERED_TARGET_MAP: Dict[str, str] = {}
    DISCOVERED_TARGETS: List[str] = []

    def __init__(self):
        """Initialise local scan state, target metadata, and the Proteus client."""
        BaseTildaScanDeviceControl.__init__(self)

        self._cm_instance = None
        self._instance = None

        self.instance_address = ""
        self.target_device = ""
        self.target_variable = "setpoint"
        self.readback_instance_address = ""
        self.readback_device = ""
        self.readback_variable = ""
        self.ready_instance_address = ""
        self.ready_device = ""
        self.ready_variable = ""
        # Raw target string selected or typed in the GUI before parsing.
        self.target_spec = ""
        self.readback_spec = ""
        self.ready_spec = ""

        self._connection = None
        self._readback_connection = None
        self._ready_connection = None

        self._known_targets: List[str] = []
        self._display_target_map: Dict[str, str] = dict(self.DISCOVERED_TARGET_MAP)
        self._known_devices_cache: Dict[str, Dict[str, Dict[str, str]]] = {}
        self._connected_instance_addresses = set()
        self._failed_instance_addresses = set()

        self.sc_start = 0.0
        self.sc_stop = 0.0
        self.sc_stepsize = 0.0
        self.sc_num_of_steps = 0
        self.sc_num_of_scans = 0
        self.sc_invert_in_odd_scans = False
        self.sc_one_scan_vals = []
        self.sc_l_cur_step = -1
        self.sc_l_cur_scan = 0
        self.sc_l_perc_compl = 0.0
        self.scan_status = "initialized"
        self._busy = False
        self.scan_dev_timeout = 10.0
        self._known_targets = list(self.DISCOVERED_TARGETS)

        if PROTEUS_AVAILABLE:
            self._create_local_instance()

    @property
    def instance(self):
        """Return the local Proteus instance used by this controller."""
        return self._instance

    @instance.setter
    def instance(self, value):
        """Store the local Proteus instance reference."""
        self._instance = value

    def available_scan_dev_types(self):
        """Discover and return selectable ``instance::device`` targets."""
        try:
            self._refresh_known_targets()
        except Exception:
            _debug_exception("ProteusScanDevControl: device discovery failed")

        dev_types = list(self._known_targets)
        current_type = self._current_device_spec()
        if current_type and current_type not in dev_types:
            dev_types.append(current_type)
        return dev_types

    def available_scan_dev_names_by_type(self, dev_type):
        """Rescan the selected target instance and return the device's variables."""
        names = []
        device_spec = self._normalise_device_spec(dev_type)
        if device_spec:
            try:
                instance_address, device_name = self._parse_device_spec(device_spec)
                device_variables = self._query_device_variables(instance_address, device_name)
                names.extend(device_variables)
            except Exception:
                _debug_exception(
                    "ProteusScanDevControl: variable discovery failed for %s",
                    dev_type,
                )

        current_type = self._current_device_spec()
        if self.target_variable and device_spec and device_spec == current_type and self.target_variable not in names:
            names.append(self.target_variable)
        return names

    def return_scan_dev_info(
        self,
        dev_type=None,
        dev_name=None,
        readback_name="",
        ready_name="",
    ):
        """Return generic scan-device metadata for the currently selected target."""
        if dev_name:
            try:
                target_spec = self._build_target_spec_from_selection(dev_type, dev_name)
            except ValueError:
                _debug_exception("ProteusScanDevControl: invalid variable selection %s", dev_name)
            else:
                if target_spec:
                    self._configure_target(target_spec, readback_name, ready_name)

        return {
            "name": dev_name or self.target_variable or "setpoint",
            "type": dev_type or self._current_device_spec() or "Proteus",
            "devClass": "Proteus",
            "stepUnitName": Units.not_defined.name,
            "start": self.sc_start,
            "stop": self.sc_stop,
            "stepSize": self.sc_stepsize if self.sc_stepsize else 1.0,
            "preScanSetPoint": None,
            "postScanSetPoint": None,
            "timeout_s": 10.0,
            "setValLimit": (-1.0 * 10 ** 30, 1.0 * 10 ** 30),
            "stepSizeLimit": (-1.0 * 10 ** 30, 1.0 * 10 ** 30),
        }

    def setup_scan_in_scan_dev(self, start, stepsize, num_of_steps, num_of_scans, invert_in_odd_scans):
        """Store scan parameters, precompute step values, and validate the target connection."""
        self.sc_start = float(start)
        self.sc_stepsize = float(stepsize)
        self.sc_num_of_steps = int(num_of_steps)
        self.sc_num_of_scans = int(num_of_scans)
        self.sc_invert_in_odd_scans = bool(invert_in_odd_scans)

        stop = self.sc_start + (self.sc_num_of_steps - 1) * self.sc_stepsize
        vals, real_step = np.linspace(self.sc_start, stop, self.sc_num_of_steps, retstep=True)
        self.sc_stop = float(stop)
        self.sc_stepsize = float(real_step)
        self.sc_one_scan_vals = vals.tolist()
        self.sc_l_cur_step = -1
        self.sc_l_cur_scan = 0
        self.sc_l_perc_compl = 0.0
        self.scan_status = "setupForScan"

        self._ensure_connection()

        self.scan_dev_has_setup_these_pars_pyqtsig.emit({
            "unitName": self.return_scan_dev_info().get("stepUnitName", Units.not_defined.name),
            "start": self.sc_start,
            "stop": self.sc_stop,
            "stepSize": self.sc_stepsize,
            "stepNums": self.sc_num_of_steps,
            "valsArrrayOneScan": self.sc_one_scan_vals,
        })

    def request_next_step(self):
        """Advance to the next scan point, write it to Proteus, and report success upstream."""
        if self._busy:
            logger.warning("ProteusScanDevControl is still busy setting the previous step")
            return False
        if not self.sc_one_scan_vals:
            logger.warning("ProteusScanDevControl has no scan values configured yet")
            return False

        self._busy = True
        try:
            next_step, next_scan = self._calc_next_position()
            value = self.sc_one_scan_vals[next_step]
            self._write_setpoint(value)
            self._wait_until_step_is_applied(value)

            self.sc_l_cur_step = next_step
            self.sc_l_cur_scan = next_scan
            self.scan_status = "complete" if self._is_complete() else "scanning"
            self.sc_l_perc_compl = self._calc_percent_complete()

            self.scan_dev_has_set_a_new_step_pyqtsig.emit({
                "curStep": self.sc_l_cur_step,
                "curScan": self.sc_l_cur_scan,
                "percentOfScan": self.sc_l_perc_compl,
                "curStepVal": value,
                "scanStatus": self.scan_status,
            })
            return True
        finally:
            self._busy = False

    def abort_scan(self):
        """Mark the current scan as aborted."""
        self.scan_status = "aborted"
        return True

    def set_pre_scan_masurement_setpoint(self, set_val):
        """Write a one-off setpoint used before or after the actual scan."""
        if set_val is None:
            return True
        try:
            self._write_setpoint(set_val)
        except Exception:
            logger.exception("ProteusScanDevControl: failed to set prescan setpoint")
            return False
        return True

    def deinit_scan_dev(self):
        """Drop cached connections and shut down the local Proteus instance."""
        self._connection = None
        self._readback_connection = None
        self._ready_connection = None
        self._known_targets = []
        self._display_target_map = {}
        self._known_devices_cache = {}
        self._connected_instance_addresses.clear()
        self._failed_instance_addresses.clear()
        if self._cm_instance is not None:
            try:
                self._cm_instance.__exit__(None, None, None)
            except Exception:
                logger.exception("ProteusScanDevControl: failed to close Proteus instance")
        self._cm_instance = None
        self._instance = None

    def connect(self, device_name: str, variable_name: str):
        """Create a Proteus Connection object for a device property."""
        self._ensure_instance()
        return InstanceObject.connect(self, device_name, variable_name)

    def known_devices(self):
        """Return the device/property view that Proteus currently knows about."""
        self._ensure_instance()
        return InstanceObject.known_devices(self)

    def _create_local_instance(self):
        """Create the local Proteus client instance if it does not already exist."""
        if not PROTEUS_AVAILABLE:
            raise RuntimeError("proteus package is not available")
        if self._instance is None:
            self._cm_instance = proteus.Instance(**PROTEUS_INSTANCE_CONFIG)
            self._instance = self._cm_instance.__enter__()
        return self._instance

    def _ensure_instance(self):
        """Guarantee that a local Proteus client instance exists."""
        if not PROTEUS_AVAILABLE:
            raise RuntimeError("proteus package is not available")
        if self._instance is None:
            self._create_local_instance()

    def _remote_instance_addresses(self):
        """Build the list of remote Proteus instances to probe or connect to."""
        addresses = list(self.DISCOVERY_INSTANCE_ADDRESSES)
        if self.instance_address and self.instance_address not in addresses:
            addresses.append(self.instance_address)
        result = []
        for address in addresses:
            address = str(address or "").strip()
            if address and address not in result:
                result.append(address)
        return result

    def _ensure_discovery_instances_connected(self):
        """Join configured remote instances once so discovery can inspect their status."""
        self._ensure_instance()
        for address in self._remote_instance_addresses():
            if address in self._connected_instance_addresses or address in self._failed_instance_addresses:
                continue
            try:
                ensure_remote_instance_connected(self.instance, address)
                self._connected_instance_addresses.add(address)
            except Exception:
                self._failed_instance_addresses.add(address)
                _debug_exception(
                    "ProteusScanDevControl: failed to connect discovery instance %s",
                    address,
                )

    def _query_remote_instance_targets(self, address: str):
        """Read the exported device/property status for one remote Proteus instance."""
        address = str(address or "").strip()
        if not address:
            return {}

        self._ensure_instance()
        if address not in self._connected_instance_addresses and address not in self._failed_instance_addresses:
            try:
                ensure_remote_instance_connected(self.instance, address)
                self._connected_instance_addresses.add(address)
            except Exception:
                self._failed_instance_addresses.add(address)
                raise

        known_instances = getattr(self.instance, "_known_instances", {})
        #print("RIs", known_instances, "\n")
        remote_ref = known_instances.get(address)
        if remote_ref is None:
            return {}
        try:
            status = remote_ref._status_json()
            #print("RI stat", status, "\n")
        except Exception:
            _debug_exception(
                "ProteusScanDevControl: failed to read status json from %s",
                address,
            )
            return {}
        return status if isinstance(status, dict) else {}

    def _refresh_known_targets(self):
        """Refresh the discovered ``instance::device`` entries used by the UI."""
        cache = {}
        targets = []
        for address in self._remote_instance_addresses():
            if address in self._failed_instance_addresses:
                continue
            try:
                known = self._query_remote_instance_targets(address)
            except Exception:
                self._failed_instance_addresses.add(address)
                _debug_exception(
                    "ProteusScanDevControl: failed to query remote instance %s for discovery",
                    address,
                )
                continue
            address_targets = {}
            for dev_name, props in known.items():
                if not isinstance(props, dict):
                    continue
                address_targets[dev_name] = dict(props)
                targets.append(f"{address}::{dev_name}")
            if address_targets:
                cache[address] = address_targets
        if not cache:
            logger.debug("ProteusScanDevControl: no remote Proteus targets discovered")
        self._known_devices_cache = cache
        self._known_targets = sorted(set(targets))
        self._display_target_map = {target: target for target in self._known_targets}
        self.__class__.DISCOVERED_TARGETS = list(self._known_targets)
        self.__class__.DISCOVERED_TARGET_MAP = dict(self._display_target_map)

    def _current_device_spec(self) -> str:
        """Return the configured target without the variable component."""
        if not self.target_device:
            return ""
        if self.instance_address:
            return f"{self.instance_address}::{self.target_device}"
        return self.target_device

    def _normalise_device_spec(self, dev_type: str) -> str:
        """Resolve discovered labels and tolerate legacy placeholder values."""
        dev_type = str(dev_type or "").strip()
        if not dev_type or dev_type == "Proteus":
            return self._current_device_spec()
        return self._display_target_map.get(
            dev_type,
            self.__class__.DISCOVERED_TARGET_MAP.get(dev_type, dev_type),
        )

    def _parse_device_spec(self, spec: str) -> Tuple[str, str]:
        """Parse a device selector into instance and device names."""
        instance_address, device_name, _variable_name, _readback, _ready = self._parse_target_spec(spec)
        return instance_address, device_name

    def _query_device_variables(self, instance_address: str, device_name: str) -> List[str]:
        """Read the variable names for one device from one remote instance."""
        if instance_address:
            self._failed_instance_addresses.discard(instance_address)
            known = self._query_remote_instance_targets(instance_address)
            device_props = known.get(device_name, {})
            if isinstance(device_props, dict):
                cached_devices = self._known_devices_cache.setdefault(instance_address, {})
                cached_devices[device_name] = dict(device_props)
                return sorted(device_props.keys())
            return []

        if not self._known_devices_cache:
            self._refresh_known_targets()
        for known in self._known_devices_cache.values():
            device_props = known.get(device_name)
            if isinstance(device_props, dict):
                return sorted(device_props.keys())
        return []

    def _build_target_spec_from_selection(self, dev_type, dev_name) -> str:
        """Combine the UI's type/name selections into one target specification."""
        device_spec = self._normalise_device_spec(dev_type)
        if not device_spec:
            return ""
        instance_address, device_name = self._parse_device_spec(device_spec)
        default_variable = (
            self.target_variable
            if device_spec == self._current_device_spec() and self.target_variable
            else "setpoint"
        )
        variable_name, readback_variable, ready_variable = self._parse_variable_spec(
            dev_name, default_variable
        )
        parts = [f"device={device_name}", f"variable={variable_name}"]
        if instance_address:
            parts.insert(0, f"instance={instance_address}")
        if readback_variable:
            parts.append(f"readback={readback_variable}")
        if ready_variable:
            parts.append(f"ready={ready_variable}")
        return ";".join(parts)

    def _parse_variable_spec(self, spec: str, default_variable: str = "setpoint") -> Tuple[str, str, str]:
        """Parse the second-box variable selector and optional readback/ready fields."""
        spec = str(spec or "").strip()
        default_variable = str(default_variable or "").strip() or "setpoint"
        if not spec:
            return default_variable, "", ""
        if "::" in spec:
            raise ValueError("variable selector must not contain an instance or device")
        if "=" not in spec:
            return spec, "", ""

        entries = {}
        for item in spec.split(";"):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(
                    "invalid Proteus variable specification segment "
                    f"{item!r}; use key=value pairs separated by ';'"
                )
            key, value = item.split("=", 1)
            key = key.strip().lower()
            if key not in {"variable", "readback", "ready"}:
                raise ValueError(f"unsupported Proteus variable specification key {key!r}")
            entries[key] = value.strip()

        return (
            entries.get("variable", default_variable) or default_variable,
            entries.get("readback", ""),
            entries.get("ready", ""),
        )

    def _parse_aux_target_spec(self, spec: str, default_device_spec: str) -> Tuple[str, str, str]:
        """Parse an optional readback/ready target, relative to the main scan device."""
        spec = str(spec or "").strip()
        if not spec:
            return "", "", ""
        if "::" in spec or "device=" in spec.lower() or "instance=" in spec.lower():
            instance_address, device_name, variable_name, _readback, _ready = self._parse_target_spec(spec)
            return instance_address, device_name, variable_name

        instance_address, device_name = self._parse_device_spec(default_device_spec)
        if "=" in spec:
            variable_name, _readback, _ready = self._parse_variable_spec(spec)
            return instance_address, device_name, variable_name
        return instance_address, device_name, spec

    def _parse_target_spec(self, spec: str) -> Tuple[str, str, str, str, str]:
        """Parse compact or explicit target syntax into instance/device/property fields."""
        spec = str(spec or "").strip()
        if "=" in spec:
            entries = {}
            for item in spec.split(";"):
                item = item.strip()
                if not item:
                    continue
                if "=" not in item:
                    raise ValueError(
                        "invalid Proteus target specification segment "
                        f"{item!r}; use key=value pairs separated by ';'"
                    )
                key, value = item.split("=", 1)
                entries[key.strip().lower()] = value.strip()

            device_name = entries.get("device", "")
            if not device_name:
                raise ValueError("Proteus target specification is missing 'device='")

            return (
                entries.get("instance", ""),
                device_name,
                entries.get("variable", "setpoint") or "setpoint",
                entries.get("readback", ""),
                entries.get("ready", ""),
            )

        parts = [part.strip() for part in spec.split("::") if part.strip()]
        if not parts:
            raise ValueError("empty Proteus target specification")

        if len(parts) == 1:
            return "", parts[0], "setpoint", "", ""
        if len(parts) == 2:
            if "://" in parts[0]:
                return parts[0], parts[1], "setpoint", "", ""
            return "", parts[0], parts[1], "", ""
        return parts[0], parts[1], parts[2], "", ""

    def _configure_target(self, target_spec: str, readback_spec: str = "", ready_spec: str = ""):
        """Resolve a UI target string and store the parsed target fields on the controller."""
        target_spec = str(target_spec or "").strip()
        target_spec = self._display_target_map.get(
            target_spec,
            self.__class__.DISCOVERED_TARGET_MAP.get(target_spec, target_spec),
        )
        instance_address, device_name, variable_name, readback_variable, ready_variable = (
            self._parse_target_spec(target_spec)
        )
        self.target_spec = target_spec
        self.instance_address = instance_address
        self.target_device = device_name
        self.target_variable = variable_name or "setpoint"
        main_device_spec = f"{instance_address}::{device_name}" if instance_address else device_name
        if readback_spec:
            rb_instance, rb_device, rb_variable = self._parse_aux_target_spec(readback_spec, main_device_spec)
        elif readback_variable:
            rb_instance, rb_device, rb_variable = instance_address, device_name, readback_variable
        else:
            rb_instance, rb_device, rb_variable = "", "", ""
        if ready_spec:
            ready_instance, ready_device, ready_variable_name = self._parse_aux_target_spec(ready_spec, main_device_spec)
        elif ready_variable:
            ready_instance, ready_device, ready_variable_name = instance_address, device_name, ready_variable
        else:
            ready_instance, ready_device, ready_variable_name = "", "", ""
        self.readback_spec = str(readback_spec or "").strip()
        self.ready_spec = str(ready_spec or "").strip()
        self.readback_instance_address = rb_instance
        self.readback_device = rb_device
        self.readback_variable = rb_variable
        self.ready_instance_address = ready_instance
        self.ready_device = ready_device
        self.ready_variable = ready_variable_name
        self._connection = None
        self._readback_connection = None
        self._ready_connection = None
        for address in {self.instance_address, self.readback_instance_address, self.ready_instance_address}:
            if not address:
                continue
            self._failed_instance_addresses.discard(address)
            if address in self._connected_instance_addresses:
                continue
            try:
                ensure_remote_instance_connected(self.instance, address)
                self._connected_instance_addresses.add(address)
            except Exception:
                self._failed_instance_addresses.add(address)
                _debug_exception(
                    "ProteusScanDevControl: failed to connect selected target instance %s",
                    address,
                )

    def _connect_property(self, device_name: str, variable_name: str):
        """Create a Proteus connection for the selected device and one property."""
        return self.connect(device_name, variable_name)

    def _ensure_connection(self):
        """Create and cache the main writable connection used for stepping the scan."""
        if self._connection is not None and getattr(self._connection, "is_connected", False):
            return self._connection

        if not self.target_device:
            raise RuntimeError("Proteus target device is not configured")

        deadline = time.perf_counter() + 2.0
        last_exc = None
        while time.perf_counter() <= deadline:
            try:
                conn = self._connect_property(self.target_device, self.target_variable)
                if conn.is_connected:
                    self._connection = conn
                    logger.info(
                        "ProteusScanDevControl: connected to %s on %s",
                        f"{self.target_device}.{self.target_variable}",
                        self.instance_address or "<auto-discovery>",
                    )
                    return self._connection
                last_exc = RuntimeError(
                    f"Target Property {self.target_variable} on Device {self.target_device} not connected yet"
                )
            except Exception as exc:
                last_exc = exc
                _debug_exception(
                    "ProteusScanDevControl: connection retry for %s.%s failed",
                    self.target_device,
                    self.target_variable,
                )
            time.sleep(0.1)

        raise RuntimeError(self._build_connection_error(last_exc))

    def _ensure_readback_connection(self):
        """Create and cache the optional readback connection if configured."""
        if not self.readback_variable:
            return None
        if self._readback_connection is not None and getattr(self._readback_connection, "is_connected", False):
            return self._readback_connection
        conn = self._connect_property(self.readback_device or self.target_device, self.readback_variable)
        if conn.is_connected:
            self._readback_connection = conn
            return self._readback_connection
        return None

    def _ensure_ready_connection(self):
        """Create and cache the optional ready-state connection if configured."""
        if not self.ready_variable:
            return None
        if self._ready_connection is not None and getattr(self._ready_connection, "is_connected", False):
            return self._ready_connection
        conn = self._connect_property(self.ready_device or self.target_device, self.ready_variable)
        if conn.is_connected:
            self._ready_connection = conn
            return self._ready_connection
        return None

    def _write_setpoint(self, value):
        """Write a scan setpoint to Proteus, retrying transient failures."""
        self._set_ready_false_before_step()
        last_exc = None
        for attempt in range(3):
            conn = self._ensure_connection()
            try:
                conn.set(value)
                return
            except Exception as set_exc:
                try:
                    conn.trigger(value)
                    return
                except Exception as trigger_exc:
                    last_exc = RuntimeError(
                        f"set failed with {set_exc!r}; trigger failed with {trigger_exc!r}"
                    )
                    logger.warning(
                        "ProteusScanDevControl: set/trigger attempt %d failed for %s.%s, retrying",
                        attempt + 1,
                        self.target_device,
                        self.target_variable,
                        exc_info=True,
                    )
                    self._connection = None
                    time.sleep(0.1)
        raise RuntimeError(self._build_connection_error(last_exc, during_write=True)) from last_exc

    def _set_ready_false_before_step(self):
        """Best-effort reset of the ready flag before sending a new step."""
        if not self.ready_variable:
            return
        try:
            conn = self._ensure_ready_connection()
            if conn is not None:
                conn.set(False)
        except Exception:
            _debug_exception(
                "ProteusScanDevControl: failed to clear ready=%s before sending step",
                self.ready_variable,
            )

    def _read_connection_now(self, connection_factory, device_name: str, variable_name: str):
        """Read the current value of one configured Proteus property."""
        if not device_name or not variable_name:
            return None
        try:
            conn = connection_factory()
            if not conn.is_connected:
                return None
            return conn.get()
        except Exception:
            _debug_exception(
                "ProteusScanDevControl: failed to read %s from %s",
                variable_name,
                device_name,
            )
            return None

    def _values_match(self, expected: Any, actual: Any, atol: float = 1e-9) -> bool:
        """Compare expected and actual values, using tolerance for numeric types."""
        if actual is None:
            return False
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            return math.isclose(float(expected), float(actual), abs_tol=atol, rel_tol=0.0)
        return expected == actual

    def _wait_until_step_is_applied(self, expected_value: Any):
        """
        Poll settle signals until the new setpoint is considered applied.

        If a ``ready`` variable is configured, it is treated as the
        authoritative indication that the device has finished moving to the
        new step. Otherwise the controller falls back to comparing the
        readback/target value against the requested setpoint.
        """
        timeout_s = max(0.0, float(getattr(self, "scan_dev_timeout", 0.0) or 0.0))
        if timeout_s <= 0.0:
            timeout_s = 10.0
        poll_interval_s = 0.05
        deadline = time.perf_counter() + timeout_s

        while time.perf_counter() <= deadline:
            if self.ready_variable:
                ready_val = self._read_connection_now(
                    self._ensure_ready_connection,
                    self.ready_device or self.target_device,
                    self.ready_variable,
                )
                if ready_val is not None and bool(ready_val):
                    return
            else:
                readback_ok = True
                if self.readback_variable:
                    readback_val = self._read_connection_now(
                        self._ensure_readback_connection,
                        self.readback_device or self.target_device,
                        self.readback_variable,
                    )
                else:
                    readback_val = self._read_connection_now(
                        self._ensure_connection,
                        self.target_device,
                        self.target_variable,
                    )
                if readback_val is not None:
                    readback_ok = self._values_match(expected_value, readback_val)
                elif self.readback_variable:
                    readback_ok = False

                if readback_ok:
                    return

            time.sleep(poll_interval_s)

        raise TimeoutError(
            "Proteus scan device did not confirm the new setpoint within "
            f"{timeout_s:.2f} s"
        )

    def _build_connection_error(self, exc: Exception, during_write: bool = False) -> str:
        """Format a consistent error message for connection or write failures."""
        action = "write to" if during_write else "connect to"
        return (
            f"Proteus scan device could not {action} "
            f"{self.target_device}.{self.target_variable} at "
            f"{self.instance_address or '<auto-discovery>'}: {exc}"
        )

    def _calc_next_position(self) -> Tuple[int, int]:
        """Compute the next step/scan indices, including inverted odd scans."""
        if self.sc_num_of_steps <= 0 or self.sc_num_of_scans <= 0:
            raise RuntimeError("scan parameters are not initialised")

        if self.sc_l_cur_step < 0:
            return (self.sc_num_of_steps - 1, 0) if self._scan_is_inverted(0) else (0, 0)

        cur_scan = self.sc_l_cur_scan
        cur_step = self.sc_l_cur_step
        scan_dir = -1 if self._scan_is_inverted(cur_scan) else 1
        candidate = cur_step + scan_dir

        if 0 <= candidate < self.sc_num_of_steps:
            return candidate, cur_scan

        next_scan = cur_scan + 1
        if next_scan >= self.sc_num_of_scans:
            return cur_step, cur_scan

        if self._scan_is_inverted(next_scan):
            return self.sc_num_of_steps - 1, next_scan
        return 0, next_scan

    def _scan_is_inverted(self, scan_index: int) -> bool:
        """Return whether a given scan index should run in reverse order."""
        return self.sc_invert_in_odd_scans and scan_index % 2 == 1

    def _calc_percent_complete(self) -> float:
        """Estimate the completed fraction of the full scan sequence."""
        if self.sc_l_cur_step < 0:
            return 0.0
        if self._scan_is_inverted(self.sc_l_cur_scan):
            completed_steps = (
                (self.sc_num_of_steps - self.sc_l_cur_step)
                + self.sc_l_cur_scan * self.sc_num_of_steps
            )
        else:
            completed_steps = (
                (self.sc_l_cur_step + 1)
                + self.sc_l_cur_scan * self.sc_num_of_steps
            )
        total_steps = max(1, self.sc_num_of_steps * self.sc_num_of_scans)
        return completed_steps / total_steps

    def _is_complete(self) -> bool:
        """Return whether the computed scan progress has reached completion."""
        return self._calc_percent_complete() >= 1.0
