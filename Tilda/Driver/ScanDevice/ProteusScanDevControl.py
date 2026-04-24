"""
Proteus-backed external scan device controller for TILDA.

This controller owns its own local Proteus instance, discovers devices on a
configured list of remote Proteus instances, and drives a selected writable
Proteus variable step-by-step for scans.

Supported target syntaxes:

Compact:
    DeviceName
    DeviceName::VariableName
    tcp://host:6000::DeviceName
    tcp://host:6000::DeviceName::VariableName

Explicit:
    instance=tcp://host:7000;device=MyDevice;variable=set_val
    instance=tcp://host:7000;device=MyDevice;variable=set_val;
    readback=scan_var;ready=ready

If the variable name is omitted, ``setpoint`` is used.
"""

import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from Tilda.Driver.ProteusListener.ProteusImport import (
    ensure_proteus_on_path,
    ensure_remote_instance_connected,
)
from Tilda.Driver.ScanDevice.BaseTildaScanDeviceControl import BaseTildaScanDeviceControl
from Tilda.PolliFit.Measurement.SpecData import SpecDataXAxisUnits as Units

logger = logging.getLogger(__name__)

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
    ]
    DISCOVERED_TARGET_MAP: Dict[str, str] = {}
    DISCOVERED_TARGETS: List[str] = []

    def __init__(self):
        BaseTildaScanDeviceControl.__init__(self)

        self._cm_instance = None
        self._instance = None

        self.instance_address = ""
        self.target_device = ""
        self.target_variable = "setpoint"
        self.readback_variable = ""
        self.ready_variable = ""
        self.target_spec = ""

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
        return self._instance

    @instance.setter
    def instance(self, value):
        self._instance = value

    def available_scan_dev_types(self):
        return ["Proteus"]

    def available_scan_dev_names_by_type(self, dev_type):
        if dev_type != "Proteus":
            return []

        names = []
        try:
            self._refresh_known_targets()
            names.extend(self._known_targets)
        except Exception:
            logger.debug("ProteusScanDevControl: device discovery failed", exc_info=True)

        if self.target_spec and self.target_spec not in names:
            names.append(self.target_spec)
        return names

    def return_scan_dev_info(self, dev_type=None, dev_name=None):
        if dev_name:
            self._configure_target(dev_name)

        return {
            "name": self.target_spec or dev_name or "ProteusDevice",
            "type": "Proteus",
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
        self.scan_status = "aborted"
        return True

    def set_pre_scan_masurement_setpoint(self, set_val):
        if set_val is None:
            return True
        try:
            self._write_setpoint(set_val)
        except Exception:
            logger.exception("ProteusScanDevControl: failed to set prescan setpoint")
            return False
        return True

    def deinit_scan_dev(self):
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
        self._ensure_instance()
        return InstanceObject.connect(self, device_name, variable_name)

    def known_devices(self):
        self._ensure_instance()
        return InstanceObject.known_devices(self)

    def _create_local_instance(self):
        if not PROTEUS_AVAILABLE:
            raise RuntimeError("proteus package is not available")
        if self._instance is None:
            self._cm_instance = proteus.Instance(**PROTEUS_INSTANCE_CONFIG)
            self._instance = self._cm_instance.__enter__()
        return self._instance

    def _ensure_instance(self):
        if not PROTEUS_AVAILABLE:
            raise RuntimeError("proteus package is not available")
        if self._instance is None:
            self._create_local_instance()

    def _remote_instance_addresses(self):
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
        self._ensure_instance()
        for address in self._remote_instance_addresses():
            if address in self._connected_instance_addresses or address in self._failed_instance_addresses:
                continue
            try:
                ensure_remote_instance_connected(self.instance, address)
                self._connected_instance_addresses.add(address)
            except Exception:
                self._failed_instance_addresses.add(address)
                logger.debug(
                    "ProteusScanDevControl: failed to connect discovery instance %s",
                    address,
                    exc_info=True,
                )

    def _query_remote_instance_targets(self, address: str):
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
        remote_ref = known_instances.get(address)
        if remote_ref is None:
            return {}
        try:
            status = remote_ref._status_json()
        except Exception:
            logger.debug(
                "ProteusScanDevControl: failed to read status json from %s",
                address,
                exc_info=True,
            )
            return {}
        return status if isinstance(status, dict) else {}

    def _refresh_known_targets(self):
        cache = {}
        targets = []
        display_target_map = {}
        for address in self._remote_instance_addresses():
            if address in self._failed_instance_addresses:
                continue
            try:
                known = self._query_remote_instance_targets(address)
            except Exception:
                self._failed_instance_addresses.add(address)
                logger.debug(
                    "ProteusScanDevControl: failed to query remote instance %s for discovery",
                    address,
                    exc_info=True,
                )
                continue
            address_targets = {}
            for dev_name, props in known.items():
                if not isinstance(props, dict):
                    continue
                address_targets[dev_name] = dict(props)
                for var_name in sorted(props.keys()):
                    short_target = f"{dev_name}::{var_name}"
                    full_target = f"{address}::{dev_name}::{var_name}"
                    targets.append(short_target)
                    display_target_map.setdefault(short_target, full_target)
            if address_targets:
                cache[address] = address_targets
        if not cache:
            logger.debug("ProteusScanDevControl: no remote Proteus targets discovered")
        self._known_devices_cache = cache
        self._known_targets = sorted(set(targets))
        self._display_target_map = display_target_map
        self.__class__.DISCOVERED_TARGETS = list(self._known_targets)
        self.__class__.DISCOVERED_TARGET_MAP = dict(self._display_target_map)

    def _parse_target_spec(self, spec: str) -> Tuple[str, str, str, str, str]:
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

    def _configure_target(self, target_spec: str):
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
        self.readback_variable = readback_variable
        self.ready_variable = ready_variable
        self._connection = None
        self._readback_connection = None
        self._ready_connection = None
        if self.instance_address:
            self._failed_instance_addresses.discard(self.instance_address)
            if self.instance_address not in self._connected_instance_addresses:
                try:
                    ensure_remote_instance_connected(self.instance, self.instance_address)
                    self._connected_instance_addresses.add(self.instance_address)
                except Exception:
                    self._failed_instance_addresses.add(self.instance_address)
                    logger.debug(
                        "ProteusScanDevControl: failed to connect selected target instance %s",
                        self.instance_address,
                        exc_info=True,
                    )

    def _connect_property(self, variable_name: str):
        return self.connect(self.target_device, variable_name)

    def _ensure_connection(self):
        if self._connection is not None and getattr(self._connection, "is_connected", False):
            return self._connection

        if not self.target_device:
            raise RuntimeError("Proteus target device is not configured")

        deadline = time.perf_counter() + 2.0
        last_exc = None
        while time.perf_counter() <= deadline:
            try:
                conn = self._connect_property(self.target_variable)
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
                logger.debug(
                    "ProteusScanDevControl: connection retry for %s.%s failed",
                    self.target_device,
                    self.target_variable,
                    exc_info=True,
                )
            time.sleep(0.1)

        raise RuntimeError(self._build_connection_error(last_exc))

    def _ensure_readback_connection(self):
        if not self.readback_variable:
            return None
        if self._readback_connection is not None and getattr(self._readback_connection, "is_connected", False):
            return self._readback_connection
        conn = self._connect_property(self.readback_variable)
        if conn.is_connected:
            self._readback_connection = conn
            return self._readback_connection
        return None

    def _ensure_ready_connection(self):
        if not self.ready_variable:
            return None
        if self._ready_connection is not None and getattr(self._ready_connection, "is_connected", False):
            return self._ready_connection
        conn = self._connect_property(self.ready_variable)
        if conn.is_connected:
            self._ready_connection = conn
            return self._ready_connection
        return None

    def _write_setpoint(self, value):
        self._set_ready_false_before_step()
        last_exc = None
        for attempt in range(3):
            conn = self._ensure_connection()
            try:
                conn.set(value)
                return
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "ProteusScanDevControl: set attempt %d failed for %s.%s, retrying",
                    attempt + 1,
                    self.target_device,
                    self.target_variable,
                    exc_info=True,
                )
                self._connection = None
                time.sleep(0.1)
        raise RuntimeError(self._build_connection_error(last_exc, during_write=True)) from last_exc

    def _set_ready_false_before_step(self):
        if not self.ready_variable:
            return
        try:
            conn = self._ensure_ready_connection()
            if conn is not None:
                conn.set(False)
        except Exception:
            logger.debug(
                "ProteusScanDevControl: failed to clear ready=%s before sending step",
                self.ready_variable,
                exc_info=True,
            )

    def _read_property_now(self, prop_name: str):
        if not prop_name or not self.target_device:
            return None
        try:
            conn = self._connect_property(prop_name)
            if not conn.is_connected:
                return None
            return conn.get()
        except Exception:
            logger.debug(
                "ProteusScanDevControl: failed to read %s from %s",
                prop_name,
                self.target_device,
                exc_info=True,
            )
            return None

    def _values_match(self, expected: Any, actual: Any, atol: float = 1e-9) -> bool:
        if actual is None:
            return False
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            return math.isclose(float(expected), float(actual), abs_tol=atol, rel_tol=0.0)
        return expected == actual

    def _wait_until_step_is_applied(self, expected_value: Any):
        timeout_s = max(0.0, float(getattr(self, "scan_dev_timeout", 0.0) or 0.0))
        if timeout_s <= 0.0:
            timeout_s = 10.0
        poll_interval_s = 0.05
        deadline = time.perf_counter() + timeout_s

        while time.perf_counter() <= deadline:
            readback_ok = True
            ready_ok = True

            readback_name = self.readback_variable or self.target_variable
            readback_val = self._read_property_now(readback_name)
            if readback_val is not None:
                readback_ok = self._values_match(expected_value, readback_val)
            elif self.readback_variable:
                readback_ok = False

            if self.ready_variable:
                ready_val = self._read_property_now(self.ready_variable)
                ready_ok = bool(ready_val) if ready_val is not None else False

            if readback_ok and ready_ok:
                return

            time.sleep(poll_interval_s)

        raise TimeoutError(
            "Proteus scan device did not confirm the new setpoint within "
            f"{timeout_s:.2f} s"
        )

    def _build_connection_error(self, exc: Exception, during_write: bool = False) -> str:
        action = "write to" if during_write else "connect to"
        return (
            f"Proteus scan device could not {action} "
            f"{self.target_device}.{self.target_variable} at "
            f"{self.instance_address or '<auto-discovery>'}: {exc}"
        )

    def _calc_next_position(self) -> Tuple[int, int]:
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
        return self.sc_invert_in_odd_scans and scan_index % 2 == 1

    def _calc_percent_complete(self) -> float:
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
        return self._calc_percent_complete() >= 1.0
