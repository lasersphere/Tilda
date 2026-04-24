"""
Proteus-backed external scan device controller for TILDA.

This controller keeps the scan stepping logic on the TILDA side and writes
the current setpoint to a writable Proteus variable on demand.

Supported target syntaxes:

Compact:
    DeviceName
    DeviceName::VariableName
    tcp://host:6000::DeviceName
    tcp://host:6000::DeviceName::VariableName

Explicit:
    instance=tcp://host:6000;device=MyDevice;variable=setpoint
    instance=tcp://host:7000;device=MyDevice;variable=set_val;
    readback=scan_var;ready=ready

If the variable name is omitted, ``setpoint`` is used.
"""

import logging
import math
import time
from typing import Any, Optional, Tuple

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
    from proteus.Connection import Connection  # type: ignore
    from proteus.InstanceObject import InstanceObject  # type: ignore
    PROTEUS_AVAILABLE = True
except Exception as exc:
    logger.info("ProteusScanDevControl: proteus import failed (%s)", exc)
    proteus = None  # type: ignore
    Connection = None  # type: ignore
    InstanceObject = None  # type: ignore
    PROTEUS_AVAILABLE = False

ProteusInstanceBase = InstanceObject if PROTEUS_AVAILABLE else object

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


class ProteusScanDevControl(BaseTildaScanDeviceControl, ProteusInstanceBase):
    """
    TILDA scan-device adapter for Proteus variables.

    The controller expects a writable Proteus variable that represents the
    device setpoint.

    Optional handshake variables can be configured through the explicit
    syntax:

    - ``readback``: variable that should match the requested setpoint
    - ``ready``: boolean variable that should become ``True`` once the
      device has settled

    The configured scan-device target itself must be the writable setpoint
    variable. Read-only measurement channels such as ``scan_var`` belong
    into ``readback=...`` instead.
    """

    def __init__(self):
        super(ProteusScanDevControl, self).__init__()
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

    def available_scan_dev_types(self):
        return ["Proteus"]

    def available_scan_dev_names_by_type(self, dev_type):
        if dev_type != "Proteus":
            return []
        names = []
        try:
            for dev_name in self._known_device_names():
                names.append(dev_name)
        except Exception:
            logger.debug("ProteusScanDevControl: device discovery failed", exc_info=True)
        if self.target_spec and self.target_spec not in names:
            names.append(self.target_spec)
        return names

    def return_scan_dev_info(self, dev_type=None, dev_name=None):
        if dev_name:
            self._configure_target(dev_name)

        info = {
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

        return info

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
        if self._cm_instance is not None:
            try:
                self._cm_instance.__exit__(None, None, None)
            except Exception:
                logger.exception("ProteusScanDevControl: failed to close Proteus instance")
        self._cm_instance = None
        self._instance = None

    def _parse_target_spec(self, spec: str) -> Tuple[str, str, str, str, str]:
        spec = (spec or "").strip()
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
        instance_address, device_name, variable_name, readback_variable, ready_variable = \
            self._parse_target_spec(target_spec)
        self.target_spec = target_spec
        self.instance_address = instance_address
        self.target_device = device_name
        self.target_variable = variable_name or "setpoint"
        self.readback_variable = readback_variable
        self.ready_variable = ready_variable
        self._connection = None
        self._readback_connection = None
        self._ready_connection = None

    def _ensure_instance(self):
        if not PROTEUS_AVAILABLE:
            raise RuntimeError("proteus package is not available")
        if self._instance is None:
            self._cm_instance = proteus.Instance(**PROTEUS_INSTANCE_CONFIG)
            self._instance = self._cm_instance.__enter__()
        if self.instance_address:
            ensure_remote_instance_connected(self.instance, self.instance_address)

    def _connect_property(self, variable_name: str):
        self._ensure_instance()
        return self.connect(self.target_device, variable_name)

    def _ensure_connection(self):
        if self._connection is not None and getattr(self._connection, "is_connected", False):
            return self._connection

        if not self.target_device:
            raise RuntimeError("Proteus target device is not configured")

        self._ensure_instance()
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
                        self.instance_address or "<local>",
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

    def _known_device_names(self):
        self._ensure_instance()
        status = self.known_devices()
        return sorted(status.keys())

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
        msg = (
            f"Proteus scan device could not {action} "
            f"{self.target_device}.{self.target_variable} at {self.instance_address or '<local>'}: {exc}"
        )
        msg += (
            ". The scan-device target must be the writable setpoint variable. "
            "If your Proteus device exposes a separate readout such as 'scan_var', "
            "configure the scan device with explicit syntax like "
            "'instance=tcp://host:7000;device=Benchmark_Dev;variable=<writable_setpoint>;readback=scan_var'."
        )
        if self.readback_variable:
            msg += f" Current readback is '{self.readback_variable}'."
        if self.ready_variable:
            msg += f" Current ready variable is '{self.ready_variable}'."
        return msg

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
