import logging
from copy import deepcopy

logger = logging.getLogger(__name__)


class ProteusLogger:
    """
    Simple logger for Proteus variables.

    Expected config structure (per track and pre/during/post block):

        cfg = {
          "instance": "tcp://192.168.15.128:6000",
          "devices": {
             "DummyDevice": {
                "random_variable": {"required": 1, "acquired": 0, "data": []},
                ...
             },
             ...
          }
        }

    Only 'required' is logically important; 'acquired' and 'data' will
    be initialised when missing.
    """

    def __init__(self, name: str = "ProteusLogger"):
        self.name = name
        self.devices = {}  # type: dict
        self.pre_dur_post_str = ""
        self.track_name = ""
        self.logging = False
        # If nothing is required, we consider logging immediately complete.
        self.logging_complete = True

    # ------------------------------------------------------------------ #
    # Setup / control
    # ------------------------------------------------------------------ #

    def setup_log(self, cfg: dict, pre_dur_post_str: str, track_name: str) -> None:
        devices_cfg = (cfg or {}).get("devices", {}) or {}

        devs = {}
        for dev_name, chan_dict in devices_cfg.items():
            if not isinstance(chan_dict, dict):
                continue
            dev_entry = {}
            for ch_name, ch_cfg in chan_dict.items():
                if not isinstance(ch_cfg, dict):
                    continue
                required = int(ch_cfg.get("required", 0))
                acquired = int(ch_cfg.get("acquired", 0))
                data = list(ch_cfg.get("data", []))
                dev_entry[ch_name] = {
                    "required": required,
                    "acquired": acquired,
                    "data": data,
                }
            if dev_entry:
                devs[dev_name] = dev_entry

        self.devices = devs
        self.pre_dur_post_str = pre_dur_post_str
        self.track_name = track_name
        self.logging = False

        self._update_complete_flag()

        logger.info(
            "%s: setup for %s/%s with devices=%s",
            self.name,
            self.pre_dur_post_str,
            self.track_name,
            list(self.devices.keys()),
        )

    def start_log(self) -> None:
        self.logging = True
        logger.info("%s: start_log", self.name)

    def stop_log(self) -> None:
        self.logging = False
        logger.info("%s: stop_log", self.name)

    # ------------------------------------------------------------------ #
    # Data updates
    # ------------------------------------------------------------------ #

    def handle_status_update(self, device, variable, value):
        """
        Called by TildaProteusBridge when a subscribed variable changes.
        """
        if not self.logging:
            return

        logger.debug(
            "%s: update %s.%s -> %r", self.name, device, variable, value
        )

        dev_dict = self.devices.setdefault(device, {})
        var_dict = dev_dict.setdefault(
            variable, {"required": 0, "acquired": 0, "data": []}
        )

        var_dict["data"].append(value)
        var_dict["acquired"] += 1
        self._update_complete_flag()

    def _update_complete_flag(self) -> None:
        complete = True
        for dev in self.devices.values():
            for ch in dev.values():
                required = int(ch.get("required", 0))
                acquired = int(ch.get("acquired", 0))
                if required > 0 and acquired < required:
                    complete = False
                    break
            if not complete:
                break

        self.logging_complete = complete
        if complete:
            logger.info(
                "%s: logging complete for %s/%s",
                self.name,
                self.pre_dur_post_str,
                self.track_name,
            )

    # ------------------------------------------------------------------ #
    # Accessor used by TildaTools.save_proteus_to_xml
    # ------------------------------------------------------------------ #

    @property
    def log(self) -> dict:
        # Deepcopy so XML writing cannot accidentally modify internal state.
        return deepcopy(self.devices)
