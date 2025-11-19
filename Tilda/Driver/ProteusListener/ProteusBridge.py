import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

# Try to import proteus and the Connection helper
try:
    import proteus  # type: ignore
    from proteus.Connection import Connection  # type: ignore
    PROTEUS_AVAILABLE = True
except Exception as exc:  # proteus missing, wrong environment, etc.
    logger.info(
        "ProteusBridge: proteus could not be imported (%s). "
        "Proteus integration will be disabled.",
        exc,
    )
    proteus = None  # type: ignore
    Connection = None  # type: ignore
    PROTEUS_AVAILABLE = False

# Reuse the same lab instance configuration as the PreScanConfigUi
try:
    from Tilda.Interface.PreScanConfigUi.PreScanConfigUi import PROTEUS_INSTANCE_CONFIG
except Exception:
    # Very conservative fallback in case the import path changes.
    PROTEUS_INSTANCE_CONFIG = {
        "instance_type": "zmq",
        "protocol": "tcp",
        "command_port": 6000,
        "publish_port": 6001,
        "port_range": 1000,
        "port_attempts": 50,
    }


class TildaProteusBridge:
    """
    Small helper that owns a Proteus Instance and a set of Connection objects.

    It:
      - connects the local Instance to the remote lab instance (add_instance),
      - subscribes to the configured device/variable pairs,
      - forwards updates into a ProteusLogger via handle_status_update().
    """

    def __init__(self, logger, instance=None):
        if not PROTEUS_AVAILABLE:
            raise RuntimeError("Proteus not available; cannot create TildaProteusBridge.")

        # Create or reuse a Proteus Instance (lab-style configuration)
        if instance is None:
            try:
                self.instance = proteus.Instance(**PROTEUS_INSTANCE_CONFIG)
            except Exception as exc:
                raise RuntimeError(f"Could not create Proteus Instance: {exc}") from exc
        else:
            self.instance = instance

        self.logger = logger
        self._connections: Dict[Tuple[str, str], object] = {}

        cmd_addr = getattr(self.instance, "command_address", None)
        pub_addr = getattr(self.instance, "publish_address", None)
        logger.info(
            "TildaProteusBridge: created Proteus client (cmd=%r, pub=%r)",
            cmd_addr,
            pub_addr,
        )

    # ------------------------------------------------------------------ #
    # Connection / subscription
    # ------------------------------------------------------------------ #

    def connect_to_remote_instance(self, address: str) -> None:
        """
        Let the local Proteus Instance know about the remote lab instance.

        The address is the one you type in the PreScan UI (e.g. 'tcp://192.168.15.128:6000').
        """
        address = (address or "").strip()
        if not address:
            logger.warning("TildaProteusBridge: empty instance address, nothing to connect.")
            return

        logger.info("TildaProteusBridge: add_instance(%s)", address)
        self.instance.add_instance(address)

    def configure_channels(self, devices_cfg: Dict[str, dict]) -> None:
        """
        Subscribe to all requested device / variable pairs.

        devices_cfg is expected to look like:
            {
              "DummyDevice": {
                  "random_variable": {"required": 1, ...},
                  ...
              },
              ...
            }
        """
        if not devices_cfg:
            logger.info("TildaProteusBridge: no devices configured, nothing to subscribe.")
            return

        if Connection is None:
            logger.warning("TildaProteusBridge: Connection class not available.")
            return

        # Unsubscribe previous connections (if any)
        for conn in self._connections.values():
            try:
                conn.unsubscribe_variable()
            except Exception:
                pass
        self._connections.clear()

        for dev_name, chan_dict in devices_cfg.items():
            if not isinstance(chan_dict, dict):
                continue

            for var_name in chan_dict.keys():
                key = (dev_name, var_name)
                if key in self._connections:
                    continue

                conn = Connection(self.instance, dev_name, var_name)

                def _make_cb(device=dev_name, var=var_name):
                    def _cb(value):
                        if self.logger is not None:
                            self.logger.handle_status_update(device, var, value)
                    return _cb

                conn.subscribe_variable(_make_cb())
                self._connections[key] = conn
                logger.info("TildaProteusBridge: subscribed to %s.%s", dev_name, var_name)

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """
        Unsubscribe all variables and close the Proteus Instance.

        (Not strictly required for your dummy tests, but nice to have.)
        """
        for conn in self._connections.values():
            try:
                conn.unsubscribe_variable()
            except Exception:
                pass
        self._connections.clear()

        try:
            # proteus.Instance supports being used as a context manager.
            # In most builds it also has a close() method; if not, this will just raise and be ignored.
            self.instance.close()
        except Exception:
            pass
