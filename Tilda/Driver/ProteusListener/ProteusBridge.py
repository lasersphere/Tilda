import logging
from typing import Dict, Tuple
import threading

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
    """

    def __init__(self, logger, instance=None):
        if not PROTEUS_AVAILABLE:
            raise RuntimeError("Proteus not available; cannot create TildaProteusBridge.")

        self.logger = logger
        self._connections: Dict[Tuple[str, str], object] = {}
        self._owns_instance = False
        self._cm_instance = None

        # Create or reuse a Proteus Instance (lab-style configuration)
        if instance is None:
            try:
                # Create the instance and start its internal threads like a context manager would
                self._cm_instance = proteus.Instance(**PROTEUS_INSTANCE_CONFIG)
                self.instance = self._cm_instance.__enter__()  # starts processing + network threads
                self._owns_instance = True
            except Exception as exc:
                raise RuntimeError(f"Could not create Proteus Instance: {exc}") from exc
        else:
            # If someone passes an already-running Instance, we assume they manage its lifetime
            self.instance = instance

        cmd_addr = getattr(self.instance, "command_address", None)
        pub_addr = getattr(self.instance, "publish_address", None)
        logging.getLogger(__name__).info(
            "TildaProteusBridge: created Proteus client (cmd=%r, pub=%r)",
            cmd_addr,
            pub_addr,
        )


    # ------------------------------------------------------------------ #
    # Connection / subscription
    # ------------------------------------------------------------------ #

    def connect_to_remote_instance(self, address: str, timeout_s: float = 5.0) -> None:
        """
        Let the local Proteus Instance know about the remote lab instance.

        The address is the one you type in the PreScan UI (e.g. 'tcp://192.168.15.128:6000').

        IMPORTANT:
        - We call Instance.add_instance() in a helper thread with a timeout so that
          the main thread (and the scan state machine) cannot freeze forever.
        - This method itself remains *logically synchronous*: it only returns
          once add_instance has either succeeded or timed out / failed.
        """
        address = (address or "").strip()
        if not address:
            logger.warning(
                "TildaProteusBridge: empty instance address, nothing to connect."
            )
            return

        logger.info("TildaProteusBridge: add_instance(%s)", address)

        result = {"ok": False, "exc": None}

        def _worker():
            try:
                self.instance.add_instance(address)
                result["ok"] = True
            except Exception as exc:
                result["exc"] = exc

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout_s)

        if t.is_alive():
            logger.warning(
                "TildaProteusBridge: add_instance(%s) did not finish within %.1f s; "
                "Proteus logging will be disabled for this scan.",
                address,
                timeout_s,
            )
            # We intentionally raise, so prepare_proteus_for_scan's try/except
            # will cleanly disable Proteus logging for this scan.
            raise RuntimeError(
                f"Timed out while trying to add_instance({address!r})"
            )

        if result["exc"] is not None:
            raise RuntimeError(
                f"Error while trying to add_instance({address!r}): {result['exc']}"
            ) from result["exc"]

        logger.info(
            "TildaProteusBridge: add_instance(%s) finished successfully", address
        )

    def configure_channels(self, devices_cfg: Dict[str, dict]) -> None:
        """
        Subscribe to all requested device/variable pairs.

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
            logger.info(
                "TildaProteusBridge: no devices configured, nothing to subscribe."
            )
            return

        # Local import to avoid problems if proteus is not installed at import time.
        from proteus.Connection import Connection

        for dev_name, chan_dict in devices_cfg.items():
            if not isinstance(chan_dict, dict):
                continue

            for var_name in chan_dict.keys():
                key = (dev_name, var_name)
                if key in self._connections:
                    # Already subscribed in this bridge
                    continue

                conn = Connection(self.instance, dev_name, var_name)

                def _make_cb(device=dev_name, var=var_name):
                    # Accept any signature that Connection/proteus might use.
                    # For current proteus (1.0.7), this is (new_value, old_value, message).
                    def _cb(*args, **kwargs):
                        if self.logger is None:
                            return

                        if args:
                            new_value = args[0]
                        else:
                            # extremely defensive fallback
                            new_value = kwargs.get("value", None)

                        self.logger.handle_status_update(device, var, new_value)

                    return _cb

                cb = _make_cb()

                # In proteus_lab 1.0.7, Connection has add_update_callback
                if hasattr(conn, "add_update_callback"):
                    conn.add_update_callback(cb)
                    logger.info(
                        "TildaProteusBridge: subscribed (add_update_callback) to %s.%s",
                        dev_name,
                        var_name,
                    )
                # Older/other variants might expose subscribe_variable directly
                elif hasattr(conn, "subscribe_variable"):
                    conn.subscribe_variable(cb)
                    logger.info(
                        "TildaProteusBridge: subscribed (subscribe_variable) to %s.%s",
                        dev_name,
                        var_name,
                    )
                else:
                    logger.warning(
                        "TildaProteusBridge: Connection for %s.%s has neither "
                        "add_update_callback nor subscribe_variable – no logging.",
                        dev_name,
                        var_name,
                    )

                self._connections[key] = conn

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #

    def close(self):
        """
        Cleanly stop the Proteus Instance if we created it.
        """
        if self._owns_instance and self._cm_instance is not None:
            try:
                # Mirror the context manager shutdown
                self._cm_instance.__exit__(None, None, None)
            except Exception:
                logger.exception("Error while shutting down Proteus Instance")

    def shutdown(self) -> None:
        """
        Detach callbacks and drop all Connection objects.

        We intentionally do *not* call Connection.exit(), because in the
        current proteus version Instance.unsubscribe_variable does not exist
        anymore and would raise an AttributeError on shutdown.
        """
        for key, conn in list(self._connections.items()):
            try:
                # Try the "new" style first
                if hasattr(conn, "remove_update_callback"):
                    conn.remove_update_callback()
                # Fallback: some versions only have unsubscribe_variable
                elif hasattr(conn, "unsubscribe_variable"):
                    conn.unsubscribe_variable()
            except Exception:
                logger.exception(
                    "TildaProteusBridge: error removing callback for %s.%s",
                    key[0],
                    key[1],
                )

        self._connections.clear()
        logger.info("TildaProteusBridge: shutdown")

