"""
Helpers for importing the Proteus package from a local source checkout.
"""

from pathlib import Path
import sys


def ensure_proteus_on_path():
    """
    Add a local Proteus source checkout to ``sys.path`` if one is present.
    """
    repo_root = Path(__file__).resolve().parents[3]
    candidates = [
        repo_root / "proteus-1.0.8" / "src",
        repo_root / "proteus" / "src",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return candidate_str
    return None


def ensure_remote_instance_connected(instance, address: str):
    """
    Call ``instance.add_instance(address)`` but tolerate Proteus failures that
    happen only while recursively joining additional stale peers.

    In practice, a remote instance can reply successfully for ``address`` and
    then include stale ephemeral client addresses in its known-instance list.
    Proteus then raises from one of those follow-up joins even though the
    originally requested remote instance is already known locally.
    """
    address = (address or "").strip()
    if not address:
        return

    try:
        instance.add_instance(address)
        return
    except Exception as exc:
        known = set()
        try:
            if hasattr(instance, "known_instance_addresses"):
                known.update(instance.known_instance_addresses())
        except Exception:
            pass
        try:
            known.update(getattr(instance, "_known_instances", {}).keys())
        except Exception:
            pass

        if address in known:
            return
        raise exc
