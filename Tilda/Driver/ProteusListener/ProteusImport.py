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
