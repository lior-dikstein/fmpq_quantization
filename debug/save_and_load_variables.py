# debug_tools.py

"""
Utilities to save and load variable state at breakpoints,
storing files under ./debug/debug_variables/<name>.pkl
"""

import pickle
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional

# determine project root as parent of the folder containing this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "debug" / "debug_variables"

def save_debug_state(name: str,
                     base_dir: str = BASE_DIR,
                     frame: Optional[Any] = None,
                     var_names: Optional[List[str]] = None) -> None:
    """
    Save variables from the given frame to debug/debug_variables/<name>.pkl.

    Args:
        name:     Identifier for this snapshot (no extension).
        base_dir: Base directory to save to.
        frame:    Frame to pull variables from; defaults to caller’s frame.
        var_names:List of variable names to save; saves all locals if None.
    """
    base_dir = Path(base_dir)
    if frame is None:
        frame = inspect.currentframe().f_back
    locals_dict = frame.f_locals
    data = ({n: locals_dict[n] for n in var_names if n in locals_dict}
            if var_names else
            locals_dict.copy())

    # ensure directory exists
    base_dir.mkdir(parents=True, exist_ok=True)
    filepath = base_dir / f"{name}.pkl"
    with filepath.open("wb") as f:
        pickle.dump(data, f)


def load_debug_state(name: str,
                     base_dir: str = BASE_DIR,
                     frame: Optional[Any] = None) -> Dict[str, Any]:
    """
    Load and inject previously saved debug state from debug/debug_variables/<name>.pkl
    into the caller’s global namespace.

    Args:
        name:  Identifier used when saving (no extension).
        base_dir: Base directory to load from.
        frame: Frame into which to inject; defaults to caller’s frame.

    Returns:
        dict mapping variable names to their saved values.
    """
    filepath = base_dir / f"{name}.pkl"
    with filepath.open("rb") as f:
        data = pickle.load(f)

    if frame is None:
        frame = inspect.currentframe().f_back
    # inject into globals of the caller
    frame.f_globals.update(data)
    return data