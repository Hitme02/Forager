"""Import NASim headless.

NASim's `envs/render.py` does a top-level `import tkinter as Tk` for its optional GUI
renderer. On Python builds without Tk (e.g. this one) that import fails at load time, even
though we never render. All real `Tk.*` use is inside renderer methods we never call, so a
stub `tkinter` module is sufficient to let NASim import. We also pin Matplotlib to the Agg
backend so NASim's `matplotlib.use('TkAgg')` attempt stays a harmless warning.

Usage:
    from ffs.realism._nasim_compat import nasim      # imported & ready
"""

from __future__ import annotations

import sys
import types
import warnings

# --- stub tkinter if it is genuinely unavailable ---
if "tkinter" not in sys.modules:
    try:  # pragma: no cover - environment dependent
        import tkinter  # noqa: F401
    except Exception:
        sys.modules["tkinter"] = types.ModuleType("tkinter")

# --- keep Matplotlib headless so the TkAgg attempt in render.py is a no-op warning ---
try:  # pragma: no cover - environment dependent
    import matplotlib
    matplotlib.use("Agg")
except Exception:
    pass

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import nasim  # noqa: E402  (re-exported)

__all__ = ["nasim"]
