"""Backward-compatibility shim.

``DataSet`` moved to :mod:`nems.tools.dataset` (it is a plain data container,
not a model, and living here created an import cycle between
``nems.preprocessing``, ``nems.models`` and ``nems.visualization``). Import
from ``nems.tools.dataset`` going forward.
"""

from nems.tools.dataset import DataSet  # noqa: F401
