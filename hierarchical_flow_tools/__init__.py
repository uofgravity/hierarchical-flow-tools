"""
hierarchical-flow-tools is a lightweight package for easily performing fully vectorised hierarchical inference with pre-trained normalising flows.
"""
from .likelihood import FlowLikelihood

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"