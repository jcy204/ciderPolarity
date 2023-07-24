from .CIDER import CIDER
import importlib.metadata

try:
    __version__ = importlib.metadata.version('ciderpolarity')
except importlib.metadata.PackageNotFoundError:
    __version__ = 'unknown-version' 
