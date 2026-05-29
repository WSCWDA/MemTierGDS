"""MemTier PyTorch UVM prototype package."""

try:
    from . import memtier_ext
except ImportError as exc:  # pragma: no cover - exercised before extension build.
    memtier_ext = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from .dataset import BaselineNumpyDataset, MemTierFileDataset


def require_extension():
    """Return the compiled extension or raise a clear setup/CUDA error."""
    if memtier_ext is None:
        raise RuntimeError(
            "memtier.memtier_ext is not available. Build the CUDA extension with "
            "`pip install -e memtier_pytorch` from the repository root, and ensure "
            "CUDA is installed and visible."
        ) from _IMPORT_ERROR
    return memtier_ext


__all__ = ["BaselineNumpyDataset", "MemTierFileDataset", "require_extension", "memtier_ext"]
