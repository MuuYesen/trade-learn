import sys
from tqdm import tqdm as _tqdm

class smart_tqdm(_tqdm):
    """A tqdm subclass that ensures fast bars are visible but keeps output compact."""

    def __init__(self, *args, **kwargs):
        # Force stderr for better stream synchronization
        kwargs.setdefault("file", sys.stderr)
        super().__init__(*args, **kwargs)

    def close(self):
        # Ensure bars that completed instantly are rendered before closing
        if self.n > 0 and getattr(self, "last_print_n", 0) == 0:
            self.display()
        super().close()

def smart_print(*args, **kwargs):
    """Simple wrapper to print to stderr by default."""
    kwargs.setdefault("file", sys.stderr)
    print(*args, **kwargs)

def mark_console_clean():
    """No-op for backward compatibility."""
    pass
