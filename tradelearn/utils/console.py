import sys

def is_notebook():
    """Detect if we are running in a Jupyter notebook/IPython environment."""
    try:
        from IPython import get_ipython
        if 'IPKernelApp' in get_ipython().config:
            return True
    except Exception:
        pass
    return False

# Choose the appropriate base tqdm class
if is_notebook():
    from tqdm.notebook import tqdm as _tqdm
    DEFAULT_FILE = sys.stdout
else:
    from tqdm import tqdm as _tqdm
    DEFAULT_FILE = sys.stderr

class smart_tqdm(_tqdm):
    """A tqdm subclass that detects environment and ensures clean output."""

    def __init__(self, *args, **kwargs):
        # Default to stdout in Notebook to avoid pink stderr background
        kwargs.setdefault("file", DEFAULT_FILE)
        super().__init__(*args, **kwargs)

    def close(self):
        # In CLI, ensure fast bars are rendered; in Notebook, tqdm handles this better
        if not is_notebook():
            if self.n > 0 and getattr(self, "last_print_n", 0) == 0:
                self.display()
        super().close()

def smart_print(*args, **kwargs):
    """Prints to the default synchronized stream (stdout for Notebook, stderr for CLI)."""
    kwargs.setdefault("file", DEFAULT_FILE)
    print(*args, **kwargs)

def mark_console_clean():
    """No-op for backward compatibility."""
    pass
