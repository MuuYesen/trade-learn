import sys

def is_notebook():
    """Detect if we are running in a Jupyter notebook/IPython environment."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        # Check if we are in a kernel environment
        if 'IPKernelApp' in shell.config:
            return True
        # Handle some cases where shell exists but isn't a full notebook
        if "google.colab" in sys.modules:
            return True
    except Exception:
        pass
    return False

# Choose the appropriate base tqdm class with safety fallback
_tqdm_class = None
DEFAULT_FILE = sys.stderr

if is_notebook():
    try:
        from tqdm.notebook import tqdm as _notebook_tqdm
        # Test if it can be initialized without error (some envs have the module but fail at runtime)
        _tqdm_class = _notebook_tqdm
        DEFAULT_FILE = sys.stdout
    except Exception:
        pass

if _tqdm_class is None:
    from tqdm import tqdm as _cli_tqdm
    _tqdm_class = _cli_tqdm
    DEFAULT_FILE = sys.stderr

class smart_tqdm(_tqdm_class):
    """A tqdm subclass that detects environment and ensures clean output."""

    def __init__(self, *args, **kwargs):
        # Default to stdout in Notebook to avoid pink stderr background
        kwargs.setdefault("file", DEFAULT_FILE)
        
        # In some weird Jupyter envs, tqdm.notebook might still fail inside __init__
        try:
            super().__init__(*args, **kwargs)
        except Exception:
            # Emergency fallback to base tqdm if initialization fails
            from tqdm import tqdm as _base_tqdm
            self.__class__ = _base_tqdm
            super().__init__(*args, **kwargs)

    def close(self):
        # In CLI, ensure fast bars are rendered; in Notebook, tqdm handles this better
        if not is_notebook():
            if hasattr(self, 'n') and self.n > 0 and getattr(self, "last_print_n", 0) == 0:
                try:
                    self.display()
                except Exception:
                    pass
        super().close()

def smart_print(*args, **kwargs):
    """Prints to the default synchronized stream (stdout for Notebook, stderr for CLI)."""
    kwargs.setdefault("file", DEFAULT_FILE)
    print(*args, **kwargs)

def mark_console_clean():
    """No-op for backward compatibility."""
    pass
