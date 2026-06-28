from __future__ import annotations

import subprocess
import sys
import textwrap


def test_engine_import_does_not_require_pynecore() -> None:
    code = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class BlockPynecore(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "pynecore" or fullname.startswith("pynecore."):
                    raise ModuleNotFoundError("No module named 'pynecore'")
                return None

        sys.meta_path.insert(0, BlockPynecore())

        import tradelearn.engine as bt

        assert bt.Cerebro is not None
        assert "tv" in bt.__all__
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True)
