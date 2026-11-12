import sys

import pytest

from tradelearn.core.config import TradelearnConfig
from tradelearn.lab import build_lab_plan, start_lab_stack


def test_lab_plan_has_only_jupyter_and_preserves_explicit_tracking(tmp_path):
    config = TradelearnConfig(mlflow_tracking_uri="sqlite:///chosen.db")
    plan = build_lab_plan(config, notebook_dir=tmp_path, no_browser=True)
    assert plan.jupyter.args[:3] == (sys.executable, "-m", "jupyterlab")
    assert not hasattr(plan, "mcp") and not hasattr(plan, "mlflow")
    assert plan.jupyter.env["MLFLOW_TRACKING_URI"] == "sqlite:///chosen.db"
    with pytest.raises(TypeError):
        build_lab_plan(config, start_mlflow=True)


def test_lab_keyboard_interrupt_closes_and_reaps_only_owned_jupyter():
    calls = []

    class Process:
        def __init__(self):
            self.alive = True

        def poll(self):
            return None if self.alive else -15

        def wait(self, timeout=None):
            if timeout is None:
                raise KeyboardInterrupt
            self.alive = False
            calls.append("wait")
            return -15

        def terminate(self):
            calls.append("terminate")

    def popen(args, *, env):
        calls.append(tuple(args[:3]))
        return Process()

    assert start_lab_stack(build_lab_plan(TradelearnConfig()), popen=popen) == 0
    assert calls == [(sys.executable, "-m", "jupyterlab"), "terminate", "wait"]


def test_lab_launch_failure_does_not_start_fallback_services():
    calls = []

    def popen(args, **kwargs):
        calls.append(args)
        raise OSError("cannot launch Jupyter")

    with pytest.raises(OSError, match="cannot launch"):
        start_lab_stack(build_lab_plan(TradelearnConfig()), popen=popen)
    assert len(calls) == 1


def test_lab_dependency_discovery_is_explicit():
    from tradelearn.lab import check_lab_dependencies, required_lab_packages

    assert required_lab_packages() == ("jupyterlab", "ipywidgets", "jupyter-ai-lite")
    assert check_lab_dependencies(find_spec=lambda name: None) == required_lab_packages()


def test_lab_sigterm_closes_owned_child_and_restores_handler():
    import signal

    previous = signal.getsignal(signal.SIGTERM)
    calls = []

    class Process:
        alive = True

        def poll(self):
            return None if self.alive else -15

        def wait(self, timeout=None):
            if timeout is None:
                handler = signal.getsignal(signal.SIGTERM)
                assert callable(handler), 'Lab must handle SIGTERM before waiting'
                handler(signal.SIGTERM, None)
            self.alive = False
            calls.append('reaped')
            return -15

        def terminate(self):
            calls.append('terminated')

    assert start_lab_stack(build_lab_plan(TradelearnConfig()), popen=lambda *a, **k: Process()) == 143
    assert calls == ['terminated', 'reaped']
    assert signal.getsignal(signal.SIGTERM) is previous
