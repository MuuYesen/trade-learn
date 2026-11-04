"""One artifact transport implementation supports both frontend lifecycles."""

from pathlib import Path
from types import SimpleNamespace


def test_shared_upload_preserves_bundle_options_and_directory_fallback(monkeypatch):
    import tradelearn.report.artifacts as artifacts
    import tradelearn.report.mlflow as module

    calls = []

    def write(stats, directory, **kwargs):
        assert stats == "stats"
        assert kwargs["strategy"] == "strategy"
        assert kwargs["market_data"] == "market"
        assert kwargs["log_report"] is False and kwargs["log_plot"] is True
        direct = directory / "report.csv"
        direct.write_text("a")
        folder = directory / "tables"
        folder.mkdir()
        (folder / "b.csv").write_text("b")
        (folder / "a.csv").write_text("a")
        return [direct, folder]

    monkeypatch.setattr(artifacts, "write_artifact_bundle", write)
    monkeypatch.setattr(artifacts, "market_data_from_strategy", lambda strategy: "market")
    client = SimpleNamespace(
        log_artifact=lambda path, artifact_path=None: calls.append((Path(path).name, artifact_path))
    )
    module.log_artifact_bundle(
        client,
        stats="stats",
        strategy="strategy",
        artifact_path="run",
        log_report=False,
        log_plot=True,
    )
    assert calls == [("report.csv", "run"), ("a.csv", "run/tables"), ("b.csv", "run/tables")]


def test_shared_upload_without_artifact_capability_does_not_build_files(monkeypatch):
    import tradelearn.report.mlflow as module

    monkeypatch.setattr(
        module,
        "write_artifact_bundle",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not render")),
        raising=False,
    )
    module.log_artifact_bundle(
        object(), stats=None, strategy=None, artifact_path=None, log_report=True, log_plot=False
    )
