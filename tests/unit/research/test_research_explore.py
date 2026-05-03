from __future__ import annotations

import pandas as pd

import tradelearn.research.explore as ex
from tradelearn.research import ResearchRun


def test_profile_summarizes_raw_dataframe_and_records_step() -> None:
    data = pd.DataFrame(
        {
            "close": [10.0, 11.0, None, 13.0],
            "volume": [100.0, 0.0, 120.0, 130.0],
            "sector": ["tech", "tech", None, "finance"],
        }
    )

    with ResearchRun("explore-demo") as run:
        profile = ex.profile(data)
        result = run.finish(artifacts={"profile": profile.to_dict()})

    assert profile.rows == 4
    assert profile.columns == 3
    assert profile.missing.loc["close"] == 1
    assert profile.missing_rate.loc["sector"] == 0.25
    assert profile.numeric.loc["close", "mean"] == 34 / 3
    assert profile.to_dict()["shape"] == {"rows": 4, "columns": 3}
    assert result.steps[0].name == "profile"
    assert result.steps[0].category == "explore"


def test_report_writes_html_profile(tmp_path) -> None:
    data = pd.DataFrame({"close": [10.0, 11.0], "volume": [100.0, 110.0]})

    output = ex.report(data, tmp_path / "explore.html")

    assert output == tmp_path / "explore.html"
    html = output.read_text()
    assert "TradeLearn Data Profile" in html
    assert "close" in html
    assert "volume" in html
