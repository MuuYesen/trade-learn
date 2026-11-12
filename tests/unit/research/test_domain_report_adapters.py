import pandas as pd

from tradelearn.data import DataExplorer
from tradelearn.research.explore import profile


def test_data_profile_is_rendered_by_common_reporter(tmp_path):
    frame = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
    explorer = DataExplorer(frame)
    result = explorer.to_analysis()
    assert "schema" in result["tables"]
    path = explorer.report(tmp_path / "data.xlsx")
    assert path.read_bytes()[:2] == b"PK"
    assert "numeric" in profile(frame).to_analysis()["tables"]


def test_profile_report_tables_keep_column_identity_as_explicit_columns():
    result = profile(pd.DataFrame({"alpha": [1.0, 2.0]})).to_analysis()
    assert "alpha" in result["tables"]["schema"]["column"].tolist()
    assert "alpha" in result["tables"]["numeric"]["column"].tolist()
