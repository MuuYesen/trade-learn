"""Public installation profiles retain capabilities without legacy entry aliases."""
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT=Path(__file__).resolve().parents[3]


def test_three_public_profiles_and_explicit_development_groups():
    data=tomllib.loads((ROOT/'pyproject.toml').read_text())
    extras=data['project']['optional-dependencies']
    assert set(extras)=={'research','lab','nautilus'}
    assert not set(extras['research']) & set(extras['lab'])
    assert {'dev','docs','ci','oracle'} <= set(data['dependency-groups'])
    assert {'Riskfolio-Lib','optuna','numba>=0.65.1','duckdb','TA-Lib'} <= set(extras['research'])
    assert 'mlflow' in extras['lab']
    assert {'boto3==1.41.2','botocore==1.41.6'} <= set(extras['nautilus'])


def test_ci_does_not_install_nautilus_or_new_research_optimizers():
    data=tomllib.loads((ROOT/'pyproject.toml').read_text())
    ci=data['dependency-groups']['ci']
    assert {'boto3==1.41.2','botocore==1.41.6','numba>=0.65.1'} <= set(ci)
    assert not any(name.startswith(('nautilus','Riskfolio','optuna')) for name in ci)
    workflow=(ROOT/'.github/workflows/ci.yml').read_text()
    assert '--extra' not in workflow
    assert 'uv sync --frozen --group dev --group ci --group oracle' in workflow
