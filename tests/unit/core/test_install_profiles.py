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
    assert set(extras)=={'research','nautilus','tl-store'}
    assert {'dev','docs','ci','oracle'} <= set(data['dependency-groups'])
    assert {'Riskfolio-Lib','optuna','numba>=0.65.1','duckdb','gplearn'} <= set(extras['research'])
    assert 'mlflow' in extras['research']
    assert 'jupyter-ai-lite==0.21.0' in extras['research']
    assert not {'jupyterlab-git','pygwalker','mcp','jupyter-ai'} & set(extras['research'])
    assert {'boto3==1.41.2','botocore==1.41.6'} <= set(extras['nautilus'])


def test_base_ci_keeps_selected_profiles_and_mandatory_lock_gate():
    data=tomllib.loads((ROOT/'pyproject.toml').read_text())
    ci=data['dependency-groups']['ci']
    assert {'boto3==1.41.2','botocore==1.41.6','numba>=0.65.1'} <= set(ci)
    # These three libraries have real component-workflow consumers in pytest.
    # Keep them in the selected CI group; never add research/UI/execution to base.
    assert {'gplearn', 'Riskfolio-Lib', 'optuna'} <= set(ci)
    assert not any(name.startswith(('nautilus', 'jupyter')) for name in ci)
    base = data['project']['dependencies']
    assert not any(name.startswith(('Riskfolio', 'optuna', 'gplearn', 'jupyter',
                                    'nautilus', 'mlflow', 'boto3')) for name in base)
    workflow=(ROOT/'.github/workflows/ci.yml').read_text()
    assert '--extra' not in workflow
    assert 'uv pip install -e . --group dev --group ci --group oracle' in workflow
    assert 'run: uv lock --check' in workflow
    assert 'official-na-boundary:' in workflow
    assert "uv pip install -e '.[nautilus]'" in workflow
