"""Public installation profiles retain capabilities without legacy entry aliases."""
from pathlib import Path

from packaging.requirements import Requirement

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
    assert {'Riskfolio-Lib','optuna','numba>=0.65.1','duckdb'} <= set(extras['research'])
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
    assert {'Riskfolio-Lib', 'optuna'} <= set(ci)
    assert 'gplearn' in {Requirement(value).name for value in ci}
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


def test_symbolic_versions_match_the_real_python_boundary():
    data = tomllib.loads((ROOT/'pyproject.toml').read_text())
    for requirements in (data['project']['optional-dependencies']['research'],
                         data['dependency-groups']['ci']):
        symbolic = [Requirement(value) for value in requirements
                    if Requirement(value).name == 'gplearn']
        for python, expected in [('3.10', '==0.4.2'), ('3.11', '==0.4.3'), ('3.14', '==0.4.3')]:
            selected = [str(value.specifier) for value in symbolic
                        if value.marker.evaluate({'python_version': python})]
            assert selected == [expected]
