"""TL owns the optional dependency; standalone adapters never need TL."""
from pathlib import Path
import pytest
from packaging.requirements import Requirement

tomllib = pytest.importorskip('tomllib')


def extra_requirement():
    root = Path(__file__).resolve().parents[3]
    metadata = tomllib.loads((root / 'pyproject.toml').read_text())['project']
    assert not any(Requirement(r).name == 'tl-store' for r in metadata['dependencies'])
    requirements = metadata['optional-dependencies']['tl-store']
    assert len(requirements) == 1  # Future adapters stay independently selectable.
    requirement = Requirement(requirements[0])
    assert requirement.name == 'tl-store'
    assert str(requirement.specifier) == '==0.2.1a2'
    return requirement


@pytest.mark.parametrize('python,system,machine,expected', [
    ('3.13', 'darwin', 'arm64', True),
    ('3.13', 'linux', 'x86_64', True),
    ('3.14', 'darwin', 'arm64', False),
    ('3.12', 'linux', 'x86_64', False),
    ('3.13', 'win32', 'AMD64', False),
    ('3.13', 'linux', 'aarch64', False),
])
def test_extra_only_selects_qualified_candidate(python, system, machine, expected):
    requirement = extra_requirement()
    assert requirement.marker.evaluate({'python_version': python, 'sys_platform': system,
                                        'platform_machine': machine}) is expected
