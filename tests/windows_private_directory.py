"""Establish production privacy prerequisites on synthetic test directories only."""
import os
from pathlib import Path
import subprocess


def make_private_test_directory(directory):
    if os.name != 'nt':
        return
    from leafquant_monitor.filesystem import _token_sids, check_private_directory

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    user = _token_sids()[0]
    commands = [
        ['icacls', str(directory), '/reset'],
        ['icacls', str(directory), '/inheritance:r', '/grant:r',
         f'*{user}:(OI)(CI)F', '*S-1-5-18:(OI)(CI)F', '*S-1-5-32-544:(OI)(CI)F'],
    ]
    for command in commands:
        result = subprocess.run(command, capture_output=True, timeout=20)
        assert result.returncode == 0, 'synthetic test directory requires a private DACL'
    check_private_directory(directory)
