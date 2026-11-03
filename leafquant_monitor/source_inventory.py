"""Explicit code-only closure for transport roles which do not create a recorder."""
from pathlib import Path

TRANSPORT_MODULES = ('__init__.py', 'codec.py', 'sqs_transport.py', 'source_file_contract.py',
                     'source_file_sender.py', 'archive_receipt_client.py', 'filesystem.py')


def transport_sources():
    root = Path(__file__).resolve().parent
    result = {'leafquant_monitor/' + name: root / name for name in TRANSPORT_MODULES}
    if any(path.is_symlink() or not path.is_file() for path in result.values()):
        raise ValueError('regular public transport source closure required')
    return result
