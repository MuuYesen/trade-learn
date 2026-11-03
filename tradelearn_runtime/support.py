"""Discover bundled official adapters without importing them or opening connections.

Discovery is an inventory, never a claim of broker, paper or live acceptance.
"""
import argparse
from importlib import metadata, resources
import json
from pathlib import PurePosixPath

NAUTILUS_VERSION = '2.0.0rc7'


def adapter_inventory(distribution=None):
    """Return installed rc7 adapter directories; import no broker modules."""
    distribution = distribution or metadata.distribution('nautilus_trader')
    if distribution.version != NAUTILUS_VERSION:
        raise ValueError('adapter inventory requires verified Nautilus ' + NAUTILUS_VERSION)
    adapters = set()
    for entry in distribution.files or ():
        parts = PurePosixPath(str(entry)).parts
        if len(parts) >= 4 and parts[:2] == ('nautilus_trader', 'adapters'):
            name = parts[2]
            if name.isidentifier() and not name.startswith('_') and parts[3] == '__init__.py':
                adapters.add(name)
    if not adapters:
        raise ValueError('installed distribution has no bundled adapter inventory')
    pinned = json.loads(resources.files(__package__).joinpath('official_adapters.json').read_text())
    declarations = {row['package']: row for row in pinned['adapters']}
    if pinned['nautilus_version'] != distribution.version or adapters - declarations.keys():
        raise ValueError('installed adapter names differ from pinned official inventory; verify distribution provenance')
    return {
        'official_interfaces': {name: declarations[name] for name in sorted(adapters)},
        'provenance': 'inventory comparison is not binary authentication',
        'nautilus_version' : distribution.version,
        'scope': 'installed_official_adapter_discovery_only',
        'adapters': sorted(adapters),
        'acceptance': 'not evidence of TL bridge, paper, live, order or platform acceptance',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        result = adapter_inventory()
    except metadata.PackageNotFoundError:
        parser.exit(2, 'Nautilus is optional and not installed; install the verified nautilus extra.\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
