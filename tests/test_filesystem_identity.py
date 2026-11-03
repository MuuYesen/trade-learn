"""File identity storage must not round unsigned Windows IDs in SQLite."""
import sqlite3
import pytest
from leafquant_monitor.filesystem import database_identity, original_identity


def test_unsigned_identities_round_trip_existing_integer_affinity_exactly():
    db=sqlite3.connect(':memory:');db.execute('CREATE TABLE ids(value INTEGER NOT NULL)')
    values=[0,123,2**63-1,2**63,2**64-1,2**128-1]
    for value in values:db.execute('INSERT INTO ids VALUES(?)',(database_identity(value),))
    rows=db.execute('SELECT value,typeof(value) FROM ids').fetchall()
    assert [original_identity(r[0]) for r in rows]==values
    assert [r[1] for r in rows]==['integer']*3+['text']*3
    for value in [-1,'u:01','u:-1','9223372036854775808',1.0]:
        with pytest.raises(ValueError):original_identity(value)
