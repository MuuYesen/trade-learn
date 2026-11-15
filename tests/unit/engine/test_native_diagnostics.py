"""Offline native diagnostics contract, runnable without pytest."""
import unittest
from pathlib import Path
import tempfile
try:
    from nautilus_trader.common import LoggerConfig, LogLevel
except ModuleNotFoundError as error:
    if error.name != 'nautilus_trader':
        raise
    LoggerConfig = LogLevel = None


@unittest.skipIf(LoggerConfig is None, 'optional Nautilus dependency is not installed')
class NativeDiagnosticsTests(unittest.TestCase):
    def test_default_is_inert_info_and_bounded(self):
        from tradelearn_runtime.diagnostics import diagnostic_config
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / 'not-created'
            cfg = diagnostic_config(directory=str(directory))
            self.assertEqual(cfg.stdout_level, LogLevel.INFO)
            self.assertEqual(cfg.fileout_level, LogLevel.INFO)
            self.assertEqual(cfg.file_config.directory, str(directory))
            self.assertEqual(cfg.file_config.file_rotate, (10485760, 5))
            self.assertFalse(directory.exists())

    def test_native_overrides_and_false_preserve_console(self):
        from tradelearn_runtime.diagnostics import diagnostic_config
        explicit = LoggerConfig(stdout_level=LogLevel.DEBUG, bypass_logging=True)
        self.assertIs(diagnostic_config(explicit), explicit)
        self.assertIs(diagnostic_config(native=explicit), explicit)
        self.assertEqual(diagnostic_config(False).stdout_level, LogLevel.INFO)
        self.assertIsNone(diagnostic_config(False).file_config)

    def test_public_debug_and_rotation_validation(self):
        from tradelearn_runtime.diagnostics import diagnostic_config
        cfg = diagnostic_config(dict(stdout_level='DEBUG', fileout_level='DEBUG',
            directory='debug', max_bytes=2048, backup_count=2))
        self.assertEqual(cfg.stdout_level, LogLevel.DEBUG)
        self.assertEqual(cfg.file_config.file_rotate, (2048, 2))
        for options in ({'max_bytes': 0}, {'backup_count': -1}, {'backup_count': True}, {'oops': 1}):
            with self.subTest(options=options), self.assertRaises((TypeError, ValueError)):
                diagnostic_config(options)


    def test_official_store_options_are_inert_and_preserve_native_logger(self):
        from tradelearn.engine.stores import NautilusStore
        from nautilus_trader.live import LiveNodeConfig
        from nautilus_trader.model import AccountId
        from nautilus_trader.adapters.binance import (BinanceDataClientConfig,
            BinanceExecutionClientConfig, BinanceDataClientFactory, BinanceExecutionClientFactory)
        def store(options=None, native=None):
            config = LiveNodeConfig(logging=native,
                data_clients={'BINANCE': BinanceDataClientConfig()},
                exec_clients={'BINANCE': BinanceExecutionClientConfig(account_id=AccountId('BINANCE-001'))})
            return NautilusStore(config, data_factories={'BINANCE': BinanceDataClientFactory()},
                exec_factories={'BINANCE': BinanceExecutionClientFactory()}, logging=options)
        self.assertEqual(store().get_logging_config().file_config.file_rotate, (10485760, 5))
        explicit = LoggerConfig(stdout_level=LogLevel.DEBUG)
        self.assertIs(store(explicit).get_logging_config(), explicit)
        self.assertEqual(store(native=explicit).get_logging_config().stdout_level, LogLevel.DEBUG)
        self.assertIsNone(store(False).get_logging_config().file_config)
        self.assertEqual(store().get_logging_config(directory='/app/logs').file_config.directory, '/app/logs')

    def test_backtest_import_does_not_load_na_or_create_logs(self):
        import subprocess
        import sys
        import os
        with tempfile.TemporaryDirectory() as folder:
            code = """import importlib.abc, sys
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'nautilus_trader' or fullname.startswith('nautilus_trader.'):
            raise AssertionError('backtest imported optional NA')
sys.meta_path.insert(0, NoNative())
from tradelearn import engine as bt
engine = bt.Cerebro()
from pathlib import Path
assert not Path('logs').exists()
assert not any(n.startswith('nautilus_trader') for n in sys.modules)
"""
            env = dict(os.environ, PYTHONPATH=os.environ.get(
                'TRADELEARN_TEST_PACKAGE_PATH', str(Path(__file__).resolve().parents[3])))
            result = subprocess.run([sys.executable, '-c', code], cwd=folder,
                env=env, text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_real_official_builder_rotates_without_connecting(self):
        # Separate process: the official NA logger is intentionally process-global.
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as folder:
            code = r'''from tradelearn.engine.stores import NautilusStore
from nautilus_trader.common import Environment, logger_flush
from nautilus_trader.live import LiveNodeConfig
from nautilus_trader.model import TraderId, AccountId
from nautilus_trader.adapters.binance import (BinanceDataClientConfig,
    BinanceExecutionClientConfig, BinanceDataClientFactory, BinanceExecutionClientFactory,
    BinanceSpotMarketDataMode)
import sys, time
from pathlib import Path
folder = sys.argv[1]
config = LiveNodeConfig(trader_id=TraderId('TL-001'), environment=Environment.SANDBOX,
    data_clients={'BINANCE': BinanceDataClientConfig(api_key='offline-placeholder',
        api_secret='offline-placeholder', spot_market_data_mode=BinanceSpotMarketDataMode.Json)},
    exec_clients={'BINANCE': BinanceExecutionClientConfig(account_id=AccountId('BINANCE-001'),
        api_key='offline-placeholder', api_secret='offline-placeholder')})
store = NautilusStore(config, data_factories={'BINANCE': BinanceDataClientFactory()},
    exec_factories={'BINANCE': BinanceExecutionClientFactory()},
    logging={'directory': folder, 'max_bytes': 4096, 'backup_count': 2})
assert not list(Path(folder).iterdir())
node = store._build_node()
node.dispose()
logger_flush()
time.sleep(0.05)
files = list(Path(folder).glob('*.log'))
assert len(files) == 3, len(files)
assert all(f.stat().st_size <= 4096 for f in files)
print('OFFLINE_ROTATION_OK')
'''
            result = subprocess.run([sys.executable, '-c', code, folder],
                text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('OFFLINE_ROTATION_OK', result.stdout)


if __name__ == '__main__':
    unittest.main()
