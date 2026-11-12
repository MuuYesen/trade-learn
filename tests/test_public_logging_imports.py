"""Offline import boundary tests; exercise actual public facade; optional NA facade has an explicit isolated stub."""
import importlib, sys, types, unittest
from pathlib import Path
from unittest.mock import patch

HERE=Path(__file__).resolve();PACKAGE=HERE.parents[1]

class PublicImports(unittest.TestCase):
    def setUp(self):
        self.saved={k:v for k,v in sys.modules.items() if k=='tradelearn' or k.startswith(('tradelearn.','leafquant_monitor','nautilus_trader'))}
        for key in self.saved:sys.modules.pop(key,None)
        self.patch=patch.dict(sys.modules,{});self.patch.start()
        self.paths=patch.object(sys,'path',[str(PACKAGE),*sys.path]);self.paths.start()
    def tearDown(self):
        self.paths.stop();self.patch.stop()
        for key in list(sys.modules):
            if key=='tradelearn' or key.startswith(('tradelearn.','leafquant_monitor','nautilus_trader')):sys.modules.pop(key,None)
        sys.modules.update(self.saved)
    def test_normal_and_star_import_do_not_load_logger_or_nautilus(self):
        from tradelearn.engine import Strategy
        self.assertIsNotNone(Strategy)
        scope={};exec('from tradelearn.engine import *',scope)
        self.assertNotIn('TradeLogStrategy',scope)
        self.assertFalse(any(k.startswith(('leafquant_monitor','nautilus_trader')) for k in sys.modules))
    def test_explicit_tradelearn_entry_is_existing_class_and_stays_nautilus_free(self):
        from tradelearn.strategy_logging.tradelearn import TradeLogStrategy
        from leafquant_monitor.tradelearn import TradeLogStrategy as Actual
        self.assertIs(TradeLogStrategy,Actual)
        self.assertFalse(any(k.startswith('nautilus_trader') for k in sys.modules))
    def test_explicit_nautilus_entry_is_existing_class(self):
        trading=types.ModuleType('nautilus_trader.trading');trading.Strategy=type('Strategy',(),{})
        root=types.ModuleType('nautilus_trader');root.__path__=[]
        with patch.dict(sys.modules,{'nautilus_trader':root,'nautilus_trader.trading':trading}):
            from tradelearn.strategy_logging.nautilustrader import TradeLogStrategy
            from leafquant_monitor.strategy import TradeLogStrategy as Actual
            self.assertIs(TradeLogStrategy,Actual)
    def test_importing_framework_namespace_alone_remains_lazy(self):
        importlib.import_module('tradelearn.strategy_logging.nautilustrader')
        self.assertFalse(any(k.startswith(('leafquant_monitor','nautilus_trader')) for k in sys.modules))
    def test_nautilus_entry_fails_if_nautilus_is_unavailable(self):
        import importlib.abc
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self,fullname,path=None,target=None):
                if fullname=='nautilus_trader':raise ModuleNotFoundError('nautilus unavailable',name=fullname)
        with patch.object(sys,'meta_path',[Block(),*sys.meta_path]):
            from tradelearn.strategy_logging.tradelearn import TradeLogStrategy as TL
            with self.assertRaises(ModuleNotFoundError):
                from tradelearn.strategy_logging.nautilustrader import TradeLogStrategy
            self.assertIsNotNone(TL)
    def test_missing_optional_dependency_fails_explicitly_without_fallback(self):
        import importlib.abc
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self,fullname,path=None,target=None):
                if fullname=='leafquant_monitor':raise ModuleNotFoundError('optional monitor unavailable',name=fullname)
        with patch.object(sys,'meta_path',[Block(),*sys.meta_path]):
            from tradelearn.engine import Strategy
            with self.assertRaises(ModuleNotFoundError):
                from tradelearn.strategy_logging.tradelearn import TradeLogStrategy
    def test_unknown_attribute_is_not_fabricated(self):
        module=importlib.import_module('tradelearn.engine')
        with self.assertRaises(AttributeError):getattr(module,'NotARealStrategy')

if __name__=='__main__':unittest.main()
