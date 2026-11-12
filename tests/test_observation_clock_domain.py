import unittest,time
class ClockDomain(unittest.TestCase):
 def test_one_real_clock_for_TL_and_standard_logger(self):
  from tradelearn.core.observation_clock import CLOCK_DOMAIN,monotonic_ns
  from leafquant_monitor.tl_timing import TLBoundaryTiming
  self.assertEqual(TLBoundaryTiming(None).clock_domain,CLOCK_DOMAIN)
  if hasattr(time,'CLOCK_MONOTONIC'):
   before=time.clock_gettime_ns(time.CLOCK_MONOTONIC);point=monotonic_ns();after=time.clock_gettime_ns(time.CLOCK_MONOTONIC)
   self.assertEqual(CLOCK_DOMAIN,'POSIX_MONOTONIC');self.assertLessEqual(before,point);self.assertLessEqual(point,after)
 def test_official_na_logger_uses_same_clock_domain(self):
  import pytest
  pytest.importorskip('nautilus_trader', reason='official NA boundary requires the explicit nautilus profile')
  from tradelearn.core.observation_clock import CLOCK_DOMAIN
  from leafquant_monitor.strategy import TradeLogStrategy
  self.assertEqual(TradeLogStrategy.monitor_clock_domain,CLOCK_DOMAIN)
if __name__=='__main__':unittest.main()
