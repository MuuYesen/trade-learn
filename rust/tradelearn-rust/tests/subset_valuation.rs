use _rust::engine::BacktestEngine;
use _rust::types::{BarEvent,OrderSide,OrderType};
fn engine()->BacktestEngine {
 BacktestEngine::new(vec![],vec![],vec![],vec![],vec![],vec![],10000.,0.,false,false,false,0.,0.,true,true,false,1.,1.,false,false,0.,0.,0.,0.)
}
fn bar(symbol:&str,price:f64)->BarEvent{BarEvent{symbol:symbol.into(),ts:1,open:price,high:price,low:price,close:price,volume:10.}}
#[test]
fn subset_uses_native_marks_and_single_raw_value_multi_cash_empty_all_and_duplicates(){
 let mut e=engine();e.submit_order("A".into(),OrderSide::Buy,OrderType::Market,2.,None,None,None,None);
 e.submit_order("B".into(),OrderSide::Sell,OrderType::Market,0.5,None,None,None,None);
 e.step_open_bars(vec![bar("A",10.),bar("B",100.)]);
 assert_eq!(e.observe_valuation(Some(&["A".into()]),false,false).unwrap(),(10030.,20.,20.));
 assert_eq!(e.observe_valuation(Some(&["B".into()]),false,false).unwrap(),(10030.,-50.,-50.));
 assert_eq!(e.observe_valuation(Some(&["A".into(),"A".into()]),true,false).unwrap(),(10030.,10070.,40.));
 assert_eq!(e.observe_valuation(None,true,false).unwrap(),(10030.,10000.,-30.));
 assert_eq!(e.observe_valuation(Some(&["EMPTY".into()]),false,false).unwrap(),(10030.,0.,0.));
}
#[test]
fn explicit_valuation_settles_only_existing_queued_cash_once_with_original_ack(){
 let mut e=engine();e.apply_cash_controls(vec![(1,"add".into(),500.)]).unwrap();
 assert_eq!(e.observe_valuation(None,true,false).unwrap().0,10000.);
 assert!(e.drain_cash_events().unwrap().is_empty());
 assert_eq!(e.observe_valuation(None,true,true).unwrap().0,10500.);
 assert_eq!(e.drain_cash_events().unwrap(),vec![(1,"add".into(),500.,10500.)]);
 assert_eq!(e.observe_valuation(None,true,true).unwrap().0,10500.);
 assert!(e.drain_cash_events().unwrap().is_empty());
}
#[test]
fn nonfinite_selected_mark_does_not_settle_cash_or_return_a_zero_value(){
 let mut e=engine();e.submit_order("A".into(),OrderSide::Buy,OrderType::Market,1.,None,None,None,None);
 e.step_open_bars(vec![bar("A",10.)]);
 let mut unknown=bar("A",10.);unknown.ts=2;unknown.close=f64::NAN;
 e.step_open_bars(vec![unknown]);
 e.apply_cash_controls(vec![(1,"add".into(),500.)]).unwrap();
 assert!(e.observe_valuation(Some(&["A".into()]),true,true).is_err());
 assert_eq!(e.get_cash(),9990.);assert!(e.drain_cash_events().unwrap().is_empty());
}
#[test]
fn columnar_native_open_marks_the_current_bar_before_business_valuation(){
 let mut e=BacktestEngine::new(vec![1,2,3],vec![10.,20.,30.],vec![10.,20.,30.],vec![10.,20.,30.],vec![10.,20.,30.],vec![10.,10.,10.],10000.,0.,false,false,false,0.,0.,true,true,false,1.,1.,false,false,0.,0.,0.,0.);
 e.step_open(0);e.submit_order("data0".into(),OrderSide::Buy,OrderType::Market,2.,None,None,None,None);
 e.step_open(1);e.step_open(2);
 assert_eq!(e.observe_valuation(Some(&["data0".into()]),false,false).unwrap(),(9960.,60.,60.));
 assert_eq!(e.get_equity(),10020.);
}
