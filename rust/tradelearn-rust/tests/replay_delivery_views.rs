use _rust::engine::BacktestEngine;
use _rust::types::{BarEvent,OrderSide,OrderType};
use std::collections::HashMap;
fn engine()->BacktestEngine {
 BacktestEngine::new(vec![],vec![],vec![],vec![],vec![],vec![],1000.,0.,false,false,false,0.,0.,true,true,false,1.,1.,false,false,0.,0.,0.,0.)
}
fn bars()->Vec<BarEvent>{vec![BarEvent{symbol:"A".into(),ts:100,open:20.,high:21.,low:19.,close:21.,volume:10.}]}
fn quotes()->Vec<BarEvent>{vec![BarEvent{symbol:"A".into(),ts:100,open:30.,high:31.,low:29.,close:31.,volume:10.}]}
fn view(turn:u64,clock:i64)->HashMap<String,(u64,i64)>{HashMap::from([("A".into(),(turn,clock))])}
#[test]
fn replay_view_market_gate_uses_bt_date_while_fill_keeps_original_quote_timestamp(){
 let mut e=engine();let id=e.submit_order("A".into(),OrderSide::Buy,OrderType::Market,0.25,None,None,None,None);
 e.configure_market_eligibility(id,10);
 assert!(e.step_bars_with_quote_views(bars(),&quotes(),&view(1,10),false).unwrap().is_empty());
 let fills=e.step_bars_with_quote_views(bars(),&quotes(),&view(2,20),false).unwrap();
 assert_eq!(fills.len(),1);assert_eq!(fills[0].ts,100);assert_eq!(fills[0].price,30.);assert_eq!(e.get_cash(),992.5);
}
#[test]
fn replay_view_exact_repeat_does_not_reconsume_quote_or_cash_flow(){
 let mut e=engine();e.step_bars_with_quote_views(bars(),&quotes(),&view(1,10),false).unwrap();
 e.apply_cash_controls(vec![(1,"add".into(),100.)]).unwrap();
 let id=e.submit_order("A".into(),OrderSide::Buy,OrderType::Market,0.25,None,None,None,None);e.configure_market_eligibility(id,0);
 assert!(e.step_bars_with_quote_views(bars(),&quotes(),&view(1,10),false).unwrap().is_empty());
 assert_eq!(e.get_cash(),1000.);assert!(e.drain_cash_events().unwrap().is_empty());
 assert_eq!(e.step_bars_with_quote_views(bars(),&quotes(),&view(2,20),false).unwrap().len(),1);
 assert_eq!(e.get_cash(),1092.5);
}
#[test]
fn replay_view_changed_same_turn_stale_turn_or_foreign_is_atomic(){
 let mut e=engine();e.step_bars_with_quote_views(bars(),&quotes(),&view(1,10),false).unwrap();
 e.apply_cash_controls(vec![(1,"add".into(),100.)]).unwrap();
 let mut changed=quotes();changed[0].close=32.;
 assert!(e.step_bars_with_quote_views(bars(),&changed,&view(1,10),false).is_err());
 assert!(e.step_bars_with_quote_views(bars(),&quotes(),&view(0,10),false).is_err());
 assert!(e.step_bars_with_quote_views(bars(),&quotes(),&view(1,9),false).is_err());
 assert!(e.step_bars_with_quote_views(bars(),&quotes(),&HashMap::from([("foreign".into(),(2,20))]),false).is_err());
 assert_eq!(e.get_cash(),1000.);assert!(e.drain_cash_events().unwrap().is_empty());
}
#[test]
fn replay_view_target_expiry_uses_bt_date_not_quote_source_date(){
 let mut e=engine();let id=e.submit_order("A".into(),OrderSide::Buy,OrderType::Limit,0.25,Some(1.),None,None,None);
 e.configure_order(id,Some(15),None);e.configure_order_expiry_clock(id,true);
 e.step_bars_with_quote_views(bars(),&quotes(),&view(1,10),false).unwrap();assert!(e.drain_order_events().is_empty());
 e.step_bars_with_quote_views(bars(),&quotes(),&view(2,20),false).unwrap();assert_eq!(e.drain_order_events(),vec![(id,"expired".into())]);
}
#[test]
fn replay_coc_uses_captured_logical_close_and_keeps_source_time(){
 let mut e=engine();e.step_bars_with_quote_views(bars(),&quotes(),&view(1,10),false).unwrap();
 let id=e.submit_order("A".into(),OrderSide::Buy,OrderType::Market,0.25,None,None,None,None);
 e.configure_market_close_reference(id,21.).unwrap();
 let fills=e.step_bars_with_quote_views(bars(),&quotes(),&view(1,10),true).unwrap();
 assert_eq!(fills[0].price,21.);assert_eq!(fills[0].ts,100);assert_eq!(e.get_cash(),994.75);
 assert_eq!(e.get_equity(),1000.);
}
#[test]
fn replay_coc_reference_rejects_foreign_nonmarket_and_nonfinite_atomically(){
 let mut e=engine();let id=e.submit_order("A".into(),OrderSide::Buy,OrderType::Limit,0.25,Some(1.),None,None,None);
 assert!(e.configure_market_close_reference(id,21.).is_err());
 assert!(e.configure_market_close_reference(987,21.).is_err());
 let market=e.submit_order("A".into(),OrderSide::Buy,OrderType::Market,0.25,None,None,None,None);
 assert!(e.configure_market_close_reference(market,f64::NAN).is_err());
 assert_eq!(e.get_cash(),1000.);
}
#[test]
fn bt_coc_applies_captured_close_at_next_broker_turn_even_equal_bt_date(){
 let mut e=engine();e.step_bars_with_quote_views(bars(),&quotes(),&view(1,10),false).unwrap();
 let id=e.submit_order("A".into(),OrderSide::Buy,OrderType::Market,0.25,None,None,None,None);
 e.configure_market_eligibility(id,10);e.configure_market_close_reference(id,21.).unwrap();
 assert_eq!(e.get_cash(),1000.);
 let fills=e.step_bars_with_quote_views(bars(),&quotes(),&view(2,10),false).unwrap();
 assert_eq!(fills.len(),1);assert_eq!(fills[0].price,21.);assert_eq!(e.get_equity(),1000.);
}

#[test]
fn clone_withdrawn_public_date_is_not_the_delivery_identity_or_market_admission(){
 let mut e=engine();e.step_bars_with_quote_views(bars(),&quotes(),&view(1,10),false).unwrap();
 let id=e.submit_order("A".into(),OrderSide::Buy,OrderType::Market,0.25,None,None,None,None);e.configure_market_eligibility(id,10);
 assert!(e.step_bars_with_quote_views(bars(),&quotes(),&view(2,9),false).unwrap().is_empty());
 assert_eq!(e.get_cash(),1000.);
 assert_eq!(e.step_bars_with_quote_views(bars(),&quotes(),&view(3,20),false).unwrap().len(),1);
 assert_eq!(e.get_cash(),992.5);
}
