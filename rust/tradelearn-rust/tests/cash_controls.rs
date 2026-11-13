use _rust::engine::BacktestEngine;
use _rust::types::{BarEvent, OrderSide, OrderType};

fn engine(cash:f64) -> BacktestEngine {
    BacktestEngine::new(vec![],vec![],vec![],vec![],vec![],vec![],cash,0.,false,false,false,0.,0.,true,true,false,1.,1.,false,false,0.,0.,0.,0.)
}
fn bar(ts:i64) -> Vec<BarEvent> {
    vec![BarEvent{symbol:"A".into(),ts,open:10.,high:10.,low:10.,close:10.,volume:100.}]
}
#[test]
fn cash_addition_is_applied_after_matching_and_never_funds_that_match() {
    let mut e=engine(100.);
    e.apply_cash_controls(vec![(1,"add".into(),500.)]).unwrap();
    assert_eq!(e.get_cash(),100.);assert!(e.drain_cash_events().unwrap().is_empty());
    e.submit_order("A".into(),OrderSide::Buy,OrderType::Market,20.,None,None,None,None);
    assert!(e.step_open_bars(bar(1)).is_empty());
    assert_eq!(e.get_cash(),600.);assert_eq!(e.get_position_for_symbol("A").0,0.);
    assert_eq!(e.drain_cash_events().unwrap(),vec![(1,"add".into(),500.,600.)]);
}
#[test]
fn cash_set_uses_the_same_portfolio_and_keeps_positions() {
    let mut e=engine(1000.);
    e.submit_order("A".into(),OrderSide::Buy,OrderType::Market,2.,None,None,None,None);
    assert_eq!(e.step_open_bars(bar(1)).len(),1);
    let position=e.get_position_for_symbol("A");
    e.apply_cash_controls(vec![(1,"set".into(),1200.)]).unwrap();
    assert_eq!(e.get_cash(),1200.);assert_eq!(e.get_position_for_symbol("A"),position);
    assert_eq!(e.get_equity(),1220.);
    assert_eq!(e.drain_cash_events().unwrap(),vec![(1,"set".into(),1200.,1200.)]);
    assert!(e.drain_cash_events().unwrap().is_empty());
    assert!(e.apply_cash_controls(vec![(1,"set".into(),5000.)]).is_err());
    assert_eq!(e.get_cash(),1200.);
}
#[test]
fn invalid_cash_batch_is_atomic_and_zero_is_a_real_flow() {
    let mut e=engine(100.);
    for commands in [vec![(1,"add".into(),50.),(1,"set".into(),400.)],vec![(1,"set".into(),f64::NAN)],vec![(1,"unknown".into(),10.)],vec![(0,"set".into(),10.)]] {
        assert!(e.apply_cash_controls(commands).is_err());assert_eq!(e.get_cash(),100.);
    }
    e.apply_cash_controls(vec![(1,"add".into(),0.),(2,"add".into(),-50.)]).unwrap();
    e.step_open_bars(bar(1));assert_eq!(e.get_cash(),50.);
    assert_eq!(e.drain_cash_events().unwrap(),vec![(1,"add".into(),0.,100.),(2,"add".into(),-50.,50.)]);
}
#[test]
fn applied_events_identify_later_set_and_prior_queued_flow_independently() {
    let mut e=engine(100.);
    e.apply_cash_controls(vec![(1,"add".into(),10.),(2,"set".into(),250.)]).unwrap();
    assert_eq!(e.drain_cash_events().unwrap(),vec![(2,"set".into(),250.,250.)]);
    e.step_open_bars(bar(1));assert_eq!(e.get_cash(),260.);
    assert_eq!(e.drain_cash_events().unwrap(),vec![(1,"add".into(),10.,260.)]);
}
#[test]
fn submission_reference_rejection_precedes_a_cheaper_execution_bar() {
    let mut e=engine(100.);
    let id=e.submit_order("A".into(),OrderSide::Buy,OrderType::Market,1.,None,None,None,None);
    e.configure_submission_check(id,200.).unwrap();
    assert!(e.step_open_bars(bar(1)).is_empty());
    assert_eq!(e.get_cash(),100.);
    assert_eq!(e.drain_order_events(),vec![(id,"submission_margin".into())]);
}
#[test]
fn submission_affordability_does_not_hide_later_execution_margin() {
    let mut e=engine(100.);
    let id=e.submit_order("A".into(),OrderSide::Buy,OrderType::Market,1.,None,None,None,None);
    e.configure_submission_check(id,10.).unwrap();
    let mut expensive=bar(1);
    expensive[0].open=200.;expensive[0].high=200.;expensive[0].low=200.;expensive[0].close=200.;
    assert!(e.step_open_bars(expensive).is_empty());
    assert_eq!(e.get_cash(),100.);
    assert_eq!(e.drain_order_events(),vec![(id,"margin".into())]);
}
