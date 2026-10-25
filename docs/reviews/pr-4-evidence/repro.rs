#![allow(dead_code)]
#[path="/private/tmp/trade-learn-pr4-review/rust/tradelearn-rust/src/types.rs"] mod types;
#[path="/private/tmp/trade-learn-pr4-review/rust/tradelearn-rust/src/matching.rs"] mod matching;
#[path="/private/tmp/trade-learn-pr4-review/rust/tradelearn-rust/src/engine.rs"] mod engine;
use types::*;
fn engine(smart: bool, opens: Vec<f64>, highs: Vec<f64>, lows: Vec<f64>, closes: Vec<f64>) -> engine::BacktestEngine {
 let n=opens.len();
 engine::BacktestEngine::new((0..n as i64).collect(),opens,highs,lows,closes,vec![100.;n],100000.,0.,false,false,false,0.,0.,true,true,false,1.,0.,smart,false,0.,0.,0.,0.)
}
fn main(){
 for smart in [false,true] {
 let mut e=engine(smart,vec![100.,110.,105.],vec![102.,120.,108.],vec![98.,108.,104.],vec![100.,115.,105.]);
 e.submit_order("data0".into(),OrderSide::Sell,OrderType::StopTrail,1.,None,Some(100.),Some(10.),None);
 println!("smart={smart}, trailing: {:?}",(0..3).map(|i|e.step(i).iter().map(|f|f.price).collect::<Vec<_>>()).collect::<Vec<_>>());
 }
 let mut e=engine(true,vec![80.],vec![85.],vec![75.],vec![82.]);
 e.submit_order("data0".into(),OrderSide::Sell,OrderType::StopTrailLimit,1.,Some(95.),Some(110.),Some(10.),None);
 println!("sell limit 95, all bar prices <=85: {:?}",e.step(0));
 let mut e=engine(false,vec![110.,96.],vec![110.,98.],vec![101.,95.],vec![105.,96.]);
 e.submit_order("data0".into(),OrderSide::Sell,OrderType::StopTrail,1.,None,Some(120.),Some(20.),None);
 println!("reference 120, initial stop 100, first bar high 110: {:?}",e.step(0));
 println!("next bar low 95 (expected to trigger 100): {:?}",e.step(1));
}
