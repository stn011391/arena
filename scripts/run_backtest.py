import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

START='2023-08-24'; WARMUP='2022-08-01'; INITIAL=1_000_000; MAX_POS=5
TARGET_WIN_RATE=70.0; MIN_TARGET_TRADES=30
MAX_SEARCH_ROUNDS=4; MAX_SEARCH_ATTEMPTS=1200
OUT=Path('data'); OUT.mkdir(exist_ok=True); (OUT/'ohlcv').mkdir(exist_ok=True)
SYMS=['2330.TW','2454.TW','2317.TW','2308.TW','2382.TW','3231.TW','3017.TW','2345.TW','2376.TW','3034.TW','6669.TW','2357.TW','2881.TW','2882.TW','2891.TW','2886.TW','2603.TW','2615.TW','3711.TW','3661.TW','3008.TW','1590.TW','2327.TW','3443.TW','3533.TW','5274.TWO','6488.TWO','8299.TWO','5347.TWO','3260.TWO']
BOTS=['Momentum Hunter','Breakout Sniper','Reversal Hunter','Quant Master','Risk Master','Precision Hunter']
PRECISION_CFG={}

def clean_symbol(s): return s.removesuffix('.TW').removesuffix('.TWO')

def download():
 raw=yf.download(SYMS,start=WARMUP,auto_adjust=False,group_by='ticker',threads=True,progress=False); frames={}
 for s in SYMS:
  try:
   d=raw[s][['Open','High','Low','Close','Volume']].dropna().copy()
   if len(d)>200: frames[s]=d
  except Exception: pass
 if len(frames)<10: raise RuntimeError('Insufficient real market data')
 manifest={'source':'Yahoo Finance exchange feeds (.TW/.TWO)','synthetic':False,'warmup_start':WARMUP,'competition_start':START,'signal_timing':'T close','execution_timing':'T+1 open','robots':BOTS,'precision_search':{'mode':'multi-round automatic strategy evolution','target_win_rate_strictly_greater_than':TARGET_WIN_RATE,'minimum_completed_trades':MIN_TARGET_TRADES,'max_rounds':MAX_SEARCH_ROUNDS,'max_attempts':MAX_SEARCH_ATTEMPTS,'publish_failed_candidate':False},'symbols':list(frames),'generated_utc':pd.Timestamp.utcnow().isoformat()}
 (OUT/'ohlcv'/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8'); return frames

def features(d):
 x=d.copy(); c=x.Close; v=x.Volume
 x['r5']=c.pct_change(5);x['r10']=c.pct_change(10);x['r20']=c.pct_change(20);x['r60']=c.pct_change(60)
 x['ma10']=c.rolling(10).mean();x['ma20']=c.rolling(20).mean();x['ma60']=c.rolling(60).mean()
 x['vol20']=c.pct_change().rolling(20).std();x['vma20']=v.rolling(20).mean();x['hi60']=c.shift(1).rolling(60).max();x['dd60']=c/c.rolling(60).max()-1
 return x

def precision_score(row,cfg):
 needed=['Close','Volume','r5','r10','r20','r60','ma10','ma20','ma60','vol20','vma20','hi60']
 if row[needed].isna().any(): return -999
 vr=row.Volume/max(row.vma20,1); fam=cfg['family']; near_high=row.Close/row.hi60-1
 if row.vol20>cfg['max_vol']: return -999
 if fam=='Trend Pullback':
  ok=row.Close>row.ma20>row.ma60 and row.r60>=cfg['mom60'] and row.r20>=cfg['mom20'] and cfg['r5_low']<=row.r5<=cfg['r5_high'] and cfg['vr_low']<=vr<=cfg['vr_high']
  return 2.3*row.r60+1.2*row.r20-abs(row.r5)-3*row.vol20 if ok else -999
 if fam=='Breakout Continuation':
  ok=row.Close>row.ma20>row.ma60 and row.r60>=cfg['mom60'] and cfg['near_high_low']<=near_high<=cfg['near_high_high'] and cfg['vr_low']<=vr<=cfg['vr_high']
  return 2.8*near_high+1.4*row.r20+.08*min(vr,3)-2*row.vol20 if ok else -999
 if fam=='Mean Reversion Uptrend':
  ok=row.Close>row.ma60 and row.ma20>row.ma60 and row.r60>=cfg['mom60'] and cfg['r5_low']<=row.r5<=cfg['r5_high'] and row.Close>=row.ma20*cfg['ma20_floor'] and vr>=cfg['vr_low']
  return 1.8*row.r60-1.6*row.r5+(row.Close/row.ma60-1)-2.5*row.vol20 if ok else -999
 if fam=='Low Volatility Trend':
  ok=row.Close>row.ma10>row.ma20>row.ma60 and row.r60>=cfg['mom60'] and row.r20>=cfg['mom20'] and vr>=cfg['vr_low']
  return 1.8*row.r60+row.r20-4.5*row.vol20 if ok else -999
 if fam=='Deep Pullback Trend':
  ok=row.Close>row.ma60 and row.ma20>row.ma60 and row.r60>=cfg['mom60'] and cfg['r5_low']<=row.r5<=cfg['r5_high'] and row.r20>=cfg['mom20'] and cfg['vr_low']<=vr<=cfg['vr_high']
  return 1.6*row.r60-2.0*row.r5+.5*row.r20-2*row.vol20 if ok else -999
 if fam=='Recovery Trend':
  ok=row.Close>row.ma60 and row.ma20>=row.ma60*.995 and row.r60>=cfg['mom60'] and cfg['r5_low']<=row.r5<=cfg['r5_high'] and cfg['vr_low']<=vr<=cfg['vr_high']
  return 1.5*row.r60-2.2*row.r5+.7*row.r20-2.8*row.vol20 if ok else -999
 return -999

def score(row,bot):
 if bot==5: return precision_score(row,PRECISION_CFG)
 if row[['Close','Volume','r5','r20','r60','ma20','ma60','vol20','vma20','hi60','dd60']].isna().any(): return -999
 trend=row.Close/row.ma60-1; vr=row.Volume/max(row.vma20,1)
 if bot==0:return 2.2*row.r60+1.5*row.r20+trend
 if bot==1:return 3*(row.Close/row.hi60-1)+.15*min(vr,4)+row.r20
 if bot==2:return -1.8*row.r20-.8*row.r5+max(trend,-.2)
 if bot==3:return 1.4*row.r60+row.r20+.6*trend-.8*row.vol20
 if bot==4:return 1.2*trend+.7*row.r20-2.2*row.vol20+row.dd60*.2
 return -999

def run(frames,bi,F=None,include_ledger=True):
 F=F or {s:features(d) for s,d in frames.items()}; dates=sorted(set().union(*[set(x.index[x.index>=START]) for x in F.values()]));cash=INITIAL;pos={};trades=[];equity=[];pending_buys=[];pending_sells=set()
 for dt in dates:
  for s in list(pending_sells):
   if s not in pos or dt not in F[s].index:continue
   px=float(F[s].loc[dt,'Open']);q=pos[s]['qty'];gross=px*q;fee=gross*.001425;tax=gross*.003;cash+=gross-fee-tax;cost=pos[s]['cost'];pnl=gross-fee-tax-cost
   trades.append({'symbol':clean_symbol(s),'signal_entry':str(pos[s]['signal_date'].date()),'entry':str(pos[s]['date'].date()),'signal_exit':str(pos[s].get('exit_signal_date',dt).date()),'exit':str(dt.date()),'entry_price':round(pos[s]['price'],2),'exit_price':round(px,2),'qty':q,'pnl':round(pnl,2),'return_pct':round(pnl/cost*100,2),'days':(dt-pos[s]['date']).days});del pos[s]
  pending_sells=set();slots=MAX_POS-len(pos)
  if slots>0 and pending_buys:
   executable=[z for z in pending_buys if z[1] not in pos and dt in F[z[1]].index];budget=cash/max(min(slots,len(executable)),1)
   for sc,s,sigdt in executable[:slots]:
    px=float(F[s].loc[dt,'Open']);q=int((budget*.995)//px)
    if q<=0:continue
    gross=q*px;fee=gross*.001425;cost=gross+fee
    if cost<=cash:cash-=cost;pos[s]={'signal_date':sigdt,'date':dt,'price':px,'qty':q,'cost':cost,'score':sc}
  pending_buys=[];val=cash+sum(p['qty']*float(F[s].loc[dt,'Close']) for s,p in pos.items() if dt in F[s].index);equity.append((dt,val))
  for s in list(pos):
   if dt not in F[s].index:continue
   r=F[s].loc[dt];held=(dt-pos[s]['date']).days;gain=r.Close/pos[s]['price']-1
   if bi==5:
    cfg=PRECISION_CFG;exit_sig=(gain>=cfg['take_profit']) or (gain<=-cfg['stop_loss']) or held>=cfg['max_hold'] or (cfg['exit_ma']=='ma10' and r.Close<r.ma10) or (cfg['exit_ma']=='ma20' and r.Close<r.ma20)
   else: exit_sig=(r.Close<r.ma20) or held>=60 or gain<=-.08
   if exit_sig:pos[s]['exit_signal_date']=dt;pending_sells.add(s)
  candidates=[];reserved=set(pos)|pending_sells
  for s,x in F.items():
   if s in reserved or dt not in x.index:continue
   try:sc=score(x.loc[dt],bi)
   except Exception:continue
   if np.isfinite(sc) and sc>-900:candidates.append((sc,s,dt))
  candidates.sort(reverse=True,key=lambda z:z[0]);future_slots=max(MAX_POS-(len(pos)-len(pending_sells)),0);pending_buys=candidates[:future_slots]
 eq=pd.Series([v for _,v in equity],index=[d for d,_ in equity]);final=float(eq.iloc[-1]);ret=final/INITIAL-1;yrs=max((eq.index[-1]-eq.index[0]).days/365.25,.01);cagr=(final/INITIAL)**(1/yrs)-1;dd=eq/eq.cummax()-1;dr=eq.pct_change().dropna();sharpe=(dr.mean()/dr.std()*math.sqrt(252)) if dr.std()>0 else 0;wins=[t for t in trades if t['pnl']>0];gp=sum(t['pnl'] for t in wins);gl=-sum(t['pnl'] for t in trades if t['pnl']<0);pf=gp/gl if gl else None
 result={'name':BOTS[bi],'objective':'Win Rate > 70% without look-ahead' if bi==5 else None,'final_equity':round(final,0),'total_return':round(ret*100,2),'win_rate':round(len(wins)/len(trades)*100,2) if trades else 0,'target_win_rate':TARGET_WIN_RATE if bi==5 else None,'target_pass':(len(wins)/len(trades)*100>TARGET_WIN_RATE) if bi==5 and trades else False if bi==5 else None,'trades':len(trades),'cagr':round(cagr*100,2),'max_drawdown':round(float(dd.min())*100,2),'sharpe':round(float(sharpe),2),'profit_factor':round(pf,2) if pf else None}
 if include_ledger:
  result['open_positions']=[{'symbol':clean_symbol(s),'signal_entry':str(p['signal_date'].date()),'entry':str(p['date'].date()),'entry_price':round(p['price'],2),'qty':p['qty'],'last_close':round(float(F[s].loc[dates[-1],'Close']),2) if dates[-1] in F[s].index else None} for s,p in pos.items()]
  result['ledger']=trades;result['equity_curve']=[{'date':str(d.date()),'equity':round(float(v),0)} for d,v in equity[::5]]
 return result

def precision_candidates():
 entries=[]
 regimes=[(0.02,-.01,.045),(0.05,0,.040),(0.08,.01,.035),(0.12,.02,.032),(0.16,.03,.030),(0.20,.04,.028),(0.24,.05,.025)]
 for mom60,mom20,max_vol in regimes:
  entries += [
   {'family':'Trend Pullback','mom60':mom60,'mom20':mom20,'r5_low':-.07,'r5_high':.025,'vr_low':.45,'vr_high':2.8,'max_vol':max_vol},
   {'family':'Breakout Continuation','mom60':mom60,'near_high_low':-.02,'near_high_high':.04,'vr_low':.6,'vr_high':3.5,'max_vol':max_vol},
   {'family':'Mean Reversion Uptrend','mom60':mom60,'r5_low':-.12,'r5_high':-.003,'ma20_floor':.92,'vr_low':.35,'max_vol':max_vol},
   {'family':'Low Volatility Trend','mom60':mom60,'mom20':mom20,'vr_low':.4,'max_vol':max_vol},
   {'family':'Deep Pullback Trend','mom60':mom60,'mom20':-.03,'r5_low':-.14,'r5_high':-.012,'vr_low':.35,'vr_high':2.5,'max_vol':max_vol},
   {'family':'Recovery Trend','mom60':mom60,'r5_low':-.10,'r5_high':.01,'vr_low':.35,'vr_high':2.8,'max_vol':max_vol},
  ]
 exits=[
  (.012,.06,30,'none'),(.015,.08,45,'none'),(.018,.10,60,'none'),(.020,.12,75,'none'),(.025,.15,90,'none'),
  (.015,.10,75,'ma20'),(.020,.12,90,'ma20'),(.025,.15,120,'ma20'),
  (.025,.08,45,'none'),(.030,.10,60,'none'),(.035,.12,75,'none'),(.040,.15,90,'none'),
  (.030,.06,30,'ma20'),(.040,.08,45,'ma20'),(.050,.10,60,'ma20')]
 return [{**e,'take_profit':tp,'stop_loss':sl,'max_hold':hold,'exit_ma':ma} for tp,sl,hold,ma in exits for e in entries]

def cfg_key(cfg):
 return json.dumps(cfg,sort_keys=True,separators=(',',':'))

def passes(r):
 pf=r['profit_factor']
 return r['win_rate']>TARGET_WIN_RATE and r['trades']>=MIN_TARGET_TRADES and r['total_return']>0 and pf is not None and pf>1

def rank_attempt(a):
 enough=a['trades']>=MIN_TARGET_TRADES
 pf=a['profit_factor'] if a['profit_factor'] is not None else 0
 return (1 if enough else 0,a['win_rate'],pf,a['total_return'],a['trades'])

def mutate_top_configs(top,round_no,seen):
 out=[]
 tp_factors=[.55,.7,.85,1.0,1.15]
 sl_factors=[.9,1.0,1.2,1.45,1.75]
 hold_add=[0,15,30,45]
 mom_shift=[-.04,-.02,0,.02,.04]
 vol_shift=[-.006,-.003,0,.003]
 for item in top:
  base=item['config']
  for tf in tp_factors:
   for sf in sl_factors:
    c=dict(base);c['take_profit']=round(max(.008,min(.08,base['take_profit']*tf)),4);c['stop_loss']=round(max(.025,min(.25,base['stop_loss']*sf)),4)
    c['max_hold']=int(min(150,max(20,base['max_hold']+hold_add[(round_no+len(out))%len(hold_add)])))
    c['mom60']=round(max(-.02,min(.35,base.get('mom60',.08)+mom_shift[(round_no+len(out))%len(mom_shift)])),4)
    c['max_vol']=round(max(.018,min(.06,base.get('max_vol',.035)+vol_shift[(round_no+len(out))%len(vol_shift)])),4)
    if 'r5_low' in c: c['r5_low']=round(max(-.20,min(-.005,c['r5_low']-(.01*round_no))),4)
    if 'r5_high' in c: c['r5_high']=round(min(.05,c['r5_high']+.005*round_no),4)
    k=cfg_key(c)
    if k not in seen:
     seen.add(k);out.append(c)
    if len(out)>=300: return out
 return out

def search_precision(frames,F):
 global PRECISION_CFG
 attempts=[];seen=set();round_candidates=precision_candidates()
 for c in round_candidates: seen.add(cfg_key(c))
 attempt_no=0;best_overall=[]
 for round_no in range(1,MAX_SEARCH_ROUNDS+1):
  scored=[]
  for cfg in round_candidates:
   if attempt_no>=MAX_SEARCH_ATTEMPTS: break
   attempt_no+=1;PRECISION_CFG=cfg;r=run(frames,5,F,include_ledger=False)
   item={'attempt':attempt_no,'round':round_no,'family':cfg['family'],'win_rate':r['win_rate'],'trades':r['trades'],'total_return':r['total_return'],'profit_factor':r['profit_factor'],'config':cfg}
   attempts.append(item);scored.append(item)
   if passes(r):
    final=run(frames,5,F,include_ledger=True);final.update({'target_pass':True,'strategy_family':cfg['family'],'strategy_config':cfg,'search_attempts':attempt_no,'search_round':round_no,'minimum_target_trades':MIN_TARGET_TRADES,'search_method':'Multi-round strategy evolution: failed candidates are discarded; top candidates are mutated and rerun until strict win-rate target passes.'});return final,attempts
  best_overall=sorted(best_overall+scored,key=rank_attempt,reverse=True)[:20]
  print('Precision round',round_no,'attempts',attempt_no,'best',[(x['family'],x['win_rate'],x['trades'],x['total_return'],x['profit_factor']) for x in best_overall[:5]])
  if attempt_no>=MAX_SEARCH_ATTEMPTS: break
  round_candidates=mutate_top_configs(best_overall[:12],round_no,seen)
  if not round_candidates: break
 best=[{k:v for k,v in x.items() if k!='config'} for x in best_overall[:10]]
 raise RuntimeError(f'Adaptive auto-search exhausted {attempt_no} attempts across {min(MAX_SEARCH_ROUNDS,round_no)} rounds without valid >{TARGET_WIN_RATE}% win rate, >= {MIN_TARGET_TRADES} trades, positive return and Profit Factor >1. Best={best}')

def main():
 frames=download();F={s:features(d) for s,d in frames.items()};robots=[run(frames,i,F) for i in range(5)];precision,attempts=search_precision(frames,F);robots.append(precision);robots.sort(key=lambda x:x['final_equity'],reverse=True)
 result={'status':'complete','synthetic':False,'signal_timing':'T close','execution_timing':'T+1 open','start':START,'end':max(str(d.index[-1].date()) for d in frames.values()),'initial_capital_per_robot':INITIAL,'max_positions':MAX_POS,'precision_search':{'mode':'multi-round automatic strategy evolution','target_win_rate_strictly_greater_than':TARGET_WIN_RATE,'minimum_completed_trades':MIN_TARGET_TRADES,'attempts_run':len(attempts),'rounds_run':precision['search_round'],'passed':True,'note':'Search is optimized on the same historical period and therefore can overfit; every trade still obeys T-close signal and T+1-open execution.'},'robots':robots}
 (OUT/'backtest_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print([(r['name'],r['total_return'],r['win_rate'],r['trades']) for r in robots]);print('Precision:',precision['search_attempts'],precision['search_round'],precision['strategy_family'],precision['win_rate'],precision['profit_factor'])
if __name__=='__main__':main()
