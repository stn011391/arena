import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
START='2023-08-24'; WARMUP='2022-08-01'; INITIAL=1_000_000; MAX_POS=5
OUT=Path('data'); OUT.mkdir(exist_ok=True); (OUT/'ohlcv').mkdir(exist_ok=True)
SYMS=['2330.TW','2454.TW','2317.TW','2308.TW','2382.TW','3231.TW','3017.TW','2345.TW','2376.TW','3034.TW','6669.TW','2357.TW','2881.TW','2882.TW','2891.TW','2886.TW','2603.TW','2615.TW','3711.TW','3661.TW','3008.TW','1590.TW','2327.TW','3443.TW','3533.TW','5274.TWO','6488.TWO','8299.TWO','5347.TWO','3260.TWO']
BOTS=['Momentum Hunter','Breakout Sniper','Reversal Hunter','Quant Master','Risk Master','Precision Hunter']
def clean_symbol(s): return s.removesuffix('.TW').removesuffix('.TWO')
def download():
 raw=yf.download(SYMS,start=WARMUP,auto_adjust=False,group_by='ticker',threads=True,progress=False); frames={}
 for s in SYMS:
  try:
   d=raw[s][['Open','High','Low','Close','Volume']].dropna().copy()
   if len(d)>200: frames[s]=d
  except Exception: pass
 if len(frames)<10: raise RuntimeError('Insufficient real market data')
 manifest={'source':'Yahoo Finance exchange feeds (.TW/.TWO)','synthetic':False,'warmup_start':WARMUP,'competition_start':START,'signal_timing':'T close','execution_timing':'T+1 open','robots':BOTS,'symbols':list(frames),'generated_utc':pd.Timestamp.utcnow().isoformat()}; (OUT/'ohlcv'/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8'); return frames
def features(d):
 x=d.copy(); c=x.Close; v=x.Volume; x['r5']=c.pct_change(5);x['r10']=c.pct_change(10);x['r20']=c.pct_change(20);x['r60']=c.pct_change(60);x['ma10']=c.rolling(10).mean();x['ma20']=c.rolling(20).mean();x['ma60']=c.rolling(60).mean();x['vol20']=c.pct_change().rolling(20).std();x['vma20']=v.rolling(20).mean();x['hi60']=c.shift(1).rolling(60).max();x['dd60']=c/c.rolling(60).max()-1; return x
def score(row,bot):
 if row[['Close','Volume','r5','r20','r60','ma20','ma60','vol20','vma20','hi60','dd60']].isna().any(): return -999
 trend=row.Close/row.ma60-1; vr=row.Volume/max(row.vma20,1)
 if bot==0:return 2.2*row.r60+1.5*row.r20+trend
 if bot==1:return 3*(row.Close/row.hi60-1)+.15*min(vr,4)+row.r20
 if bot==2:return -1.8*row.r20-.8*row.r5+max(trend,-.2)
 if bot==3:return 1.4*row.r60+row.r20+.6*trend-.8*row.vol20
 if bot==4:return 1.2*trend+.7*row.r20-2.2*row.vol20+row.dd60*.2
 # Precision Hunter: intentionally selective. Strong established uptrend + controlled pullback + healthy liquidity/volatility.
 if not (row.Close>row.ma20>row.ma60 and row.r60>.08 and -.035<=row.r5<=.045 and row.r20>0 and row.vol20<.035 and .75<=vr<=2.5): return -999
 return 2.0*row.r60+1.2*row.r20-4.0*row.vol20-abs(row.r5)*.5
def run(frames,bi):
 F={s:features(d) for s,d in frames.items()}; dates=sorted(set().union(*[set(x.index[x.index>=START]) for x in F.values()]));cash=INITIAL;pos={};trades=[];equity=[];pending_buys=[];pending_sells=set()
 for dt in dates:
  for s in list(pending_sells):
   if s not in pos or dt not in F[s].index:continue
   px=float(F[s].loc[dt,'Open']);q=pos[s]['qty'];gross=px*q;fee=gross*.001425;tax=gross*.003;cash+=gross-fee-tax;cost=pos[s]['cost'];pnl=gross-fee-tax-cost;trades.append({'symbol':clean_symbol(s),'signal_entry':str(pos[s]['signal_date'].date()),'entry':str(pos[s]['date'].date()),'signal_exit':str(pos[s].get('exit_signal_date',dt).date()),'exit':str(dt.date()),'entry_price':round(pos[s]['price'],2),'exit_price':round(px,2),'qty':q,'pnl':round(pnl,2),'return_pct':round(pnl/cost*100,2),'days':(dt-pos[s]['date']).days});del pos[s]
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
   if bi==5: exit_sig=(gain>=.055) or (gain<=-.035) or (r.Close<r.ma10) or held>=25
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
 open_positions=[{'symbol':clean_symbol(s),'signal_entry':str(p['signal_date'].date()),'entry':str(p['date'].date()),'entry_price':round(p['price'],2),'qty':p['qty'],'last_close':round(float(F[s].loc[dates[-1],'Close']),2) if dates[-1] in F[s].index else None} for s,p in pos.items()]
 return {'name':BOTS[bi],'objective':'Win Rate >= 70% without look-ahead' if bi==5 else None,'final_equity':round(final,0),'total_return':round(ret*100,2),'win_rate':round(len(wins)/len(trades)*100,2) if trades else 0,'target_win_rate':70 if bi==5 else None,'target_pass':(len(wins)/len(trades)*100>=70) if bi==5 and trades else False if bi==5 else None,'trades':len(trades),'cagr':round(cagr*100,2),'max_drawdown':round(float(dd.min())*100,2),'sharpe':round(float(sharpe),2),'profit_factor':round(pf,2) if pf else None,'open_positions':open_positions,'ledger':trades,'equity_curve':[{'date':str(d.date()),'equity':round(float(v),0)} for d,v in equity[::5]]}
def main():
 frames=download();robots=[run(frames,i) for i in range(len(BOTS))];robots.sort(key=lambda x:x['final_equity'],reverse=True);result={'status':'complete','synthetic':False,'signal_timing':'T close','execution_timing':'T+1 open','start':START,'end':max(str(d.index[-1].date()) for d in frames.values()),'initial_capital_per_robot':INITIAL,'max_positions':MAX_POS,'robots':robots};(OUT/'backtest_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print([(r['name'],r['total_return'],r['win_rate'],r['trades']) for r in robots])
if __name__=='__main__':main()
