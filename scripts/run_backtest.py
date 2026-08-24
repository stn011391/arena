import json, math, os
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

START='2023-08-24'; WARMUP='2022-08-01'; INITIAL=1_000_000; MAX_POS=5
OUT=Path('data'); OUT.mkdir(exist_ok=True); (OUT/'ohlcv').mkdir(exist_ok=True)
# Liquid TWSE/TPEx universe. Prices are fetched from Yahoo's Taiwan exchange feeds; no synthetic prices.
SYMS=['2330.TW','2454.TW','2317.TW','2308.TW','2382.TW','3231.TW','3017.TW','2345.TW','2376.TW','3034.TW','6669.TW','2357.TW','2881.TW','2882.TW','2891.TW','2886.TW','2603.TW','2615.TW','3711.TW','3661.TW','3008.TW','1590.TW','2327.TW','3443.TW','3533.TW','5274.TWO','6488.TWO','8299.TWO','5347.TWO','3260.TWO']
BOTS=['Momentum Hunter','Breakout Sniper','Reversal Hunter','Quant Master','Risk Master']

def download():
    raw=yf.download(SYMS,start=WARMUP,auto_adjust=False,group_by='ticker',threads=True,progress=False)
    frames={}
    for s in SYMS:
        try:
            d=raw[s][['Open','High','Low','Close','Volume']].dropna().copy()
            if len(d)>200: frames[s]=d
        except Exception: pass
    if len(frames)<10: raise RuntimeError(f'Insufficient real market data: {len(frames)} symbols')
    manifest={'source':'Yahoo Finance exchange feeds (.TW/.TWO)','synthetic':False,'warmup_start':WARMUP,'competition_start':START,'symbols':list(frames),'generated_utc':pd.Timestamp.utcnow().isoformat()}
    (OUT/'ohlcv'/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return frames

def features(d):
    x=d.copy(); c=x.Close; v=x.Volume
    x['r5']=c.pct_change(5); x['r20']=c.pct_change(20); x['r60']=c.pct_change(60)
    x['ma20']=c.rolling(20).mean(); x['ma60']=c.rolling(60).mean(); x['vol20']=c.pct_change().rolling(20).std(); x['vma20']=v.rolling(20).mean(); x['hi60']=c.shift(1).rolling(60).max(); x['dd60']=c/x.Close.rolling(60).max()-1
    return x

def score(row,bot):
    if row.isna().any(): return -999
    trend=(row.Close/row.ma60-1); vr=row.Volume/max(row.vma20,1)
    if bot==0:return 2.2*row.r60+1.5*row.r20+trend
    if bot==1:return 3*(row.Close/row.hi60-1)+.15*min(vr,4)+row.r20
    if bot==2:return -1.8*row.r20-.8*row.r5+max(trend,-.2)
    if bot==3:return 1.4*row.r60+row.r20+.6*trend-.8*row.vol20
    return 1.2*trend+.7*row.r20-2.2*row.vol20+row.dd60*.2

def run(frames,bi):
    F={s:features(d) for s,d in frames.items()}; dates=sorted(set().union(*[set(x.index[x.index>=START]) for x in F.values()]))
    cash=INITIAL; pos={}; trades=[]; equity=[]
    for dt in dates:
        # exits decided only from data available at dt close; execution uses that close consistently for all bots.
        for s in list(pos):
            if dt not in F[s].index: continue
            r=F[s].loc[dt]; held=(dt-pos[s]['date']).days
            exit_sig=(r.Close<r.ma20) or held>=60 or (r.Close/pos[s]['price']-1<=-.08)
            if exit_sig:
                px=float(r.Close); q=pos[s]['qty']; gross=px*q; fee=gross*.001425; tax=gross*.003; cash+=gross-fee-tax
                cost=pos[s]['cost']; pnl=(gross-fee-tax)-cost
                trades.append({'symbol':s.replace('.TW','').replace('.TWO',''),'entry':str(pos[s]['date'].date()),'exit':str(dt.date()),'entry_price':round(pos[s]['price'],2),'exit_price':round(px,2),'qty':q,'pnl':round(pnl,2),'return_pct':round(pnl/cost*100,2),'days':held})
                del pos[s]
        candidates=[]
        for s,x in F.items():
            if s in pos or dt not in x.index:continue
            r=x.loc[dt]
            try: sc=score(r,bi)
            except: continue
            if np.isfinite(sc):candidates.append((sc,s,r))
        candidates.sort(reverse=True,key=lambda z:z[0])
        slots=MAX_POS-len(pos)
        if slots>0 and candidates:
            budget=cash/max(slots,1)
            for sc,s,r in candidates[:slots]:
                px=float(r.Close); q=int((budget*.995)//px)
                if q<=0:continue
                gross=q*px; fee=gross*.001425; cost=gross+fee
                if cost<=cash: cash-=cost; pos[s]={'date':dt,'price':px,'qty':q,'cost':cost,'score':sc}
        val=cash+sum(p['qty']*float(F[s].loc[dt].Close) for s,p in pos.items() if dt in F[s].index)
        equity.append((dt,val))
    if dates:
        dt=dates[-1]
        for s in list(pos):
            if dt not in F[s].index:continue
            px=float(F[s].loc[dt].Close);q=pos[s]['qty'];gross=px*q;fee=gross*.001425;tax=gross*.003;cost=pos[s]['cost'];pnl=gross-fee-tax-cost
            trades.append({'symbol':s.replace('.TW','').replace('.TWO',''),'entry':str(pos[s]['date'].date()),'exit':str(dt.date()),'entry_price':round(pos[s]['price'],2),'exit_price':round(px,2),'qty':q,'pnl':round(pnl,2),'return_pct':round(pnl/cost*100,2),'days':(dt-pos[s]['date']).days})
    eq=pd.Series([v for _,v in equity],index=[d for d,_ in equity]); final=float(eq.iloc[-1]); ret=final/INITIAL-1; yrs=max((eq.index[-1]-eq.index[0]).days/365.25,.01); cagr=(final/INITIAL)**(1/yrs)-1; dd=eq/eq.cummax()-1; dr=eq.pct_change().dropna(); sharpe=(dr.mean()/dr.std()*math.sqrt(252)) if dr.std()>0 else 0; wins=[t for t in trades if t['pnl']>0]; gp=sum(t['pnl'] for t in wins); gl=-sum(t['pnl'] for t in trades if t['pnl']<0); pf=gp/gl if gl else None
    return {'name':BOTS[bi],'final_equity':round(final,0),'total_return':round(ret*100,2),'win_rate':round(len(wins)/len(trades)*100,2) if trades else 0,'trades':len(trades),'cagr':round(cagr*100,2),'max_drawdown':round(float(dd.min())*100,2),'sharpe':round(float(sharpe),2),'profit_factor':round(pf,2) if pf else None,'ledger':trades,'equity_curve':[{'date':str(d.date()),'equity':round(float(v),0)} for d,v in equity[::5]]}

def main():
    frames=download(); robots=[run(frames,i) for i in range(5)]; robots.sort(key=lambda x:x['final_equity'],reverse=True)
    result={'status':'complete','synthetic':False,'start':START,'end':max(str(d.index[-1].date()) for d in frames.values()),'initial_capital_per_robot':INITIAL,'max_positions':MAX_POS,'robots':robots}
    (OUT/'backtest_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='robots'},ensure_ascii=False));print([(r['name'],r['total_return'],r['win_rate'],r['trades']) for r in robots])
if __name__=='__main__':main()
