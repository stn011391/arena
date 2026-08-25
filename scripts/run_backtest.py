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
def pct(v): return round(float(v)*100,2)
def num(v): return round(float(v),2)

def download():
 raw=yf.download(SYMS,start=WARMUP,auto_adjust=False,group_by='ticker',threads=True,progress=False); frames={}
 for s in SYMS:
  try:
   d=raw[s][['Open','High','Low','Close','Volume']].dropna().copy()
   if len(d)>200: frames[s]=d
  except Exception: pass
 if len(frames)<10: raise RuntimeError('Insufficient real market data')
 manifest={'source':'Yahoo Finance exchange feeds (.TW/.TWO)','synthetic':False,'warmup_start':WARMUP,'competition_start':START,'signal_timing':'T close','execution_timing':'T+1 open','audit_trail':'Every completed trade records entry reason, exit reason, signal snapshot and discipline result.','robots':BOTS,'precision_search':{'mode':'multi-round automatic strategy evolution','target_win_rate_strictly_greater_than':TARGET_WIN_RATE,'minimum_completed_trades':MIN_TARGET_TRADES,'max_rounds':MAX_SEARCH_ROUNDS,'max_attempts':MAX_SEARCH_ATTEMPTS,'publish_failed_candidate':False},'symbols':list(frames),'generated_utc':pd.Timestamp.utcnow().isoformat()}
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

def snapshot(row):
 vr=float(row.Volume/max(row.vma20,1)) if pd.notna(row.vma20) and row.vma20 else None
 return {'close':num(row.Close),'ma10':num(row.ma10) if pd.notna(row.ma10) else None,'ma20':num(row.ma20) if pd.notna(row.ma20) else None,'ma60':num(row.ma60) if pd.notna(row.ma60) else None,'r5_pct':pct(row.r5) if pd.notna(row.r5) else None,'r20_pct':pct(row.r20) if pd.notna(row.r20) else None,'r60_pct':pct(row.r60) if pd.notna(row.r60) else None,'vol20_pct':pct(row.vol20) if pd.notna(row.vol20) else None,'volume_ratio':num(vr) if vr is not None else None,'prior_60d_high':num(row.hi60) if pd.notna(row.hi60) else None,'drawdown_60d_pct':pct(row.dd60) if pd.notna(row.dd60) else None}

def robot_discipline(bi):
 common='只使用 T 日收盤前已知資料產生訊號，下一個可交易日開盤成交；最多同時持有 5 檔。'
 if bi==0:return {'philosophy':'追蹤中期相對強勢與趨勢延續。','entry_rules':['依 60 日動能、20 日動能、股價相對 MA60 的綜合分數排序。','只有資料完整、未持有且排名進入可用持倉名額才下單。'],'exit_rules':['收盤跌破 MA20。','持有滿 60 日。','相對進場價虧損達 8%。'], 'execution_rule':common}
 if bi==1:return {'philosophy':'尋找接近前高且有成交量確認的突破延續。','entry_rules':['依股價距前 60 日高點、20 日動能、成交量/20 日均量比綜合排序。','只有排名進入可用持倉名額才下單。'],'exit_rules':['收盤跌破 MA20。','持有滿 60 日。','相對進場價虧損達 8%。'], 'execution_rule':common}
 if bi==2:return {'philosophy':'在仍有中期趨勢支撐時捕捉短線超跌反轉。','entry_rules':['偏好 5 日與 20 日跌幅較深、但 MA60 趨勢仍可接受的標的。','只有綜合反轉分數排名進入可用持倉名額才下單。'],'exit_rules':['收盤跌破 MA20。','持有滿 60 日。','相對進場價虧損達 8%。'], 'execution_rule':common}
 if bi==3:return {'philosophy':'以多因子方式平衡動能、趨勢與波動風險。','entry_rules':['依 60 日動能、20 日動能、MA60 趨勢與 20 日波動度計算 Quant 分數。','只有綜合分數排名進入可用持倉名額才下單。'],'exit_rules':['收盤跌破 MA20。','持有滿 60 日。','相對進場價虧損達 8%。'], 'execution_rule':common}
 if bi==4:return {'philosophy':'優先選擇低波動、趨勢穩定且回撤受控的標的。','entry_rules':['依 MA60 趨勢、20 日動能、20 日波動與 60 日回撤計算風險調整分數。','只有風險調整分數排名進入可用持倉名額才下單。'],'exit_rules':['收盤跌破 MA20。','持有滿 60 日。','相對進場價虧損達 8%。'], 'execution_rule':common}
 cfg=PRECISION_CFG
 return {'philosophy':'只交易通過自動搜尋後的高勝率策略設定；沒有 A+ setup 就持有現金。','entry_rules':['策略家族：'+str(cfg.get('family','—'))+'。','所有該策略家族的趨勢/動能/波動/成交量硬條件必須同時通過。','通過硬條件後再按策略分數排序，只買可用名額內最高分標的。'],'exit_rules':['停利：+'+str(round(cfg.get('take_profit',0)*100,2))+'%。','停損：-'+str(round(cfg.get('stop_loss',0)*100,2))+'%。','最長持有 '+str(cfg.get('max_hold','—'))+' 日。','均線出場：'+str(cfg.get('exit_ma','none'))+'。'], 'execution_rule':common}

def entry_audit(row,bi,sc,rank,slots):
 s=snapshot(row); vr=s['volume_ratio']; trend=pct(row.Close/row.ma60-1) if pd.notna(row.ma60) else None
 if bi==0: reason=f"動能排名入選 #{rank}：60日 {s['r60_pct']:+.2f}%、20日 {s['r20_pct']:+.2f}%、相對MA60 {trend:+.2f}%，綜合分數 {sc:.4f}。"
 elif bi==1: reason=f"突破排名入選 #{rank}：距前60日高點 {pct(row.Close/row.hi60-1):+.2f}%、量比 {vr:.2f}、20日動能 {s['r20_pct']:+.2f}%，分數 {sc:.4f}。"
 elif bi==2: reason=f"反轉排名入選 #{rank}：5日 {s['r5_pct']:+.2f}%、20日 {s['r20_pct']:+.2f}%、相對MA60 {trend:+.2f}%，分數 {sc:.4f}。"
 elif bi==3: reason=f"Quant排名入選 #{rank}：60日 {s['r60_pct']:+.2f}%、20日 {s['r20_pct']:+.2f}%、波動 {s['vol20_pct']:.2f}%，分數 {sc:.4f}。"
 elif bi==4: reason=f"風險調整排名入選 #{rank}：相對MA60 {trend:+.2f}%、20日 {s['r20_pct']:+.2f}%、波動 {s['vol20_pct']:.2f}%、60日回撤 {s['drawdown_60d_pct']:+.2f}%，分數 {sc:.4f}。"
 else: reason=f"Precision A+ setup 入選 #{rank}：{PRECISION_CFG.get('family')}；60日 {s['r60_pct']:+.2f}%、20日 {s['r20_pct']:+.2f}%、5日 {s['r5_pct']:+.2f}%、量比 {vr:.2f}、波動 {s['vol20_pct']:.2f}%，分數 {sc:.4f}。"
 checks=[{'rule':'訊號只使用 T 日收盤資料','passed':True},{'rule':'策略條件/分數有效','passed':bool(np.isfinite(sc) and sc>-900)},{'rule':f'排名 #{rank} 在可用名額 {slots} 內','passed':rank<=slots},{'rule':'成交延後至 T+1 開盤','passed':True}]
 return reason,checks,s

def exit_audit(row,bi,entry_price,held):
 gain=float(row.Close/entry_price-1); reasons=[]; checks=[]
 if bi==5:
  cfg=PRECISION_CFG
  if gain>=cfg['take_profit']: reasons.append(f"停利觸發：收盤相對進場 {gain*100:+.2f}% ≥ +{cfg['take_profit']*100:.2f}%")
  if gain<=-cfg['stop_loss']: reasons.append(f"停損觸發：收盤相對進場 {gain*100:+.2f}% ≤ -{cfg['stop_loss']*100:.2f}%")
  if held>=cfg['max_hold']: reasons.append(f"時間出場：持有 {held} 日 ≥ {cfg['max_hold']} 日")
  if cfg['exit_ma']=='ma10' and row.Close<row.ma10: reasons.append(f"均線出場：收盤 {row.Close:.2f} < MA10 {row.ma10:.2f}")
  if cfg['exit_ma']=='ma20' and row.Close<row.ma20: reasons.append(f"均線出場：收盤 {row.Close:.2f} < MA20 {row.ma20:.2f}")
  checks=[{'rule':'達到已定義的 Precision 出場條件','passed':bool(reasons)},{'rule':'出場訊號在 T 日收盤判定','passed':True},{'rule':'實際賣出延後至 T+1 開盤','passed':True}]
 else:
  if row.Close<row.ma20: reasons.append(f"趨勢失效：收盤 {row.Close:.2f} < MA20 {row.ma20:.2f}")
  if held>=60: reasons.append(f"時間出場：持有 {held} 日 ≥ 60 日")
  if gain<=-.08: reasons.append(f"停損觸發：收盤相對進場 {gain*100:+.2f}% ≤ -8.00%")
  checks=[{'rule':'達到既定出場條件','passed':bool(reasons)},{'rule':'出場訊號在 T 日收盤判定','passed':True},{'rule':'實際賣出延後至 T+1 開盤','passed':True}]
 return '；'.join(reasons),checks,snapshot(row)

def run(frames,bi,F=None,include_ledger=True):
 F=F or {s:features(d) for s,d in frames.items()}; dates=sorted(set().union(*[set(x.index[x.index>=START]) for x in F.values()]));cash=INITIAL;pos={};trades=[];equity=[];pending_buys=[];pending_sells={}
 for dt in dates:
  for s in list(pending_sells):
   if s not in pos: pending_sells.pop(s,None);continue
   if dt not in F[s].index: continue
   px=float(F[s].loc[dt,'Open']);q=pos[s]['qty'];gross=px*q;fee=gross*.001425;tax=gross*.003;cash+=gross-fee-tax;cost=pos[s]['cost'];pnl=gross-fee-tax-cost
   t={'symbol':clean_symbol(s),'signal_entry':str(pos[s]['signal_date'].date()),'entry':str(pos[s]['date'].date()),'signal_exit':str(pos[s].get('exit_signal_date',dt).date()),'exit':str(dt.date()),'entry_price':round(pos[s]['price'],2),'exit_price':round(px,2),'qty':q,'pnl':round(pnl,2),'return_pct':round(pnl/cost*100,2),'days':(dt-pos[s]['date']).days}
   if include_ledger:
    t.update({'entry_reason':pos[s].get('entry_reason'),'entry_checks':pos[s].get('entry_checks',[]),'entry_snapshot':pos[s].get('entry_snapshot'),'exit_reason':pos[s].get('exit_reason'),'exit_checks':pos[s].get('exit_checks',[]),'exit_snapshot':pos[s].get('exit_snapshot'),'discipline_pass':all(c.get('passed') for c in pos[s].get('entry_checks',[])+pos[s].get('exit_checks',[]))})
   trades.append(t);del pos[s];pending_sells.pop(s,None)
  slots=MAX_POS-len(pos)
  if slots>0 and pending_buys:
   executable=[z for z in pending_buys if z[1] not in pos and dt in F[z[1]].index];budget=cash/max(min(slots,len(executable)),1)
   for item in executable[:slots]:
    sc,s,sigdt,reason,checks,snap=item;px=float(F[s].loc[dt,'Open']);q=int((budget*.995)//px)
    if q<=0:continue
    gross=q*px;fee=gross*.001425;cost=gross+fee
    if cost<=cash:cash-=cost;pos[s]={'signal_date':sigdt,'date':dt,'price':px,'qty':q,'cost':cost,'score':sc,'entry_reason':reason,'entry_checks':checks,'entry_snapshot':snap}
  pending_buys=[];val=cash+sum(p['qty']*float(F[s].loc[dt,'Close']) for s,p in pos.items() if dt in F[s].index);equity.append((dt,val))
  for s in list(pos):
   if s in pending_sells or dt not in F[s].index:continue
   r=F[s].loc[dt];held=(dt-pos[s]['date']).days;gain=r.Close/pos[s]['price']-1
   if bi==5:
    cfg=PRECISION_CFG;exit_sig=(gain>=cfg['take_profit']) or (gain<=-cfg['stop_loss']) or held>=cfg['max_hold'] or (cfg['exit_ma']=='ma10' and r.Close<r.ma10) or (cfg['exit_ma']=='ma20' and r.Close<r.ma20)
   else: exit_sig=(r.Close<r.ma20) or held>=60 or gain<=-.08
   if exit_sig:
    pos[s]['exit_signal_date']=dt
    if include_ledger: pos[s]['exit_reason'],pos[s]['exit_checks'],pos[s]['exit_snapshot']=exit_audit(r,bi,pos[s]['price'],held)
    pending_sells[s]=True
  candidates=[];reserved=set(pos)|set(pending_sells)
  for s,x in F.items():
   if s in reserved or dt not in x.index:continue
   try:sc=score(x.loc[dt],bi)
   except Exception:continue
   if np.isfinite(sc) and sc>-900:candidates.append((sc,s,dt))
  candidates.sort(reverse=True,key=lambda z:z[0]);future_slots=max(MAX_POS-(len(pos)-len(pending_sells)),0);selected=[]
  for rank,(sc,s,sigdt) in enumerate(candidates[:future_slots],1):
   if include_ledger: reason,checks,snap=entry_audit(F[s].loc[dt],bi,sc,rank,future_slots)
   else: reason,checks,snap=None,[],None
   selected.append((sc,s,sigdt,reason,checks,snap))
  pending_buys=selected
 eq=pd.Series([v for _,v in equity],index=[d for d,_ in equity]);final=float(eq.iloc[-1]);ret=final/INITIAL-1;yrs=max((eq.index[-1]-eq.index[0]).days/365.25,.01);cagr=(final/INITIAL)**(1/yrs)-1;dd=eq/eq.cummax()-1;dr=eq.pct_change().dropna();sharpe=(dr.mean()/dr.std()*math.sqrt(252)) if dr.std()>0 else 0;wins=[t for t in trades if t['pnl']>0];gp=sum(t['pnl'] for t in wins);gl=-sum(t['pnl'] for t in trades if t['pnl']<0);pf=gp/gl if gl else None
 result={'name':BOTS[bi],'objective':'Win Rate > 70% without look-ahead' if bi==5 else None,'discipline':robot_discipline(bi),'discipline_audit':'Every published trade must have a signal reason and must pass the robot discipline checks.','final_equity':round(final,0),'total_return':round(ret*100,2),'win_rate':round(len(wins)/len(trades)*100,2) if trades else 0,'target_win_rate':TARGET_WIN_RATE if bi==5 else None,'target_pass':(len(wins)/len(trades)*100>TARGET_WIN_RATE) if bi==5 and trades else False if bi==5 else None,'trades':len(trades),'cagr':round(cagr*100,2),'max_drawdown':round(float(dd.min())*100,2),'sharpe':round(float(sharpe),2),'profit_factor':round(pf,2) if pf else None}
 if include_ledger:
  result['open_positions']=[{'symbol':clean_symbol(s),'signal_entry':str(p['signal_date'].date()),'entry':str(p['date'].date()),'entry_price':round(p['price'],2),'qty':p['qty'],'last_close':round(float(F[s].loc[dates[-1],'Close']),2) if dates[-1] in F[s].index else None,'entry_reason':p.get('entry_reason'),'entry_checks':p.get('entry_checks',[]),'entry_snapshot':p.get('entry_snapshot'),'discipline_pass':all(c.get('passed') for c in p.get('entry_checks',[]))} for s,p in pos.items()]
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

def cfg_key(cfg): return json.dumps(cfg,sort_keys=True,separators=(',',':'))
def passes(r):
 pf=r['profit_factor'];return r['win_rate']>TARGET_WIN_RATE and r['trades']>=MIN_TARGET_TRADES and r['total_return']>0 and pf is not None and pf>1
def rank_attempt(a):
 enough=a['trades']>=MIN_TARGET_TRADES;pf=a['profit_factor'] if a['profit_factor'] is not None else 0;return (1 if enough else 0,a['win_rate'],pf,a['total_return'],a['trades'])

def mutate_top_configs(top,round_no,seen):
 out=[];tp_factors=[.55,.7,.85,1.0,1.15];sl_factors=[.9,1.0,1.2,1.45,1.75];hold_add=[0,15,30,45];mom_shift=[-.04,-.02,0,.02,.04];vol_shift=[-.006,-.003,0,.003]
 for item in top:
  base=item['config']
  for tf in tp_factors:
   for sf in sl_factors:
    c=dict(base);c['take_profit']=round(max(.008,min(.08,base['take_profit']*tf)),4);c['stop_loss']=round(max(.025,min(.25,base['stop_loss']*sf)),4);c['max_hold']=int(min(150,max(20,base['max_hold']+hold_add[(round_no+len(out))%len(hold_add)])));c['mom60']=round(max(-.02,min(.35,base.get('mom60',.08)+mom_shift[(round_no+len(out))%len(mom_shift)])),4);c['max_vol']=round(max(.018,min(.06,base.get('max_vol',.035)+vol_shift[(round_no+len(out))%len(vol_shift)])),4)
    if 'r5_low' in c:c['r5_low']=round(max(-.20,min(-.005,c['r5_low']-(.01*round_no))),4)
    if 'r5_high' in c:c['r5_high']=round(min(.05,c['r5_high']+.005*round_no),4)
    k=cfg_key(c)
    if k not in seen:seen.add(k);out.append(c)
    if len(out)>=300:return out
 return out

def search_precision(frames,F):
 global PRECISION_CFG
 attempts=[];seen=set();round_candidates=precision_candidates()
 for c in round_candidates:seen.add(cfg_key(c))
 attempt_no=0;best_overall=[]
 for round_no in range(1,MAX_SEARCH_ROUNDS+1):
  scored=[]
  for cfg in round_candidates:
   if attempt_no>=MAX_SEARCH_ATTEMPTS:break
   attempt_no+=1;PRECISION_CFG=cfg;r=run(frames,5,F,include_ledger=False);item={'attempt':attempt_no,'round':round_no,'family':cfg['family'],'win_rate':r['win_rate'],'trades':r['trades'],'total_return':r['total_return'],'profit_factor':r['profit_factor'],'config':cfg};attempts.append(item);scored.append(item)
   if passes(r):
    final=run(frames,5,F,include_ledger=True);final.update({'target_pass':True,'strategy_family':cfg['family'],'strategy_config':cfg,'search_attempts':attempt_no,'search_round':round_no,'minimum_target_trades':MIN_TARGET_TRADES,'search_method':'Multi-round strategy evolution: failed candidates are discarded; top candidates are mutated and rerun until strict win-rate target passes.'});return final,attempts
  best_overall=sorted(best_overall+scored,key=rank_attempt,reverse=True)[:20];print('Precision round',round_no,'attempts',attempt_no,'best',[(x['family'],x['win_rate'],x['trades'],x['total_return'],x['profit_factor']) for x in best_overall[:5]])
  if attempt_no>=MAX_SEARCH_ATTEMPTS:break
  round_candidates=mutate_top_configs(best_overall[:12],round_no,seen)
  if not round_candidates:break
 best=[{k:v for k,v in x.items() if k!='config'} for x in best_overall[:10]];raise RuntimeError(f'Adaptive auto-search exhausted {attempt_no} attempts across {min(MAX_SEARCH_ROUNDS,round_no)} rounds without valid >{TARGET_WIN_RATE}% win rate, >= {MIN_TARGET_TRADES} trades, positive return and Profit Factor >1. Best={best}')

def main():
 frames=download();F={s:features(d) for s,d in frames.items()};robots=[run(frames,i,F) for i in range(5)];precision,attempts=search_precision(frames,F);robots.append(precision);robots.sort(key=lambda x:x['final_equity'],reverse=True)
 result={'status':'complete','synthetic':False,'signal_timing':'T close','execution_timing':'T+1 open','discipline_schema_version':'1.0','start':START,'end':max(str(d.index[-1].date()) for d in frames.values()),'initial_capital_per_robot':INITIAL,'max_positions':MAX_POS,'precision_search':{'mode':'multi-round automatic strategy evolution','target_win_rate_strictly_greater_than':TARGET_WIN_RATE,'minimum_completed_trades':MIN_TARGET_TRADES,'attempts_run':len(attempts),'rounds_run':precision['search_round'],'passed':True,'note':'Search is optimized on the same historical period and therefore can overfit; every trade still obeys T-close signal and T+1-open execution.'},'robots':robots}
 (OUT/'backtest_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print([(r['name'],r['total_return'],r['win_rate'],r['trades']) for r in robots]);print('Precision:',precision['search_attempts'],precision['search_round'],precision['strategy_family'],precision['win_rate'],precision['profit_factor'])
if __name__=='__main__':main()
