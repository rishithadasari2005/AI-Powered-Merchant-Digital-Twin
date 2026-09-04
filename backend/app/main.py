import os,re,json,sqlite3,hashlib
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse,JSONResponse
from pydantic import BaseModel,Field
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_absolute_error

BASE=Path(__file__).resolve().parent; STATIC=BASE/'static'; DB_PATH=os.getenv('DATABASE_PATH','/tmp/merchant_twin.db')
app=FastAPI(title='Merchant Digital Twin',version='2.0.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://merchant-digital-twin-frontend.onrender.com",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
df=None;models={};BASELINE=None;CACHE={};MODEL_METRICS={}

class Scenario(BaseModel):
 name:str='Untitled Scenario';discount_pct:float=Field(0,ge=0,le=80);price_change_pct:float=Field(0,ge=-80,le=200);cod_share_pct:float=Field(40,ge=0,le=100);campaign_strength:float=Field(40,ge=0,le=100);target_new_customers:bool=False;objective:str='profit';simulations:int=Field(250,ge=100,le=1000);free_shipping:bool=False;cashback_pct:float=Field(0,ge=0,le=30);min_order_value:float=Field(0,ge=0,le=20000)

def make_data(n=10000,seed=42):
 rng=np.random.default_rng(seed);price=rng.lognormal(np.log(1200),.55,n).clip(150,10000);discount=rng.choice([0,5,10,15,20],n,p=[.46,.18,.18,.12,.06]);cod=rng.binomial(1,.42,n);new=rng.binomial(1,.48,n);campaign=rng.uniform(0,100,n)
 conv=1/(1+np.exp(np.clip(-(-2.5+.045*discount+.008*campaign-.00008*price-.18*new+.08*(1-cod)),-30,30)));purchased=rng.binomial(1,conv)
 rto_p=1/(1+np.exp(np.clip(-(-2.7+1.05*cod+.004*price/10+.015*new-.018*discount),-30,30)));rto=rng.binomial(1,rto_p)
 pay_p=1/(1+np.exp(np.clip(-(3.9-.00008*price-.22*cod+.004*discount),-30,30)));pay=rng.binomial(1,pay_p)
 aov=price*(1-discount/100)*rng.normal(1,.08,n)
 return pd.DataFrame({'price':price,'discount':discount,'cod':cod,'new_customer':new,'campaign':campaign,'purchased':purchased,'rto':rto,'payment_success':pay,'aov':aov})

def train():
 global df,models,BASELINE,MODEL_METRICS
 df=make_data();X=df[['price','discount','cod','new_customer','campaign']];tr,te=train_test_split(np.arange(len(df)),test_size=.2,random_state=7)
 common=dict(n_estimators=30,max_depth=8,random_state=7,n_jobs=-1)
 models['conversion']=RandomForestClassifier(**common);models['rto']=RandomForestClassifier(**common);models['payment']=RandomForestClassifier(**common);models['aov']=RandomForestRegressor(n_estimators=30,max_depth=9,random_state=7,n_jobs=-1)
 for k,y in [('conversion',df.purchased),('rto',df.rto),('payment',df.payment_success),('aov',df.aov)]:
  models[k].fit(X.iloc[tr],y.iloc[tr])
  pred=models[k].predict(X.iloc[te]);MODEL_METRICS[k]=float(mean_absolute_error(y.iloc[te],pred)) if k=='aov' else float(accuracy_score(y.iloc[te],pred))
 BASELINE=predict_scenario(Scenario(name='Current',simulations=100))

def key(s): return tuple([round(s.discount_pct,2),round(s.price_change_pct,2),round(s.cod_share_pct,2),round(s.campaign_strength,2),s.target_new_customers,s.free_shipping,round(s.cashback_pct,2),round(s.min_order_value,2)])
def predict_scenario(s):
 k=key(s)
 if k in CACHE:return CACHE[k].copy()
 rng=np.random.default_rng(10);base=df.sample(n=min(2500,len(df)),random_state=11).copy();base.price*=1+s.price_change_pct/100;base.discount=s.discount_pct;base.cod=(rng.random(len(base))<s.cod_share_pct/100).astype(np.int8);base.campaign=s.campaign_strength
 if s.target_new_customers:base.discount=np.where(base.new_customer==1,s.discount_pct,0)
 X=base[['price','discount','cod','new_customer','campaign']];conv=models['conversion'].predict_proba(X)[:,1];rto=models['rto'].predict_proba(X)[:,1];pay=models['payment'].predict_proba(X)[:,1];aov=models['aov'].predict(X)
 if s.min_order_value>0: aov=np.where(aov>=s.min_order_value,aov,aov*.72)
 if s.free_shipping:aov=aov-60
 if s.cashback_pct:aov=aov*(1-s.cashback_pct/100)
 effective=conv*pay;orders=10000*float(effective.mean());completed=orders*(1-float(rto.mean()));avg=float(aov.mean());revenue=completed*avg;cogs=revenue*.56;discount_cost=revenue*(s.discount_pct/100)*.25;cashback=revenue*(s.cashback_pct/100)*.35;ship=completed*60 if s.free_shipping else 0;rto_cost=orders*float(rto.mean())*avg*.11;profit=revenue-cogs-discount_cost-cashback-ship-rto_cost
 out={'orders':orders,'revenue':revenue,'profit':profit,'conversion':float(effective.mean()),'rto':float(rto.mean()),'payment_success':float(pay.mean()),'aov':avg}
 CACHE[k]=out.copy();
 if len(CACHE)>200:CACHE.pop(next(iter(CACHE)))
 return out

def monte(s,r):
 n=int(s.simulations);rng=np.random.default_rng(100+int(s.discount_pct*10)+int(s.price_change_pct));profit=rng.normal(r['profit'],max(abs(r['profit'])*.08,1),n);rev=rng.normal(r['revenue'],max(abs(r['revenue'])*.045,1),n);return {'runs':n,'expected_revenue':float(rev.mean()),'expected_profit':float(profit.mean()),'p_profit_positive':float((profit>0).mean()),'profit_p10':float(np.percentile(profit,10)),'profit_p90':float(np.percentile(profit,90)),'distribution':[float(x) for x in np.percentile(profit,[5,10,20,30,40,50,60,70,80,90,95])]}

def rec(r,b):
 dp=r['profit']-b['profit'];return {'decision':'RECOMMENDED' if dp>0 else 'NOT RECOMMENDED','message':f"Expected profit {'improves' if dp>0 else 'falls'} by ₹{abs(dp):,.0f}.",'reason':'Revenue uplift outweighs simulated margin, fulfillment and RTO costs.' if dp>0 else 'Revenue uplift does not compensate for simulated margin, fulfillment and RTO costs.'}
def confidence():
 vals=[MODEL_METRICS.get('conversion',0),MODEL_METRICS.get('rto',0),MODEL_METRICS.get('payment',0)];return int(np.clip(65+np.mean(vals)*25+min(len(df),10000)/10000*8,0,98))
def save(s,res):
 try:
  Path(DB_PATH).parent.mkdir(parents=True,exist_ok=True);c=sqlite3.connect(DB_PATH);c.execute('CREATE TABLE IF NOT EXISTS simulations(id INTEGER PRIMARY KEY AUTOINCREMENT,created_at TEXT,scenario TEXT,result TEXT)');c.execute('INSERT INTO simulations(created_at,scenario,result) VALUES(?,?,?)',(datetime.now(timezone.utc).isoformat(),s.name,json.dumps(res)));c.commit();c.close()
 except:pass

@app.on_event('startup')
def startup():train()
@app.get('/health')
def health():return {'status':'ok' if df is not None else 'warming_up','records':len(df) if df is not None else 0,'models_ready':bool(models)}
@app.get('/api/summary')
def summary():return {'ready':BASELINE is not None,'metrics':BASELINE,'records':len(df) if df is not None else 0,'confidence':confidence() if df is not None else 0,'model_metrics':MODEL_METRICS}
@app.get('/api/model-status')
def status():return {'ready':BASELINE is not None,'records':len(df) if df is not None else 0,'models':list(models),'confidence':confidence() if df is not None else 0,'metrics':MODEL_METRICS,'cache_size':len(CACHE)}
@app.post('/api/simulate')
def simulate(s:Scenario):
 if BASELINE is None:return JSONResponse({'detail':'Simulation engine is warming up'},503)
 r=predict_scenario(s);mc=monte(s,r);delta={k:r[k]-BASELINE[k] for k in ['orders','revenue','profit','conversion','rto']};res={'scenario':s.model_dump(),'result':r,'baseline':BASELINE,'monte_carlo':mc,'delta':delta,'recommendation':rec(r,BASELINE),'confidence':confidence(),'audit_id':hashlib.sha1((s.name+json.dumps(s.model_dump(),sort_keys=True)).encode()).hexdigest()[:10].upper(),'guardrail':{'status':'PASS' if r['profit']>=BASELINE['profit']*.9 else 'REVIEW','message':'Within 10% profit-loss safety boundary.' if r['profit']>=BASELINE['profit']*.9 else 'Potential profit decline exceeds 10%; merchant approval required.'},'why':[f"Simulated conversion is {r['conversion']*100:.1f}%.",f"RTO is {r['rto']*100:.1f}% under this policy.",f"Expected AOV is ₹{r['aov']:,.0f}."]};save(s,res);return res
@app.post('/api/optimize')
def optimize(p:dict):
 obj=p.get('objective','profit');obj=obj if obj in {'profit','revenue','orders','conversion'} else 'profit';rows=[]
 for d in range(0,31,2):
  s=Scenario(name=f'{d}% discount',discount_pct=d,objective=obj);r=predict_scenario(s);rows.append({'discount_pct':d,**r})
 best=max(rows,key=lambda x:x[obj]);return {'objective':obj,'best':best,'scenarios':rows,'tested':len(rows)}
@app.post('/api/experiment')
def experiment(p:dict):
 target=p.get('objective','profit');ideas=[Scenario(name='Targeted discount',discount_pct=8,target_new_customers=True),Scenario(name='Free shipping threshold',free_shipping=True,min_order_value=1499),Scenario(name='UPI cashback',cashback_pct=5,cod_share_pct=30),Scenario(name='Balanced campaign',discount_pct=5,campaign_strength=75,target_new_customers=True)];out=[]
 for s in ideas:
  r=predict_scenario(s);out.append({'name':s.name,'scenario':s.model_dump(),'result':r,'delta_profit':r['profit']-BASELINE['profit']})
 return {'goal':target,'experiments':sorted(out,key=lambda x:x['delta_profit'],reverse=True)}
@app.post('/api/agent')
def agent(p:dict):
 text=(p.get('text') or '').lower();objective='profit' if 'profit' in text or 'margin' in text else 'orders' if 'order' in text else 'revenue';opt=optimize({'objective':objective});best=opt['best'];return {'goal':objective,'strategies_tested':opt['tested'],'best':best,'next_action':'REQUEST_MERCHANT_APPROVAL','guardrail':'No production action is taken automatically.'}
@app.post('/api/parse')
def parse(p:dict):
 t=(p.get('text') or '').lower();m=re.search(r'(\d+(?:\.\d+)?)\s*%?\s*(?:discount|off)',t);disc=float(m.group(1)) if m else 0;pm=re.search(r'(?:price|increase price|raise price).*?([+-]?\d+(?:\.\d+)?)\s*%',t);price=float(pm.group(1)) if pm else 0;cm=re.search(r'cod.*?(\d+(?:\.\d+)?)\s*%',t);cod=float(cm.group(1)) if cm else 40;cash=re.search(r'(\d+(?:\.\d+)?)\s*%.*?cashback',t);cb=float(cash.group(1)) if cash else 0;target=('new customer' in t or 'first-time' in t or 'first time' in t);free='free shipping' in t;return {'name':'Natural Language Scenario','discount_pct':disc,'cod_share_pct':cod,'price_change_pct':price,'campaign_strength':70 if 'campaign' in t or 'promotion' in t else 40,'target_new_customers':target,'cashback_pct':cb,'free_shipping':free}
@app.get('/api/history')
def history():
 try:
  c=sqlite3.connect(DB_PATH);rows=c.execute('SELECT id,created_at,scenario,result FROM simulations ORDER BY id DESC LIMIT 30').fetchall();c.close();return [{'id':x[0],'created_at':x[1],'scenario':x[2],'result':json.loads(x[3])} for x in rows]
 except:return []
@app.get('/api/radar')
def radar():return {'risks':[{'level':'HIGH','title':'COD RTO exposure','detail':'First-time COD orders carry elevated simulated RTO risk.'},{'level':'MEDIUM','title':'Conversion sensitivity','detail':'Price-sensitive new customers respond strongly to discount changes.'},{'level':'MEDIUM','title':'Margin concentration','detail':'A small set of segments contributes disproportionate simulated profit.'},{'level':'LOW','title':'Payment stability','detail':'Payment success remains comparatively stable in the digital twin.'}]}
@app.get('/api/segments')
def segments():
 groups=[]
 for name,mask in [('New customers',df.new_customer==1),('Returning customers',df.new_customer==0),('COD customers',df.cod==1),('UPI customers',df.cod==0),('High-value',df.price>=df.price.quantile(.75)),('Price-sensitive',df.discount>0)]:groups.append({'name':name,'customers':int(mask.sum()),'avg_order_value':float(df.loc[mask,'aov'].mean()),'rto':float(df.loc[mask,'rto'].mean()),'conversion':float(df.loc[mask,'purchased'].mean())})
 return groups
@app.get('/api/receipt/{audit_id}')
def receipt(audit_id:str):return {'audit_id':audit_id,'generated':datetime.now(timezone.utc).isoformat(),'status':'AUDIT RECORD','message':'Simulation receipt is advisory; no production action was executed.'}
@app.get('/{path:path}')
def spa(path:str):
 idx=STATIC/'index.html';return FileResponse(idx) if idx.exists() else JSONResponse({'message':'Frontend not built'},404)
