import sys,json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score,f1_score
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from run import ROOT,read,sha,load,verify,analysis_labels

def main():
 torch.set_num_threads(4);r=read(ROOT/'receipt.json');verify(r['signature']);verify(read(ROOT/'features_locked.json')['signature'])
 final=read(ROOT/'formal/final.json');assert final['status']=='complete' and final['scored']==120
 assert final['audit_sha256']==sha(ROOT/'analysis/audit.json')
 rows=read(ROOT/'analysis/scores.json');assert len(rows)==len({tuple(x[k] for k in ['dataset','seed','scale','condition','head']) for x in rows})==120
 truth={};err=0.;summary_err=0.
 for x in rows:
  key=(x['dataset'],x['seed'])
  if key not in truth:truth[key]=analysis_labels(*key)
  t=truth[key];y=t['y'].numpy();idx=t['test'].numpy();p=load(Path(x['source'])/'probabilities.pt').numpy();pred=p.argmax(1)
  values=dict(test_accuracy=accuracy_score(y[idx],pred[idx]),test_macro_f1=f1_score(y[idx],pred[idx],labels=np.arange(p.shape[1]),average='macro',zero_division=0))
  for k,v in values.items():err=max(err,abs(v-x[k]));assert abs(v-x[k])<1e-12
 for x in read(ROOT/'analysis/summary.json'):
  sub=[z for z in rows if all(z[k]==x[k] for k in ['dataset','scale','head'])]
  for m in ['test_accuracy','test_macro_f1','validation_accuracy','target_brier']:
   values=[np.mean([z[m] for z in sub if z['seed']==s])*(1 if m=='target_brier' else 100) for s in [0,1]]
   for k,v in [('mean',np.mean(values)),('sd',np.std(values,ddof=1))]:summary_err=max(summary_err,abs(v-x[m][k]));assert abs(v-x[m][k])<1e-10
 index={(z['dataset'],z['seed'],z['scale'],z['condition'],z['head']):z for z in rows};paired_err=0.
 for x in read(ROOT/'analysis/paired.json'):
  for m in ['test_accuracy','test_macro_f1','validation_accuracy','target_brier']:
   a=index[(x['dataset'],x['seed'],'unit',x['condition'],x['head'])][m];b=index[(x['dataset'],x['seed'],'raw',x['condition'],x['head'])][m]
   delta=(a-b)*(1 if m=='target_brier' else 100);paired_err=max(paired_err,abs(delta-x[m]));assert abs(delta-x[m])<1e-12
 result=dict(status='passed',scored=120,max_score_error=err,max_summary_error=summary_err,max_paired_error=paired_err,source_hashes=len(r['signature']),script_sha256=sha(__file__))
 Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))

if __name__=='__main__':main()
