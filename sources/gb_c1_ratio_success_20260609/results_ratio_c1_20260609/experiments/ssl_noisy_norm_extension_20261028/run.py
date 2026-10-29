"""Noisy-only extension; reuse CS/Photo and the verified linear fitting kernel."""
import sys,json,time,os,gc,argparse,traceback
from pathlib import Path
from contextlib import contextmanager
import torch
import torch.nn.functional as F
CODE=Path(__file__).resolve().parent
W=Path('C:/Users/Admin/Desktop/实验8去噪5.19-/gb_c1_ratio_success_20260609/results_ratio_c1_20260609')
CFG=json.loads((CODE/'config.json').read_text(encoding='utf-8-sig'));ROOT=Path(CFG['runtime'])
SCREEN=Path('D:/codex_temp/ssl_backbone_screen_20261026');CROSS=Path('D:/codex_temp/ssl_readout_factorial_20261027')
SCODE=W/'experiments/ssl_backbone_screen_20261026';CCODE=W/'experiments/ssl_readout_factorial_20261027'
sys.path[1:1]=[str(SCODE),str(CCODE)]
import artifacts as art
import downstream as down
from data_bridge import load_features,load_supervised,analysis_labels,seed_all,state_hash
from fit import fit_linear
from contract import tasks
art.ROOT=ROOT
read,sha,save,save_tensor=art.read,art.sha,art.save,art.save_tensor

def load(p):return torch.load(p,map_location='cpu',weights_only=True)
def verify(sig):
 for p,h in sig.items():
  if sha(p)!=h:raise RuntimeError('Changed source '+p)
def ref(receipt,d,s):return next(r for r in receipt['references'] if (r['dataset'],r['seed'])==(d,s))
def features_dir(d,s):return ROOT/'features'/d/f'seed{s}'
def folder(t,prefix='formal'):return ROOT/prefix/t['dataset']/f"seed{t['seed']}"/t['scale']/t['condition']/t['head']

def checked_head(p,files):
 r=read(p/'result.json')
 if (p/'complete.json').exists():
  m=read(p/'complete.json');pairs=list(m['files'].items());files.add(str(p/'complete.json'))
 else:pairs=[(n,r[k]) for n,k in [('student.pt','student_sha256'),('probabilities.pt','probabilities_sha256'),('target.pt','target_file_sha256'),('config.json','config.json_sha256'),('epochs.jsonl','epochs.jsonl_sha256')]]
 for name,h in pairs:assert sha(p/name)==h;files.add(str(p/name))
 files.add(str(p/'result.json'))

def prepare():
 ROOT.mkdir(parents=True,exist_ok=True);dest=ROOT/'receipt.json'
 if dest.exists():r=read(dest);verify(r['signature']);return r
 files={str(p.resolve()) for p in CODE.glob('*.py')}|{str(CODE/'config.json'),str(CCODE/'fit.py')}
 for base in [SCREEN,CROSS]:
  r=read(base/'receipt.json');verify(r['signature']);files.update(r['signature']);files.add(str(base/'receipt.json'))
 for item in read(art.PREP/'input_manifest.json')['files']:
  assert sha(item['path'])==item['sha256'];files.add(item['path'])
 refs=[];aliases=[]
 for d in CFG['new_datasets']:
  for s in CFG['seeds']:
   base=SCREEN/'downstream'/d/f'seed{s}'/'graphecl';r=read(base/'r2/result.json');enc=base/'r2/encoder.pt';assignment=base/'r2/support32.pt'
   assert r['status']=='complete' and sha(enc)==r['checkpoint_sha256'];verify(r['source_hashes']);files.update(r['source_hashes'])
   hp=[p for p in (SCREEN/'formal/ssl'/d/f'seed{s}'/'graphecl').glob('H_epoch*.pt') if sha(p)==r['h_sha256']];assert len(hp)==1
   files.update(map(str,[enc,assignment,base/'r2/result.json',hp[0]]))
   rr=dict(dataset=d,seed=s,classes=r['classes'],encoder=str(enc),assignment=str(assignment),heads={})
   for c in CFG['conditions']:
    rr['heads'][c]={}
    for head in CFG['heads']:
     p=base/'conditions'/c/head;checked_head(p,files);rr['heads'][c][head]=str(p)
     aliases.append(dict(task=dict(dataset=d,seed=s,scale='raw',condition=c,head=head),path=str(p)))
   refs.append(rr)
 cs=read(CROSS/'analysis/scores.json');files.add(str(CROSS/'analysis/scores.json'))
 cr=read(CROSS/'receipt.json')
 for row in cs:
  if row['dataset'] in CFG['reuse_datasets'] and row['initialization']=='legacy' and row['condition'] in CFG['conditions']:
   p=Path(row['source']);checked_head(p,files);aliases.append(dict(task={k:row[k] for k in ['dataset','seed','scale','condition','head']},path=str(p)))
 for rr in cr['references']:
  if rr['dataset'] in CFG['reuse_datasets']:refs.append(dict(dataset=rr['dataset'],seed=rr['seed'],classes=rr['classes'],assignment=rr['assignment'],reused=True))
 r=dict(config=CFG,references=refs,tasks=tasks(CFG),aliases=aliases,signature={p:sha(p) for p in sorted(files)},created=time.strftime('%Y-%m-%d %H:%M:%S'))
 assert len(r['tasks'])==36 and len(aliases)==84 and len(refs)==10
 save(dest,r);return r

def extract(receipt):
 for rr in receipt['references']:
  if rr.get('reused'):continue
  d,s=rr['dataset'],rr['seed'];dest=features_dir(d,s);expected=dict(receipt_sha256=sha(ROOT/'receipt.json'),dataset=d,seed=s)
  if art.reusable(dest,expected):continue
  if dest.exists() and any(dest.iterdir()):raise RuntimeError('Incomplete extraction retained')
  enc=load(rr['encoder']);x=load_features(d,s)['x'];seed_all(s)
  m=down._CORE.AlignedMLP(x.shape[1],rr['classes'],enc['projection.weight'].shape[0],64,.5).cuda();down._copy_r2_encoder(m,enc)
  ih=state_hash(m.deployment_state());initial={k:v.detach().cpu().clone() for k,v in m.backbone.layers[-1].state_dict().items()}
  for c in CFG['conditions']:
   for head in CFG['heads']:assert ih==read(Path(rr['heads'][c][head])/'result.json')['initial_hash']
  m.freeze_encoder();before=state_hash(m.encoder_state())
  with torch.no_grad():raw=m.encode(x.cuda()).cpu()
  assert before==state_hash(m.encoder_state())
  unit=F.normalize(raw,dim=1);p=Path(rr['heads']['uniform0.4']['CE']);old=load(p/'student.pt')
  with torch.no_grad():replay=F.linear(raw.cuda(),old['layers.1.weight'].cuda(),old['layers.1.bias'].cuda()).softmax(1).cpu()
  error=float((replay-load(p/'probabilities.pt')).abs().max());assert error<=CFG['replay_atol']
  for name,value in [('raw.pt',raw),('unit.pt',unit),('initial.pt',initial)]:save_tensor(dest/name,value)
  save(dest/'stats.json',dict(dataset=d,seed=s,initial_deployment_hash=ih,initial_head_hash=state_hash(initial),encoder_hash=before,extraction_error=error,nodes=len(raw)))
  art.complete(dest,expected,['raw.pt','unit.pt','initial.pt','stats.json']);print('EXTRACTED',d,s,error,flush=True)
  del m,x,raw,unit,replay;gc.collect();torch.cuda.empty_cache()
 sig={str(p):sha(p) for rr in receipt['references'] if not rr.get('reused') for p in features_dir(rr['dataset'],rr['seed']).iterdir() if p.is_file()}
 obj=dict(receipt_sha256=sha(ROOT/'receipt.json'),signature=sig);path=ROOT/'features_locked.json'
 if path.exists():assert read(path)==obj
 else:save(path,obj)

def fit(t,receipt,prefix='formal'):
 d,s,c,head=t['dataset'],t['seed'],t['condition'],t['head'];rr=ref(receipt,d,s);f=features_dir(d,s)
 hpath=f/(t['scale']+'.pt');initial=load(f/'initial.pt');target_path=Path(rr['heads'][c][head])/'target.pt';target=load(target_path);data=load_supervised(d,s,c)
 expected=F.one_hot(data['observed'],data['classes']).float()
 if head=='SUPPORT32':expected=down.build_support32_targets(data['observed'],load(rr['assignment'])[data['train']],classes=data['classes'])
 assert torch.equal(expected,target)
 cfg=dict(t,receipt_sha256=sha(ROOT/'receipt.json'),feature_path=str(hpath),feature_sha256=sha(hpath),initial_hash=state_hash(initial),target_source=str(target_path),target_source_sha256=sha(target_path),epochs=200,lr=.01,weight_decay=.0005)
 return fit_linear(folder(t,prefix),load(hpath),target,data,initial,cfg)

def smoke(receipt):
 extract(receipt);checks=[]
 for rr in receipt['references']:
  if rr.get('reused'):continue
  for head in CFG['heads']:
   t=dict(dataset=rr['dataset'],seed=rr['seed'],scale='raw',condition='uniform0.4',head=head);r=fit(t,receipt,'engineering');p=folder(t,'engineering');old=Path(rr['heads']['uniform0.4'][head])
   error=float((load(p/'probabilities.pt')-load(old/'probabilities.pt')).abs().max())
   a=[json.loads(v)['validation_correct'] for v in (p/'epochs.jsonl').read_text().splitlines()];b=[json.loads(v)['validation_correct'] for v in (old/'epochs.jsonl').read_text().splitlines()]
   check=dict(task=t,probability_error=error,validation_trajectory_match=a==b,best_epoch_match=r['best_epoch']==read(old/'result.json')['best_epoch']);checks.append(check);save(ROOT/'engineering/replay_progress.json',checks)
   assert error<=CFG['replay_atol'] and a==b and check['best_epoch_match'];print('REPLAY',json.dumps(check),flush=True)
 save(ROOT/'engineering/smoke_complete.json',dict(receipt_sha256=sha(ROOT/'receipt.json'),features_lock_sha256=sha(ROOT/'features_locked.json'),checks=checks))

def execute(receipt):
 gate=read(ROOT/'engineering/smoke_complete.json');assert len(gate['checks'])==12 and gate['receipt_sha256']==sha(ROOT/'receipt.json') and gate['features_lock_sha256']==sha(ROOT/'features_locked.json')
 verify(read(ROOT/'features_locked.json')['signature']);done=[];failures=[]
 for t in tasks(CFG):
  save(ROOT/'formal/status.json',dict(status='running',completed=len(done),total=36,task=t,failures=failures,pid=os.getpid()))
  try:r=fit(t,receipt);done.append(t);print('COMPLETE',json.dumps(t),r['valid_correct'],flush=True)
  except Exception:
   e=dict(task=t,traceback=traceback.format_exc());failures.append(e);save(folder(t)/'failure.json',e);print(e['traceback'],flush=True)
  finally:gc.collect();torch.cuda.empty_cache()
 verify(receipt['signature']);verify(read(ROOT/'features_locked.json')['signature']);save(ROOT/'formal/fits_locked.json',dict(receipt_sha256=sha(ROOT/'receipt.json'),completed=done,failures=failures))
 from analyze import analyze
 analyze()

@contextmanager
def lock():
 import msvcrt
 ROOT.mkdir(parents=True,exist_ok=True)
 with (ROOT/'runner.lock').open('a+b') as f:
  f.seek(0);f.write(b'0');f.flush();f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
  try:yield
  finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)

def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','smoke','all']);args=p.parse_args()
 with lock():
  r=prepare();torch.set_num_threads(CFG['threads']);torch.use_deterministic_algorithms(True)
  if args.command=='prepare':print(json.dumps(dict(tasks=len(r['tasks']),aliases=len(r['aliases']),hashes=len(r['signature']))));return
  if not torch.cuda.is_available():raise RuntimeError('GPU unavailable')
  smoke(r) if args.command=='smoke' else execute(r)

if __name__=='__main__':sys.modules['run']=sys.modules[__name__];main()
