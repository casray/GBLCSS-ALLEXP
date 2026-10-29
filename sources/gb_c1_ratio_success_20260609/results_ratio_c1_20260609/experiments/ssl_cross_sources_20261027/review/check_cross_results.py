import json, statistics as st, hashlib
from pathlib import Path
R=Path('D:/codex_temp/ssl_cross_sources_20261027')
P=Path('D:/codex_temp/ssl_backbone_csphoto_20261027')
def read(p):return json.loads(p.read_text())
rows=read(R/'analysis/scores.json');old=read(P/'analysis/scores.json')
lookup={(x['dataset'],x['seed'],x['arm'],x['condition'],x['head']):x for x in old if x['path']=='R2'}
diff=0.
for r in rows:
    if not r['new']:
        o=lookup[(r['dataset'],r['seed'],r['features'],r['condition'],r['head'])]
        for k in ['test_accuracy','test_macro_f1','target_brier']:diff=max(diff,abs(r[k]-o[k]))
final=read(R/'formal/final.json');assert hashlib.sha256((R/'analysis/audit.json').read_bytes()).hexdigest()==final['audit_sha256']
assert len(rows)==96 and sum(r['new'] for r in rows)==32
table=[]
for ds,pair in [('coauthorcs',['grace','graphecl']),('amazonpho',['gca_degree','graphecl'])]:
    for f in pair:
        for t in pair:
            rr=[r for r in rows if (r['dataset'],r['features'],r['targets'],r['head'])==(ds,f,t,'SUPPORT32')]
            item=dict(dataset=ds,features=f,targets=t)
            for c in ['clean','uniform0.4','uniform0.6','pair0.4']:item[c]=100*st.mean(r['test_accuracy'] for r in rr if r['condition']==c)
            item['macro_f1']=100*st.mean(r['test_macro_f1'] for r in rr if r['condition']!='clean')
            table.append(item)
effect=[]
for ds,pair in [('coauthorcs',['grace','graphecl']),('amazonpho',['gca_degree','graphecl'])]:
    vals={}
    for f in pair:
        for t in pair:vals[(f,t)]=100*st.mean(r['test_accuracy'] for r in rows if (r['dataset'],r['features'],r['targets'],r['head'])==(ds,f,t,'SUPPORT32') and r['condition']!='clean')
    a,b=pair;aa,ab,ba,bb=[vals[k] for k in [(a,a),(a,b),(b,a),(b,b)]]
    effect.append(dict(dataset=ds,feature_effect_A_target=ba-aa,feature_effect_B_target=bb-ab,target_B_effect_A_feature=ab-aa,target_B_effect_B_feature=bb-ba,interaction=bb-ba-ab+aa,symmetric_feature_effect=(ba-aa+bb-ab)/2,symmetric_target_effect=(ab-aa+bb-ba)/2))
result=dict(alias_max_score_difference=diff,audit_sha_verified=True,table=table,effects=effect)
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
