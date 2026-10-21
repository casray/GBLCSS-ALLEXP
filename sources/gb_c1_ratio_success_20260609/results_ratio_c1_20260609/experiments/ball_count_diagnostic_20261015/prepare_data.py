"""Download only the three named official artifacts, pinned to one repository commit."""
import json
import urllib.request
from data_tools import ROOT,CONFIG,DATA,load_new_graph,describe,d


def fetch(url):
    return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'research-reproducibility'}),timeout=60).read()


def main():
    manifest=ROOT/'dataset_manifest.json'
    if manifest.exists():
        prior=json.loads(manifest.read_text(encoding='utf-8'))
        for name,item in prior['datasets'].items():
            if d.sha(DATA[name])!=item['sha256']:raise ValueError('Frozen dataset changed: '+name)
        print(json.dumps(prior,ensure_ascii=False),flush=True);return
    metadata=json.loads(fetch('https://api.github.com/repos/yandex-research/heterophilous-graphs/commits/main'))
    commit=metadata['sha']; records={}
    for name in CONFIG['new_datasets']:
        url=f'https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/{commit}/data/{name}.npz'
        content=fetch(url); path=DATA[name];path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and path.read_bytes()!=content:raise ValueError('Will not overwrite different data')
        path.write_bytes(content)
        graph=load_new_graph(path,name,0)
        with __import__('numpy').load(path,allow_pickle=False) as raw:split_count=len(raw['train_masks'])
        for seed in CONFIG['protocol_seeds']:load_new_graph(path,name,seed)
        records[name]=dict(source_url=url,sha256=d.sha(path),bytes=len(content),split_count=split_count,**describe(graph))
        print(json.dumps(dict(dataset=name,**records[name]),ensure_ascii=False),flush=True)
    d.write(manifest,dict(repository='yandex-research/heterophilous-graphs',commit=commit,datasets=records,
                         no_dataset_selected_by_test_accuracy=True))


if __name__=='__main__':main()
