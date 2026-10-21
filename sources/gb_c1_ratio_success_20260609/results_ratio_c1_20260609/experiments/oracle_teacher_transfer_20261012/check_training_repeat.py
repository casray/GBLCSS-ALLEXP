import json

import torch

import run_oracle as run


def main():
    torch.set_num_threads(4)
    folder = run.ROOT / 'engineering' / 'short_repeat'
    folder.mkdir(parents=True, exist_ok=True)
    observations = []
    for tag in ('original1', 'original2', 'sanitized'):
        if tag == 'sanitized':
            data, _ = run.data_for('cora', .4, 'A_noisy')
        else:
            data, _, _ = run.common.load_data('cora', .4, 1)
        data = run.previous.gpu_data(data)
        run.previous.set_seed(1)
        with (folder / f'{tag}.jsonl').open('w', encoding='utf-8') as log:
            model, probability, epoch, settings = run.previous.train_erase(data, 3, log)
        torch.save(probability.cpu(), folder / f'{tag}.pt')
        observations.append(dict(tag=tag, losses=[r['loss'] for r in map(json.loads,
                     (folder / f'{tag}.jsonl').read_text().splitlines())]))
        del model, data, probability
        torch.cuda.empty_cache()
    pairs = []
    for name in ('original2', 'sanitized'):
        a = torch.load(folder / 'original1.pt', weights_only=True)
        b = torch.load(folder / f'{name}.pt', weights_only=True)
        pairs.append(dict(first='original1', second=name, max_difference=float((a-b).abs().max()),
                          argmax_differences=int((a.argmax(1) != b.argmax(1)).sum())))
    run.write(folder / 'result.json', dict(traces=observations, comparisons=pairs,
          purpose='isolate retraining variability; no test-label accuracy computed'))
    print(observations, pairs, flush=True)


if __name__ == '__main__':
    main()
