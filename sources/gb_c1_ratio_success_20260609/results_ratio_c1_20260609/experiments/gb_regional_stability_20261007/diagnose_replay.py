import io
import json
from contextlib import redirect_stdout

import torch

import run_stability as run

run.old.require_gpu()
data, _ = run.previous.load_data('cora', .4)
data = run.previous.gpu_data(data)
cache = torch.load(run.old.cache_folder('cora', .4)/'regional.pt', map_location='cpu', weights_only=True)
teacher = cache['teacher'].cuda()
region = run.RegionalKL(teacher, cache['assignments']['R1'].cuda())
original = torch.load(run.old.run_folder('cora', .4, 'R1')/'student_probabilities.pt',
                      map_location='cpu', weights_only=True)
records, probabilities = [], []
for repeat in range(2):
    log = io.StringIO()
    with redirect_stdout(io.StringIO()):
        model, prob, epoch, initial = run.old.train_modified(data, teacher, region, 1., log)
    p = prob.detach().cpu()
    probabilities.append(p)
    rows = [json.loads(line) for line in log.getvalue().splitlines()]
    records.append(dict(repeat=repeat, best_epoch=epoch, initial=initial,
                        max_abs_from_original=float((p-original).abs().max()),
                        metrics=run.old.exact_metrics(prob,data),
                        validation=[r['valid'] for r in rows]))
    del model, prob
result = dict(original_runner_repeats=records,
              repeat_max_abs_difference=float((probabilities[0]-probabilities[1]).abs().max()),
              repeat_argmax_differences=int((probabilities[0].argmax(1)!=probabilities[1].argmax(1)).sum()),
              repeat_validation_mismatches=sum(a!=b for a,b in zip(records[0]['validation'],records[1]['validation'])))
run.write(run.ROOT/'replay_diagnosis.json',result)
for record in records:
    record.pop('validation')
print(json.dumps(result,indent=2),flush=True)
