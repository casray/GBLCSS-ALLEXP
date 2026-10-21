import copy
import json
from pathlib import Path
import tempfile
import unittest
import sys

import run_suite
import summarize


CONFIG = Path(__file__).with_name('config_v2.json')


class SuiteTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG.read_text(encoding='utf-8'))

    def test_manifest_count_unique_and_seed_first(self):
        jobs = run_suite.build_jobs(self.config)
        self.assertEqual(len(jobs), 1080)
        self.assertEqual(len({j['id'] for j in jobs}), len(jobs))
        self.assertEqual(jobs, run_suite.build_jobs(self.config))
        self.assertEqual([j['seed'] for j in jobs], sorted(j['seed'] for j in jobs))
        self.assertEqual(jobs[0]['dataset'], 'cora')
        self.assertEqual(jobs[0]['noise'], 'u0.4')
        self.assertEqual(jobs[0]['backbone'], 'cached_mlp')

    def test_a_once_and_cost_only_arms(self):
        jobs = run_suite.build_jobs(self.config)
        a_jobs = [j for j in jobs if j['arm'] == 'A']
        keys = [(j['dataset'], j['noise'], j['seed'], j['backbone']) for j in a_jobs]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(all(j['ratio'] == .1 for j in a_jobs))
        self.assertEqual({j['arm'] for j in jobs if j['dataset'] == 'coauthorcs'}, {'A', 'C', 'D'})

    def test_e2_only_new_c_rules_and_main_ratio(self):
        jobs = run_suite.build_jobs(self.config)
        e2 = [j for j in jobs if j['experiment'] == 'E2']
        self.assertEqual(len(e2), 48)
        self.assertEqual({j['rule'] for j in e2}, {'R3', 'O2'})
        self.assertTrue(all(j['arm'] == 'C' and j['ratio'] == .1 for j in e2))
        self.assertTrue(all(j['rule'] == 'R0' for j in jobs if j['experiment'] == 'E0'))
        first_e2 = next(i for i, j in enumerate(jobs) if j['experiment'] == 'E2')
        self.assertEqual(first_e2, 12)

    def test_completed_result_rejects_wrong_config_job_and_source(self):
        job = run_suite.build_jobs(self.config)[0]
        result = {'status': 'completed', 'job': job, 'config_sha256': 'cfg', 'source_sha256': 'src'}
        run_suite.validate_completed_result(result, job, 'cfg', 'src')
        for field, value in [('config_sha256', 'wrong'), ('source_sha256', 'wrong'), ('job', {})]:
            bad = copy.deepcopy(result)
            bad[field] = value
            with self.assertRaises(ValueError):
                run_suite.validate_completed_result(bad, job, 'cfg', 'src')

    def test_protocol_refuses_source_or_configuration_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source'
            source.mkdir()
            module = source / 'module.py'
            module.write_text('x = 1\n', encoding='utf-8')
            config = root / 'input.json'
            config.write_text('{"a": 1}\n', encoding='utf-8')
            run_suite.freeze_protocol(config, root / 'run', source)
            run_suite.freeze_protocol(config, root / 'run', source)
            module.write_text('x = 2\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                run_suite.freeze_protocol(config, root / 'run', source)

    def test_source_fingerprint_includes_imported_support_files(self):
        hashes, _ = run_suite.source_fingerprint(Path(__file__).parent)
        for name in ['__init__.py', 'data.py', 'granular_ball.py', 'targets.py']:
            self.assertIn('support/gblcss/' + name, hashes)

    def test_pairs_use_exact_seeds_and_preserve_negative_delta(self):
        rows = []
        for seed, arm, accuracy in [(0, 'C', .7), (1, 'B', .9), (0, 'B', .8), (2, 'C', .99)]:
            rows.append({'job': {'dataset': 'cora', 'noise': 'u0.4', 'seed': seed,
                                'backbone': 'gcn', 'arm': arm, 'ratio': .1, 'rule': 'R0'},
                         'test_accuracy': accuracy})
        pairs = summarize.paired_rows(rows)
        row = next(row for row in pairs if row['comparison'] == 'C_R0-B_R0')
        self.assertEqual(row['n_pairs'], 1)
        self.assertAlmostEqual(row['delta_pp_mean'], -10)
        self.assertEqual(json.loads(row['paired_seeds']), [0])

    def test_cost_only_and_sensitivity_pair_with_exact_a_reference(self):
        rows = []
        for dataset, seed, arm, ratio, accuracy in [('coauthorcs', 0, 'A', .1, .7),
                ('coauthorcs', 0, 'C', .1, .6), ('coauthorcs', 0, 'D', .1, .8),
                ('corafull', 1, 'A', .1, .7), ('corafull', 1, 'C', .05, .75),
                ('corafull', 2, 'C', .3, .9)]:
            rows.append({'job': {'dataset': dataset, 'noise': 'u0.4', 'seed': seed,
                                'backbone': 'gcn', 'arm': arm, 'ratio': ratio, 'rule': 'R0'},
                         'test_accuracy': accuracy})
        pairs = summarize.paired_rows(rows)
        cs = {r['comparison']: r for r in pairs if r['dataset'] == 'coauthorcs'}
        self.assertAlmostEqual(cs['C_R0-A_R0']['delta_pp_mean'], -10)
        self.assertAlmostEqual(cs['D_R0-A_R0']['delta_pp_mean'], 10)
        sensitivity = [r for r in pairs if r['dataset'] == 'corafull']
        self.assertEqual(len(sensitivity), 1)
        self.assertEqual(sensitivity[0]['ratio'], .05)
        self.assertEqual(sensitivity[0]['reference_ratio'], .1)
        self.assertAlmostEqual(sensitivity[0]['delta_pp_mean'], 5)

    def test_prediction_artifact_integrity_in_resume_and_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'results').mkdir()
            job = run_suite.build_jobs(self.config)[0]
            prediction = root / 'results' / 'prediction.npz'
            prediction.write_bytes(b'original predictions')
            result = {'status': 'completed', 'job': job, 'config_sha256': 'cfg', 'source_sha256': 'src',
                      'prediction_file': prediction.name, 'prediction_sha256': run_suite.sha256_bytes(prediction.read_bytes())}
            run_suite.validate_completed_result(result, job, 'cfg', 'src', root)
            (root / 'jobs.json').write_text(json.dumps([job]), encoding='utf-8')
            (root / 'results' / (job['id'] + '.json')).write_text(json.dumps(result), encoding='utf-8')
            self.assertEqual(summarize.summarize(root)['completed'], 1)
            prediction.write_bytes(b'changed predictions')
            with self.assertRaises(ValueError):
                run_suite.validate_completed_result(result, job, 'cfg', 'src', root)
            self.assertEqual(summarize.summarize(root)['completed'], 0)
            prediction.unlink()
            with self.assertRaises(ValueError):
                run_suite.validate_completed_result(result, job, 'cfg', 'src', root)
            outside = root / 'outside.npz'
            outside.write_bytes(b'original predictions')
            result['prediction_file'] = '../outside.npz'
            with self.assertRaises(ValueError):
                run_suite.validate_completed_result(result, job, 'cfg', 'src', root)

    def test_partial_summary_handles_missing_fields_and_failed_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'results').mkdir()
            jobs = run_suite.build_jobs(self.config)[:2]
            (root / 'jobs.json').write_text(json.dumps(jobs), encoding='utf-8')
            result = {'status': 'completed', 'job': jobs[0], 'test_accuracy': .7}
            (root / 'results' / (jobs[0]['id'] + '.json')).write_text(json.dumps(result), encoding='utf-8')
            (root / 'status.json').write_text(json.dumps({'status': 'failed', 'reason': 'test failure'}), encoding='utf-8')
            info = summarize.summarize(root)
            self.assertEqual(info['completed'], 1)
            self.assertEqual(info['total'], 2)
            report = (root / 'PROGRESS.md').read_text(encoding='utf-8')
            self.assertIn('test failure', report)
            self.assertIn('NA', (root / 'summary.csv').read_text(encoding='utf-8'))
            self.assertTrue((root / 'runs.csv').exists())

    def test_cpu_child_queue_cost_resume_and_timeout(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source'
            source.mkdir()
            worker = source / 'run_arm.py'
            worker.write_text('''import argparse, hashlib, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--config'); p.add_argument('--job'); p.add_argument('--run-root')
a = p.parse_args(); r = Path(a.run_root)
j = json.loads(Path(a.job).read_text(encoding='utf-8'))
protocol = json.loads((r / 'protocol.json').read_text(encoding='utf-8'))
out = {'status':'completed', 'job':j, 'config_sha256':hashlib.sha256(Path(a.config).read_bytes()).hexdigest(), 'source_sha256':protocol['source_sha256'], 'test_accuracy':.5, 'run_total_s':.01, 'cold_total_s':2.0, 'shared_cold_s':1.2}
(r / 'results' / (j['id'] + '.json')).write_text(json.dumps(out), encoding='utf-8')
''', encoding='utf-8')
            source.joinpath('prepare.py').write_text('pass\n', encoding='utf-8')
            config = root / 'input.json'
            config.write_text(json.dumps(self.config), encoding='utf-8')
            run = root / 'run'
            result = run_suite.run_suite(config, run, limit=1, source_dir=source)
            self.assertEqual(result['status'], 'completed')
            data = json.loads(next((run / 'results').glob('*.json')).read_text(encoding='utf-8'))
            self.assertEqual(data['body_cold_total_s'], 2.0)
            self.assertGreaterEqual(data['startup_overhead_s'], 0)
            self.assertAlmostEqual(data['cold_total_s'], 2.0 + data['startup_overhead_s'])
            self.assertAlmostEqual(data['amortized_total_s']['1'], data['cold_total_s'])
            self.assertAlmostEqual(data['amortized_total_s']['3'], data['cold_total_s'] - 1.2 + 1.2 / 3)
            self.assertEqual(run_suite.run_suite(config, run, limit=1, source_dir=source)['completed'], 1)
            with self.assertRaises(TimeoutError):
                run_suite._execute_child([sys.executable, '-c', 'import time; time.sleep(20)'],
                                         root / 'timeout.log', .05, root / 'timeout.json', {})


if __name__ == '__main__':
    unittest.main()
