import importlib.util
from pathlib import Path
import sys
import unittest

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
spec=importlib.util.spec_from_file_location('csphoto_contract',CODE/'run.py')
run=importlib.util.module_from_spec(spec);sys.modules[spec.name]=run;spec.loader.exec_module(run)


class Contracts(unittest.TestCase):
    def test_grid(self):
        self.assertEqual(len(run.base.expected(new=True)),60)
        self.assertEqual(run.CONFIG['seeds'],[0,1,2,3,4])

    def test_roots(self):
        self.assertEqual(run.base.ROOT,run.ROOT)
        self.assertEqual(run.base.fitter.ROOT,run.ROOT)
        with self.assertRaises(ValueError):run.p.output(run.CODE.parent/'forbidden.txt')

    def test_no_test_training(self):
        with self.assertRaises(ValueError):run.p.training_data({'test_y':None})

    def test_seed_first_summary(self):
        a=run.base.imported('csphoto_analysis_test',CODE/'analysis.py')
        rs=[dict(dataset='coauthorcs',condition=c,seed=s,route='x',kind='student',accuracy=v)
            for s,vs in [(0,[0.,100.]),(1,[2.,102.])] for c,v in zip(['uniform0.4','uniform0.6'],vs)]
        rows=a.summarize(rs,['dataset','condition','route','kind'],['accuracy'],[0,1])
        row=next(r for r in rows if r['dataset']=='ALL' and r['condition']=='NOISY_MACRO')
        self.assertEqual(row['accuracy_mean'],51.)
        self.assertAlmostEqual(row['accuracy_std_ddof1'],2**.5)

    def test_queue_owner_family(self):
        queue=run.base.imported('csphoto_queue_test',CODE/'gpu_queue.py')
        rows=[dict(ProcessId=10,ParentProcessId=1),dict(ProcessId=11,ParentProcessId=10),
              dict(ProcessId=12,ParentProcessId=11),dict(ProcessId=99,ParentProcessId=1)]
        self.assertEqual(queue.other_owners(rows,11),[rows[-1]])


if __name__=='__main__':unittest.main()
