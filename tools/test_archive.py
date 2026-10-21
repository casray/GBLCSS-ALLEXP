import tempfile
from pathlib import Path
import unittest

import build_archive as a


class ArchiveTests(unittest.TestCase):
    def row(self,path,category='text',size=100):
        return dict(relative=path,category=category,bytes=size)

    def test_weights_retained(self):
        self.assertEqual(a.policy(self.row('experiment/student.pt','model_or_array'))[0],'local_artifact')

    def test_dataset_json_retained(self):
        self.assertEqual(a.policy(self.row('external/repo/data/raw/data.json'))[0],'local_dataset')

    def test_dataset_loader_code_preserved(self):
        self.assertEqual(a.policy(self.row('external/repo/data/loader.py'))[0],'upload')

    def test_compressed_experiment_record_preserved(self):
        self.assertEqual(a.policy(self.row('experiment/raw_results.jsonl.gz','compressed_package'))[0],'upload')

    def test_package_duplicate_retained(self):
        self.assertEqual(a.policy(self.row('manuscript.zip','compressed_package'))[0],'local_package')

    def test_secret_signature_quarantines_without_returning_value(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'fixture.txt'
            path.write_text('gh'+'p_'+'0'*40)
            self.assertEqual(a.scan(path),['github_token'])

    def test_oversize_retained(self):
        self.assertEqual(a.policy(self.row('results.log',size=101*1024*1024))[0],'local_oversize')


if __name__=='__main__':
    unittest.main(verbosity=2)
