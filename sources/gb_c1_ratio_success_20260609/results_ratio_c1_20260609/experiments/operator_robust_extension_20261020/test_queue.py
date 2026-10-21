import json
from pathlib import Path
import tempfile
import unittest
import os
import psutil

from experiment import file_hash
import run_queue
import prepare_data


class QueueTests(unittest.TestCase):
    def test_freeze_rejects_config_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);config=p/'input.json';config.write_text('{}',encoding='utf8')
            run_queue.freeze(config,p/'run')
            config.write_text('{"x":1}',encoding='utf8')
            with self.assertRaises(ValueError):
                run_queue.freeze(config,p/'run')

    def test_resume_checks_same_size_original_input_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);old=p/'old';old.mkdir();(old/'x.npy').write_bytes(b'abcd')
            (old/'metadata.json').write_text('{}',encoding='utf8')
            config=p/'config.json';config.write_text(json.dumps({'source_cache_root':str(p/'unused')}),encoding='utf8')
            cache=p/'run'/'cache'/'fake';cache.mkdir(parents=True)
            m=dict(config_sha256=file_hash(config),old_directory=str(old),new_files={},
                original_cache_files={'x.npy':{'bytes':4,'sha256':file_hash(old/'x.npy')}},
                original_metadata_sha256=file_hash(old/'metadata.json'))
            (cache/'metadata.json').write_text(json.dumps(m),encoding='utf8')
            (old/'x.npy').write_bytes(b'zzzz')
            with self.assertRaisesRegex(ValueError,'original'):
                prepare_data.prepare(config,p/'run','fake')

    def test_artifact_digest_checked_before_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'results').mkdir();(root/'checkpoints').mkdir()
            pp=root/'results'/'pred.npz';cp=root/'checkpoints'/'model.pt'
            pp.write_bytes(b'pred');cp.write_bytes(b'model')
            (root/'traces').mkdir();tp=root/'traces'/'x.json';tp.write_bytes(b'[]')
            j={'id':'x'};protocol={'config_sha256':'c','source_sha256':'s'}
            r=dict(status='completed',job=j,**protocol,prediction_file='pred.npz',prediction_sha256=file_hash(pp),
                checkpoint_file='checkpoints/model.pt',checkpoint_sha256=file_hash(cp),
                trace_file='traces/x.json',trace_sha256=file_hash(tp))
            run_queue.validate(r,j,protocol,root)
            tp.write_bytes(b'[0]')
            with self.assertRaises(ValueError):
                run_queue.validate(r,j,protocol,root)
            tp.write_bytes(b'[]')
            pp.write_bytes(b'evil')
            with self.assertRaises(ValueError):
                run_queue.validate(r,j,protocol,root)

    def test_queue_rejects_second_active_owner(self):
        self.assertTrue(hasattr(run_queue,'claim_lock'),'queue locking not implemented')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            run_queue.claim_lock(root)
            with self.assertRaises(RuntimeError):
                run_queue.claim_lock(root)

    def test_dead_parent_does_not_release_live_worker_ownership(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            self.assertFalse(psutil.pid_exists(2147483646))
            (root/'queue.lock').write_text(json.dumps(dict(pid=2147483646,created=0,
                child_pid=os.getpid(),child_created=psutil.Process().create_time())),encoding='utf8')
            with self.assertRaisesRegex(RuntimeError,'worker'):
                run_queue.claim_lock(root)


if __name__=='__main__':
    unittest.main()
