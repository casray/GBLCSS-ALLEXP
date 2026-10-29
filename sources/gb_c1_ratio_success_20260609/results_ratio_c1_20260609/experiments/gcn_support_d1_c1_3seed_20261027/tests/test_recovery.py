import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('recovery_contract', Path(__file__).resolve().parents[1]/'recover_interruption.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class RecoveryTests(unittest.TestCase):
    def test_retains_exact_prefix_bytes(self):
        raw = b''.join((json.dumps(dict(epoch=i,loss=i/10))+'\r\n').encode() for i in range(1,139))
        prefix, extra = r.prefix_bytes(raw,75)
        self.assertEqual(extra,63)
        self.assertEqual(prefix, b''.join(raw.splitlines(keepends=True)[:75]))
        self.assertEqual(json.loads(prefix.splitlines()[-1])['epoch'],75)

    def test_rejects_gap_or_short_log(self):
        for raw,epoch in ((b'{"epoch":2}\n',1),(b'{"epoch":1}\n',2)):
            with self.assertRaises(ValueError):
                r.prefix_bytes(raw,epoch)

    def test_unchanged_complete_prefix(self):
        raw = b'{"epoch":1}\n'
        self.assertEqual(r.prefix_bytes(raw,1),(raw,0))
