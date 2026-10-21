import json
from pathlib import Path
import unittest

from experiment import VARIANTS, build_jobs, paired_summary


class ManifestTests(unittest.TestCase):
    def test_manifest_and_three_arm_plugin_control(self):
        config=json.loads((Path(__file__).parent/'config.json').read_text())
        jobs=build_jobs(config)
        self.assertEqual(len(jobs),192)
        self.assertEqual(len({j['id'] for j in jobs}),192)
        self.assertEqual(sum(VARIANTS[j['variant']]['family']=='tfr' for j in jobs),72)
        self.assertEqual(sum(VARIANTS[j['variant']]['family']=='ce' for j in jobs),120)
        self.assertEqual(VARIANTS['tfr_P_shared']['rho'],0)
        self.assertEqual(VARIANTS['tfr_S_shared']['mask'],'shared')
        self.assertEqual(VARIANTS['ce_M_shared']['rho'],.5)
        self.assertTrue(all(v['target']=='A' for v in VARIANTS.values()))

    def test_pairing_does_not_mix_protocol_or_seed(self):
        rows=[]
        for variant,acc in [('tfr_P_ind',.5),('tfr_P_shared',.51),('tfr_S_shared',.53)]:
            rows.append(dict(job=dict(dataset='cora',noise='u0.6',seed=0,variant=variant),test_accuracy=acc,
                             split_sha256='s',noise_sha256='n',config_sha256='c',source_sha256='p',assignment_sha256='a'))
        result=paired_summary(rows)
        item=next(r for r in result if r['comparison']=='tfr_S_shared-tfr_P_shared')
        self.assertAlmostEqual(item['delta_pp_mean'],2)
        rows[-1]['noise_sha256']='different'
        self.assertFalse(any(r['comparison']=='tfr_S_shared-tfr_P_shared' for r in paired_summary(rows)))


if __name__=='__main__':unittest.main()
