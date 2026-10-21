import tempfile
from pathlib import Path
import unittest
import torch
from prepare_greet import train_representation


class PretrainingTests(unittest.TestCase):
    def test_label_free_training_saves_frozen_export(self):
        torch.manual_seed(4); x=torch.rand(12,5)
        u=torch.arange(12); edges=torch.stack((torch.cat((u,u)),torch.cat(((u+1)%12,(u-1)%12))))
        se=torch.zeros(12,2); knn=torch.stack(((u+1)%12,(u-1)%12),1)
        config=dict(seed=0,export_seed=4,epochs=2,half_dim=4,projection_dim=4,layers=2,
            learning_rate=.001,highpass_alpha=.1,margin=.5,cl_rounds=2,batch_size=12,
            maskfeat_rates=[.1,.5],dropedge_rates=[.5,.1])
        with tempfile.TemporaryDirectory() as directory:
            summary=train_representation(x,edges,se,knn,config,Path(directory))
            self.assertEqual(summary['status'],'completed')
            self.assertEqual(summary['label_usage'],'none')
            h=torch.load(Path(directory)/'H.pt',weights_only=True)
            self.assertEqual(tuple(h.shape),(12,8))
            self.assertTrue(bool(torch.isfinite(h).all()))
            self.assertTrue((Path(directory)/'checkpoint.pt').exists())


if __name__=='__main__': unittest.main()
