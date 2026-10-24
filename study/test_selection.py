import unittest
import numpy as np
from selection import select, ktv_frames, MODES

class SelectionTests(unittest.TestCase):
    def test_fixed_budget_and_collisions(self):
        for n in (6, 7, 30):
            for anchor in range(n):
                scores = np.zeros(n); scores[anchor] = 1
                for mode in MODES:
                    s = select(mode, [0,1,2,3], [0,1,2,3,4,5], scores, np.arange(n)*.1, "what happened after the event")
                    self.assertEqual(len(set(s['indices'])), 6)
                    self.assertEqual(s['indices'], sorted(s['indices']))
                    self.assertEqual(sum(s['tokens_per_frame']), 936)
                    self.assertEqual(sorted(s['ranks']), list(range(6)))
                    if mode.startswith('hybrid'):
                        self.assertTrue(set([0,1,2,3]).issubset(s['indices']))
                        self.assertIn(anchor, s['indices'])
    def test_neighbor_direction(self):
        scores = np.zeros(40); scores[20] = 1
        for word, expected in [('before',10),('after',30)]:
            s = select('hybrid4_neighbor',[0,1,2,3],[0,1,2,3,4,5],scores,np.arange(40)*.1,word)
            self.assertEqual(s['neighbor'],expected)
    def test_degenerate(self):
        self.assertEqual(len(ktv_frames(np.ones((8,3)),6)),6)
    def test_upstream_clustering_parity(self):
        from sklearn.cluster import KMeans
        f = np.random.default_rng(42).normal(size=(50,8))
        k = KMeans(n_clusters=6,random_state=0,init='k-means++',n_init=10).fit(f)
        expected = sorted(np.argmin(np.linalg.norm(f-k.cluster_centers_[:,None,:],axis=2),axis=1).tolist())
        self.assertEqual(ktv_frames(f,6), expected)

if __name__ == '__main__': unittest.main()
