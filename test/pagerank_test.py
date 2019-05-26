
import os
import sys
import scipy as sp
import scipy.sparse as sparse
from numpy.testing import assert_allclose
import unittest

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..')))

from fastpagerank.pagerank import pagerank_power
from fastpagerank.pagerank import pagerank

class TestMolerPageRank(unittest.TestCase):
    def setUp(self):
        # ---G1---
        n1 = 5
        rows1 = [0, 1, 2, 2, 2, 3, 3, 4, 4, 4]
        cols1 = [1, 2, 1, 3, 4, 0, 2, 0, 2, 3]
        data1 = [
            0.4923, 0.0999, 0.2132, 0.0178, 0.5694,
            0.0406, 0.2047, 0.861, 0.3849, 0.4829]

        self.p1 = 0.83
        self.personalize1 = sp.array([0.6005, 0.1221, 0.2542, 0.4778, 0.4275])
        self.G1 = sparse.csr_matrix((data1, (rows1, cols1)), shape=(n1, n1))
        self.pr1 = sp.array([0.1592, 0.2114, 0.3085, 0.1, 0.2208])

        # ---G2---
        n2 = 10
        rows2 = [2, 2, 4, 5, 5, 5, 6, 6, 9, 9]
        cols2 = [4, 5, 5, 3, 4, 9, 1, 2, 2, 4]
        data2 = [
            0.4565, 0.2861, 0.573, 0.0025, 0.4829,
            0.3866, 0.3041, 0.3407, 0.2653, 0.8079]
        self.G2 = sparse.csr_matrix((data2, (rows2, cols2)), shape=(n2, n2))
        self.personalize2 = sp.array([0.8887, 0.6491, 0.7843, 0.7103, 0.7428,
                                      0.6632, 0.7351, 0.3006, 0.8722, 0.1652])
        self.p2 = 0.92
        self.pr2 = sp.array([0.0234, 0.0255, 0.0629, 0.0196, 0.3303,
                             0.3436, 0.0194, 0.0079, 0.023, 0.1445])

        # ---G3---
        n3 = 5
        rows3 = [2]
        cols3 = [4]
        data3 = [0.5441]
        self.G3 = sparse.csr_matrix((data3, (rows3, cols3)), shape=(n3, n3))

        self.personalize3 = sp.array([0.0884, 0.2797, 0.3093, 0.5533, 0.985])
        self.p3 = 0.81
        self.pr3 = sp.array([0.0358, 0.1134, 0.1254, 0.2244, 0.501])

        # ---G4---
        n4 = 5
        rows4 = []
        cols4 = []
        data4 = []
        self.G4 = sparse.csr_matrix((data4, (rows4, cols4)), shape=(n4, n4))

        self.personalize4 = sp.array([0.2534, 0.8945, 0.9562, 0.056, 0.9439])
        self.p4 = 0.70
        self.pr4 = sp.array([0.0816, 0.2882, 0.3081, 0.018, 0.3041])

        # ---G5---
        n5 = 0
        rows5 = []
        cols5 = []
        data5 = []
        self.G5 = sparse.csr_matrix((data5, (rows5, cols5)), shape=(n5, n5))

        self.personalize5 = sp.array([])
        self.p5 = 0.70
        self.pr5 = sp.array([])

    def test_pagerank_1(self):
        calculated_pagerank = pagerank(self.G1, p=self.p1,
                                       personalize=self.personalize1)
        assert_allclose(calculated_pagerank, self.pr1, rtol=0, atol=1e-04)

    def test_pagerank_2(self):

        calculated_pagerank = pagerank(self.G2, p=self.p2,
                                       personalize=self.personalize2)
        assert_allclose(calculated_pagerank, self.pr2, rtol=0, atol=1e-04)

    def test_single_edge(self):
        calculated_pagerank = pagerank(self.G3, p=self.p3,
                                       personalize=self.personalize3)
        assert_allclose(calculated_pagerank, self.pr3, rtol=0, atol=1e-04)

    def test_zero_edge(self):
        calculated_pagerank = pagerank(self.G4, p=self.p4,
                                       personalize=self.personalize4)
        assert_allclose(calculated_pagerank, self.pr4, rtol=0, atol=1e-04)

    def test_empty_graph(self):
        calculated_pagerank = pagerank(self.G5, p=self.p5,
                                       personalize=self.personalize5)
        self.assertEqual(calculated_pagerank.size, 0)

    def test_power_pagerank_1(self):
        calculated_pagerank = pagerank_power(self.G1, p=self.p1,
                                             personalize=self.personalize1)
        assert_allclose(calculated_pagerank, self.pr1, rtol=0, atol=1e-04)

    def test_power_pagerank_2(self):

        calculated_pagerank = pagerank_power(self.G2, p=self.p2,
                                             personalize=self.personalize2)
        assert_allclose(calculated_pagerank, self.pr2, rtol=0, atol=1e-04)

    def test_power_single_edge(self):
        calculated_pagerank = pagerank_power(self.G3, p=self.p3,
                                             personalize=self.personalize3)
        assert_allclose(calculated_pagerank, self.pr3, rtol=0, atol=1e-04)

    def test_power_zero_edge(self):
        calculated_pagerank = pagerank_power(self.G4, p=self.p4,
                                             personalize=self.personalize4)
        assert_allclose(calculated_pagerank, self.pr4, rtol=0, atol=1e-04)

    def test_power_empty_graph(self):
        calculated_pagerank = pagerank_power(self.G5, p=self.p5,
                                             personalize=self.personalize5)
        self.assertEqual(calculated_pagerank.size, 0)


#             assert_array_almost_equal(Ynx,  Yml, decimal = 5)
if __name__ == '__main__':
    unittest.main()