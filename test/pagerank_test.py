
import networkx as nx
import random
import timeit
import numpy as np
import igraph
from numpy.testing import assert_array_almost_equal

import os, sys
current = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(current, '..')
sys.path.insert(0,src_path)

from src.pagerank import *

def get_random_graph(min_size=100, max_size=300, min_sparsity = 0.1, max_sparsity = 0.5):
    ''' Creates a random graph and a teleport vector and output them in different formats for different algorithms
    
    Inputs
    ------
    
    min_size and max_size: The size of the graph will be a random number in the range of (min_size, max_size)
    min_sparsity and max_sparsity: The sparcity of the graph will be a random number in the range of (min_sparsity, max_sparsity)
    
    Returns
    -------
    
    nxG: A random Graph for NetworkX
    A: The equivallent csr Adjacency matrix, for our moler_pagerank
    iG: The equivallent iGraph
    customization_vector: Personalization probabily vector
    customization_dict: Personalization probabily vector, in the form of a dictionary for NetworkX
    
    '''
    passed=True
    G_size = random.randint(min_size,max_size)
    p=random.uniform(min_sparsity, max_sparsity)
    nxG = nx.fast_gnp_random_graph(G_size, p, seed=None, directed=True)
    for e in nxG.edges():
         nxG[e[0]][e[1]]['weight']=sp.rand()

    A=nx.to_scipy_sparse_matrix(nxG)

    iG=igraph.Graph(list(nxG.edges()), directed=True)
    iG.es['weight'] = A.data
    
    customization_vector = np.random.random(G_size)
    customization_dict = dict(enumerate(customization_vector.reshape(-1)))
    return nxG, A, iG, customization_vector, customization_dict

import unittest

class TestMolerPageRank(unittest.TestCase):
    def setUp(self):
        self.number_of_tests = 10
    def test_exact_pagerank(self):
        damping_factor = 0.85
        for i in range(self.number_of_tests):
            nxG, A, iG, customization_vector, customization_dict = get_random_graph()

            Xnx  = nx.pagerank_numpy(nxG, alpha=damping_factor, personalization = customization_dict) 
            Xnx =  np.array([v for k,v in Xnx.items() ])

            Xml =  moler_pagerank_sparse(A, p=damping_factor, personalize=customization_vector)

            assert_array_almost_equal(Xnx,  Xml, decimal = 5)
        
    def test_power_pagerank(self):
        damping_factor = 0.85
        tol = 1e-06
        for i in range(self.number_of_tests):
            nxG, A, iG, customization_vector, customization_dict = get_random_graph()

            Ynx =  nx.pagerank_scipy(nxG, alpha=damping_factor, tol=tol, personalization=customization_dict)
            Ynx =  np.array([v for k,v in Ynx.items() ])

            Yml =  moler_pagerank_sparse_power(A, p=damping_factor, tol=tol, personalize=customization_vector)


            assert_array_almost_equal(Ynx,  Yml, decimal = 5)
            
if __name__ == '__main__':
    unittest.main()