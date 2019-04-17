"""Two "fast" implementations of PageRank.

Pythom implementations of Matlab original in:
Cleve Moler, Experiments with MATLAB.
"""
# uncomment
from __future__ import division

import scipy as sp
import scipy.sparse as sprs
import scipy.spatial
import scipy.sparse.linalg

__author__ = "Armin Sajadi"
__copyright__ = "Copyright 215, The Wikisim Project"
__credits__ = ["Armin Sajadi"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Armin Sajadi"
__email__ = "sajadi@cs.dal.ca"
__status__ = "Development"


def pagerank(G, p=0.85,
             personalize=None, reverse=False):
    """ Calculates pagerank given a csr graph

    Inputs:
    -------

    G: a csr graph.
    p: damping factor
    personlize: if not None, should be an array with the size of the nodes
                containing probability distributions.
                It will be normalized automatically
    reverse: If true, returns the reversed-pagerank

    outputs
    -------

    Pagerank Scores for the nodes

    """
    # In Moler's algorithm, $G_{ij}$ represents the existences of an edge
    # from node $j$ to $i$, while we have assumed the opposite!
    if not reverse:
        G = G.T

    n, _ = G.shape
    c = sp.asarray(G.sum(axis=0)).reshape(-1)

    k = c.nonzero()[0]

    D = sprs.csr_matrix((1 / c[k], (k, k)), shape=(n, n))

    if personalize is None:
        personalize = sp.ones(n)
    personalize = personalize.reshape(n, 1)
    e = (personalize / personalize.sum()) * n

    I = sprs.eye(n)
    x = sprs.linalg.spsolve((I - p * G.dot(D)), e)

    x = x / x.sum()
    return x


def pagerank_power(G, p=0.85, max_iter=100,
                   tol=1e-06, personalize=None, reverse=False):
    """ Calculates pagerank given a csr graph

    Inputs:
    -------
    G: a csr graph.
    p: damping factor
    max_iter: maximum number of iterations
    personlize: if not None, should be an array with the size of the nodes
                containing probability distributions.
                It will be normalized automatically.
    reverse: If true, returns the reversed-pagerank

    Returns:
    --------
    Pagerank Scores for the nodes

    """
    # In Moler's algorithm, $G_{ij}$ represents the existences of an edge
    # from node $j$ to $i$, while we have assumed the opposite!
    if not reverse:
        G = G.T

    n, _ = G.shape
    c = sp.asarray(G.sum(axis=0)).reshape(-1)

    k = c.nonzero()[0]

    D = sprs.csr_matrix((1 / c[k], (k, k)), shape=(n, n))

    if personalize is None:
        personalize = sp.ones(n)
    personalize = personalize.reshape(n, 1)
    e = (personalize / personalize.sum()) * n

    z = (((1 - p) * (c != 0) + (c == 0)) / n)[sp.newaxis, :]
    G = p * G.dot(D)

    x = e / n
    oldx = sp.zeros((n, 1))

    iteration = 0

    while sp.linalg.norm(x - oldx) > tol:
        oldx = x
        x = G.dot(x) + e.dot(z.dot(x))
        iteration += 1
        if iteration >= max_iter:
            break
    x = x / sum(x)

    return x.reshape(-1)
