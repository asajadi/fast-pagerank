# Fast Personalized Pagerank Implementation (Moler PageRank)
I needed a fast PageRank for Wikisim project, it has to be fast enough that can run in real time on relatively small graphs. I started from optimizing the networkx, however, I found a very nice algorithm by **Cleve Mole** which takes the full advantage of sparse matrix operations. 
I implemented two versions of the algorithm in Python, both inspired  by the sparse fast solutions given in **Cleve Moler**'s book, [*Experiments with MATLAB*](http://www.mathworks.com/moler/index_ncm.html). The power method is much faster with enough precision for our task. Our benchmarsk shows that this implementation is **faster than networkx** implementation magnititude of times

## Personalized Pagerank
I modified the algorithm a little bit to be able to calculate **personalized Pagerank** as well. 

## Input Format
The input is a 2d array, each row of the array is an edge of the graph $[[a,b], [c,d]]$, $a$ and $b$ are the node numbers. The **personalization vector** is probability distribution over the nodes, a.k.a **teleporting vector**.
## Comparison with Networkx
Both of the implementation (Exact Solution and PowerMethod) are much faster than their correspondent method in NetworkX
