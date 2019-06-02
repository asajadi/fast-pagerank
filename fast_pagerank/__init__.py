#from .pagerank import Page
name = "fast_pagerank"
from .fast_pagerank import pagerank
from .fast_pagerank import pagerank_power
__all__ = ["pagerank", "pagerank_power"]