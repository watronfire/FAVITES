#! /usr/bin/env python3
'''
Niema Moshiri 2017

"TreeUnit" module, where each branch's length (in time units) is multiplied by
a rate that is sampled from a user-parameterized Gamma distribution.
'''
from TreeUnit import TreeUnit
import FAVITES_GlobalContext as GC

class TreeUnit_Gamma(TreeUnit):
    def cite():
        return GC.CITATION_NUMPY

    def init():
        try:
            global gamma
            from numpy.random import gamma
        except:
            from os import chdir
            chdir(GC.START_DIR)
            assert False, "Error loading NumPy. Install with: pip3 install numpy"
        try:
            global read_tree_newick
            from treeswift import read_tree_newick
        except:
            from os import chdir
            chdir(GC.START_DIR)
            assert False, "Error loading TreeSwift. Install with: pip3 install treeswift"
        GC.tree_rate_shape = float(GC.tree_rate_shape)
        assert GC.tree_rate_shape > 0, "tree_rate_shape must be positive"
        GC.tree_rate_scale = float(GC.tree_rate_scale)
        assert GC.tree_rate_scale > 0, "tree_rate_scale must be positive"

    def time_to_mutation_rate(tree):
        if not hasattr(GC,"NUMPY_SEEDED"):
            from numpy.random import seed as numpy_seed
            numpy_seed(seed=GC.random_number_seed)
            GC.random_number_seed += 1
            GC.NUMPY_SEEDED = True
        t = read_tree_newick(tree)
        for node in t.traverse_preorder():
            if node.edge_length is not None:
                node.edge_length *= gamma(shape=GC.tree_rate_shape,scale=GC.tree_rate_scale)
        return str(t)
