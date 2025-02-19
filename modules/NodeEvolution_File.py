#! /usr/bin/env python3
'''
Niema Moshiri 2016

"NodeEvolution" module, where the phylogenetic tree(s) is/are loaded from file
in the Newick format. The module accepts a single file, so to include more than
one tree, all trees must be in the same file (one line per tree).

Leaf labels must be in the following format: "N?|INDIVIDUAL|TIME"
where "?" can be any integer (it's ignored), "INDIVIDUAL" is the label of an
infected individual in the contact network, and "TIME" is a sample time for
that individual. For example, N10|7|2.0 is a valid leaf label because N10 is
ignored (but it's N followed by an integer), 7 is the label of an individual in
the contact network, and 2.0 is the time at which individual 7 was sampled.
'''
from NodeEvolution import NodeEvolution
import modules.FAVITES_ModuleFactory as MF
import FAVITES_GlobalContext as GC
from os.path import abspath,expanduser

class NodeEvolution_File(NodeEvolution):
    def cite():
        return [GC.CITATION_FAVITES, GC.CITATION_TREESWIFT]

    def init():
        assert "ContactNetworkGenerator_File" in str(MF.modules['ContactNetworkGenerator']), "Must use ContactNetworkGenerator_File module"
        assert "EndCriteria_TransmissionFile" in str(MF.modules['EndCriteria']), "Must use EndCriteria_TransmissionFile module"
        assert "NumBranchSample_All" in str(MF.modules['NumBranchSample']), "Must use NumBranchSample_All module"
        assert "NumTimeSample_None" in str(MF.modules['NumTimeSample']), "Must use NumTimeSample_None module"
        assert "SeedSelection_TransmissionFile" in str(MF.modules['SeedSelection']), "Must use SeedSelection_TransmissionFile module"
        assert "TimeSample_None" in str(MF.modules['TimeSample']), "Must use TimeSample_None module"
        assert "TransmissionNodeSample_TransmissionFile" in str(MF.modules['TransmissionNodeSample']), "Must use TransmissionNodeSample_TransmissionFile module"
        assert "TransmissionTimeSample_TransmissionFile" in str(MF.modules['TransmissionTimeSample']), "Must use TransmissionTimeSample_TransmissionFile module"
        GC.tree_file = abspath(expanduser(GC.tree_file.strip()))
        if GC.tree_file.lower().endswith('.gz'):
            from gzip import open as gopen
            GC.tree_file = gopen(GC.tree_file)
        else:
            GC.tree_file = open(GC.tree_file)
        try:
            global read_tree_newick
            from treeswift import read_tree_newick
        except:
            from os import chdir
            chdir(GC.START_DIR)
            assert False, "Error loading TreeSwift. Install with: pip3 install treeswift"

    def evolve_to_current_time(node, finalize=False):
        # if it's not the end yet, just dummy
        if not finalize:
            if node is None:
                return
            for virus in node.viruses():
                virus.set_time(GC.time)
        # otherwise, store trees
        elif not hasattr(GC,'sampled_trees'):
            seed_to_root_virus = {v.get_name():GC.seed_to_first_virus[v].get_root() for u,v,t in GC.transmissions if u is None}
            inf_to_seed = {}
            for u,v,t in GC.transmissions:
                if u is None:
                    inf_to_seed[v.get_name()] = v.get_name()
                else:
                    inf_to_seed[v.get_name()] = inf_to_seed[u.get_name()]
            trees = {l.decode().strip() if isinstance(l,bytes) else l.strip() for l in GC.tree_file}
            trees = {tree for tree in trees if len(tree) != 0}
            GC.sampled_trees = set()
            for tree in trees:
                t = read_tree_newick(tree)
                seeds = {inf_to_seed[str(leaf).split('|')[1]] for leaf in t.traverse_leaves()}
                assert len(seeds) == 1, "More than 1 seed in tree: %s" % tree
                seed = seeds.pop()
                GC.sampled_trees.add((seed_to_root_virus[seed],tree))
            GC.PRUNE_TREES = False
