#! /usr/bin/env python3
'''
Niema Moshiri 2016

"SeedSequence" module, which samples a single viral sequence using a profile HMM
generated from a multiple sequence alignment from some public dataset, then
simulates a non-homogeneous Yule tree with "num_seeds" leaves and "seed_height" root-to-tip
distance, and then evolves the sequence down this tree under a GTR+Gamma model
to get seed sequences for each seed.
'''
from SeedSequence import SeedSequence
from SeedSequence_Virus import SeedSequence_Virus
from SequenceEvolution_GTRGammaSeqGen import SequenceEvolution_GTRGammaSeqGen
import FAVITES_GlobalContext as GC
import modules.FAVITES_ModuleFactory as MF
from math import exp
from os.path import expanduser
from os import chdir,makedirs
from subprocess import CalledProcessError,check_output
from treesap import nonhomogeneous_yule_tree
from treeswift import read_tree_newick

OUT_FOLDER = "seed_sequences"
class SeedSequence_VirusNonHomYuleHeightGTRGamma(SeedSequence):
    def cite():
        return [GC.CITATION_HMMER, GC.CITATION_TREESWIFT, GC.CITATION_SEQGEN]

    def init():
        SeedSequence_Virus.init()
        SequenceEvolution_GTRGammaSeqGen.init()
        GC.seed_height = float(GC.seed_height)
        assert GC.seed_height > 0, "seed_height must be positive"
        assert GC.check_eval_str(GC.seed_speciation_rate_func), "Potentially dangerous seed_speciation_rate_func"
        GC.check_seqgen_executable()

    def generate():
        if not hasattr(GC, "seed_sequences"):
            assert len(GC.seed_nodes) > 1, "Must have at least 2 seed nodes"
            rootseq = SeedSequence_Virus.generate()
            tmp = nonhomogeneous_yule_tree(lambda t: eval(GC.seed_speciation_rate_func), end_num_leaves=len(GC.seed_nodes))
            for node in tmp.traverse_preorder(leaves=False):
                node.label = None
            tmp.root.edge_length = 0 # don't count root edge in height
            tmp.scale_edges(GC.seed_height/tmp.height())
            treestr = tmp.newick().replace('[&R] ','')
            makedirs(OUT_FOLDER, exist_ok=True)
            f = open(OUT_FOLDER + '/time_tree.tre','w')
            f.write(treestr)
            f.close()
            treestr = MF.modules['TreeUnit'].time_to_mutation_rate(treestr).replace('[&R] ','')
            seqgen_file = OUT_FOLDER + '/seed.txt'
            f = open(seqgen_file, 'w')
            f.write("1 %d\nROOT %s\n1\n%s" % (len(rootseq),rootseq,treestr))
            f.close()
            command = [GC.seqgen_path,'-or','-k1']
            if GC.random_number_seed is not None:
                command += ['-z%d'%GC.random_number_seed]
            command += GC.seqgen_args.split()
            try:
                seqgen_out = check_output(command, stdin=open(seqgen_file), stderr=open(OUT_FOLDER + '/log_seqgen.txt','w')).decode('ascii')
                f = open(OUT_FOLDER + '/seqgen.out','w')
                f.write(seqgen_out)
                f.close()
            except CalledProcessError as e:
                f = open('seqgen.err','w'); f.write(str(e)); f.close()
                chdir(GC.START_DIR)
                assert False, "Seq-Gen encountered an error"
            GC.seed_sequences = [line.split()[-1].strip() for line in seqgen_out.splitlines()[1:]]
        try:
            return GC.seed_sequences.pop()
        except IndexError:
            assert False, "Late seeds are not supported at this time"

    def merge_trees():
        return GC.merge_trees_seqgen()
