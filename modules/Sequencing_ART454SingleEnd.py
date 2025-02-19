#! /usr/bin/env python3
'''
Niema Moshiri 2016

"Sequencing" module, using ART to simulate Roche 454 reads (single-end)
'''
from Sequencing import Sequencing
import FAVITES_GlobalContext as GC
from gzip import open as gopen
from subprocess import call,STDOUT
from tempfile import NamedTemporaryFile
from os.path import expanduser
from os import getcwd,makedirs,chdir,listdir,rename

class Sequencing_ART454SingleEnd(Sequencing):
    def cite():
        return GC.CITATION_ART

    def init():
        GC.out_dir = expanduser(GC.out_dir)
        GC.art_454_options = [i.strip() for i in GC.art_454_options.strip().split()]
        GC.art_454_path = expanduser(GC.art_454_path.strip())

    def introduce_sequencing_error(node):
        if not hasattr(GC,"sequencing_file"):
            GC.sequencing_file = gopen('%s/error_prone_files/sequence_data_subsampled_errorprone_read1.fastq.gz'%GC.out_dir, 'wb', 9)
        orig_dir = getcwd()
        chdir(GC.out_dir)
        makedirs("ART_output", exist_ok=True)
        chdir("ART_output")
        cn_label = node.get_name()
        for t in GC.final_sequences[cn_label]:
            f = NamedTemporaryFile(mode='w')
            for l,s in GC.final_sequences[cn_label][t]:
                f.write(">%s\n%s\n" % (l,s))
            f.flush()

            command = [GC.art_454_path] + GC.art_454_options
            if GC.random_number_seed is not None:
                command += ['-r',str(GC.random_number_seed)]
                GC.random_number_seed += 1
            command.append(f.name)
            command.append('%s_%f' % (cn_label,t))
            command.append(str(GC.art_454_fold_coverage))
            try:
                call(command, stdout=open('%s_%f.log' % (cn_label,t), 'w'), stderr=STDOUT)
            except FileNotFoundError:
                chdir(GC.START_DIR)
                assert False, "art_454 executable was not found: %s" % GC.art_454_path
            f.close()
            for l in open('%s_%f.fq' % (cn_label,t)):
                GC.sequencing_file.write(l.encode())
        chdir(orig_dir)

    def finalize():
        if hasattr(GC,"sequencing_file"):
            GC.sequencing_file.close()