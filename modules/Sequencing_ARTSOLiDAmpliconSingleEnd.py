#! /usr/bin/env python3
'''
Niema Moshiri 2016

"Sequencing" module, using ART to simulate SOLiD reads (amplicon single-end, F3)
'''
from Sequencing import Sequencing
import FAVITES_GlobalContext as GC
from gzip import open as gopen
from subprocess import call
from tempfile import NamedTemporaryFile
from os.path import expanduser
from os import getcwd,makedirs,chdir,listdir

class Sequencing_ARTSOLiDAmpliconSingleEnd(Sequencing):
    def cite():
        return GC.CITATION_ART

    def init():
        GC.out_dir = expanduser(GC.out_dir)
        GC.art_SOLiD_options = [i.strip() for i in GC.art_SOLiD_options.strip().split()]
        GC.art_SOLiD_path = expanduser(GC.art_SOLiD_path.strip())
        assert GC.art_SOLiD_len_read <= 75, "Maximum ART SOLiD read length is 75"

    def introduce_sequencing_error(node):
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
            command = [GC.art_SOLiD_path] + GC.art_SOLiD_options
            if GC.random_number_seed is not None:
                command += ['-r',str(GC.random_number_seed)]
                GC.random_number_seed += 1
            command += ['-A','s']
            command.append(f.name)
            command.append('%s_%f' % (cn_label,t))
            command.append(str(GC.art_SOLiD_len_read))
            command.append(str(GC.art_SOLiD_reads_per_amplicon))
            try:
                call(command, stdout=open('%s_%f.log' % (cn_label,t), 'w'))
            except FileNotFoundError:
                chdir(GC.START_DIR)
                assert False, "art_SOLiD executable was not found: %s" % GC.art_illumina_path
            f.close()
            if not hasattr(GC,"sequencing_file"):
                GC.sequencing_file = gopen('%s/error_prone_files/sequence_data_subsampled_errorprone.fastq.gz'%GC.out_dir, 'wb', 9)
            for l in open('%s_%f.fq' % (cn_label,t)):
                GC.sequencing_file.write(l.encode())
        chdir(orig_dir)

    def finalize():
        if hasattr(GC,"sequencing_file"):
            GC.sequencing_file.close()