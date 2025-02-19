#! /usr/bin/env python3
'''
Niema Moshiri 2017

"TimeSample" module, where no nodes are sampled
'''
from TimeSample import TimeSample
import FAVITES_GlobalContext as GC
import modules.FAVITES_ModuleFactory as MF

class TimeSample_None(TimeSample):
    def cite():
        return GC.CITATION_FAVITES

    def init():
        if "NodeEvolution_File" not in str(MF.modules['NodeEvolution']):
            assert "NodeAvailability_None" in str(MF.modules['NodeAvailability']), "Must use NodeAvailability_None module"
            assert "NumTimeSample_None" in str(MF.modules['NumTimeSample']), "Must use NumTimeSample_None module"

    def sample_times(node, num_times):
        return []