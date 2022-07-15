#! /usr/bin/env python3
'''
Niema Moshiri 2016

"ContactNetworkGenerator" module, where the graph that is generated is a
tripartite community with specified inter-community connectivity.
'''
from ContactNetworkGenerator import ContactNetworkGenerator
import FAVITES_GlobalContext as GC
from gzip import open as gopen
from os.path import expanduser

class ContactNetworkGenerator_NateBipartite(ContactNetworkGenerator):
    def cite():
        return GC.CITATION_NETWORKX

    def init():
        try:
            global barabasi_albert_graph
            global disjoint_union
            from networkx import barabasi_albert_graph, disjoint_union
        except:
            from os import chdir
            chdir(GC.START_DIR)
            assert False, "Error loading NetworkX. Install with: pip3 install networkx"
        try:
            global np
            import numpy as np
        except:
            from os import chdir
            chdir( GC.START_DIR )
            assert False, "Error loading NumPy. Install with: pip3 install numpy"
        try:
            global reduce
            from functools import reduce
        except:
            from os import chdir
            chdir( GC.START_DIR )
            assert False, "Error loading itertools"

        assert isinstance(GC.comm_sizes, list), "comm_sizes must be a list of positive integers"
        assert len( GC.comm_sizes ) == 2, "comm_sizes must be of length 2"
        for e in GC.comm_sizes:
            assert isinstance(e, int) and e > 0, "comm_sizes must be a list of positive integers"
        assert isinstance(GC.comm_m, list), "comm_m must be a list of positive integers"
        assert len( GC.comm_m ) == 2, "comm_m must be of length 2"
        for e in GC.comm_m:
            assert isinstance(e, int) and e > 0, "comm_m must be a list of positive integers"
        assert isinstance( GC.prob_ab, float ), "prob_ab must be a float"
        assert GC.prob_ab >= 0 and GC.prob_ab <= 1, "prob_ab must be between 0 and 1"

    def get_edge_list():
        def _calculate_maximum_connectivity( size ):
            return (size * (size - 1)) / 2

        graphs = list()
        average_connectivity = list()
        for i, params in enumerate( zip( GC.comm_sizes, GC.comm_m ) ):
            cn_comm = barabasi_albert_graph( *params, seed=GC.random_number_seed )
            average_connectivity.append( len( cn_comm.edges ) / _calculate_maximum_connectivity( params[0] ) )
            graphs.append( cn_comm )
            GC.random_number_seed += 1
        cn = reduce( disjoint_union, graphs )

        # encode parition information in graph attributes.
        nodelist = range( 0, sum( GC.comm_sizes ) )
        size_cumsum = [sum( GC.comm_sizes[0:x] ) for x in range( 0, len( GC.comm_sizes ) + 1 )]
        cn.graph["partition"] = [
            set( nodelist[size_cumsum[x]: size_cumsum[x + 1]] )
            for x in range( 0, len( size_cumsum ) - 1 )
        ]

        # Calculate inter-community connectivity prob as a fraction of intra-community connectivity.
        average_connectivity = sum( average_connectivity ) / len( average_connectivity )
        community_connectivity = GC.prob_ab * average_connectivity

        # For each community pair, connected edges with the calculated probability
        random_draw = np.random.random_sample( size=(GC.comm_sizes[0], GC.comm_sizes[1]) )
        edge_x, edge_y = np.where( random_draw < community_connectivity )
        edge_x += size_cumsum[0]
        edge_y += size_cumsum[1]
        cn.add_edges_from( zip( edge_x, edge_y ) )

        # Output contact network
        out = GC.nx2favites( cn, 'u' )
        f = gopen( expanduser( "%s/contact_network.txt.gz" % GC.out_dir ), 'wb', 9 )
        f.write( '\n'.join( out ).encode() )
        f.write( b'\n' )
        f.close()

        # Output community information
        f = gopen(expanduser("%s/contact_network_partitions.txt.gz" % GC.out_dir),'wb',9)
        f.write(str(cn.graph['partition']).encode()); f.write(b'\n')
        f.close()
        GC.cn_communities = [{str(n) for n in c} for c in cn.graph['partition']]
        return out