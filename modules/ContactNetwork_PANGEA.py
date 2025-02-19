#! /usr/bin/env python3
'''
Niema Moshiri 2016

"ContactNetwork" module, using the PANGEA HIV Simulation Model
(https://github.com/olli0601/PANGEA.HIV.sim)
'''
from ContactNetwork import ContactNetwork
from ContactNetworkEdge_PANGEA import ContactNetworkEdge_PANGEA as Edge
from ContactNetworkNode_PANGEA import ContactNetworkNode_PANGEA as Node
import modules.FAVITES_ModuleFactory as MF
import FAVITES_GlobalContext as GC

class ContactNetwork_PANGEA(ContactNetwork):
    def init():
        GC.pangea_module_check()

    def cite():
        return GC.CITATION_PANGEA

    def __init__(self, edge_list=None):
        if hasattr(GC,'PANGEA_TRANSMISSION_NETWORK'):
            self.transmissions = [(Node(None,u,None), Node(None,v,None), float(t)) for u,v,t in GC.PANGEA_TRANSMISSION_NETWORK]
            self.nodes = set()
            for u,v,t in self.transmissions:
                self.nodes.add(u)
                self.nodes.add(v)
            tmp = {(u,v) for u,v,t in self.transmissions}
            self.edges = set()
            for u,v in tmp:
                self.edges.add(Edge(u,v,None))
                self.edges.add(Edge(v,u,None))

    def is_directed(self):
        return False

    def num_transmissions(self):
        return None

    def num_nodes(self):
        return 2

    def get_nodes(self):
        return set()

    def get_node(self, name):
        return None

    def num_infected_nodes(self):
        return None

    def get_infected_nodes(self):
        return set()

    def num_uninfected_nodes(self):
        return None

    def get_uninfected_nodes(self):
        return set()

    def num_edges(self):
        return 1

    def nodes_iter(self):
        for node in self.nodes:
            yield node

    def edges_iter(self):
        for edge in self.edges:
            yield edge

    def get_edges_from(self, node):
        return []

    def get_edges_to(self,node):
        return []

    def get_transmissions(self):
        return self.transmissions

    def add_transmission(self,u,v,time):
        pass

    def add_to_infected(self,node):
        pass

    def remove_from_infected(self,node):
        pass