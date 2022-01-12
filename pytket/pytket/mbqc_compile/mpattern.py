# Copyright 2019-2021 Cambridge Quantum Computing
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pytket.transform import Transform
from pytket.circuit import Circuit as Circuit
from pytket.pyzx import pyzx_to_tk
from pytket.pyzx import tk_to_pyzx
from pyzx.simplify import interior_clifford_simp
from pyzx.circuit import Circuit as pyzxCircuit
from pyzx.graph.graph_s import GraphS as Graph
from pyzx.circuit.graphparser import circuit_to_graph
from pyzx.gflow import gflow as gflow
#from typing import None

class MPattern:
    """
    A series of tools that allow circuits to be transformed into 
    equivalent circuits representing measurement patterns.
    """
    def __init__(self):
        pass
    
    def single_conversion(c: Circuit) -> Circuit:
        """
        Converts a tket circuit to another with reduced depth and higher width.
        
        :param c:       A pytket circuit.
        :param type:    Circuit
        
        :returns:       A pytket circuit.
        :rtype:         Circuit
        """
        pyzxc = MPattern.zx_convert(c)
        g = MPattern.zx_diagram(pyzxc)
        c2 = MPattern.entangle(g)
        return c2
    
    def zx_convert(c: Circuit) -> pyzxCircuit:
        """
        Converts a tket circuit to a pyzx circuit.
        
        :param c:       A pytket circuit.
        :param type:    Circuit
        
        :returns:       A pyzx circuit.
        :rtype:         pyzxCircuit
        """
        Transform.RebaseToPyZX().apply(c)
        return tk_to_pyzx(c)
    
    def zx_diagram(pyzxc: pyzxCircuit) -> Graph:
        """
        Converts a pyzx circuit to a zx diagram.
        
        :param pyzxc:   A pyzx circuit.
        :param type:    pyzxCircuit
        
        :returns:       A zx diagram.
        :rtype:         Graph
        """
        g = circuit_to_graph(pyzxc)
        interior_clifford_simp(g, quiet=True)
        MPattern.remove_redundant(g)
        return g.copy()
    
    def entangle(g: Graph) -> Circuit:
        """
        Creates a tket circuit which implements the edges of a zx diagram
        via CZ gates on pairs of qubits.
        
        :param g:       A zx diagram.
        :param type:    Graph
        
        :returns:       A pytket circuit.
        :rtype:         Circuit
        """
        c = Circuit(len(g.vertices()))
        vlist = list(g.vertices())
        dlist = []
        for v in vlist:
            dlist.append(len(g.neighbors(v)))
        vlist = [x for _, x in sorted(zip(dlist, vlist))]
        vlist.reverse()
        edge_pool = set(g.edge_set())
        finished_edge_pool = {}
        doneround = {}
        for v in vlist:
            doneround[v] = False
        while len(edge_pool)>0:
            for vid in range(len(vlist)):
                if not doneround[vlist[vid]]:
                    for vid2 in range(vid+1,len(vlist)):
                        if ((not doneround[vlist[vid]]) and (not doneround[vlist[vid2]])):
                            if (((vlist[vid],vlist[vid2]) in edge_pool) or ((vlist[vid2],vlist[vid]) in edge_pool)):
                                c.CZ(vlist[vid],vlist[vid2])
                                doneround[vlist[vid]] = True
                                doneround[vlist[vid2]] = True
                                edge_pool -= set([(vlist[vid],vlist[vid2])])
                                edge_pool -= set([(vlist[vid2],vlist[vid])])
                                if vlist[vid] in finished_edge_pool.keys():
                                    finished_edge_pool[vlist[vid]] += 1
                                else:
                                    finished_edge_pool[vlist[vid]] = 1
                                if vlist[vid2] in finished_edge_pool.keys():
                                    finished_edge_pool[vlist[vid2]] += 1
                                else:
                                    finished_edge_pool[vlist[vid2]] = 1
                                break
                        else:
                            continue
                else:
                    continue
            for key in doneround.keys():
                doneround[key] = False
            dlist = []
            for v in vlist:
                if v in finished_edge_pool.keys():
                    dlist.append(len(g.neighbors(v))-finished_edge_pool[v])
                else:
                    dlist.append(len(g.neighbors(v)))
                    finished_edge_pool[v] = 0
            vlist = [x for _, x in sorted(zip(dlist, vlist))]
            vlist.reverse()
        return c

    def remove_redundant(g: Graph) -> None:
        """
        Removes simples edges from a zx diagram by merging the connected
        vertices.
        
        :param g:       A zx diagram.
        :param type:    Graph
        """
        simple_edges = set()
        for edge in g.edge_set():
            if g.edge_type(edge)== 1:
                simple_edges.add(edge)
        for edge in simple_edges:
            v1,v2 = edge
            g.remove_edge(edge)
            is_boundary = True
            removing_input = True
            remove_vertex = v1
            keep_vertex = v2
            if v1 in g.inputs:
                remove_vertex = v1
                keep_vertex = v2
            elif v2 in g.inputs:
                remove_vertex = v2
                keep_vertex = v1
            elif v1 in g.outputs:
                removing_input = False
            elif v2 in g.outputs:
                removing_input = False
                remove_vertex = v2
                keep_vertex = v1
            else:
                is_boundary = False
            g.set_row(keep_vertex, g.row(remove_vertex))
            g.set_qubit(keep_vertex,g.qubit(remove_vertex))
            neighbors = g.neighbors(remove_vertex) - [keep_vertex]
            for neighbor in neighbors:
                this_type = g.edge_type((remove_vertex, neighbor))
                if g.connected(keep_vertex,neighbor):
                    if (g.edge_type((keep_vertex, neighbor)) == this_type):
                        g.remove_edge((keep_vertex,neighbor))
                    else:
                        g.add_edge((keep_vertex,neighbor),this_type)
                else:
                    g.add_edge((keep_vertex,neighbor),this_type)
            g.add_to_phase(keep_vertex,g.phase(remove_vertex))
            g.remove_vertex(remove_vertex)
            if is_boundary:
                if removing_input:
                    g.inputs.remove(remove_vertex)
                    g.inputs.append(keep_vertex)
                else:
                    g.outputs.remove(remove_vertex)
                    g.outputs.append(keep_vertex)
                g.set_type(keep_vertex, 0)
        MPattern.identity_cleanup(g)

    def identity_cleanup(g: Graph) -> None:
        """
        Removes identity vertices from a zx diagram if any exist.
        
        :param g:       A zx diagram.
        :param type:    Graph
        """
        v_list = []
        for v in g.vertices():
            if (len(g.neighbors(v)) == 2) and (g.phase(v) == 0) and not (v in g.inputs+g.outputs):
                v_list.append(v)
                neigh = list(g.neighbors(v))
                new_edge = (neigh[0],neigh[1])
                g.add_edge(new_edge,1)
        for v in v_list:
            g.remove_vertex(v)
        if len(v_list)>0:
            MPattern.remove_redundant(g)
            
    def split_subgraphs(g: Graph) -> list:
        """
        If a zx diagram contains sub-diagrams which are not connected to each
        other, it splits them into multiple zx diagrams. It returns a list of
        all the irreducible zx diagrams contained by the original.
        
        :param g:       A zx diagram.
        :param type:    Graph
        
        :returns:       A list of zx diagrams.
        :rtype:         list (of 'Graph' objects)
        """
        g2 = g.copy()
        cluster_list = []
        for v in g2.vertices():
            found = False
            for cluster in cluster_list:
                if v in cluster:
                    found = True
            if (not found):
                new_set = set([v])
                new_nodes = set([v])
                while True:
                    temp = set()
                    for v2 in new_nodes:
                        temp |= set(g2.neighbors(v2))
                    new_nodes = temp - new_set
                    if (len(new_nodes) == 0):
                        break
                    new_set |= new_nodes
                cluster_list.append(new_set)
        graph_list = []
        for cluster in range(len(cluster_list)):
            curr_cluster = cluster_list[cluster]
            new_g = g2.copy()
            new_vertices = set(new_g.vertices())
            for v in new_vertices:
                if not (v in curr_cluster):
                    new_g.remove_edges(new_g.incident_edges(v))
                    if v in new_g.inputs:
                        new_g.inputs.remove(v)
                    if v in new_g.outputs:
                        new_g.outputs.remove(v)
                    new_g.remove_vertex(v)
            graph_list.append(new_g)
        return graph_list
    
    #def correction_layers(glist: list) -> list:
    #    for g in glist:
    #    gf_layers = gflow(g)[0]
    #    max_layer = 0
    #    if len(gf_layers.values()) > 0:
    #        max_layer = max(gf_layers.values())
    #    clifford_layers = max_layer
    #    for l in layer_list(gf_layers)[1:]:
    #        for v in l:
    #            if g.phase(v) not in {0,1/2,1,3/2}:
    #                clifford_layers -= 1
    #                break   
    #    layers =