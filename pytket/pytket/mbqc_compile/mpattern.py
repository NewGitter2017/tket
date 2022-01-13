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
    Class with tools to convert a pytket circuit into a new pytket circuit
    with lower depth and higher width, by using MBQC techniques.
    
    :param c:       A pytket circuit.
    :param type:    Circuit
    """
    def __init__(self, c: Circuit) -> None:
        self.c = c
        self.qubits = len(c.qubits)
        self.inputs = {}
        self.outputs = {}
        for qubit in c.qubits:
            self.inputs[qubit.index[0]] = qubit.index[0]
            self.outputs[qubit.index[0]] = qubit.index[0]
        pass
    
    def single_conversion(self) -> None:
        """
        Converts a tket circuit to another with reduced depth and higher width.
        """
        pyzxc = MPattern.zx_convert(self.c)
        g = self.zx_diagram(pyzxc)
        c2 = MPattern.entangle(g)
        #At this point we have the new qubit register, the CZ gates, and the
        #mapping for each input/output but we are missing the conditional
        #measurements.
        pass
    
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
    
    def zx_diagram(self, pyzxc: pyzxCircuit) -> Graph:
        """
        Converts a pyzx circuit to a zx diagram.
        
        :param pyzxc:   A pyzx circuit.
        :param type:    pyzxCircuit
        
        :returns:       A zx diagram.
        :rtype:         Graph
        """
        g = circuit_to_graph(pyzxc)
        interior_clifford_simp(g, quiet=True)
        for q in range(self.qubits):
            self.inputs[q] = sorted(list(g.vertices()))[q]
            self.outputs[q] = sorted(list(g.vertices()))[-self.qubits+q]
        self.remove_redundant(g)
        #The following code assumes that "g.copy()" will squash the vertex
        #labels and thus keeps track of the new input/output vertices. If
        #pyzx is updated such that graph.copy() no longer changes vertex labels
        #then comment out the remaining commands in this method, before
        #'return'.
        original_labels = sorted(list(g.vertices()))
        for i in self.inputs.keys():
            for v in range(len(original_labels)):
                if self.inputs[i] == original_labels[v]:
                    self.inputs[i] = v
                    break
        for o in self.outputs.keys():
            for v in range(len(original_labels)):
                if self.outputs[o] == original_labels[v]:
                    self.outputs[o] = v
                    break
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

    def remove_redundant(self, g: Graph) -> None:
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
                    for i in self.inputs.keys():
                        if self.inputs[i] == remove_vertex:
                            self.inputs[i] = keep_vertex
                            break
                else:
                    g.outputs.remove(remove_vertex)
                    g.outputs.append(keep_vertex)
                    for o in self.outputs.keys():
                        if self.outputs[o] == remove_vertex:
                            self.outputs[o] = keep_vertex
                            break
                g.set_type(keep_vertex, 0)
        self.identity_cleanup(g)
        pass

    def identity_cleanup(self, g: Graph) -> None:
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
            self.remove_redundant(g)
        pass
            
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
    
    #This function needs to be reviewed for correctness
    def correct(g,produce_string=False):
        product = ""
        gf = gflow(g)
        if gflow == None:
            print("Non-deterministic graph.")
            return (None, None)
        else:
            l = layer_list(gf[0])
            S = {}
            for vo in l[0]:
                S[("z"+str(vo))]=set()
                S[("x"+str(vo))]=set()
                S[("m"+str(vo))]=set()
            for layer in l[1:]:
                for v in layer:
                    S[v]=set([v])
                    #if g.phase(v) < 1:
                    #    S[v]=set([v])
                    #else:
                    #    S[v]=set([True,v])
            for cl in range(len(l)-1):
                for i in l[-1-cl]:
                    Gi = gf[1][i]
                    A = Gi - set([i])
                    B = set()
                    Ngi = set()
                    for gi in A:
                        Ngi |= g.neighbors(gi)
                    for u in Ngi:
                        if len(g.neighbors(u) & (A-set([u])))%2:
                            B |= set([u])
                    for u in B:
                        if not (u in l[0]):
                            S[u] = (S[u]|S[i]) - (S[u]&S[i])
                        else:
                            zu = "z" + str(u)
                            S[zu] = (S[zu]|S[i]) - (S[zu]&S[i])
                            new = {(w,x) for w in S[i] for x in S["x"+str(u)]}
                            mu = "m" + str(u)
                            S[mu] = (S[mu]|new) - (S[mu]&new)
                        if produce_string:
                            print("Z(" + str(u) + "," + str(i) + ")")
                            product = "Z(" + str(u) + "," + str(i) + ")" + product
                    for u in A:
                        if (u in l[0]):
                            xu = "x" + str(u)
                            S[xu] = (S[xu]|S[i]) - (S[xu]&S[i])
                        elif g.phase(u) in {1/2, 3/2}:
                            S[u] = (S[u]|S[i]) - (S[u]&S[i])
                        if produce_string:
                            print("X(" + str(u) + "," + str(i) + ")")
                            print(u, g.neighbors(u))
                            product = "X(" + str(u) + "," + str(i) + ")" + product
            clean_corrections = {}
            for key in S.keys():
                if (str(key)[0] in "xz") and (len(S[key])>0):
                    clean_corrections[key]=S[key]
            return clean_corrections