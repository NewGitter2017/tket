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
#from pytket.pyzx import pyzx_to_tk
from pytket.extensions.pyzx import tk_to_pyzx
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
    """
    def __init__(self, c: Circuit) -> None:
        """
        :param c:       A pytket circuit.
        :param type:    Circuit
        """
        self.c = c
        self.qubits = len(c.qubits)
        self.inputs = {}
        self.outputs = {}
        for qubit in c.qubits:
            self.inputs[qubit.index[0]] = qubit.index[0]
            self.outputs[qubit.index[0]] = qubit.index[0]
        pass
    
    def single_conversion(self) -> Circuit:
        """
        Converts a tket circuit to another with reduced depth and higher width.
        
        :returns:       A pytket circuit.
        :rtype:         Circuit
        """
        pyzxc = MPattern.zx_convert(self.c)
        g = self.zx_diagram(pyzxc)
        c2 = MPattern.entangle(g)
        #At this point we have the new qubit register, the CZ gates, and the
        #mapping for each input/output but we are missing the conditional
        #measurements.
        subs = self.split_subgraphs(g)
        c3 = MPattern.correct(subs)
        c2.add_circuit(c3,c3.qubits,c3.bits[:len(c3.qubits)])
        return c3
    
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
        #We assume that g.copy() will squash the vertex
        #labels and thus we keep track of the new input/output vertices. If
        #pyzx is updated such that graph.copy() no longer changes vertex labels
        #then comment out the next line (label_squish(g)).
        self.label_squish(g)
        return g.copy()
    
    def label_squish(self, g: Graph) -> None:
        """
        Updates the input/output labels of the MPattern to matched a squished
        graph cause by g.copy().
        
        :param g:       A pyzx graph.
        :param type:    Graph
        """
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
    
    def entangle(g: Graph) -> Circuit:
        """
        Creates a tket circuit which implements the edges of a zx diagram
        via CZ gates on pairs of qubits.
        
        :param g:       A zx diagram.
        :param type:    Graph
        
        :returns:       A pytket circuit.
        :rtype:         Circuit
        """
        c = Circuit(len(g.vertices()),len(g.vertices()))
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
            c.add_barrier(range(len(g.vertices())), range(len(g.vertices())))
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
                neigh = g.neighbors(v)
                new_edge = (neigh[0],neigh[1])
                g.add_edge(new_edge,1)
        for v in v_list:
            g.remove_vertex(v)
        if len(v_list)>0:
            self.remove_redundant(g)
        pass
            
    def split_subgraphs(self, g: Graph) -> list:
        """
        If a zx diagram contains sub-diagrams which are not connected to each
        other, it splits them into multiple zx diagrams. It returns a list of
        all the irreducible zx diagrams contained by the original.
        
        :param g:       A zx diagram.
        :param type:    Graph
        
        :returns:       A list of zx diagrams.
        :rtype:         list (of 'Graph' objects)
        """
        #'label_squish()' is ran before 'g.copy()' to keep track of input/
        #output qubit labels.
        self.label_squish(g)
        g1 = g.copy()
        cluster_list = []
        for v in g1.vertices():
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
                        temp |= set(g1.neighbors(v2))
                    new_nodes = temp - new_set
                    if (len(new_nodes) == 0):
                        break
                    new_set |= new_nodes
                cluster_list.append(new_set)
        graph_list = []
        for cluster in range(len(cluster_list)):
            curr_cluster = cluster_list[cluster]
            new_g = g1.copy()
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
   
    def layer_list(layers: dict) -> list:
        """
        This method takes a dictionary which maps each vertex in a zx diagram
        to a correction layer as input. It then produces a list of sets of 
        vertices as output. The sets in the list represent the layers of
        measurements for the diagram in ascending order.
        
        :param layers:  Dictionary mapping vertices to their correction layer.
        :param type:    dict
        
        :returns:       A list of sets containing integers.
        :rtype:         list (of integers)
        """
        new_list = []
        depth = -1
        for vertex in layers.keys():
            layer = layers[vertex]
            if layer > depth:
                diff = layer - depth
                layer_set += [set() for i in range(diff)]
                depth = layer
            new_list[layer] |= {vertex}
        return new_list
    
    #This function needs to be reviewed for correctness
    def correct(glist: list) -> Circuit:
        """
        This method takes a list of subgraphs as input and produces a circuit
        of measurements and corrections which ensures that the underlying
        graphs are implemented deterministically.
        
        :param glist:   A list of unconnected graphs.
        :param type:    Graph
        
        :returns:       A circuit containing measurements and conditional gates.
        :rtype:         Circuit
        """
        S = {}
        total_v = 0
        for g in glist:
            total_v += len(g.vertice())
            for v in g.vertices():
                S[v] = {"x":set(),"z":set()}
        new_c = Circuit(total_v,total_v)
        for g in glist:
            gf = gflow(g)
            if gflow == None:
                return None
            else:
                l_list = MPattern.layer_list(gf[0])
                layer_num = len(l_list)
                for corr_layer in range(layer_num-1):
                    if corr_layer > 0:
                        isClifford = True
                        for v in l_list[-1-corr_layer]:
                            if not (g.phase(v) in {0,1/2,1,3/2}):
                                isClifford = False
                                break
                        if not isClifford:
                            new_c.add_barrier(list(g.vertices()),list(g.vertices()))
                    for v in l_list[-1-corr_layer]:
                        my_result = {v}
                        if g.phase(v) in {0,1/2,1,3/2}:
                            my_result ^= S[v]["z"]
                        if g.phase(v) in {1/2,1,3/2}:
                            my_result ^= S[v]["x"]
                        if g.phase(v) in {1/2,3/2}:
                            my_result ^= {True}
                        for u in (gf[1][v] - {v}):
                            S[u]["x"] ^= my_result
                        for u in g.vertices():
                            Nu = set(g.neighbors(u))
                            if (len(Nu & gf[1][v])%2) == 1:
                                S[u]["z"] ^= my_result
                        if g.phase(v) in {0,1}:
                            new_c.H(v)
                            new_c.Measure(v,v)
                        elif(g.phase(v) in {1/2,3/2}):
                            new_c.Rx(-g.phase(v),v)
                            new_c.Measure(v,v)
                        else:
                            new_c.H(v)
                            #theta = zi-(((-1)**xi)*g.phase(v))
                            zi = False
                            for val in S[v]["z"]:
                                if type(val)==bool:
                                    zi ^= val
                                else:
                                    zi ^= new_c.bits[val]
                            xi = False
                            for val in S[v]["x"]:
                                if type(val)==bool:
                                    xi ^= val
                                else:
                                    xi ^= new_c.bits[val]
                            new_c.X(v, condition=zi)
                            new_c.Rx(g.phase(v), v, condition=xi)
                            new_c.Rx(-g.phase(v),v, condition=(xi^True))
                            new_c.Measure(v,v)
                if len(l_list)>1:
                    new_c.add_barrier(list(g.vertices()),list(g.vertices()))
                for v in l_list[0]:
                    new_c.H(v)
                    zi = False
                    for val in S[v]["z"]:
                        if type(val)==bool:
                            zi ^= val
                        else:
                            zi ^= new_c.bits[val]
                    xi = False
                    for val in S[v]["x"]:
                        if type(val)==bool:
                            xi ^= val
                        else:
                            xi ^= new_c.bits[val]
                    new_c.Z(v, condition=xi)
                    new_c.X(v, condition=zi)
                    new_c.Rx(-g.phase(v),v)
        return new_c