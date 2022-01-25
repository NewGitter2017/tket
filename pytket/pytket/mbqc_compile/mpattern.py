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
from pytket.circuit import Circuit, Command
from pytket.extensions.pyzx import tk_to_pyzx
from pyzx.simplify import interior_clifford_simp
from pyzx.circuit import Circuit as pyzxCircuit
from pyzx.graph.graph_s import GraphS
from pyzx.circuit.graphparser import circuit_to_graph
from pyzx.gflow import gflow
from pytket.routing import route
from pytket.routing import Architecture
import math

class MPattern:
    """
    Class with tools to convert a pytket circuit into a new pytket circuit
    with lower depth and higher width, by using MBQC techniques.
    """
    def __init__(self, c: Circuit, arch: Architecture = None) -> None:
        """
        Initialises the MPattern object by giving it a pytket circuit and 
        optionally a device architecture.
        
        :param c:       An arbitrary pytket circuit that we want converted to 
                            a measurement pattern.
        :param type:    Circuit
        
        :param arch:    A pytket architecture that we want the measurement pattern to fit into.
        :param type:    Architecture
        """
        self.c = c
        self.arch = arch
    
    def set_architecture(self, arch: Architecture) -> None:
        """
        Method to change the device architecture of a measurement pattern.
        
        :param arch:    A new pytket architecture to fit the pattern onto.
        :param type:    Architecture
        """
        self.arch = arch
    
    def single_conversion(self) -> Circuit:
        """
        Converts a pytket circuit to another with reduced depth and higher width.
        
        :returns:       A tuple containing the new circuit and the i/o map.
        :rtype:         tuple
        """
        (g,io_map) = self.zx_diagram() #Creates a simplified ZX diagram.
        subs = self.split_subgraphs(g,io_map) #Returns list of disjoint subgraphs.
        cz_circ = MPattern.entangle(g) #Creates the CZ circuit part of the pattern.
        #Figure out how to add self.route_circ(cz_circuit) here and how it's
        #going to affect the next command. Also how to keep track of mapping.
        m_circ = MPattern.correct(subs) #Circuit implementing measurements/corrections.
        cz_circ.add_circuit(m_circ,m_circ.qubits,m_circ.bits[:len(m_circ.qubits)])
        return (cz_circ, io_map)
    
    def multi_conversion(self, n: int = 2, strategy: str = "Depth", ) -> Circuit:
        depth_structure = self.depth_structure()
        size = len(depth_structure)
        if strategy == "Gates":
            non_cliff = 0
            for d in depth_structure:
                for gate in d:
                    if not MPattern.is_Clifford(gate):
                        non_cliff += 1
            size = non_cliff
        slice_size = math.ceil(size/n)
        done_depth = 0
        output = []
        if strategy == "Depth":
            for curr in range(n):
                finish_at = min(done_depth + slice_size,size)
                subcircuit = Circuit()
                for qubit in self.c.qubits:
                    subcircuit.add_qubit(qubit)
                for bit in self.c.bits:
                    subcircuit.add_bit(bit)
                for depth_list in depth_structure[done_depth,finish_at]:
                    for gate in depth_list:
                        subcircuit.add_gate(Op=gate.op, args=gate.args)
                sub_pattern = MPattern(subcircuit)
                output.append(sub_pattern.single_conversion())
                if finish_at >= size:
                    break
                else:
                    done_depth = finish_at
        elif strategy == "Gates":
            for curr in range(n):
                ncliff_total = 0
                added_depths = 0
                stop_at_next_nClifford = False
                stopped = False
                for depth in depth_structure[done_depth:]:
                    for gate in depth:
                        if not MPattern.is_Clifford(gate):
                            if stop_at_next_nClifford:
                                stopped = True
                                break
                            else:
                                ncliff_total += 1
                    if stopped:
                        break
                    else:
                        added_depths += 1
                        if ncliff_total >= slice_size:
                            stop_at_next_nClifford = True
                subcircuit = Circuit()
                for qubit in self.c.qubits:
                    subcircuit.add_qubit(qubit)
                for bit in self.c.bits:
                    subcircuit.add_bit(bit)
                for depth_list in depth_structure[done_depth:done_depth+added_depths]:
                    for gate in depth_list:
                        subcircuit.add_gate(Op=gate.op, args=gate.args)
                sub_pattern = MPattern(subcircuit)
                output.append(sub_pattern.single_conversion())
                if done_depth+added_depths >= size:
                    break
                else:
                    done_depth += added_depths
        return output
    
    def is_Clifford(aGate: Command) -> bool:
        if aGate.op.get_name() in {'Z','X','Y','H','S','V','Sdg','Vdg','SX','SXdg','CX','CY','CZ','CH','CV','CVdg','CSX','CSXdg','CCX','SWAP','CSWAP','noop','BRIDGE','Reset'}:
            return True
        elif aGate.op.get_name() in {'T','Tdg'}:
            return False
        elif aGate.op.get_name() in {'Rx','Rz','Ry','CRx','CRy','CRz'}:
            if aGate.op.params[0] in {0,1/2,1,3/2,2}:
                return True
        else:
            return False
    
    def depth_structure(self) -> list:
        gates = self.c.get_commands()
        qubits = self.c.qubits
        depth = self.c.depth()
        qn = self.c.n_qubits
        current_frontiers = [0]*qn
        depth_slices = [[] for d in range(depth)]
        for gate in gates:
            involved_qubits = gate.qubits
            qubit_indices = []
            for qubit in involved_qubits:
                qubit_indices.append(qubits.index(qubit))
            max_frontier = max([current_frontiers[qid] for qid in qubit_indices])
            for qid in qubit_indices:
                current_frontiers[qid] = max_frontier + 1
            depth_slices[max_frontier].append(gate)
        return depth_slices
    
    def zx_diagram(self) -> GraphS:
        """
        Converts a pytket circuit to a zx diagram.
        
        :returns:       A tuple containing the zx diagram and new i/o maps.
        :rtype:         tuple
        """
        rebased_c = self.c.copy() #Creates copy of original circuit on which we will work.
        rebased_c.flatten_registers() #Pyzx can't handle pytket qubit labelling so we have to flatten the register.
        Transform.RebaseToPyZX().apply(rebased_c) #Rebasing the gates into something pyzx can read.
        pyzxc = tk_to_pyzx(rebased_c) #Convert to pyzx circuit.
        g = circuit_to_graph(pyzxc) #Get graph (zx diagram) from pyzx circuit.
        interior_clifford_simp(g, quiet=True) #Remove interior clifford spiders.
        new_inputs = {}
        new_outputs = {}
        sorted_vertices = sorted(list(g.vertices()))
        for q in range(self.c.n_qubits):
            new_inputs[self.c.qubits[q]] = sorted_vertices[q]
            new_outputs[self.c.qubits[q]] = sorted_vertices[q-self.c.n_qubits]
        io_map = {"i":new_inputs, "o":new_outputs} #Keeps track of the vertices corresponding to the original inputs/outputs.
        self.remove_redundant(g,io_map) #Removes the last few remaining simple edges in the diagram.
        #We assume that g.copy() will squash the vertex
        #labels and thus we keep track of the new input/output vertices. If
        #pyzx is updated such that graph.copy() no longer changes vertex labels
        #then comment out the next line (label_squish(g)).
        self.label_squish(g,io_map) #Squishes the labels of the graph vertices to fill in empty vertex indices.
        return (g.copy(),io_map)
    
    def label_squish(self, g: GraphS, io_map: dict) -> None:
        """
        Updates the input/output labels of the MPattern to matched a squished
        graph caused by g.copy().
        
        :param g:       A pyzx graph representing a zx diagram.
        :param type:    GraphS
        
        :param io_map:  A dictionary containing the current i/o mappings.
        :param type:    dict
        """
        original_labels = sorted(list(g.vertices())) #Sort the original vertices in ascending order.
        #The following code will update the i/o map with the new labels of the
        #input/output qubits after the squish.
        for i in io_map["i"].keys():
            for v in range(len(original_labels)):
                if io_map["i"][i] == original_labels[v]:
                    io_map["i"][i] = v
                    break
        for o in io_map["o"].keys():
            for v in range(len(original_labels)):
                if io_map["o"][o] == original_labels[v]:
                    io_map["o"][o] = v
                    break
    
    @staticmethod
    def entangle(g: GraphS) -> Circuit:
        """
        Creates a pytket circuit which implements the edges of a zx diagram
        via CZ gates on pairs of qubits.
        
        :param g:       A zx diagram the edges of which we want to implement.
        :param type:    GraphS
        
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
        doneround = [False for v in vlist] #Which qubits have had CZ gates placed on them during this round of application.
        while len(edge_pool)>0:
            for vid in range(len(vlist)):
                if not doneround[vid]:
                    for vid2 in range(vid+1,len(vlist)):
                        if not doneround[vid2]:
                            if (((vlist[vid],vlist[vid2]) in edge_pool) or ((vlist[vid2],vlist[vid]) in edge_pool)):
                                c.CZ(vlist[vid],vlist[vid2])
                                doneround[vid] = True
                                doneround[vid2] = True
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
            for v in range(len(doneround)):
                doneround[v] = False
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
    
    def route_circ(self, c: Circuit) -> Circuit:
        if not self.arch == None:
            return route(c, self.arch)
        #routedCirc.flatten_registers() Need to figure out why this was here.
        #EDIT: I think it was here precisely because I couldn't feed routed
        #pytket circuits back to pyzx without flattening the qubits. I'll 
        #revise this entire method when I work on routing later, for now just
        #ignore this.
        else:
            return c

    def remove_redundant(self, g: GraphS, io_map: dict) -> None:
        """
        Removes simples edges from a zx diagram by merging the connected
        vertices.
        
        :param g:       A zx diagram with some remaining simple edges we want to remove.
        :param type:    GraphS
        
        :param io_map:  A dictionary containing the current i/o mapping.
        :param type:    dict
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
                    for i in io_map["i"].keys():
                        if io_map["i"][i] == remove_vertex:
                            io_map["i"][i] = keep_vertex
                            break
                else:
                    g.outputs.remove(remove_vertex)
                    g.outputs.append(keep_vertex)
                    for o in io_map["o"].keys():
                        if io_map["o"][o] == remove_vertex:
                            io_map["o"][o] = keep_vertex
                            break
                g.set_type(keep_vertex, 0)
        self.identity_cleanup(g)

    def identity_cleanup(self, g: GraphS) -> None:
        """
        Removes identity vertices from a zx diagram if any exist.
        
        :param g:       A zx diagram with a few possible remaining identity vertices.
        :param type:    GraphS
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
            
    def split_subgraphs(self, g: GraphS, io_map: dict) -> list:
        """
        If a zx diagram contains sub-diagrams which are not connected to each
        other, it splits them into multiple zx diagrams. It returns a list of
        all the irreducible zx diagrams contained by the original.
        
        :param g:       A zx diagram which may contain disjointed sub-diagrams.
        :param type:    GraphS
        
        :param io_map:  A dictionary containing the current i/o mapping.
        :param type:    dict
        
        :returns:       A list of zx diagrams.
        :rtype:         list (of 'GraphS' objects)
        """
        #'label_squish()' is ran before 'g.copy()' to keep track of input/
        #output qubit labels.
        self.label_squish(g, io_map) #Must squish labels again because we are going to use graph.copy()
        g1 = g.copy() #Create copy of the graph to work on.
        cluster_list = [] #Will contain all the sets of interconnected vertices.
        for v in g1.vertices():
            found = False
            for cluster in cluster_list:
                if v in cluster:
                    found = True #Adds a flag if the current vertex exists in any cluster.
            if (not found): #If the current vertex isn't in a cluster, create new cluster.
                new_set = set([v]) #Current state of new cluster.
                new_nodes = set([v]) #The latest additions to the new cluster.
                while True:
                    temp = set()
                    for v2 in new_nodes:
                        temp |= set(g1.neighbors(v2)) #Get neighbors of new additions.
                    new_nodes = temp - new_set #If they have already been added to the cluster they are not new additions.
                    if (len(new_nodes) == 0): #If there are no more neighbors not in the cluster then the cluster is complete.
                        break
                    new_set |= new_nodes #Add new additions to the cluster.
                cluster_list.append(new_set) #Add new cluster to the list.
        graph_list = [] #This is a list of all the disjoint subgraphs.
        for cluster in range(len(cluster_list)): #We will extract one subgraph for each cluster.
            curr_cluster = cluster_list[cluster]
            new_g = g1.copy() #The subgraph can start as a copy of the full graph.
            new_vertices = set(new_g.vertices())
            for v in new_vertices: #Remove each vertex not in the current cluster from the current subgraph.
                if not (v in curr_cluster):
                    new_g.remove_edges(new_g.incident_edges(v))
                    if v in new_g.inputs:
                        new_g.inputs.remove(v)
                    if v in new_g.outputs:
                        new_g.outputs.remove(v)
                    new_g.remove_vertex(v)
            graph_list.append(new_g) #Add new subgraph to the list.
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
                new_list += [set() for i in range(diff)]
                depth = layer
            new_list[layer] |= {vertex}
        return new_list
    
    def correct(glist: list) -> Circuit:
        """
        This method takes a list of subgraphs as input and produces a circuit
        of measurements and corrections which ensures that the underlying
        graphs are implemented deterministically.
        
        :param glist:   A list of unconnected graphs.
        :param type:    list (GraphS)
        
        :returns:       A circuit containing measurements and conditional gates.
        :rtype:         Circuit
        """
        signals = {}
        total_v = 0
        for g in glist:
            total_v += len(g.vertices())
            for v in g.vertices():
                signals[v] = {"x":set(),"z":set()}
        new_c = Circuit(total_v,total_v)
        for g in glist:
            gf = gflow(g)
            if gflow == None:
                return None
            else:
                l_list = MPattern.layer_list(gf[0])
                layer_num = len(l_list)
                reset_list = []
                for corr_layer in range(layer_num-1):
                    if corr_layer > 0:
                        isClifford = True
                        for v in l_list[-1-corr_layer]:
                            if not (g.phase(v) in {0,1/2,1,3/2}):
                                isClifford = False
                                break
                        if not isClifford:
                            new_c.add_barrier(list(g.vertices()),list(g.vertices()))
                        for v in reset_list:
                            #Add command to reset qubit 'v'
                            pass
                        reset_list = []
                    for v in l_list[-1-corr_layer]:
                        my_result = {v}
                        if g.phase(v) in {0,1/2,1,3/2}:
                            my_result ^= signals[v]["z"]
                        if g.phase(v) in {1/2,1,3/2}:
                            my_result ^= signals[v]["x"]
                        if g.phase(v) in {1/2,3/2}:
                            my_result ^= {True}
                        for u in (gf[1][v] - {v}):
                            signals[u]["x"] ^= my_result
                        for u in g.vertices():
                            Nu = set(g.neighbors(u))
                            if (len(Nu & gf[1][v])%2) == 1:
                                signals[u]["z"] ^= my_result
                        if g.phase(v) in {0,1}:
                            new_c.H(v)
                        elif(g.phase(v) in {1/2,3/2}):
                            new_c.Rx(-g.phase(v),v)
                        else:
                            new_c.H(v)
                            #theta = zi-(((-1)**xi)*g.phase(v))
                            zi = False
                            for val in signals[v]["z"]:
                                if type(val)==bool:
                                    zi ^= val
                                else:
                                    zi ^= new_c.bits[val]
                            xi = False
                            for val in signals[v]["x"]:
                                if type(val)==bool:
                                    xi ^= val
                                else:
                                    xi ^= new_c.bits[val]
                            if type(zi) == bool:
                                if zi:
                                    new_c.X(v)
                            else:
                                new_c.X(v, condition=zi)
                            if type(xi) == bool:
                                if xi:
                                    new_c.Rx(g.phase(v), v)
                                else:        
                                    new_c.Rx(-g.phase(v), v)
                            else:
                                new_c.Rx(g.phase(v), v, condition=xi)
                                new_c.Rx(-g.phase(v),v, condition=(xi^True))
                        new_c.Measure(v,v)
                        reset_list.append(v)
                if len(l_list)>1:
                    new_c.add_barrier(list(g.vertices()),list(g.vertices()))
                for v in reset_list:
                    #Add command to reset qubit 'v'
                    pass
                for v in l_list[0]:
                    zi = False
                    for val in signals[v]["z"]:
                        if type(val)==bool:
                            zi ^= val
                        else:
                            zi ^= new_c.bits[val]
                    xi = False
                    for val in signals[v]["x"]:
                        if type(val)==bool:
                            xi ^= val
                        else:
                            xi ^= new_c.bits[val]
                    if type(xi) == bool:
                        if xi:
                            new_c.X(v)
                    else:
                        new_c.X(v, condition=xi)
                    if type(zi) == bool:
                        if zi:
                            new_c.Z(v)
                    else:
                        new_c.Z(v, condition=zi)
                    if g.phase(v) == 1:
                        new_c.Z(v)
                    elif(g.phase(v) == 0):
                        pass
                    else:
                        new_c.Rz(-g.phase(v),v)
        return new_c