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

from pytket.mapping import MappingManager, LexiRouteRoutingMethod, get_token_swapping_network
from pytket.transform import Transform
from pytket.circuit import Circuit, Command, OpType, Bit, Qubit
from pytket.extensions.pyzx import tk_to_pyzx
from pyzx.simplify import interior_clifford_simp
from pyzx.graph.graph_s import GraphS
from pyzx.circuit.graphparser import circuit_to_graph
from pyzx.gflow import gflow
from pytket.architecture import Architecture, SquareGrid, FullyConnected
from pytket.placement import place_with_map
import math
from pytket.passes import DefaultMappingPass, RoutingPass
from pytket.predicates import CompilationUnit
"""

"""

class MPattern:
    """
    Class with tools to convert a pytket circuit into a new pytket circuit
    with lower depth and higher width, by using MBQC techniques.
    """
    def __init__(self, c: Circuit) -> None:
        """
        Initialises the MPattern object by giving it a pytket circuit.
        
        :param c:       An arbitrary pytket circuit that we want converted to 
                            a measurement pattern.
        :param type:    Circuit
        """
        self.c = c
    
    def single_conversion(self) -> Circuit:
        """
        Converts a pytket circuit to another with reduced depth and higher width.
        
        :returns:       A tuple containing the new circuit and the i/o map.
        :rtype:         tuple
        """
        (g,io_map) = self.zx_diagram() #Creates a simplified ZX diagram.
        subs = self.split_subgraphs(g,io_map) #Returns list of disjoint subgraphs.
        cz_circ = MPattern.entangle(g) #Creates the CZ circuit part of the pattern.
        m_circ = MPattern.correct(subs) #Circuit implementing measurements/corrections.
        cz_circ.add_circuit(m_circ,[])
        return (cz_circ, io_map)
    
    def multi_conversion(self, n: int = 1, strategy: str = "Gates", ) -> list:
        #Currently 'strategy' takes a 'str' type parameter that is either "Depth"
        #or "Gates". Might want to consider additional strategies and switch to
        #an enum in the future.
        """
        Splits a pytket circuit into 'n' subcircuits, each subcircuit containing
        either an approximately equal depth or an approximately equal number of
        non-Clifford gates. Then converts each subcircuit to a measurement pattern,
        extracts a new circuit from the measurement pattern, and ultimately
        returns a list of tuples containing the new circuits and the dictionaries
        mapping the inputs and outputs of the new circuits to the original.
        
        :param n:        Number of segments to attempt to split into (May return fewer).
        :param type:     int
        
        :param strategy: Splitting strategy either by "Depth" or by "Gates".
        :param type:     str
        
        :returns:        A list of tuples containing circuits and i/o maps.
        :rtype:          list
        """
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
                for depth_list in depth_structure[done_depth:finish_at]:
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
                if done_depth+added_depths >= len(depth_structure):
                    break
                else:
                    done_depth += added_depths
        return output
    
    """
    def routed_conversion(self, arch: Architecture, n: int = 1, splitStrat: str = "Gates", routeStrat: str = "Separate") -> list:
        pattern_list = self.multi_conversion(n, splitStrat)
        new_pattern_list = []
        new_c = Circuit()
        for q in arch.nodes:
            new_c.add_qubit(q)
        for pattern in pattern_list:
            cu = CompilationUnit(pattern[0])
            DefaultMappingPass(arch).apply(cu)
            for k in pattern[1]["i"].keys():
                pattern[1]["i"][k] = cu.initial_map[pattern[0].qubits[pattern[1]["i"][k]]]
            for k in pattern[1]["o"].keys():
                pattern[1]["o"][k] = cu.final_map[pattern[0].qubits[pattern[1]["o"][k]]]
            segment_circuit = cu.circuit.copy()
            print(segment_circuit.valid_connectivity(arch,False,True))
            new_tuple = (segment_circuit,pattern[1])
            new_pattern_list.append(new_tuple)
            for q in arch.nodes:
                if not q in segment_circuit.qubits:
                    segment_circuit.add_qubit(q)
            if len(new_pattern_list)>1:
                previous_qubits = new_pattern_list[-2][1]["o"]
                current_qubits = new_pattern_list[-1][1]["i"]
                #print(new_c.valid_connectivity(arch,True))
                for key in previous_qubits.keys():
                    if not previous_qubits[key]==current_qubits[key]:
                        new_c.SWAP(previous_qubits[key],current_qubits[key])
            new_c.add_circuit(segment_circuit,[])
        #print(new_c.valid_connectivity(arch,False))
        final_cu = CompilationUnit(new_c)
        DefaultMappingPass(arch).apply(final_cu)
        final_map = {"i":new_pattern_list[0][1]["i"],"o":new_pattern_list[-1][1]["o"]}
        #print(len(new_c.get_commands()),new_c.get_commands())
        #print(len(final_cu.circuit.get_commands()),final_cu.circuit.get_commands())
        #print(len(final_cu.circuit.get_commands()))
        for k in final_map["i"].keys():
            final_map["i"][k] = final_cu.initial_map[final_map["i"][k]]
        for k in final_map["o"].keys():
            final_map["o"][k] = final_cu.final_map[final_map["o"][k]]
        #return new_pattern_list
        return (final_cu.circuit,final_map)
        """
        
    def unrouted_conversion(self, n: int = 1, splitStrat: str = "Gates") -> tuple:
        pattern_list = self.multi_conversion(n, splitStrat)
        new_c = Circuit()
        prev_map = {}
        for p in range(len(pattern_list)):
            (curr_circ,curr_map) = pattern_list[p]
            if len(prev_map)==0:
                new_c.add_circuit(curr_circ,[])
                prev_map = curr_map.copy()
            else:
                q_map = {}
                for k in prev_map["o"].keys():
                    q_map[curr_circ.qubits[curr_map["i"][k]]] = new_c.qubits[prev_map["o"][k]]
                prev_ancillas = []
                curr_ancillas = []
                for q in new_c.qubits:
                    if not q in list(q_map.values()):
                        prev_ancillas.append(q)
                for q in curr_circ.qubits:
                    if not q in q_map.keys():
                        curr_ancillas.append(q)
                while len(prev_ancillas)>0 and len(curr_ancillas)>0:
                    q_map[curr_ancillas.pop()]=prev_ancillas.pop()
                if len(curr_ancillas)>0:
                    unused_id_pool = []
                    for q in curr_circ.qubits:
                        if not q in list(q_map.values()):
                            unused_id_pool.append(q)
                    for q in curr_circ.qubits:
                        if not q in q_map.keys():
                            q_map[q] = unused_id_pool.pop()
                for k in curr_map["o"].keys():
                    curr_map["o"][k] = q_map[curr_circ.qubits[curr_map["o"][k]]]
                curr_circ.rename_units(q_map)
                new_c.add_circuit(curr_circ,[])
                for k in curr_map["o"].keys():
                    curr_map["o"][k] = new_c.qubits.index(curr_map["o"][k])
                prev_map = curr_map.copy()
        final_map = {"i":pattern_list[0][1]["i"],"o":prev_map["o"]}
        for io in final_map.keys():
            for q in final_map[io].keys():
                final_map[io][q] = new_c.qubits[final_map[io][q]]
        return (new_c,final_map)
    
    def routed_conversion(self, arch: Architecture = None, n: int = 1, splitStrat: str = "Gates", routeStrat: str = "Separate") -> tuple:
        if (type(arch)==type(FullyConnected(0))) or (arch == None):
            return self.unrouted_conversion(n,splitStrat)
        else:
            pattern_list = self.multi_conversion(n, splitStrat)
            new_c = Circuit()
            for q in arch.nodes:
                new_c.add_qubit(q)
            for p in range(len(pattern_list)):
                print("________________________________________________________")
                print("SEGMENT ", str(p), " :")
                print("________________________________________________________")
                print("Map:")
                print(pattern_list[p][1])
                print("Unrouted circuit:")
                print(pattern_list[p][0].get_commands())
                (curr_circ,curr_map) = pattern_list[p]
                if p>0:
                    (prev_circ,prev_map) = pattern_list[p-1]
                    route_map = {}
                    for k in curr_map["i"].keys():
                        route_map[curr_circ.qubits[curr_map["i"][k]]]=prev_map["o"][k]
                        curr_map["i"][k] = prev_map["o"][k]
                    output_list = [curr_circ.qubits[q] for q in list(curr_map["o"].values())]
                    already_placed = 0
                    unplaced_qubit_map = {}
                    for q in range(curr_circ.n_qubits):
                        if curr_circ.qubits[q] in output_list:
                            if curr_circ.qubits[q] in route_map.keys():
                                already_placed += 1
                                for k in curr_map["o"].keys():
                                    if type(curr_map["o"][k])==int:
                                        if curr_map["o"][k]==q:
                                            curr_map["o"][k]=route_map[curr_circ.qubits[q]]
                                            break
                            else:
                                for k in curr_map["o"].keys():
                                    if type(curr_map["o"][k])==int:
                                        if curr_map["o"][k]==q:
                                            unplaced_qubit_map[k] = len(route_map.keys()) + q - already_placed
                                            break
                        elif curr_circ.qubits[q] in route_map.keys():
                            already_placed += 1
                    place_with_map(curr_circ,route_map)
                    for unplaced_qubit in unplaced_qubit_map.keys():
                        curr_map["o"][unplaced_qubit]=curr_circ.qubits[unplaced_qubit_map[unplaced_qubit]]  
                else:
                    for i in curr_map["i"].keys():
                        curr_map["i"][i] = curr_circ.qubits[curr_map["i"][i]]
                    for o in curr_map["o"].keys():
                        curr_map["o"][o] = curr_circ.qubits[curr_map["o"][o]]
                #for k in curr_map["i"].keys():
                #    temp = Bit(name="i" + str(k))
                #    curr_circ.add_bit(temp)
                #    curr_circ.add_barrier(units = [curr_map["i"][k], temp])
                #for k in curr_map["o"].keys():
                #    temp = Bit(name="o" + str(k))
                #    curr_circ.add_bit(temp)
                #    curr_circ.add_barrier(units = [curr_map["o"][k], temp])
                #mm = MappingManager(arch)
                #routed_c = mm.route_circuit(curr_circ, [LexiRouteRoutingMethod()])
                #routed_c = route(curr_circ,arch)
                #for q in routed_c.qubits:
                #    print(q,type(q))
                #barriers = routed_c.commands_of_type(OpType.Barrier)
                #for k in curr_map["i"].keys():
                #    for b in barriers:
                #        if len(b.bits) == 1 and b.bits[0].reg_name == "i" + str(k):
                #            curr_map["i"][k] = b.qubits[0]
                #            print("a barrier", b)
                #            break
                #for k in curr_map["o"].keys():
                #    for b in barriers:
                #        if len(b.bits) == 1 and b.bits[0].reg_name == "o" + str(k):
                #            curr_map["o"][k] = b.qubits[0]
                #            break
                #permutation = {x:x for x in routed_c.qubits}
                #for com in routed_c.commands_of_type(OpType.SWAP):
                #    permutation[com.qubits[0]] = com.qubits[1]
                #    permutation[com.qubits[1]] = com.qubits[0]
                #for k in curr_map["i"].keys():
                #    curr_map["i"][k] = permutation[curr_map["i"][k]]
                #print("Routed circuit:")
                #print(routed_c.get_commands())
                #print("Map:")
                #print(curr_map)
                
                #print("PERMUTATION:")
                #print(permutation)
                #print(routed_c.get_commands())
                #return(curr_circ,arch)
                cu = CompilationUnit(curr_circ)
                #DefaultMappingPass(arch).apply(cu)
                RoutingPass(arch).apply(cu)
                #if p==1:
                    #print("-------HERE")
                    #print(cu.initial_map)
                    #print("------END HERE")    
                #for k in curr_map["i"].keys():
                #    curr_map["i"][k] = cu.initial_map[curr_map["i"][k]]
                #for k in curr_map["o"].keys():
                #    curr_map["o"][k] = cu.final_map[curr_map["o"][k]]
                #if p==1:
                    #print("Partially mapped circuit qubits:")
                    #print(curr_circ.qubits)
                    #print("Partially mapped circuit:")
                    #print(curr_circ.get_commands())
                    #print("Routed circuit:")
                    #print(route(curr_circ,arch).get_commands())
                    #print("Map:")
                    #print(curr_map)
                #if p==0:
                    #print("Routed circuit:")
                    #print(cu.circuit.get_commands())
                    #print("Map:")
                    #print(curr_map)
                new_tuple = (cu.circuit,curr_map)
                #new_tuple = (routed_c,curr_map)
                pattern_list[p] = new_tuple
                segment_circuit = cu.circuit.copy()
                for q in arch.nodes:
                    if not q in segment_circuit.qubits:
                        segment_circuit.add_qubit(q)
                new_c.add_circuit(segment_circuit,[])
            final_map = {"i":pattern_list[0][1]["i"],"o":pattern_list[-1][1]["o"]}
            return (new_c,final_map)
        
        def routed_conversion_separate(self, pattern_list: list, arch: Architecture = None) -> tuple:
            new_c = Circuit()
            for q in arch.nodes:
                new_c.add_qubit(q)
            for p in range(len(pattern_list)):
                (curr_circ,curr_map) = pattern_list[p]
                cu = CompilationUnit(curr_circ)
                DefaultMappingPass(arch).apply(cu)
                for k in curr_map["i"].keys():
                    curr_map["i"][k] = cu.initial_map[curr_circ.qubits[curr_map["i"][k]]]
                for k in curr_map["o"].keys():
                    curr_map["o"][k] = cu.final_map[curr_circ.qubits[curr_map["o"][k]]]
                new_tuple = (cu.circuit,curr_map)
                pattern_list[p] = new_tuple
                segment_circuit = cu.circuit.copy() 
                for q in arch.nodes:
                    if not q in segment_circuit.qubits:
                        segment_circuit.add_qubit(q)
                if p>0:
                    prev_map = pattern_list[p-1][1]
                    matching_dict = {}
                    for k in curr_map["i"].keys():
                        matching_dict[prev_map["o"][k]]=curr_map["i"][k]
                    swaps_as_pairs = get_token_swapping_network(arch, matching_dict)
                    for pair in swaps_as_pairs:
                        new_c.SWAP(new_c.qubits.index(pair[0]),new_c.qubits.index(pair[1]))                
                new_c.add_circuit(segment_circuit,[])
            final_map = {"i":pattern_list[0][1]["i"],"o":pattern_list[-1][1]["o"]}
            return (new_c,final_map)
        
        def routed_conversion_sequential(self, pattern_list: list, arch: Architecture = None) -> tuple:
            new_c = Circuit()
            for q in arch.nodes:
                new_c.add_qubit(q)
            for p in range(len(pattern_list)):
                (curr_circ,curr_map) = pattern_list[p]
                if p>0:
                    (prev_circ,prev_map) = pattern_list[p-1]
                    route_map = {}
                    for k in curr_map["i"].keys():
                        route_map[curr_circ.qubits[curr_map["i"][k]]]=prev_map["o"][k]
                        curr_map["i"][k] = prev_map["o"][k]
                    output_list = [curr_circ.qubits[q] for q in list(curr_map["o"].values())]
                    already_placed = 0
                    unplaced_qubit_map = {}
                    for q in range(curr_circ.n_qubits):
                        if curr_circ.qubits[q] in output_list:
                            if curr_circ.qubits[q] in route_map.keys():
                                already_placed += 1
                                for k in curr_map["o"].keys():
                                    if type(curr_map["o"][k])==int:
                                        if curr_map["o"][k]==q:
                                            curr_map["o"][k]=route_map[curr_circ.qubits[q]]
                                            break
                            else:
                                for k in curr_map["o"].keys():
                                    if type(curr_map["o"][k])==int:
                                        if curr_map["o"][k]==q:
                                            unplaced_qubit_map[k] = len(route_map.keys()) + q - already_placed
                                            break
                        elif curr_circ.qubits[q] in route_map.keys():
                            already_placed += 1
                    place_with_map(curr_circ,route_map)
                    for unplaced_qubit in unplaced_qubit_map.keys():
                        curr_map["o"][unplaced_qubit]=curr_circ.qubits[unplaced_qubit_map[unplaced_qubit]]  
                else:
                    for i in curr_map["i"].keys():
                        curr_map["i"][i] = curr_circ.qubits[curr_map["i"][i]]
                    for o in curr_map["o"].keys():
                        curr_map["o"][o] = curr_circ.qubits[curr_map["o"][o]]
                cu = CompilationUnit(curr_circ)
                DefaultMappingPass(arch).apply(cu)
                new_tuple = (cu.circuit,curr_map)
                pattern_list[p] = new_tuple
                segment_circuit = cu.circuit.copy()
                for q in arch.nodes:
                    if not q in segment_circuit.qubits:
                        segment_circuit.add_qubit(q)
                new_c.add_circuit(segment_circuit,[])
            final_map = {"i":pattern_list[0][1]["i"],"o":pattern_list[-1][1]["o"]}
            return (new_c,final_map)
        
    @staticmethod
    def is_Clifford(aGate: Command) -> bool:
        """
        Placeholder method which checks if a gate is Clifford or not.
        
        :param n:        Number of segments to attempt to split into (May return fewer).
        :param type:     Command
        
        :returns:        The result of the check.
        :rtype:          bool
        """
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
        """
        Converts a pytket circuit to a list containing 'x' lists, each containing
        'y' gates, where 'x' is the depth of the circuit and 'y' is the number
        of gates acting in a given timestep. Essentially we split the circuit
        into a list of 'timeslices'.
        
        :returns:       A list containing lists of gates.
        :rtype:         list
        """
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
   
    @staticmethod
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
    
    @staticmethod
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
                                new_c.add_gate(OpType.Reset, [v])
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
                        for u in g.vertices()-{v}:
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
                    new_c.add_gate(OpType.Reset, [v])
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