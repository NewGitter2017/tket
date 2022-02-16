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

import pyzx
from pytket.circuit import Circuit, Qubit
from pytket.mbqc_compile.mpattern import MPattern
from pytket.routing import SquareGrid

def test_init() -> None:
    c1 = Circuit(6)
    c1.H(0)
    c1.CZ(3,2)
    c1.H(3)
    c1.CX(0,3)
    mp1 = MPattern(c1)
    assert mp1.c.n_qubits == 6

def test_zx_diagram() -> None:
    c1 = Circuit(4)
    c1.H(0)
    c1.CZ(3,2)
    c1.H(3)
    c1.CX(0,3)
    mp1 = MPattern(c1)
    (g1,io_map) = mp1.zx_diagram()
    assert type(g1) == pyzx.graph.graph_s.GraphS
    assert len(g1.edge_set()) == 4
    assert io_map["i"][c1.qubits[0]] == 0
    assert io_map["i"][c1.qubits[1]] in [3,4,5]
    assert io_map["i"][c1.qubits[2]] in [1,2,4]
    assert io_map["i"][c1.qubits[3]] in [2,3]
    assert io_map["o"][c1.qubits[0]] in [1,2,3,4]
    assert io_map["o"][c1.qubits[1]] in [3,4,5]
    assert io_map["o"][c1.qubits[2]] in [1,2,4]
    assert io_map["o"][c1.qubits[3]] in [3,4,5]
    assert io_map["o"][c1.qubits[3]] > io_map["i"][c1.qubits[3]]
    assert not (io_map["o"][c1.qubits[0]] == io_map["i"][c1.qubits[2]])
    assert not (io_map["o"][c1.qubits[0]] == io_map["i"][c1.qubits[3]])
    assert not (io_map["o"][c1.qubits[0]] == io_map["o"][c1.qubits[3]])
    assert not (io_map["i"][c1.qubits[2]] == io_map["i"][c1.qubits[3]])
    assert not (io_map["i"][c1.qubits[2]] == io_map["o"][c1.qubits[3]])
    
def test_split_subgraphs() -> None:
    c1 = Circuit(4)
    c1.H(0)
    c1.CZ(3,2)
    c1.H(3)
    c1.CX(0,3)
    mp1 = MPattern(c1)
    (g1,io_map) = mp1.zx_diagram()
    glist = mp1.split_subgraphs(g1,io_map)
    assert len(glist) == 2
    assert ((len(glist[0].vertices()) == 1) and (len(glist[1].vertices()) == 5)) or ((len(glist[0].vertices()) == 5) and (len(glist[1].vertices()) == 1))
    
def test_entangle() -> None:
    c1 = Circuit(4)
    c1.H(0)
    c1.CZ(3,2)
    c1.H(3)
    c1.CX(0,3)
    mp1 = MPattern(c1)
    (g1,io_map) = mp1.zx_diagram()
    c2 = MPattern.entangle(g1)
    assert len(c2.qubits) == 6
    assert len(c2.get_commands()) == 4
    
def test_single_conversion() -> None:
    c1 = Circuit(4)
    c1.H(0)
    c1.CZ(3,2)
    c1.H(3)
    c1.CX(0,3)
    mp1 = MPattern(c1)
    (c2,io_map) = mp1.single_conversion()
    assert len(c2.qubits) == 6
    assert len(c2.bits) == 8
    assert len(c2.get_commands()) == 14
    assert len(c2.get_commands()[0].args) == 2
    assert len(c2.get_commands()[1].args) == 2
    assert len(c2.get_commands()[2].args) == 2
    assert len(c2.get_commands()[3].args) == 1
    assert len(c2.get_commands()[4].args) == 2
    assert len(c2.get_commands()[5].args) == 2
    assert len(c2.get_commands()[6].args) == 1
    assert len(c2.get_commands()[7].args) == 2
    assert len(c2.get_commands()[8].args) == 1
    assert len(c2.get_commands()[9].args) == 2
    assert len(c2.get_commands()[10].args) == 2
    assert len(c2.get_commands()[11].args) == 1
    assert len(c2.get_commands()[12].args) == 3
    assert len(c2.get_commands()[13].args) == 2
    c = Circuit(0)
    qubits = [Qubit("a",0), Qubit("b",0), Qubit("b", 23), Qubit("a", 24), Qubit("c", 7), Qubit("d", 7)]
    for q in qubits:
       c.add_qubit(q)
       c.H(q)
    c.CX(qubits[5], qubits[3])
    c.CX(qubits[0], qubits[1])
    c.CX(qubits[2], qubits[4])
    c.CZ(qubits[2], qubits[3])
    c.CZ(qubits[4], qubits[0])
    mp = MPattern(c)
    (c_,io_map_) = mp.single_conversion()
    assert len(c_.qubits) == 12
    assert len(c_.bits) == 22
    assert len(c_.get_commands()) == 49
    assert len(c_.get_commands()[0].args) == 2
    assert len(c_.get_commands()[9].args) == 2
    assert len(c_.get_commands()[10].args) == 1
    assert len(c_.get_commands()[13].args) == 2
    assert len(c_.get_commands()[16].args) == 2
    assert len(c_.get_commands()[17].args) == 1
    assert len(c_.get_commands()[22].args) == 2
    assert len(c_.get_commands()[23].args) == 1
    assert len(c_.get_commands()[28].args) == 3
    assert len(c_.get_commands()[29].args) == 1
    assert len(c_.get_commands()[30].args) == 1
    assert len(c_.get_commands()[48].args) == 2
    
def test_multi_conversion() -> None:
    c1 = Circuit(4)
    c1.H(0)
    c1.CZ(0,1)
    c1.T(2)
    c1.X(1)
    c1.CX(0,2)
    c1.H(2)
    c1.T(2)
    c1.CZ(2,1)
    c1.CX(1,0)
    c1.T(0)
    c1.T(1)
    c1.CZ(0,2)
    c1.CZ(1,3)
    c1.T(0)
    c1.T(2)
    c1.T(3)
    c1.X(1)
    mp1 = MPattern(c1)
    output = mp1.multi_conversion(n=2,strategy="Depth")
    assert len(output) == 2
    output = mp1.multi_conversion(n=3,strategy="Depth")
    assert len(output) == 3
    output = mp1.multi_conversion(n=4,strategy="Depth")
    assert len(output) == 4
    output = mp1.multi_conversion(n=5,strategy="Depth")
    assert len(output) == 5
    output = mp1.multi_conversion(n=6,strategy="Depth")
    assert len(output) == 5
    output = mp1.multi_conversion(n=2,strategy="Gates")
    assert len(output) == 2
    output = mp1.multi_conversion(n=3,strategy="Gates")
    assert len(output) == 2
    output = mp1.multi_conversion(n=4,strategy="Gates")
    assert len(output) == 3
    output = mp1.multi_conversion(n=5,strategy="Gates")
    assert len(output) == 3
    output = mp1.multi_conversion(n=6,strategy="Gates")
    assert len(output) == 3
    output = mp1.multi_conversion(n=7,strategy="Gates")
    assert len(output) == 4
    
def test_unrouted_conversion() -> None:
    c1 = Circuit(4)
    c1.H(0)
    c1.CZ(0,1)
    c1.T(2)
    c1.X(1)
    c1.CX(0,2)
    c1.H(2)
    c1.T(2)
    c1.CZ(2,1)
    c1.CX(1,0)
    c1.T(0)
    c1.T(1)
    c1.CZ(0,2)
    c1.CZ(1,3)
    c1.T(0)
    c1.T(2)
    c1.T(3)
    c1.X(1)
    mp1 = MPattern(c1)
    (nc,nm) = mp1.unrouted_conversion(2,"Gates")
    assert len(nc.get_commands()) == 50
    q = nc.qubits
    i = nm["i"]
    o = nm["o"]
    assert i[q[0]] == q[5]
    assert i[q[1]] == q[0]
    assert i[q[2]] == q[1]
    assert i[q[3]] == q[7]
    assert o[q[0]] == q[6]
    assert o[q[1]] == q[5]
    assert o[q[2]] == q[2]
    assert o[q[3]] == q[7]
    
def test_routed_conversion_sequential() -> None:
    c1 = Circuit(4)
    c1.H(0)
    c1.CZ(0,1)
    c1.T(2)
    c1.X(1)
    c1.CX(0,2)
    c1.H(2)
    c1.T(2)
    c1.CZ(2,1)
    c1.CX(1,0)
    c1.T(0)
    c1.T(1)
    c1.CZ(0,2)
    c1.CZ(1,3)
    c1.T(0)
    c1.T(2)
    c1.T(3)
    c1.X(1)
    sg = SquareGrid(3,3)
    c1.rename_units({c1.qubits[0]:sg.nodes[4]})
    (nc_seq,nm_seq) = MPattern(c1).routed_conversion(sg,3,"Depth","Sequential")
    assert len(nc_seq.get_commands()) == 71
    i = nm_seq["i"]
    o = nm_seq["o"]
    dummy_circ = Circuit(4)
    assert i[sg.nodes[4]] == sg.nodes[4]
    assert i[dummy_circ.qubits[1]] == sg.nodes[3]
    assert i[dummy_circ.qubits[2]] == sg.nodes[2]
    assert i[dummy_circ.qubits[3]] == sg.nodes[8]
    assert o[sg.nodes[4]] == sg.nodes[6]
    assert o[dummy_circ.qubits[1]] == sg.nodes[0]
    assert o[dummy_circ.qubits[2]] == sg.nodes[4]
    assert o[dummy_circ.qubits[3]] == sg.nodes[8]
    
def test_routed_conversion_separate() -> None:
    c1 = Circuit(4)
    c1.H(0)
    c1.CZ(0,1)
    c1.T(2)
    c1.X(1)
    c1.CX(0,2)
    c1.H(2)
    c1.T(2)
    c1.CZ(2,1)
    c1.CX(1,0)
    c1.T(0)
    c1.T(1)
    c1.CZ(0,2)
    c1.CZ(1,3)
    c1.T(0)
    c1.T(2)
    c1.T(3)
    c1.X(1)
    sg = SquareGrid(3,3)
    c1.rename_units({c1.qubits[0]:sg.nodes[4]})
    (nc_sep,nm_sep) = MPattern(c1).routed_conversion(sg,3,"Depth","Separate")
    assert len(nc_sep.get_commands()) == 76
    i = nm_sep["i"]
    o = nm_sep["o"]
    dummy_circ = Circuit(4)
    assert i[sg.nodes[4]] == sg.nodes[4]
    assert i[dummy_circ.qubits[1]] == sg.nodes[3]
    assert i[dummy_circ.qubits[2]] == sg.nodes[2]
    assert i[dummy_circ.qubits[3]] == sg.nodes[8]
    assert o[sg.nodes[4]] == sg.nodes[0]
    assert o[dummy_circ.qubits[1]] == sg.nodes[2]
    assert o[dummy_circ.qubits[2]] == sg.nodes[3]
    assert o[dummy_circ.qubits[3]] == sg.nodes[5]