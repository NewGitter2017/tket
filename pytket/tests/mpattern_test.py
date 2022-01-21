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
    assert io_map["i"][c1.qubits[1]] in [4,5]
    assert io_map["i"][c1.qubits[2]] in [1,2]
    assert io_map["i"][c1.qubits[3]] in [2,3]
    assert io_map["o"][c1.qubits[0]] in [1,2,3,4]
    assert io_map["o"][c1.qubits[1]] in [4,5]
    assert io_map["o"][c1.qubits[2]] in [1,2]
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
    assert len(c2.get_commands()) == 5
    
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
    assert len(c2.get_commands()[3].args) == 2
    assert len(c2.get_commands()[4].args) == 12
    assert len(c2.get_commands()[5].args) == 1
    assert len(c2.get_commands()[6].args) == 1
    assert len(c2.get_commands()[7].args) == 2
    assert len(c2.get_commands()[8].args) == 2
    assert len(c2.get_commands()[9].args) == 10
    assert len(c2.get_commands()[10].args) == 2
    assert len(c2.get_commands()[11].args) == 3
    assert len(c2.get_commands()[12].args) == 2
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
    assert len(c_.get_commands()) == 45
    assert len(c_.get_commands()[0].args) == 2
    assert len(c_.get_commands()[10].args) == 2
    assert len(c_.get_commands()[11].args) == 24
    assert len(c_.get_commands()[12].args) == 1
    assert len(c_.get_commands()[17].args) == 1
    assert len(c_.get_commands()[18].args) == 2
    assert len(c_.get_commands()[23].args) == 2
    assert len(c_.get_commands()[24].args) == 24
    assert len(c_.get_commands()[25].args) == 2
    assert len(c_.get_commands()[26].args) == 3
    assert len(c_.get_commands()[44].args) == 2