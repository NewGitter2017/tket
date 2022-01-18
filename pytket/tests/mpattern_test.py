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
from pytket.circuit import Circuit as Circuit
from pytket.mbqc_compile.mpattern import MPattern
#import pytest  # type: ignore
#import platform
#from typing import Any


def test_init() -> None:
    c1 = Circuit(6)
    c1.H(0)
    c1.CZ(3,2)
    c1.H(3)
    c1.CX(0,3)
    mp1 = MPattern(c1)
    assert mp1.qubits == 6
    assert mp1.inputs == {0:0,1:1,2:2,3:3,4:4,5:5}
    assert mp1.outputs == {0:0,1:1,2:2,3:3,4:4,5:5}


def test_zx_convert() -> None:
    c1 = Circuit(4)
    c1.H(0)
    c1.CZ(3,2)
    c1.H(3)
    c1.CX(0,3)
    zxc = MPattern.zx_convert(c1)
    assert type(zxc) == pyzx.circuit.Circuit
    assert zxc.qubits == 4
    assert type(zxc.gates[0]) == pyzx.circuit.gates.HAD
    assert type(zxc.gates[1]) == pyzx.circuit.gates.CZ
    assert type(zxc.gates[2]) == pyzx.circuit.gates.HAD
    assert type(zxc.gates[3]) == pyzx.circuit.gates.CNOT


def test_zx_diagram() -> None:
    c1 = Circuit(4)
    c1.H(0)
    c1.CZ(3,2)
    c1.H(3)
    c1.CX(0,3)
    mp1 = MPattern(c1)
    zxc = MPattern.zx_convert(c1)
    g1 = mp1.zx_diagram(zxc)
    assert type(g1) == pyzx.graph.graph_s.GraphS
    assert len(g1.edge_set()) == 4
    assert mp1.inputs[0] == 0
    assert mp1.inputs[1] in [4,5]
    assert mp1.inputs[2] in [1,2]
    assert mp1.inputs[3] in [2,3]
    assert mp1.outputs[0] in [1,2,3,4]
    assert mp1.outputs[1] in [4,5]
    assert mp1.outputs[2] in [1,2]
    assert mp1.outputs[3] in [3,4,5]
    assert mp1.outputs[3] > mp1.inputs[3]
    assert not (mp1.outputs[0] == mp1.inputs[2])
    assert not (mp1.outputs[0] == mp1.inputs[3])
    assert not (mp1.outputs[0] == mp1.outputs[3])
    assert not (mp1.inputs[2] == mp1.inputs[3])
    assert not (mp1.inputs[2] == mp1.outputs[3])
    
def test_split_subgraphs() -> None:
    c1 = Circuit(4)
    c1.H(0)
    c1.CZ(3,2)
    c1.H(3)
    c1.CX(0,3)
    mp1 = MPattern(c1)
    zxc = MPattern.zx_convert(c1)
    g1 = mp1.zx_diagram(zxc)
    glist = mp1.split_subgraphs(g1)
    assert mp1.inputs[0] == 0
    assert mp1.inputs[1] in [4,5]
    assert mp1.inputs[2] in [1,2]
    assert mp1.inputs[3] in [2,3]
    assert mp1.outputs[0] in [1,2,3,4]
    assert mp1.outputs[1] in [4,5]
    assert mp1.outputs[2] in [1,2]
    assert mp1.outputs[3] in [3,4,5]
    assert mp1.outputs[3] > mp1.inputs[3]
    assert not (mp1.outputs[0] == mp1.inputs[2])
    assert not (mp1.outputs[0] == mp1.inputs[3])
    assert not (mp1.outputs[0] == mp1.outputs[3])
    assert not (mp1.inputs[2] == mp1.inputs[3])
    assert not (mp1.inputs[2] == mp1.outputs[3])
    assert len(glist) == 2
    assert ((len(glist[0].vertices()) == 1) and (len(glist[1].vertices()) == 5)) or ((len(glist[0].vertices()) == 5) and (len(glist[1].vertices()) == 1))
    
def test_entangle() -> None:
    c1 = Circuit(4)
    c1.H(0)
    c1.CZ(3,2)
    c1.H(3)
    c1.CX(0,3)
    mp1 = MPattern(c1)
    zxc = MPattern.zx_convert(c1)
    g1 = mp1.zx_diagram(zxc)
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
    c2 = mp1.single_conversion()
    assert len(c2.bits) == 6