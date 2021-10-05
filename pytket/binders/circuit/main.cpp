// Copyright 2019-2021 Cambridge Quantum Computing
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <sstream>
#include "Utils/UnitID.hpp"

namespace py = pybind11;

namespace tket {

PYBIND11_MODULE(circuit, m) {
  py::enum_<UnitType>(
      m, "UnitType",
      "Enum for data types of units in circuits (e.g. Qubits vs Bits).")
      .value("qubit", UnitType::Qubit, "A single Qubit");
  py::class_<UnitID>(m, "UnitID", "A handle to a computational unit (e.g. qubit, bit)");
  py::class_<Qubit, UnitID>(m, "Qubit", "A handle to a qubit");
}

}  // namespace tket
