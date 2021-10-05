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

#ifndef _TKET__unit_downcast_H_
#define _TKET__unit_downcast_H_
#include <pybind11/pybind11.h>

#include "Utils/UnitID.hpp"

namespace pybind11 {

/** Enable automatic downcasting of UnitIDs as required for some Circuit methods
 */
template <>
struct polymorphic_type_hook<tket::UnitID> {
  static const void* get(const tket::UnitID* src, const std::type_info*& type) {
    if (src) {
      type = &typeid(tket::Qubit);
    } else
      type = nullptr;
    return src;
  }
};
}  // namespace pybind11

#endif  //_TKET__unit_downcast_H_
