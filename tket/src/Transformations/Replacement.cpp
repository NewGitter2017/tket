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

#include "Replacement.hpp"

#include "Circuit/CircPool.hpp"
#include "Circuit/CircUtils.hpp"
#include "Transform.hpp"

namespace tket {

Circuit CX_circ_from_multiq(const Op_ptr op) {
  OpDesc desc = op->get_desc();
  if (!desc.is_gate())
    throw NotImplemented(
        "Can only build replacement circuits for basic gates; given " +
        desc.name());
  unsigned n_qubits = op->n_qubits();
  switch (desc.type()) {
    case OpType::CZ: {
      return CircPool::CZ_using_CX();
    }
    case OpType::CY: {
      return CircPool::CY_using_CX();
    }
    case OpType::CH: {
      return CircPool::CH_using_CX();
    }
    case OpType::CV: {
      return CircPool::CV_using_CX();
    }
    case OpType::CVdg: {
      return CircPool::CVdg_using_CX();
    }
    case OpType::CSX: {
      return CircPool::CSX_using_CX();
    }
    case OpType::CSXdg: {
      return CircPool::CSXdg_using_CX();
    }
    case OpType::CRz: {
      return CircPool::CRz_using_CX(op->get_params()[0]);
    }
    case OpType::CRx: {
      return CircPool::CRx_using_CX(op->get_params()[0]);
    }
    case OpType::CRy: {
      return CircPool::CRy_using_CX(op->get_params()[0]);
    }
    case OpType::CU1: {
      return CircPool::CU1_using_CX(op->get_params()[0]);
    }
    case OpType::CU3: {
      std::vector<Expr> params = op->get_params();
      return CircPool::CU3_using_CX(params[0], params[1], params[2]);
    }
    case OpType::SWAP: {
      return CircPool::SWAP_using_CX_0();
    }
    case OpType::CSWAP: {
      return CircPool::CSWAP_using_CX();
    }
    case OpType::PhaseGadget: {
      return phase_gadget(n_qubits, op->get_params()[0], CXConfigType::Snake);
    }
    case OpType::ISWAP: {
      return CircPool::ISWAP_using_CX(op->get_params()[0]);
    }
    case OpType::XXPhase: {
      return CircPool::XXPhase_using_CX(op->get_params()[0]);
    }
    case OpType::ECR: {
      return CircPool::ECR_using_CX();
    }
    case OpType::ZZMax: {
      return CircPool::ZZMax_using_CX();
    }
    case OpType::ZZPhase: {
      return CircPool::ZZPhase_using_CX(op->get_params()[0]);
    }
    case OpType::YYPhase: {
      return CircPool::YYPhase_using_CX(op->get_params()[0]);
    }
    case OpType::XXPhase3: {
      return CircPool::XXPhase3_using_CX(op->get_params()[0]);
    }
    case OpType::BRIDGE: {
      return CircPool::BRIDGE_using_CX_0();
    }
    case OpType::CnRy: {
      return Transform::decomposed_CnRy(op, n_qubits);
    }
    case OpType::CCX: {
      return CircPool::CCX_normal_decomp();
    }
    case OpType::CnX: {
      return Transform::cnx_normal_decomp(n_qubits - 1);
    }
    case OpType::ESWAP: {
      return CircPool::ESWAP_using_CX(op->get_params()[0]);
    }
    case OpType::FSim: {
      return CircPool::FSim_using_CX(op->get_params()[0], op->get_params()[1]);
    }
    case OpType::Sycamore: {
      return CircPool::FSim_using_CX(0.5, 1. / 6.);
    }
    case OpType::ISWAPMax: {
      return CircPool::ISWAP_using_CX(1.);
    }
    case OpType::PhasedISWAP: {
      return CircPool::PhasedISWAP_using_CX(
          op->get_params()[0], op->get_params()[1]);
    }
    default: {
      throw NotImplemented(
          "Cannot find replacement circuit for OpType::" + desc.name());
    }
  }
}

Circuit CX_ZX_circ_from_op(const Op_ptr op) {
  OpDesc desc = op->get_desc();
  if (!desc.is_gate())
    throw NotImplemented(
        "Can only build replacement circuits for basic gates; given " +
        desc.name());
  switch (desc.type()) {
    case OpType::Z: {
      Circuit replacement(1);
      replacement.add_op<unsigned>(OpType::Rz, 1., {0});
      replacement.add_phase(0.5);
      return replacement;
    }
    case OpType::X: {
      Circuit replacement(1);
      replacement.add_op<unsigned>(OpType::Rx, 1., {0});
      replacement.add_phase(0.5);
      return replacement;
    }
    case OpType::Y: {
      Circuit replacement(1);
      replacement.add_op<unsigned>(OpType::Rz, 1., {0});
      replacement.add_op<unsigned>(OpType::Rx, 1., {0});
      replacement.add_phase(-0.5);
      return replacement;
    }
    case OpType::S: {
      Circuit replacement(1);
      replacement.add_op<unsigned>(OpType::Rz, 0.5, {0});
      replacement.add_phase(0.25);
      return replacement;
    }
    case OpType::Sdg: {
      Circuit replacement(1);
      replacement.add_op<unsigned>(OpType::Rz, -0.5, {0});
      replacement.add_phase(-0.25);
      return replacement;
    }
    case OpType::T: {
      Circuit replacement(1);
      replacement.add_op<unsigned>(OpType::Rz, 0.25, {0});
      replacement.add_phase(0.125);
      return replacement;
    }
    case OpType::Tdg: {
      Circuit replacement(1);
      replacement.add_op<unsigned>(OpType::Rz, -0.25, {0});
      replacement.add_phase(-0.125);
      return replacement;
    }
    case OpType::V: {
      Circuit replacement(1);
      replacement.add_op<unsigned>(OpType::Rx, 0.5, {0});
      return replacement;
    }
    case OpType::Vdg: {
      Circuit replacement(1);
      replacement.add_op<unsigned>(OpType::Rx, -0.5, {0});
      return replacement;
    }
    case OpType::SX: {
      Circuit replacement(1);
      replacement.add_op<unsigned>(OpType::Rx, 0.5, {0});
      replacement.add_phase(0.25);
      return replacement;
    }
    case OpType::SXdg: {
      Circuit replacement(1);
      replacement.add_op<unsigned>(OpType::Rx, -0.5, {0});
      replacement.add_phase(-0.25);
      return replacement;
    }
    case OpType::H: {
      Circuit replacement(1);
      replacement.add_op<unsigned>(OpType::Rz, 0.5, {0});
      replacement.add_op<unsigned>(OpType::Rx, 0.5, {0});
      replacement.add_op<unsigned>(OpType::Rz, 0.5, {0});
      replacement.add_phase(0.5);
      return replacement;
    }
    case OpType::Ry: {
      Expr angle = op->get_params()[0];
      Circuit replacement(1);
      replacement.add_op<unsigned>(OpType::Rz, -0.5, {0});
      replacement.add_op<unsigned>(OpType::Rx, angle, {0});
      replacement.add_op<unsigned>(OpType::Rz, 0.5, {0});
      return replacement;
    }
    case OpType::Rx:
    case OpType::Rz:
    case OpType::Measure:
    case OpType::Collapse: {
      Circuit replacement(1);
      replacement.add_op<unsigned>(op, {0});
      return replacement;
    }
    case OpType::U3: {
      Circuit replacement(1);
      Expr angle_z1 = op->get_params()[2];
      Expr angle_y = op->get_params()[0];
      Expr angle_z2 = op->get_params()[1];
      replacement.add_op<unsigned>(OpType::Rz, angle_z1 - 0.5, {0});
      replacement.add_op<unsigned>(OpType::Rx, angle_y, {0});
      replacement.add_op<unsigned>(OpType::Rz, angle_z2 + 0.5, {0});
      replacement.add_phase((angle_z1 + angle_z2) / 2);
      return replacement;
    }
    case OpType::U2: {
      Circuit replacement(1);
      Expr angle_z1 = op->get_params()[1];
      Expr angle_z2 = op->get_params()[0];
      replacement.add_op<unsigned>(OpType::Rz, angle_z1 - 0.5, {0});
      replacement.add_op<unsigned>(OpType::Rx, 0.5, {0});
      replacement.add_op<unsigned>(OpType::Rz, angle_z2 + 0.5, {0});
      replacement.add_phase((angle_z1 + angle_z2) / 2);
      return replacement;
    }
    case OpType::U1: {
      Circuit replacement(1);
      Expr angle = op->get_params()[0];
      replacement.add_op<unsigned>(OpType::Rz, angle, {0});
      replacement.add_phase(angle / 2);
      return replacement;
    }
    case OpType::PhasedX: {
      Circuit replacement(1);
      Expr theta = op->get_params()[0];
      Expr phi = op->get_params()[1];
      replacement.add_op<unsigned>(OpType::Rz, -phi, {0});
      replacement.add_op<unsigned>(OpType::Rx, theta, {0});
      replacement.add_op<unsigned>(OpType::Rz, phi, {0});
      return replacement;
    }
    case OpType::CX: {
      Circuit replacement(2);
      replacement.add_op<unsigned>(op, {0, 1});
      return replacement;
    }
    case OpType::CY:
    case OpType::CZ:
    case OpType::CH:
    case OpType::CV:
    case OpType::CVdg:
    case OpType::CSX:
    case OpType::CSXdg:
    case OpType::CRz:
    case OpType::CRx:
    case OpType::CRy:
    case OpType::CU1:
    case OpType::CU3:
    case OpType::PhaseGadget:
    case OpType::CCX:
    case OpType::SWAP:
    case OpType::CSWAP:
    case OpType::ECR:
    case OpType::ISWAP:
    case OpType::XXPhase:
    case OpType::XXPhase3:
    case OpType::ZZMax:
    case OpType::ZZPhase:
    case OpType::YYPhase:
    case OpType::CnRy:
    case OpType::CnX:
    case OpType::ESWAP:
    case OpType::FSim:
    case OpType::Sycamore:
    case OpType::ISWAPMax:
    case OpType::BRIDGE: {
      Circuit replacement = CX_circ_from_multiq(op);
      Transform::decompose_ZX().apply(replacement);
      return replacement;
    }
    case OpType::tk1: {
      Circuit replacement(1);
      std::vector<Expr> params = op->get_params();
      replacement.add_op<unsigned>(OpType::Rz, params[2], {0});
      replacement.add_op<unsigned>(OpType::Rx, params[1], {0});
      replacement.add_op<unsigned>(OpType::Rz, params[0], {0});
      return replacement;
    }
    default:
      throw NotImplemented(
          "Cannot find replacement circuit for OpType::" + desc.name());
  }
}

}  // namespace tket