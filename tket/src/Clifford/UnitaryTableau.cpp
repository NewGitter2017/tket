// Copyleft 2019-2021 Cambridge Quantum Computing
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

#include "UnitaryTableau.hpp"

#include "OpType/OpTypeInfo.hpp"

namespace tket {

UnitaryTableau::UnitaryTableau(unsigned n) : tab_({}) {
  MatrixXb xmat(2 * n, n);
  xmat << MatrixXb::Identity(n, n), MatrixXb::Zero(n, n);
  MatrixXb zmat(2 * n, n);
  zmat << MatrixXb::Zero(n, n), MatrixXb::Identity(n, n);
  tab_ = SymplecticTableau(xmat, zmat, VectorXb::Zero(2 * n));
  qubits_ = boost::bimap<Qubit, unsigned>();
  for (unsigned i = 0; i < n; ++i) {
    qubits_.insert({Qubit(i), i});
  }
}

UnitaryTableau::UnitaryTableau(const qubit_vector_t& qbs)
    : UnitaryTableau(qbs.size()) {
  qubits_ = boost::bimap<Qubit, unsigned>();
  for (unsigned i = 0; i < qbs.size(); ++i) {
    qubits_.insert({qbs[i], i});
  }
}

UnitaryTableau::UnitaryTableau(
    const MatrixXb& xx, const MatrixXb& xz, const VectorXb& xph,
    const MatrixXb& zx, const MatrixXb& zz, const VectorXb& zph)
    : tab_({}) {
  unsigned n_qubits = xx.rows();
  if ((xx.cols() != n_qubits) || (xz.rows() != n_qubits) ||
      (xz.cols() != n_qubits) || (xph.size() != n_qubits) ||
      (zx.rows() != n_qubits) || (zx.cols() != n_qubits) ||
      (zz.rows() != n_qubits) || (zz.cols() != n_qubits) ||
      (zph.size() != n_qubits))
    throw NotValid(
        "Unitary tableau requires equally-sized square matrices and vectors");
  MatrixXb xmat(2 * n_qubits, n_qubits);
  xmat << xx, zx;
  MatrixXb zmat(2 * n_qubits, n_qubits);
  zmat << zx, zz;
  VectorXb phase(2 * n_qubits);
  phase << xph, zph;
  tab_ = SymplecticTableau(xmat, zmat, phase);
  qubits_ = boost::bimap<Qubit, unsigned>();
  for (unsigned i = 0; i < n_qubits; ++i) {
    qubits_.insert({Qubit(i), i});
  }
}

QubitPauliTensor UnitaryTableau::get_xrow(const Qubit& qb) const {
  unsigned uq = qubits_.left.at(qb);
  PauliStabiliser stab = tab_.get_pauli(uq);
  std::list<Qubit> qbs;
  for (unsigned i = 0; i < qubits_.size(); ++i) {
    qbs.push_back(qubits_.right.at(i));
  }
  std::list<Pauli> string = {stab.string.begin(), stab.string.end()};
  Complex coeff = 1.;
  if (!stab.coeff) coeff *= -1.;
  return QubitPauliTensor(QubitPauliString(qbs, string), coeff);
}

QubitPauliTensor UnitaryTableau::get_zrow(const Qubit& qb) const {
  unsigned uqb = qubits_.left.at(qb);
  PauliStabiliser stab = tab_.get_pauli(uqb + qubits_.size());
  std::list<Qubit> qbs;
  for (unsigned i = 0; i < qubits_.size(); ++i) {
    qbs.push_back(qubits_.right.at(i));
  }
  std::list<Pauli> string = {stab.string.begin(), stab.string.end()};
  Complex coeff = 1.;
  if (!stab.coeff) coeff *= -1.;
  return QubitPauliTensor(QubitPauliString(qbs, string), coeff);
}

QubitPauliTensor UnitaryTableau::get_row_product(
    const QubitPauliTensor& qpt) const {
  QubitPauliTensor result(qpt.coeff);
  for (const std::pair<const Qubit, Pauli>& p : qpt.string.map) {
    auto qks_it = qubits_.left.find(p.first);
    if (qks_it == qubits_.left.end()) {
      // Acts as identity on p.first
      result = result * QubitPauliTensor(p.first, p.second);
    } else {
      switch (p.second) {
        case Pauli::I: {
          break;
        }
        case Pauli::X: {
          result = result * get_xrow(p.first);
          break;
        }
        case Pauli::Y: {
          // Y = iXZ
          result = result * get_xrow(p.first);
          result = result * get_zrow(p.first);
          result.coeff *= i_;
          break;
        }
        case Pauli::Z: {
          result = result * get_zrow(p.first);
          break;
        }
      }
    }
  }
  return result;
}

std::set<Qubit> UnitaryTableau::get_qubits() const {
  std::set<Qubit> result;
  for (boost::bimap<Qubit, unsigned>::const_iterator iter = qubits_.begin(),
                                                     iend = qubits_.end();
       iter != iend; ++iter) {
    result.insert(iter->left);
  }
  return result;
}

void UnitaryTableau::apply_S_at_front(const Qubit& qb) {
  unsigned uqb = qubits_.left.at(qb);
  tab_.row_mult(uqb + qubits_.size(), uqb, i_);
}

void UnitaryTableau::apply_S_at_end(const Qubit& qb) {
  unsigned uqb = qubits_.left.at(qb);
  tab_.apply_S(uqb);
}

void UnitaryTableau::apply_V_at_front(const Qubit& qb) {
  unsigned uqb = qubits_.left.at(qb);
  tab_.row_mult(uqb, uqb + qubits_.size(), i_);
}

void UnitaryTableau::apply_V_at_end(const Qubit& qb) {
  unsigned uqb = qubits_.left.at(qb);
  tab_.apply_V(uqb);
}

void UnitaryTableau::apply_CX_at_front(
    const Qubit& control, const Qubit& target) {
  unsigned uc = qubits_.left.at(control);
  unsigned ut = qubits_.left.at(target);
  tab_.row_mult(ut, uc, 1.);
  tab_.row_mult(uc + qubits_.size(), ut + qubits_.size());
}

void UnitaryTableau::apply_CX_at_end(
    const Qubit& control, const Qubit& target) {
  unsigned uc = qubits_.left.at(control);
  unsigned ut = qubits_.left.at(target);
  tab_.apply_CX(uc, ut);
}

void UnitaryTableau::apply_gate_at_front(
    OpType type, const qubit_vector_t& qbs) {
  switch (type) {
    case OpType::Z: {
      apply_S_at_front(qbs.at(0));
      apply_S_at_front(qbs.at(0));
      break;
    }
    case OpType::X: {
      apply_V_at_front(qbs.at(0));
      apply_V_at_front(qbs.at(0));
      break;
    }
    case OpType::Y: {
      apply_S_at_front(qbs.at(0));
      apply_S_at_front(qbs.at(0));
      apply_V_at_front(qbs.at(0));
      apply_V_at_front(qbs.at(0));
      break;
    }
    case OpType::S: {
      apply_S_at_front(qbs.at(0));
      break;
    }
    case OpType::Sdg: {
      apply_S_at_front(qbs.at(0));
      apply_S_at_front(qbs.at(0));
      apply_S_at_front(qbs.at(0));
      break;
    }
    case OpType::V: {
      apply_V_at_front(qbs.at(0));
      break;
    }
    case OpType::Vdg: {
      apply_V_at_front(qbs.at(0));
      apply_V_at_front(qbs.at(0));
      apply_V_at_front(qbs.at(0));
      break;
    }
    case OpType::H: {
      apply_S_at_front(qbs.at(0));
      apply_V_at_front(qbs.at(0));
      apply_S_at_front(qbs.at(0));
      break;
    }
    case OpType::CX: {
      apply_CX_at_front(qbs.at(0), qbs.at(1));
      break;
    }
    case OpType::CY: {
      apply_V_at_front(qbs.at(1));
      apply_V_at_front(qbs.at(1));
      apply_V_at_front(qbs.at(1));
      apply_CX_at_front(qbs.at(0), qbs.at(1));
      apply_V_at_front(qbs.at(1));
      break;
    }
    case OpType::CZ: {
      apply_S_at_front(qbs.at(1));
      apply_V_at_front(qbs.at(1));
      apply_S_at_front(qbs.at(1));
      apply_CX_at_front(qbs.at(0), qbs.at(1));
      apply_S_at_front(qbs.at(1));
      apply_V_at_front(qbs.at(1));
      apply_S_at_front(qbs.at(1));
      break;
    }
    case OpType::SWAP: {
      apply_CX_at_front(qbs.at(0), qbs.at(1));
      apply_CX_at_front(qbs.at(1), qbs.at(0));
      apply_CX_at_front(qbs.at(0), qbs.at(1));
      break;
    }
    case OpType::BRIDGE: {
      apply_CX_at_front(qbs.at(0), qbs.at(2));
      break;
    }
    case OpType::noop: {
      break;
    }
    default: {
      throw NotValid(
          optypeinfo().at(type).name +
          " cannot be applied to a UnitaryTableau; it is not a Clifford gate");
    }
  }
}

void UnitaryTableau::apply_gate_at_end(OpType type, const qubit_vector_t& qbs) {
  std::vector<unsigned> uqbs;
  for (const Qubit& q : qbs) {
    uqbs.push_back(qubits_.left.at(q));
  }
  tab_.apply_gate(type, uqbs);
}

void UnitaryTableau::apply_pauli_at_front(
    const QubitPauliTensor& pauli, unsigned half_pis) {
  half_pis = half_pis % 4;
  if (half_pis == 0) return;  // Identity
  if (half_pis == 2) {        // Degenerates to product of PI rotations
    for (const std::pair<const Qubit, Pauli>& term : pauli.string.map) {
      switch (term.second) {
        case Pauli::I: {
          break;
        }
        case Pauli::X: {
          apply_gate_at_front(OpType::X, {term.first});
          break;
        }
        case Pauli::Y: {
          apply_gate_at_front(OpType::Y, {term.first});
          break;
        }
        case Pauli::Z: {
          apply_gate_at_front(OpType::Z, {term.first});
          break;
        }
      }
    }
    return;
  }

  // From here, half_pis == 1 or 3
  // They act the same except for a phase flip on the product term
  MatrixXb product_x = MatrixXb::Zero(1, qubits_.size());
  MatrixXb product_z = MatrixXb::Zero(1, qubits_.size());
  MatrixXb::RowXpr px = product_x.row(0);
  MatrixXb::RowXpr pz = product_z.row(0);
  if (pauli.coeff != 1. && pauli.coeff != -1.)
    throw NotValid(
        "Can only apply Pauli gadgets with real unit coefficients to "
        "UnitaryTableaux");
  bool phase = (pauli.coeff == -1.) ^ (half_pis == 1);

  // Collect the product term
  for (const std::pair<const Qubit, Pauli>& term : pauli.string.map) {
    unsigned uqb = qubits_.left.at(term.first);
    switch (term.second) {
      case Pauli::I: {
        break;
      }
      case Pauli::X: {
        tab_.row_mult(
            tab_.xmat_.row(uqb), tab_.zmat_.row(uqb), tab_.phase_(uqb), px, pz,
            phase, 1., px, pz, phase);
        break;
      }
      case Pauli::Y: {
        tab_.row_mult(
            tab_.xmat_.row(uqb), tab_.zmat_.row(uqb), tab_.phase_(uqb), px, pz,
            phase, 1., px, pz, phase);
        tab_.row_mult(
            tab_.xmat_.row(uqb + qubits_.size()),
            tab_.zmat_.row(uqb + qubits_.size()),
            tab_.phase_(uqb + qubits_.size()), px, pz, phase, 1., px, pz,
            phase);
        break;
      }
      case Pauli::Z: {
        tab_.row_mult(
            tab_.xmat_.row(uqb + qubits_.size()),
            tab_.zmat_.row(uqb + qubits_.size()),
            tab_.phase_(uqb + qubits_.size()), px, pz, phase, 1., px, pz,
            phase);
        break;
      }
    }
  }

  // Apply the product term on the anti-commuting rows
  for (const std::pair<const Qubit, Pauli>& term : pauli.string.map) {
    unsigned uqb = qubits_.left.at(term.first);
    MatrixXb::RowXpr xx = tab_.xmat_.row(uqb);
    MatrixXb::RowXpr xz = tab_.zmat_.row(uqb);
    MatrixXb::RowXpr zx = tab_.xmat_.row(uqb + qubits_.size());
    MatrixXb::RowXpr zz = tab_.zmat_.row(uqb + qubits_.size());
    switch (term.second) {
      case Pauli::I: {
        break;
      }
      case Pauli::X: {
        tab_.row_mult(
            px, pz, phase, zx, zz, tab_.phase_(uqb + qubits_.size()), -i_, zx,
            zz, tab_.phase_(uqb + qubits_.size()));
        break;
      }
      case Pauli::Y: {
        tab_.row_mult(
            px, pz, phase, zx, zz, tab_.phase_(uqb + qubits_.size()), -i_, zx,
            zz, tab_.phase_(uqb + qubits_.size()));
        tab_.row_mult(
            px, pz, phase, xx, xz, tab_.phase_(uqb), -i_, xx, xz,
            tab_.phase_(uqb));
        break;
      }
      case Pauli::Z: {
        tab_.row_mult(
            px, pz, phase, xx, xz, tab_.phase_(uqb), -i_, xx, xz,
            tab_.phase_(uqb));
        break;
      }
    }
  }
}

void UnitaryTableau::apply_pauli_at_end(
    const QubitPauliTensor& pauli, unsigned half_pis) {
  std::vector<Pauli> string(qubits_.size(), Pauli::I);
  for (const std::pair<const Qubit, Pauli>& pair : pauli.string.map) {
    unsigned uqb = qubits_.left.at(pair.first);
    string.at(uqb) = pair.second;
  }
  if (pauli.coeff != 1. && pauli.coeff != -1.)
    throw NotValid(
        "Can only apply Pauli gadgets with real unit coefficients to "
        "UnitaryTableaux");
  tab_.apply_pauli_gadget({string, pauli.coeff == 1.}, half_pis);
}

UnitaryTableau UnitaryTableau::compose(
    const UnitaryTableau& first, const UnitaryTableau& second) {
  std::set<Qubit> qbs = first.get_qubits();
  for (const Qubit& q : second.get_qubits()) {
    qbs.insert(q);
  }
  UnitaryTableau result = UnitaryTableau({});
  const unsigned nqb = qbs.size();

  std::vector<QubitPauliTensor> rows;

  unsigned qir = 0;
  for (const Qubit& qi : qbs) {
    auto qif_it = first.qubits_.left.find(qi);
    if (qif_it == first.qubits_.left.end()) {
      // First acts as identity on qi, so just take effect of second
      rows.push_back(second.get_xrow(qi));
    } else {
      // Sum rows of second according to entries of first
      QubitPauliTensor fxrow = first.get_xrow(qi);
      QubitPauliTensor rxrow = second.get_row_product(fxrow);
      rows.push_back(rxrow);
    }

    result.qubits_.insert({qi, qir});
    ++qir;
  }

  // Do the same for the Z rows
  for (const Qubit& qi : qbs) {
    auto qif_it = first.qubits_.left.find(qi);
    if (qif_it == first.qubits_.left.end()) {
      // First acts as identity on qi, so just take effect of second
      rows.push_back(second.get_zrow(qi));
    } else {
      // Sum rows of second according to entries of first
      QubitPauliTensor fzrow = first.get_zrow(qi);
      QubitPauliTensor rzrow = second.get_row_product(fzrow);
      rows.push_back(rzrow);
    }
  }

  // Combine row lists and convert to PauliStabilisers
  PauliStabiliserList all_rows;
  for (const QubitPauliTensor& row : rows) {
    if (row.coeff != 1. && row.coeff != -1.)
      throw NotValid("Coefficient error in Tableau composition");
    std::vector<Pauli> ps(nqb, Pauli::I);
    for (const std::pair<const Qubit, Pauli>& p : row.string.map) {
      unsigned q = result.qubits_.left.at(p.first);
      ps[q] = p.second;
    }
    all_rows.push_back(PauliStabiliser(ps, row.coeff == 1.));
  }

  result.tab_ = SymplecticTableau(all_rows);

  return result;
}

std::ostream& operator<<(std::ostream& os, const UnitaryTableau& tab) {
  unsigned nqs = tab.qubits_.size();
  for (unsigned i = 0; i < nqs; ++i) {
    Qubit qi = tab.qubits_.right.at(i);
    os << "X@" << qi.repr() << "\t->\t" << tab.tab_.xmat_.row(i) << "   "
       << tab.tab_.zmat_.row(i) << "   " << tab.tab_.phase_(i) << std::endl;
  }
  os << "--" << std::endl;
  for (unsigned i = 0; i < nqs; ++i) {
    Qubit qi = tab.qubits_.right.at(i);
    os << "Z@" << qi.repr() << "\t->\t" << tab.tab_.xmat_.row(i + nqs) << "   "
       << tab.tab_.zmat_.row(i + nqs) << "   " << tab.tab_.phase_(i + nqs)
       << std::endl;
  }
  return os;
}

bool UnitaryTableau::operator==(const UnitaryTableau& other) const {
  if (get_qubits() != other.get_qubits()) return false;

  unsigned nq = qubits_.size();

  for (unsigned i = 0; i < nq; ++i) {
    Qubit qi = qubits_.right.at(i);
    unsigned oi = other.qubits_.left.at(qi);
    for (unsigned j = 0; j < nq; ++j) {
      Qubit qj = qubits_.right.at(j);
      unsigned oj = other.qubits_.left.at(qj);
      if (tab_.xmat_(i, j) != other.tab_.xmat_(oi, oj)) return false;
      if (tab_.zmat_(i, j) != other.tab_.zmat_(oi, oj)) return false;
      if (tab_.xmat_(i + nq, j) != other.tab_.xmat_(oi + nq, oj)) return false;
      if (tab_.zmat_(i + nq, j) != other.tab_.zmat_(oi + nq, oj)) return false;
    }
    if (tab_.phase_(i) != other.tab_.phase_(oi)) return false;
    if (tab_.phase_(i + nq) != other.tab_.phase_(oi + nq)) return false;
  }

  return true;
}

}  // namespace tket
