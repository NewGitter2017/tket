#ifndef _TKET_TESTS_TokenSwapping_TableLookup_SwapSequenceReductionTester_H_
#define _TKET_TESTS_TokenSwapping_TableLookup_SwapSequenceReductionTester_H_

#include <string>

#include "../TestUtils/DecodedProblemData.hpp"
#include "TokenSwapping/SwapListOptimiser.hpp"
#include "TokenSwapping/TableLookup/SwapListTableOptimiser.hpp"

namespace tket {
namespace tsa_internal {
namespace tests {

/** Directly test the results of table reductions on fixed swap sequences. */
class SwapSequenceReductionTester {
 public:
  struct Options {
    bool optimise_initial_segment_only;
  };

  // Reduces the sequence of swaps, checks it, and returns the size.
  size_t get_checked_solution_size(
      const DecodedProblemData& problem_data,
      const DecodedArchitectureData& architecture_data, const Options& options);

  size_t get_checked_solution_size(
      const DecodedProblemData& problem_data, const Options& options);

 private:
  SwapListOptimiser m_general_optimiser;
  // SwapList m_raw_swap_list;
};

struct SequenceReductionStats {
  size_t problems;
  size_t reduced_problems;
  size_t total_original_swaps;

  // This only includes problems where the number of swaps strictly decreased
  // after table reduction.
  size_t total_original_swaps_for_reduced_problems;

  // This is the sum of "reduced_swaps" passed in, over all problems (including
  // those where there was no decrease).
  size_t total_reduced_swaps;

  SequenceReductionStats();

  void add_solution(size_t original_swaps, size_t reduced_swaps);

  std::string str() const;
};

}  // namespace tests
}  // namespace tsa_internal
}  // namespace tket
#endif