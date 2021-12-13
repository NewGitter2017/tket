#include "PartialTsaTesting.hpp"

#include <catch2/catch.hpp>

#include "TestStatsStructs.hpp"
#include "TokenSwapping/ArchitectureMapping.hpp"
#include "TokenSwapping/DistancesFromArchitecture.hpp"
#include "TokenSwapping/NeighboursFromArchitecture.hpp"
#include "TokenSwapping/RiverFlowPathFinder.hpp"
#include "TokenSwapping/TSAUtils/DistanceFunctions.hpp"
#include "TokenSwapping/TSAUtils/VertexSwapResult.hpp"

;
using std::vector;

namespace tket {
namespace tsa_internal {
namespace tests {

// Also checks if an empty token pair swap occurs.
static size_t get_recalculated_final_L(
    VertexMapping problem, const SwapList& swap_list,
    DistancesInterface& distances, TokenOption token_option) {
  bool empty_tok_swap = false;

  for (auto id_opt = swap_list.front_id(); id_opt;
       id_opt = swap_list.next(id_opt)) {
    const auto swap = swap_list.at(id_opt.value());
    const VertexSwapResult swap_result(swap, problem);
    if (swap_result.tokens_moved == 0 &&
        token_option == TokenOption::DO_NOT_ALLOW_EMPTY_TOKEN_SWAP) {
      empty_tok_swap = true;
    }
  }
  if (empty_tok_swap) {
    REQUIRE(false);
  }
  return get_total_home_distances(problem, distances);
}

static void check_progress(
    size_t init_L, size_t final_L, RequiredTsaProgress progress) {
  REQUIRE(final_L <= init_L);
  switch (progress) {
    case RequiredTsaProgress::FULL:
      REQUIRE(final_L == 0);
      return;
    case RequiredTsaProgress::NONZERO:
      if (init_L > 0) {
        REQUIRE(final_L < init_L);
      }
    // Fall through
    case RequiredTsaProgress::NONE:
      return;
    default:
      REQUIRE(false);
  }
}

static std::string run_tests(
    const std::vector<VertexMapping>& problems, DistancesInterface& distances,
    NeighboursInterface& neighbours, PathFinderInterface& path_finder,
    PartialTsaInterface& partial_tsa, RequiredTsaProgress progress,
    TokenOption token_option) {
  REQUIRE(!problems.empty());
  PartialTsaStatistics statistics;
  SwapList swap_list;

  for (const auto& problem : problems) {
    const auto init_L = get_total_home_distances(problem, distances);
    swap_list.clear();

    // Will be destructively altered
    auto problem_copy = problem;
    path_finder.reset();
    partial_tsa.append_partial_solution(
        swap_list, problem_copy, distances, neighbours, path_finder);

    const auto final_L = get_total_home_distances(problem_copy, distances);
    check_progress(init_L, final_L, progress);

    REQUIRE(
        get_recalculated_final_L(problem, swap_list, distances, token_option) ==
        final_L);

    statistics.add_problem_result(
        init_L, final_L, problem.size(), swap_list.size());
  }
  std::stringstream ss;
  ss << "[TSA=" << partial_tsa.name();
  switch (progress) {
    case RequiredTsaProgress::FULL:
      ss << " FULL";
      break;

    case RequiredTsaProgress::NONZERO:
      ss << " NONZERO";
      break;

    // Fall through
    case RequiredTsaProgress::NONE:
    default:
      break;
  }
  ss << " PF=" << path_finder.name() << "\n"
     << statistics.str(problems.size()) << "]";
  return ss.str();
}

std::string run_tests(
    const Architecture& arch, const std::vector<VertexMapping>& problems,
    PathFinderInterface& path_finder, PartialTsaInterface& partial_tsa,
    RequiredTsaProgress progress, TokenOption token_option) {
  const ArchitectureMapping arch_mapping(arch);
  DistancesFromArchitecture distances(arch_mapping);
  NeighboursFromArchitecture neighbours(arch_mapping);
  return run_tests(
      problems, distances, neighbours, path_finder, partial_tsa, progress,
      token_option);
}

std::string run_tests(
    const Architecture& arch, const std::vector<VertexMapping>& problems,
    RNG& rng, PartialTsaInterface& partial_tsa, RequiredTsaProgress progress,
    TokenOption token_option) {
  const ArchitectureMapping arch_mapping(arch);
  DistancesFromArchitecture distances(arch_mapping);
  NeighboursFromArchitecture neighbours(arch_mapping);
  RiverFlowPathFinder path_finder(distances, neighbours, rng);

  return run_tests(
      problems, distances, neighbours, path_finder, partial_tsa, progress,
      token_option);
}

}  // namespace tests
}  // namespace tsa_internal
}  // namespace tket