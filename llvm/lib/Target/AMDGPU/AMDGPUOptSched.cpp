//===- AMDGPUOptSched.cpp - Exact AMDGPU instruction scheduler ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Solvers, and the driver that applies what they return.
///
/// The driver's rule is that a schedule has to earn its way in. It is applied
/// only if it verifies, only if it beats the order the region already had on
/// Cost -- fitting the register budget first, then cycles, then demand as a
/// tie-break -- and only if measuring it with the backend's own register
/// trackers agrees that it raises pressure on no file. Anything else falls
/// back to the existing order. A scheduler that is sometimes worse is not
/// usable, however good its best case is.
///
/// Note the two different register measurements, which is deliberate.
/// Problem::pressureOf is a cheap model the search can afford to call on
/// every candidate order; demandOf below is the expensive truth, and it is
/// what decides. The model is allowed to be wrong and is only allowed to
/// steer.
///
//===----------------------------------------------------------------------===//

#include "AMDGPUOptSched.h"
#include "GCNRegPressure.h"
#include "GCNSubtarget.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveRegUnits.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include <numeric>

using namespace llvm;
using namespace llvm::AMDGPUOptSched;

#define DEBUG_TYPE "amdgpu-optsched"

STATISTIC(NumRegions, "Number of regions seen by the optimal scheduler");
STATISTIC(NumProved, "Number of regions scheduled with a proof of optimality");
STATISTIC(NumApplied, "Number of regions whose schedule was changed");
STATISTIC(NumUnverified, "Number of answers rejected by the verifier");
STATISTIC(NumPressureRejects,
          "Number of answers rejected for raising pressure");
STATISTIC(NumShortfallRejects,
          "Number of answers rejected for leaving a value with no register it "
          "could be allocated to");
STATISTIC(NumModelDisagreements,
          "Number of answers the demand model called an improvement and the "
          "register trackers did not");

static cl::opt<unsigned> ExactNodeLimit(
    "amdgpu-optsched-exact-limit", cl::init(16), cl::Hidden,
    cl::desc("Attempt an exhaustive search on regions up to this many "
             "instructions"));

static cl::opt<uint64_t> ExactStepLimit(
    "amdgpu-optsched-step-limit", cl::init(2000000), cl::Hidden,
    cl::desc("Search nodes the exhaustive solver may expand before it gives "
             "up its claim to optimality"));

static cl::opt<unsigned> WindowWidth(
    "amdgpu-optsched-window", cl::init(16), cl::Hidden,
    cl::desc("On regions too wide to enumerate, search this many consecutive "
             "instructions at a time, exhaustively, and sweep the window "
             "along the region. Zero falls back to list scheduling"));

static cl::opt<unsigned> WindowPasses(
    "amdgpu-optsched-window-passes", cl::init(4), cl::Hidden,
    cl::desc("How many times to sweep the window across a region before "
             "settling. One pass can only carry an instruction about a "
             "window's length"));

static cl::opt<uint64_t> WindowStepLimit(
    "amdgpu-optsched-window-step-limit", cl::init(50000), cl::Hidden,
    cl::desc("Search nodes one window may expand before it gives up and "
             "leaves that window alone. Per window, so one awkward window "
             "cannot stop the sweep"));

static cl::opt<uint64_t> WindowTotalStepLimit(
    "amdgpu-optsched-window-total-step-limit", cl::init(20000000), cl::Hidden,
    cl::desc("Search nodes the windowed solver may expand across a whole "
             "region before it stops sweeping"));

static cl::opt<bool> AllowPressureIncrease(
    "amdgpu-optsched-allow-pressure-increase", cl::init(false), cl::Hidden,
    cl::desc("Apply a shorter schedule even when it raises register "
             "pressure"));

static cl::opt<bool>
    AllowTies("amdgpu-optsched-allow-ties", cl::init(false), cl::Hidden,
              cl::desc("Apply a schedule that merely ties the existing one on "
                       "cycles and demand"));

static cl::opt<bool> UseListFloor(
    "amdgpu-optsched-list-floor", cl::init(true), cl::Hidden,
    cl::desc("When the chosen solver's answer loses, fall back to the list "
             "schedule rather than to the order the region arrived in"));

//===----------------------------------------------------------------------===//
// List scheduling
//===----------------------------------------------------------------------===//

namespace {

/// Critical-path list scheduling: among the instructions whose predecessors
/// are all placed, take the one with the most work left below it.
///
/// Here to be the baseline, and to be the fallback on regions the exhaustive
/// search cannot touch. It never claims more than Witness.
class ListSolver : public Solver {
public:
  StringRef getName() const override { return "list"; }
  Answer solve(const Problem &P) override;
};

Answer ListSolver::solve(const Problem &P) {
  const unsigned N = P.size();
  SmallVector<unsigned, 0> PredsLeft(N, 0);
  for (const Edge &E : P.edges())
    ++PredsLeft[E.Succ];

  SmallVector<bool, 0> Placed(N, false);
  SmallVector<unsigned, 0> Cycles(N, 0);
  SmallVector<unsigned, 16> PipeFree(P.numPipes(), 0);
  Answer A;
  A.Solver = "list";
  A.Order.reserve(N);

  unsigned Last = 0;
  for (unsigned Step = 0; Step != N; ++Step) {
    unsigned Pick = ~0u;
    unsigned PickCycle = 0;
    for (unsigned I = 0; I != N; ++I) {
      if (Placed[I] || PredsLeft[I])
        continue;
      unsigned T = Last + 1;
      for (unsigned EI : P.predEdgesOf(I)) {
        const Edge &E = P.edges()[EI];
        T = std::max(T, Cycles[E.Pred] + E.Latency);
      }
      for (const PipeUse &U : P.pipesOf(I))
        T = std::max(T, PipeFree[U.Pipe]);
      // Ready soonest, then most work below it. Node number last, so the
      // result does not depend on iteration order. Reading the pipes into T
      // first is what makes "soonest" prefer an instruction that can actually
      // start over one that would only sit waiting for a busy unit.
      if (Pick == ~0u || std::tuple(T, ~P.heights()[I], I) <
                             std::tuple(PickCycle, ~P.heights()[Pick], Pick)) {
        Pick = I;
        PickCycle = T;
      }
    }
    assert(Pick != ~0u && "no ready instruction: the region graph has a cycle");

    Placed[Pick] = true;
    Cycles[Pick] = PickCycle;
    for (const PipeUse &U : P.pipesOf(Pick))
      PipeFree[U.Pipe] = PickCycle + U.Cycles;
    Last = PickCycle;
    A.Order.push_back(Pick);
    for (unsigned EI : P.succEdgesOf(Pick))
      --PredsLeft[P.edges()[EI].Succ];
  }

  A.Cycles = Cycles;
  A.Makespan = P.makespanOf(Cycles);
  A.St = Status::Witness;
  return A;
}

/// Open \p N's definitions and close whatever its reads were keeping alive.
/// Returns the demand as it was, to hand back to \ref reopenValues.
///
/// Both searches step liveness this way and there is exactly one copy of the
/// rule, deliberately: two walks of the same liveness that drift apart would
/// make one search's pruning unsound against the other's cost, and nothing
/// would report it.
static Pressure placeValues(const Problem &P, unsigned N,
                            MutableArrayRef<unsigned> UsesLeft, Pressure &Cur,
                            Pressure &Peak) {
  Pressure Saved = Cur;
  for (unsigned V : P.defsOf(N))
    if (!P.values()[V].LiveIn)
      Cur[P.values()[V].File] += P.values()[V].Size;
  Peak.takeMax(Cur);
  for (unsigned V : P.defsOf(N)) {
    const RegValue &RV = P.values()[V];
    if (!RV.LiveIn && !RV.LiveOut && RV.Uses == 0)
      Cur[RV.File] -= RV.Size;
  }
  for (unsigned V : P.usesOf(N)) {
    const RegValue &RV = P.values()[V];
    if (--UsesLeft[V] == 0 && !RV.LiveOut)
      Cur[RV.File] -= RV.Size;
  }
  return Saved;
}

static void reopenValues(const Problem &P, unsigned N,
                         MutableArrayRef<unsigned> UsesLeft, Pressure &Cur,
                         Pressure &Peak, const Pressure &SavedCur,
                         const Pressure &SavedPeak) {
  for (unsigned V : P.usesOf(N))
    ++UsesLeft[V];
  Cur = SavedCur;
  Peak = SavedPeak;
}

/// Live-in demand, and the reads each value starts with.
static Pressure startValues(const Problem &P,
                            SmallVectorImpl<unsigned> &UsesLeft) {
  UsesLeft.assign(P.values().size(), 0);
  Pressure Cur;
  for (unsigned V = 0, E = P.values().size(); V != E; ++V) {
    UsesLeft[V] = P.values()[V].Uses;
    if (P.values()[V].LiveIn)
      Cur[P.values()[V].File] += P.values()[V].Size;
  }
  return Cur;
}

//===----------------------------------------------------------------------===//
// Exhaustive search
//===----------------------------------------------------------------------===//

/// Depth-first over legal orders, keeping the cheapest.
///
/// The search space is orders, not (order, timing) pairs, because for a fixed
/// order the earliest-issue timing is optimal -- so timing each order as it is
/// built loses nothing. See Problem::timeOrder.
///
/// Register demand is carried down the recursion the same way the timing is,
/// because it has to be: the cost of an order is not a function of its last
/// instruction, and re-measuring a whole order at every leaf would cost more
/// than the search.
struct Search {
  const Problem &P;
  const uint64_t StepLimit;
  uint64_t Steps = 0;
  /// Set when the budget runs out. Once set, the answer is an upper bound and
  /// nothing may call it a minimum.
  bool Exhausted = false;

  SmallVector<unsigned, 0> PredsLeft;
  SmallVector<bool, 0> Placed;
  SmallVector<unsigned, 0> Cycles;
  SmallVector<unsigned, 0> Order;
  /// When each functional unit frees up, carried down the recursion and
  /// restored on the way back out, like Cycles.
  SmallVector<unsigned, 16> PipeFree;

  /// Reads left before each value dies, and the demand so far. Peak only ever
  /// grows as the order is extended, which is what makes it safe to prune on.
  SmallVector<unsigned, 0> UsesLeft;
  Pressure Cur, Peak;

  SmallVector<unsigned, 0> BestOrder;
  Cost Best;

  Search(const Problem &P, uint64_t SL) : P(P), StepLimit(SL) {}

  void run();
  void recurse(unsigned Last, unsigned Makespan);
  unsigned lowerBound(unsigned Last, unsigned Makespan) const;
  /// Open \p N's definitions and close whatever its reads were keeping alive.
  /// Returns the state to hand back to \ref unplace.
  Pressure place(unsigned N);
  void unplace(unsigned N, const Pressure &SavedCur, const Pressure &SavedPeak);
};

void Search::run() {
  const unsigned N = P.size();
  PredsLeft.assign(N, 0);
  for (const Edge &E : P.edges())
    ++PredsLeft[E.Succ];
  Placed.assign(N, false);
  Cycles.assign(N, 0);
  PipeFree.assign(P.numPipes(), 0);
  Order.reserve(N);

  Cur = startValues(P, UsesLeft);
  Peak = Cur;

  // Start from the order the region already has. It is legal, so the search
  // can only improve on it, and a budget that runs out immediately still
  // leaves a usable answer behind.
  P.sourceOrder(BestOrder);
  Best = P.costOf(BestOrder);

  recurse(/*Last=*/0, /*Makespan=*/0);
}

Pressure Search::place(unsigned N) {
  return placeValues(P, N, UsesLeft, Cur, Peak);
}

void Search::unplace(unsigned N, const Pressure &SavedCur,
                     const Pressure &SavedPeak) {
  reopenValues(P, N, UsesLeft, Cur, Peak, SavedCur, SavedPeak);
}

unsigned Search::lowerBound(unsigned Last, unsigned Makespan) const {
  const unsigned N = P.size();
  unsigned Remaining = N - Order.size();
  // The last instruction cannot issue before Last + Remaining, since each one
  // takes a cycle of the single issue port.
  unsigned LB = std::max(Makespan, Last + Remaining);
  for (unsigned I = 0; I != N; ++I) {
    if (Placed[I])
      continue;
    unsigned Ready = Last + 1;
    for (unsigned EI : P.predEdgesOf(I)) {
      const Edge &E = P.edges()[EI];
      if (Placed[E.Pred])
        Ready = std::max(Ready, Cycles[E.Pred] + E.Latency);
    }
    // A unit already booked out past that is just as hard a floor as a
    // dependence, and on a region full of MFMAs it is usually the harder one.
    for (const PipeUse &U : P.pipesOf(I))
      Ready = std::max(Ready, PipeFree[U.Pipe]);
    LB = std::max(LB, Ready + P.heights()[I]);
  }
  return LB;
}

void Search::recurse(unsigned Last, unsigned Makespan) {
  if (Exhausted)
    return;
  if (++Steps > StepLimit) {
    Exhausted = true;
    return;
  }

  const unsigned N = P.size();
  if (Order.size() == N) {
    Cost C{P.budget().excessOf(Peak), Makespan, Peak.sum()};
    if (C < Best) {
      Best = C;
      BestOrder = Order;
    }
    return;
  }

  // Every term here is a lower bound on the corresponding term of any
  // completion of this prefix -- Peak because demand never falls as more
  // instructions are added, Makespan by the usual critical-path argument. So
  // if this does not beat the best complete order lexicographically, nothing
  // below it can either: a completion is at least this on every term, and
  // raising any term of a lexicographic tuple that is already not smaller
  // leaves it not smaller.
  Cost LB{P.budget().excessOf(Peak), lowerBound(Last, Makespan), Peak.sum()};
  if (!(LB < Best))
    return;

  for (unsigned I = 0; I != N; ++I) {
    if (Placed[I] || PredsLeft[I])
      continue;

    unsigned T = Last + 1;
    for (unsigned EI : P.predEdgesOf(I)) {
      const Edge &E = P.edges()[EI];
      T = std::max(T, Cycles[E.Pred] + E.Latency);
    }
    for (const PipeUse &U : P.pipesOf(I))
      T = std::max(T, PipeFree[U.Pipe]);

    Placed[I] = true;
    Cycles[I] = T;
    SmallVector<unsigned, 2> SavedPipes;
    for (const PipeUse &U : P.pipesOf(I)) {
      SavedPipes.push_back(PipeFree[U.Pipe]);
      PipeFree[U.Pipe] = T + U.Cycles;
    }
    Order.push_back(I);
    for (unsigned EI : P.succEdgesOf(I))
      --PredsLeft[P.edges()[EI].Succ];
    Pressure SavedPeak = Peak;
    Pressure SavedCur = place(I);

    recurse(T, std::max(Makespan, T + P.latencyOf(I)));

    unplace(I, SavedCur, SavedPeak);
    for (unsigned EI : P.succEdgesOf(I))
      ++PredsLeft[P.edges()[EI].Succ];
    Order.pop_back();
    for (auto [K, U] : enumerate(P.pipesOf(I)))
      PipeFree[U.Pipe] = SavedPipes[K];
    Placed[I] = false;

    if (Exhausted)
      return;
  }
}

class ExactSolver : public Solver {
  unsigned NodeLimit;
  uint64_t StepLimit;

public:
  ExactSolver(unsigned NodeLimit, uint64_t StepLimit)
      : NodeLimit(NodeLimit), StepLimit(StepLimit) {}

  StringRef getName() const override { return "exact"; }
  Answer solve(const Problem &P) override;
};

Answer ExactSolver::solve(const Problem &P) {
  Answer A;
  A.Solver = "exact";

  if (P.size() > NodeLimit) {
    // Too wide to enumerate. Say so rather than returning a partial search
    // dressed up as a minimum.
    A = ListSolver().solve(P);
    A.Solver = "exact(fell back to list)";
    return A;
  }

  Search S(P, StepLimit);
  S.run();

  A.Order = S.BestOrder;
  P.timeOrder(A.Order, A.Cycles);
  A.Makespan = P.makespanOf(A.Cycles);
  if (S.Exhausted) {
    A.St = Status::UpperBound;
  } else {
    A.St = Status::Optimal;
    // Say what it is optimal *for*. The objective is no longer just cycles,
    // and its demand term comes from a model, so a reader who takes "optimal"
    // to mean "no legal schedule is shorter" would be reading more into it
    // than is there.
    A.Proof = ("exhausted all legal orders of " + Twine(P.size()) +
               " instructions in " + Twine(S.Steps) +
               " steps; none costs less than (excess " + Twine(S.Best.Excess) +
               ", " + Twine(S.Best.Makespan) + " cycles, peak " +
               Twine(S.Best.Peak) + ") under the modelled demand")
                  .str();
  }
  return A;
}

//===----------------------------------------------------------------------===//
// Windowed search
//===----------------------------------------------------------------------===//

/// One window's worth of exhaustive search, over state the sweep owns.
///
/// Everything it needs to know about the region before the window -- the cycle
/// the last instruction issued at, when each unit frees, what is live and how
/// many reads each value has left -- is the sweep's running state, borrowed by
/// reference and left exactly as it was found. That is what makes this
/// affordable: the part of the region before the window is walked once for the
/// whole sweep, not once per window.
///
/// The search is over permutations of the window's members only. A member's
/// predecessors that lie outside the window are all in the prefix already,
/// because the order being rearranged is legal and so puts every predecessor
/// earlier; their cycles are therefore settled and can be read straight out of
/// the sweep's Cycles.
struct WindowSearch {
  const Problem &P;
  const uint64_t StepLimit;
  uint64_t Steps = 0;

  /// Sweep state, borrowed.
  MutableArrayRef<unsigned> Cycles;
  MutableArrayRef<unsigned> PipeFree;
  MutableArrayRef<unsigned> UsesLeft;
  Pressure &Cur;
  Pressure &Peak;

  /// The window, in the order it currently has. Held by value rather than as
  /// a slice of the order being rearranged, because the sweep replaces that
  /// order whenever a window is kept and a slice of it would then be looking
  /// at the old one.
  SmallVector<unsigned, 16> Members;
  /// Per member, how many of its predecessors are also members and not yet
  /// placed.
  SmallVector<unsigned, 16> PredsLeft;
  SmallVector<bool, 16> Taken;
  /// How many instructions follow the window. They still need a cycle each,
  /// which is a floor on what any rearrangement here can lead to.
  unsigned Tail = 0;

  SmallVector<unsigned, 16> Cand;
  /// The best rearrangement found, empty if none beat the window as it stands.
  SmallVector<unsigned, 16> Best;
  Cost BestCost{~0u, ~0u, ~0u};
  bool OutOfBudget = false;

  WindowSearch(const Problem &P, uint64_t StepLimit,
               MutableArrayRef<unsigned> Cycles,
               MutableArrayRef<unsigned> PipeFree,
               MutableArrayRef<unsigned> UsesLeft, Pressure &Cur,
               Pressure &Peak)
      : P(P), StepLimit(StepLimit), Cycles(Cycles), PipeFree(PipeFree),
        UsesLeft(UsesLeft), Cur(Cur), Peak(Peak) {}

  /// Index of \p N among the members, or ~0u if it is not one. A linear scan
  /// because a window is a handful of instructions; a map would cost more to
  /// build than it saves.
  unsigned memberOf(unsigned N) const {
    for (unsigned K = 0, E = Members.size(); K != E; ++K)
      if (Members[K] == N)
        return K;
    return ~0u;
  }

  void run(unsigned Last);
  void recurse(unsigned Last, unsigned Reach);
  /// What the window costs in the order it already has, by the same measure
  /// the search ranks rearrangements with. Leaves the sweep state as it found
  /// it.
  Cost baseCost(unsigned Last);
};

Cost WindowSearch::baseCost(unsigned Last) {
  SmallVector<unsigned, 16> SavedPipes(PipeFree.begin(), PipeFree.end());
  SmallVector<unsigned, 16> SavedCycles;
  SmallVector<Pressure, 16> SavedCur;
  Pressure SavedPeak = Peak;

  unsigned Reach = 0;
  for (unsigned I : Members) {
    unsigned T = Last + 1;
    for (unsigned EI : P.predEdgesOf(I)) {
      const Edge &E = P.edges()[EI];
      T = std::max(T, Cycles[E.Pred] + E.Latency);
    }
    for (const PipeUse &U : P.pipesOf(I))
      T = std::max(T, PipeFree[U.Pipe]);
    for (const PipeUse &U : P.pipesOf(I))
      PipeFree[U.Pipe] = T + U.Cycles;
    SavedCycles.push_back(Cycles[I]);
    Cycles[I] = T;
    SavedCur.push_back(placeValues(P, I, UsesLeft, Cur, Peak));
    Reach = std::max(Reach, T + P.heights()[I]);
    Last = T;
  }
  Cost C{P.budget().excessOf(Peak), std::max(Reach, Last + Tail), Peak.sum()};

  for (unsigned K = Members.size(); K--;) {
    reopenValues(P, Members[K], UsesLeft, Cur, Peak, SavedCur[K], SavedPeak);
    Cycles[Members[K]] = SavedCycles[K];
  }
  llvm::copy(SavedPipes, PipeFree.begin());
  return C;
}

void WindowSearch::run(unsigned Last) {
  PredsLeft.assign(Members.size(), 0);
  Taken.assign(Members.size(), false);
  for (unsigned K = 0, E = Members.size(); K != E; ++K)
    for (unsigned EI : P.predEdgesOf(Members[K]))
      if (memberOf(P.edges()[EI].Pred) != ~0u)
        ++PredsLeft[K];
  Cand.clear();
  Best.clear();
  // Start from what the window already costs, so the search reports only a
  // rearrangement that beats it. Starting from infinity would report the
  // window back to the sweep unchanged and make every tie look like a move.
  BestCost = baseCost(Last);
  recurse(Last, /*Reach=*/0);
}

void WindowSearch::recurse(unsigned Last, unsigned Reach) {
  if (OutOfBudget)
    return;
  if (Cand.size() == Members.size()) {
    Cost C{P.budget().excessOf(Peak), std::max(Reach, Last + Tail), Peak.sum()};
    if (C < BestCost) {
      BestCost = C;
      Best.assign(Cand.begin(), Cand.end());
    }
    return;
  }
  if (++Steps > StepLimit) {
    OutOfBudget = true;
    return;
  }

  // The same argument as the whole-region search: demand never falls as the
  // window is filled in, and neither does the reach or the issue cycle, so a
  // partial window that is already not cheaper than the best complete one
  // cannot be completed into one that is.
  Cost LB{P.budget().excessOf(Peak),
          std::max(Reach, Last + unsigned(Members.size() - Cand.size()) + Tail),
          Peak.sum()};
  if (!(LB < BestCost))
    return;

  for (unsigned K = 0, E = Members.size(); K != E; ++K) {
    if (Taken[K] || PredsLeft[K])
      continue;
    unsigned I = Members[K];

    unsigned T = Last + 1;
    for (unsigned EI : P.predEdgesOf(I)) {
      const Edge &Ed = P.edges()[EI];
      T = std::max(T, Cycles[Ed.Pred] + Ed.Latency);
    }
    for (const PipeUse &U : P.pipesOf(I))
      T = std::max(T, PipeFree[U.Pipe]);

    Taken[K] = true;
    unsigned SavedCycle = Cycles[I];
    Cycles[I] = T;
    SmallVector<unsigned, 2> SavedPipes;
    for (const PipeUse &U : P.pipesOf(I)) {
      SavedPipes.push_back(PipeFree[U.Pipe]);
      PipeFree[U.Pipe] = T + U.Cycles;
    }
    Cand.push_back(I);
    for (unsigned EI : P.succEdgesOf(I))
      if (unsigned M = memberOf(P.edges()[EI].Succ); M != ~0u)
        --PredsLeft[M];
    Pressure SavedPeak = Peak;
    Pressure SavedCur = placeValues(P, I, UsesLeft, Cur, Peak);

    // Height, not latency, because what this is ranking is how long the region
    // still has to run after the window, and the longest path out of I is the
    // part of that the window is responsible for. It is a lower bound on the
    // makespan rather than the makespan itself, which is fine for choosing
    // between windows -- the choice is checked against the real cost of the
    // whole order before anything is kept.
    recurse(T, std::max(Reach, T + P.heights()[I]));

    reopenValues(P, I, UsesLeft, Cur, Peak, SavedCur, SavedPeak);
    for (unsigned EI : P.succEdgesOf(I))
      if (unsigned M = memberOf(P.edges()[EI].Succ); M != ~0u)
        ++PredsLeft[M];
    Cand.pop_back();
    for (auto [J, U] : enumerate(P.pipesOf(I)))
      PipeFree[U.Pipe] = SavedPipes[J];
    Cycles[I] = SavedCycle;
    Taken[K] = false;

    if (OutOfBudget)
      return;
  }
}

class WindowSolver : public Solver {
  unsigned Width;
  unsigned Passes;
  uint64_t StepLimit;
  uint64_t TotalStepLimit;

public:
  WindowSolver(unsigned Width, unsigned Passes, uint64_t StepLimit,
               uint64_t TotalStepLimit)
      : Width(Width), Passes(Passes), StepLimit(StepLimit),
        TotalStepLimit(TotalStepLimit) {}

  StringRef getName() const override { return "window"; }
  Answer solve(const Problem &P) override;
};

Answer WindowSolver::solve(const Problem &P) {
  const unsigned N = P.size();
  Answer A;
  A.Solver = "window";
  P.sourceOrder(A.Order);
  A.St = Status::UpperBound;

  if (N >= 2 && Width >= 2) {
    Cost BestCost = P.costOf(A.Order);
    uint64_t Total = 0;
    bool OutOfBudget = false;

    SmallVector<unsigned, 0> Cycles(N, 0);
    SmallVector<unsigned, 16> PipeFree;
    SmallVector<unsigned, 0> UsesLeft;
    SmallVector<unsigned, 0> Trial;

    for (unsigned Pass = 0; Pass != Passes && !OutOfBudget; ++Pass) {
      bool Changed = false;

      // Start the sweep over. The order may have changed under it, and the
      // state is relative to a position in a particular order.
      llvm::fill(Cycles, 0u);
      PipeFree.assign(P.numPipes(), 0);
      Pressure Cur = startValues(P, UsesLeft);
      Pressure Peak = Cur;
      unsigned Last = 0;

      for (unsigned I = 0; I + 1 < N; ++I) {
        unsigned W = std::min<unsigned>(Width, N - I);
        if (W >= 2) {
          WindowSearch WS(P, StepLimit, Cycles, PipeFree, UsesLeft, Cur, Peak);
          ArrayRef<unsigned> Window = ArrayRef(A.Order).slice(I, W);
          WS.Members.assign(Window.begin(), Window.end());
          WS.Tail = N - I - W;
          WS.run(Last);
          // A window that ran out of budget still reports the best it found,
          // and the sweep carries on: the per-window cap is there so that one
          // wide window costs one window's time, not the region's.
          Total += WS.Steps;
          OutOfBudget = Total > TotalStepLimit;

          // The window's own ranking is a proxy -- it prices the window and
          // guesses at the rest of the region through heights. What decides is
          // the cost of the whole order with the window spliced in, measured
          // the way the driver will measure it. So a proxy that is wrong costs
          // a little time and never a worse schedule.
          //
          // Not worse, rather than better, and only because the window search
          // has already insisted on a strict improvement by its own measure.
          // A move that leaves the region's cost where it was but takes the
          // window off a local best is how the sweep crosses a plateau, and
          // there are plateaus: half of a two-instruction improvement is
          // usually worth nothing on its own. The driver still requires a
          // strict improvement over the source order before anything is
          // applied, so a sweep that only ever moves sideways changes nothing.
          if (!WS.Best.empty() && !llvm::equal(WS.Best, WS.Members)) {
            Trial.assign(A.Order.begin(), A.Order.end());
            llvm::copy(WS.Best, Trial.begin() + I);
            Cost C = P.costOf(Trial);
            if (!(BestCost < C)) {
              Changed |= C < BestCost;
              BestCost = C;
              A.Order = Trial;
            }
          }
        }

        // Step the sweep past this position, using whatever sits there now.
        unsigned Node = A.Order[I];
        unsigned T = Last + 1;
        for (unsigned EI : P.predEdgesOf(Node)) {
          const Edge &E = P.edges()[EI];
          T = std::max(T, Cycles[E.Pred] + E.Latency);
        }
        for (const PipeUse &U : P.pipesOf(Node))
          T = std::max(T, PipeFree[U.Pipe]);
        for (const PipeUse &U : P.pipesOf(Node))
          PipeFree[U.Pipe] = T + U.Cycles;
        Cycles[Node] = T;
        Last = T;
        placeValues(P, Node, UsesLeft, Cur, Peak);

        if (OutOfBudget)
          break;
      }

      // A pass that moved nothing means every window is already at its own
      // best, and another pass would retrace it exactly.
      if (!Changed)
        break;
    }
  }

  P.timeOrder(A.Order, A.Cycles);
  A.Makespan = P.makespanOf(A.Cycles);
  return A;
}

} // end anonymous namespace

std::unique_ptr<Solver> llvm::AMDGPUOptSched::createListSolver() {
  return std::make_unique<ListSolver>();
}

std::unique_ptr<Solver>
llvm::AMDGPUOptSched::createExactSolver(unsigned NodeLimit,
                                        uint64_t StepLimit) {
  return std::make_unique<ExactSolver>(NodeLimit, StepLimit);
}

std::unique_ptr<Solver>
llvm::AMDGPUOptSched::createWindowSolver(unsigned Width, unsigned Passes,
                                         uint64_t StepLimit,
                                         uint64_t TotalStepLimit) {
  return std::make_unique<WindowSolver>(Width, Passes, StepLimit,
                                        TotalStepLimit);
}

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

namespace {

/// What a schedule asks of each register file, counted in 32-bit registers.
///
/// Both halves of the demand, because the allocator pays both: the virtual
/// values that need somewhere to live, and the physical registers already
/// nailed down at that point by inline asm, calling conventions and the like.
struct Demand {
  unsigned SGPR = 0;
  unsigned ArchVGPR = 0;
  unsigned AGPR = 0;
  unsigned VGPR = 0; ///< Unified, where the two share a file.

  /// Per register file, never across files: trading 4 SGPRs for 4 VGPRs is
  /// not free and must not look free.
  ///
  /// The unified VGPR count is not enough on its own, even though VGPRs and
  /// AGPRs do share one file on gfx90a and later. Found the hard way on
  /// a-v-flat-atomicrmw.ll, where a function reserves every AGPR with inline
  /// asm: a schedule that moved demand out of AGPRs and into arch VGPRs left
  /// the unified total unchanged and still ran the allocator out of
  /// registers. So each sub-count has to hold as well.
  bool noWorseThan(const Demand &Old) const {
    return SGPR <= Old.SGPR && ArchVGPR <= Old.ArchVGPR && AGPR <= Old.AGPR &&
           VGPR <= Old.VGPR;
  }

  void takeMax(const Demand &O) {
    SGPR = std::max(SGPR, O.SGPR);
    ArchVGPR = std::max(ArchVGPR, O.ArchVGPR);
    AGPR = std::max(AGPR, O.AGPR);
    VGPR = std::max(VGPR, O.VGPR);
  }
};

/// Which file each register unit belongs to, so the physical half of the
/// demand can be counted per file. Built once per target: a unit's class
/// never changes.
struct UnitFiles {
  BitVector SGPR, ArchVGPR, AGPR;

  explicit UnitFiles(const SIRegisterInfo &TRI, const MachineRegisterInfo &MRI)
      : SGPR(TRI.getNumRegUnits()), ArchVGPR(TRI.getNumRegUnits()),
        AGPR(TRI.getNumRegUnits()) {
    for (unsigned U = 0, E = TRI.getNumRegUnits(); U != E; ++U) {
      MCRegUnitRootIterator Root(MCRegUnit(U), &TRI);
      if (!Root.isValid())
        continue;
      MCRegister Reg = *Root;
      // Reserved units are unavailable no matter what we schedule, so they
      // are a constant that only adds noise to a comparison.
      if (MRI.isReserved(Reg))
        continue;
      const TargetRegisterClass *RC = TRI.getPhysRegBaseClass(Reg);
      if (!RC)
        continue;
      if (TRI.isSGPRClass(RC))
        SGPR.set(U);
      else if (TRI.isAGPRClass(RC))
        AGPR.set(U);
      else if (TRI.isVGPRClass(RC))
        ArchVGPR.set(U);
    }
  }
};

/// Peak demand of a candidate order, without committing to it.
///
/// Measured with the backend's own trackers rather than from the Problem: the
/// Problem's view of liveness is a model, and a model that disagrees with the
/// allocator is the kind of bug that only shows up as a spill thousands of
/// instructions later.
///
/// The two trackers walk in lockstep and the peak is taken over the *sum at
/// each point*, not over each half separately. Summing the two peaks would be
/// blind to exactly the case this is here for: a schedule that moves a short
/// virtual live range into the one window where every physical register is
/// already taken raises neither peak on its own, and is still fatal.
/// \p Trace, if given, receives the demand at each position of \p Order, so
/// the measurement can be lined up against Problem::pressureOf position by
/// position. The walk is upward, so it is filled back to front.
Demand demandOf(ScheduleDAGMILive &DAG, ArrayRef<unsigned> Order,
                const UnitFiles &Files, bool Unified,
                SmallVectorImpl<Demand> *Trace = nullptr) {
  MachineBasicBlock *MBB = DAG.SUnits[0].getInstr()->getParent();
  auto BBEnd = MBB->end();
  const SIRegisterInfo &TRI = *static_cast<const SIRegisterInfo *>(DAG.TRI);

  GCNUpwardRPTracker RPT(*DAG.getLIS());
  LiveRegUnits Phys(TRI);
  Phys.addLiveOuts(*MBB);

  // Both trackers start just below the region: the virtual one by stepping
  // over the boundary instruction, the physical one by walking back from the
  // end of the block, whose tail this schedule does not touch.
  // Debug instructions carry operands but no liveness, and stepBackward says
  // so with an assertion.
  for (auto It = BBEnd; It != DAG.end();) {
    const MachineInstr &MI = *--It;
    if (!MI.isDebugInstr())
      Phys.stepBackward(MI);
  }
  if (DAG.end() != BBEnd) {
    RPT.reset(*DAG.end());
    RPT.recede(*DAG.end());
  } else {
    RPT.reset(*std::prev(BBEnd));
  }

  const BitVector &Live = Phys.getBitVector();
  BitVector Scratch;
  auto Overlap = [&](const BitVector &File) {
    Scratch = Live;
    Scratch &= File;
    return unsigned(Scratch.count());
  };

  Demand Max;
  Demand Last; ///< The most recent sample, for \p Trace.
  /// \p Dead is the demand of \p MI's own dead defs, which a live-range view
  /// misses entirely: a value nobody reads is live nowhere, but the register
  /// holding it still has to exist for the one cycle the instruction writes
  /// it. Charge it where it is written.
  auto Sample = [&](const MachineInstr *MI) {
    GCNRegPressure V = RPT.getPressure();
    Demand D;
    D.SGPR = V.getSGPRNum() + Overlap(Files.SGPR);
    D.ArchVGPR = V.getArchVGPRNum() + Overlap(Files.ArchVGPR);
    D.AGPR = V.getAGPRNum() + Overlap(Files.AGPR);
    if (MI)
      for (const MachineOperand &MO : MI->all_defs()) {
        if (!MO.isDead())
          continue;
        Register Reg = MO.getReg();
        if (Reg.isVirtual()) {
          const TargetRegisterClass *RC = DAG.MRI.getRegClass(Reg);
          unsigned Size = TRI.getRegSizeInBits(*RC) / 32;
          if (TRI.isSGPRClass(RC))
            D.SGPR += Size;
          else if (TRI.isAGPRClass(RC))
            D.AGPR += Size;
          else
            D.ArchVGPR += Size;
        } else {
          // stepBackward has already cleared these, so they are not in Live.
          for (MCRegUnit RU : TRI.regunits(Reg.asMCReg())) {
            unsigned U = unsigned(RU);
            if (Live[U])
              continue;
            if (Files.SGPR[U])
              ++D.SGPR;
            else if (Files.AGPR[U])
              ++D.AGPR;
            else if (Files.ArchVGPR[U])
              ++D.ArchVGPR;
          }
        }
      }
    D.VGPR = Unified ? D.ArchVGPR + D.AGPR : D.ArchVGPR;
    Last = D;
    Max.takeMax(D);
  };

  Sample(nullptr);
  if (Trace)
    Trace->assign(Order.size(), Demand());
  for (unsigned I = Order.size(); I-- > 0;) {
    const MachineInstr &MI = *DAG.SUnits[Order[I]].getInstr();
    RPT.recede(MI);
    Phys.stepBackward(MI);
    Sample(&MI);
    if (Trace)
      (*Trace)[I] = Last;
  }
  return Max;
}

/// Replays a schedule the solver already produced. All of the thinking
/// happens in initialize(); pickNode just reads it out, so that the search
/// sees a whole region at once instead of a sequence of local choices.
class OptSchedStrategy : public MachineSchedStrategy {
  SmallVector<unsigned, 0> Order;
  unsigned Next = 0;

public:
  // We measure pressure ourselves, on whole candidate orders. The incremental
  // tracker would only duplicate that at every step.
  bool shouldTrackPressure() const override { return false; }
  bool shouldTrackLaneMasks() const override { return false; }

  void initialize(ScheduleDAGMI *DAGMI) override;

  SUnit *pickNode(bool &IsTopNode) override {
    IsTopNode = true;
    if (Next == Order.size())
      return nullptr;
    return const_cast<SUnit *>(&CurrentDAG->SUnits[Order[Next++]]);
  }

  void schedNode(SUnit *, bool) override {}
  void releaseTopNode(SUnit *) override {}
  void releaseBottomNode(SUnit *) override {}

private:
  ScheduleDAGMI *CurrentDAG = nullptr;
};

void OptSchedStrategy::initialize(ScheduleDAGMI *DAGMI) {
  CurrentDAG = DAGMI;
  Next = 0;
  Order.clear();

  auto &DAG = *static_cast<ScheduleDAGMILive *>(DAGMI);
  if (DAG.SUnits.empty())
    return;

  // -misched names a strategy out of one global registry, so this one can be
  // picked for any target, and is: a .ll file that takes its triple from the
  // RUN line rather than from the module compiles for the host. Nothing below
  // is target-neutral -- pressureOf reads GCN register classes and would read
  // x86 ones as if they were VGPRs -- so hand the region back unscheduled.
  //
  // r600 is in the same position despite sharing the backend. It has no
  // GCNSubtarget, so the register budget cannot even be asked for: its
  // getMaxNumSGPRs asserts rather than answer.
  if (!DAG.MF.getTarget().getTargetTriple().isAMDGCN()) {
    LLVM_DEBUG(dbgs() << "optsched: not an AMDGCN target; standing down\n");
    // SUnits are numbered in source order, so this reissues what is there.
    Order.resize(DAG.SUnits.size());
    std::iota(Order.begin(), Order.end(), 0u);
    return;
  }

  ++NumRegions;
  Problem P = Problem::build(DAG);
  P.sourceOrder(Order);

  SmallVector<unsigned, 0> SourceCycles;
  P.timeOrder(Order, SourceCycles);
  unsigned SourceMakespan = P.makespanOf(SourceCycles);

  // The decomposition does not influence the schedule yet; it is the
  // structure the eventual global search will run on, and this is how we look
  // at it.
  DEBUG_WITH_TYPE("amdgpu-optsched-chain", {
    const TargetInstrInfo *TII = DAG.TII;
    Decomposition::build(P, Order).print(dbgs(), P, [&](unsigned N) {
      return TII->getName(DAG.SUnits[N].getInstr()->getOpcode());
    });
  });

  // Exhaustive while that is affordable; above it, exhaustive on a window at
  // a time, which is the same search asked a question it can finish. List
  // scheduling is what is left when even that is turned off.
  std::unique_ptr<Solver> S;
  bool SolverIsList = false;
  if (P.size() <= ExactNodeLimit) {
    S = createExactSolver(ExactNodeLimit, ExactStepLimit);
  } else if (WindowWidth >= 2) {
    S = createWindowSolver(WindowWidth, WindowPasses, WindowStepLimit,
                           WindowTotalStepLimit);
  } else {
    S = createListSolver();
    SolverIsList = true;
  }
  Answer A = S->solve(P);

  const Pressure &Thru = P.liveThrough();
  LLVM_DEBUG(dbgs() << "optsched: " << P.Name << " n=" << P.size()
                    << " edges=" << P.edges().size() << " live-through "
                    << Thru[RegFile::ArchVGPR] << "+" << Thru[RegFile::AGPR]
                    << " VGPR / " << Thru[RegFile::SGPR] << " SGPR"
                    << " solver=" << A.Solver << " status=" << toString(A.St)
                    << " makespan=" << A.Makespan << " (source "
                    << SourceMakespan << ")\n");

  // The order we are holding, and what it costs. Both move if an answer is
  // adopted: a second answer is then measured against what we already have,
  // not against what the region arrived with.
  unsigned CurMakespan = SourceMakespan;

  // Gate one answer and adopt it if it wins on every count. Every answer goes
  // through this, so the floor below cannot enter by a side door that the
  // solver's own answer would not have been allowed through.
  auto Adopt = [&](const Answer &Cand, const Twine &What) -> bool {
    if (Error E = verify(P, Cand)) {
      ++NumUnverified;
      LLVM_DEBUG(dbgs() << "optsched: rejected, " << toStringWithoutConsuming(E)
                        << "; keeping the existing order\n");
      consumeError(std::move(E));
      return false;
    }
    if (Cand.St == Status::Optimal)
      ++NumProved;

    // Strictly better, not merely as good. An equal-cost reorder buys nothing
    // and bets the whole region on our model being complete, which it is not:
    // every later pass reads the order too, and some of them read things this
    // model has no name for. On materialize-frame-index-sgpr.ll moving an SCC
    // def three slots down shortens $scc's live range, which makes frame-index
    // lowering pick its scalar path, which needs an SGPR that is not there.
    // Nothing in a cycle count can see that coming, so a tie has to lose.
    //
    // "Equal cost" now includes demand, so an order that is the same length but
    // holds fewer registers is no longer a tie -- it is a win, and a win with a
    // reason behind it rather than an arbitrary tie-break.
    Cost CurCost = P.costOf(Order);
    Cost NewCost = P.costOf(Cand.Order);
    auto ShowCost = [](const Cost &C) {
      return formatv("excess {0}, {1} cycles, peak {2}", C.Excess, C.Makespan,
                     C.Peak);
    };
    LLVM_DEBUG(dbgs() << "optsched: cost (" << ShowCost(CurCost) << ") -> ("
                      << ShowCost(NewCost) << ")\n");
    if (AllowTies ? CurCost < NewCost : !(NewCost < CurCost)) {
      LLVM_DEBUG(
          dbgs() << "optsched: (" << ShowCost(NewCost) << ") vs source ("
                 << ShowCost(CurCost)
                 << "); not strictly better, keeping the existing order\n");
      return false;
    }

    // Room, as opposed to pressure. Pressure says how many registers a schedule
    // wants; this says whether some value has anywhere at all to go, which is a
    // different question and the one the allocator fails on. Stretching a value
    // across a pre-coloured range is not expensive, it is impossible -- there
    // is nothing to evict and nothing to spill that would help -- so this is a
    // gate and not a term in Cost.
    //
    // Relative, not absolute: the order we hold is allocatable by assumption,
    // so what matters is that we did not make it worse. An absolute test would
    // refuse regions that were already tight and that we left alone.
    unsigned CurShort = P.shortfallOf(Order);
    unsigned NewShort = P.shortfallOf(Cand.Order);
    LLVM_DEBUG(dbgs() << "optsched: budget " << P.budget()[RegFile::SGPR]
                      << " SGPR / " << P.budget()[RegFile::ArchVGPR] << "+"
                      << P.budget()[RegFile::AGPR] << " VGPR; shortfall "
                      << CurShort << " -> " << NewShort << "\n");
    if (NewShort > CurShort) {
      ++NumShortfallRejects;
      LLVM_DEBUG(dbgs() << "optsched: shortfall " << CurShort << " -> "
                        << NewShort
                        << "; the new order leaves a value with no register to "
                           "live in, keeping the existing order\n");
      return false;
    }

    if (!AllowPressureIncrease) {
      bool Unified = DAG.MF.getSubtarget<GCNSubtarget>().hasGFX90AInsts();
      UnitFiles Files(*static_cast<const SIRegisterInfo *>(DAG.TRI), DAG.MRI);
      Demand Before = demandOf(DAG, Order, Files, Unified);
      Demand After = demandOf(DAG, Cand.Order, Files, Unified);
      auto Show = [](const Demand &D) {
        return formatv("{0}+{1} VGPR / {2} SGPR", D.ArchVGPR, D.AGPR, D.SGPR);
      };
      LLVM_DEBUG(dbgs() << "optsched: demand " << Show(Before) << " -> "
                        << Show(After) << "\n");
      // What the model thought, in the same units, and where it first parts
      // company with the trackers. The gate above is the truth; the model only
      // steers, but a model that is wrong by eighty registers steers into a
      // spill, and finding where it goes wrong beats guessing at it.
      DEBUG_WITH_TYPE("amdgpu-optsched-regs", {
        auto ShowP = [](const Pressure &Pr) {
          return formatv("{0}+{1} VGPR / {2} SGPR", Pr[RegFile::ArchVGPR],
                         Pr[RegFile::AGPR], Pr[RegFile::SGPR]);
        };
        SmallVector<Pressure, 0> Model;
        SmallVector<Demand, 0> Meas;
        dbgs() << "optsched: model " << ShowP(P.pressureOf(Order, &Model))
               << " -> " << ShowP(P.pressureOf(Cand.Order)) << "\n";
        demandOf(DAG, Order, Files, Unified, &Meas);
        unsigned Worst = 0, WorstAt = ~0u;
        for (unsigned I = 0, E = Model.size(); I != E; ++I) {
          unsigned M = Model[I][RegFile::ArchVGPR] + Model[I][RegFile::AGPR];
          unsigned T = Meas[I].ArchVGPR + Meas[I].AGPR;
          if (T > M && T - M > Worst) {
            Worst = T - M;
            WorstAt = I;
          }
        }
        if (WorstAt != ~0u)
          dbgs() << "optsched: model low by " << Worst << " VGPR at position "
                 << WorstAt << " of " << Model.size() << ": "
                 << *DAG.SUnits[Order[WorstAt]].getInstr();
      });
      if (!After.noWorseThan(Before)) {
        ++NumPressureRejects;
        ++NumModelDisagreements;
        // Worth naming, not just counting. The search believed this order was
        // no worse on demand and the trackers say otherwise, so Problem's model
        // and the allocator's view have come apart somewhere -- most likely a
        // physical range the pinning edges did not cover, or a subregister lane
        // the model flattens. The gate catches it either way; this line is how
        // to find out that it happened.
        LLVM_DEBUG(dbgs() << "optsched: model said (" << ShowCost(NewCost)
                          << ") beats (" << ShowCost(CurCost)
                          << ") but measured demand " << Show(Before) << " -> "
                          << Show(After) << "; keeping the existing order\n");
        return false;
      }
    }

    if (Cand.Order != Order) {
      ++NumApplied;
      LLVM_DEBUG(dbgs() << "optsched: applying " << What << ", " << CurMakespan
                        << " -> " << Cand.Makespan << " cycles\n");
      Order = Cand.Order;
      CurMakespan = Cand.Makespan;
    }
    return true;
  };

  if (Adopt(A, toString(A.St) + " schedule"))
    return;

  // The answer lost, and the order we are holding is the one the region
  // arrived with. That is pre-scheduler order: -misched=amdgpu-optsched
  // replaces GCNSchedStrategy outright, so nothing downstream will improve it,
  // and leaving it alone is a choice to ship whatever the earlier passes
  // happened to leave behind.
  //
  // The list schedule is not optimal and is not claimed to be, but it is legal,
  // it is cheap to build, and it is what this scheduler would have produced on
  // a region the search was never going to fit. So it is the floor. It faces
  // the same gates as any other answer, so a region whose list schedule is
  // worse than what it arrived with still keeps what it arrived with.
  //
  // Skipped when the solver already was the list solver: that answer has just
  // been through these gates and lost, and it would lose again.
  if (UseListFloor && !SolverIsList) {
    Answer L = createListSolver()->solve(P);
    LLVM_DEBUG(dbgs() << "optsched: falling back, solver=" << L.Solver
                      << " status=" << toString(L.St) << " makespan="
                      << L.Makespan << " (holding " << CurMakespan << ")\n");
    Adopt(L, "list floor");
  }
}

} // end anonymous namespace

ScheduleDAGInstrs *llvm::createAMDGPUOptSchedDAG(MachineSchedContext *C) {
  return new ScheduleDAGMILive(C, std::make_unique<OptSchedStrategy>());
}
