//===- AMDGPUOptSchedChain.cpp - Decompose a region around its chain ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Cutting a region into slots along its longest dependence chain, and
/// measuring what is left over: where each remaining instruction may go, what
/// each choice costs, and how large the resulting search space is.
///
/// This is a change of coordinates, not a decision. Nothing here picks a
/// schedule. It produces the structure a search needs in order to be able to
/// claim a global optimum: a fixed spine that no schedule can reorder, and an
/// explicit, finite, measurable space of everything that can still vary.
///
//===----------------------------------------------------------------------===//

#include "AMDGPUOptSched.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>

using namespace llvm;
using namespace llvm::AMDGPUOptSched;

#define DEBUG_TYPE "amdgpu-optsched-chain"

//===----------------------------------------------------------------------===//
// Counting orders
//===----------------------------------------------------------------------===//

/// log10 of n!.
static double logFactorial(unsigned N) {
  return N < 2 ? 0.0 : std::lgamma((double)N + 1.0) / std::log(10.0);
}

Width llvm::AMDGPUOptSched::countOrders(const Problem &P,
                                        ArrayRef<unsigned> Nodes,
                                        unsigned ExactLimit) {
  const unsigned K = Nodes.size();
  Width W;
  if (K < 2)
    return W;

  // Local numbering, so the sub-order can be handled with bitmasks and small
  // arrays regardless of how large the enclosing region is.
  DenseMap<unsigned, unsigned> Local;
  for (unsigned I = 0; I != K; ++I)
    Local[Nodes[I]] = I;

  // Direct edges only. Transitive ones add no constraint that the direct
  // edges do not already imply, and every intermediate node is inside this
  // slot -- see the note on self-containment in the header.
  SmallVector<uint64_t, 32> SuccMask(K, 0);
  SmallVector<SmallVector<unsigned, 4>, 32> Preds(K), Succs(K);
  for (unsigned I = 0; I != K; ++I) {
    for (unsigned EI : P.succEdgesOf(Nodes[I])) {
      const Edge &E = P.edges()[EI];
      auto It = Local.find(E.Succ);
      if (It == Local.end())
        continue;
      unsigned J = It->second;
      SuccMask[I] |= 1ULL << J;
      Succs[I].push_back(J);
      Preds[J].push_back(I);
    }
  }

  if (K <= ExactLimit && K <= 20) {
    // DP over subsets. Count[S] is the number of orders of S; peel a maximal
    // element, which keeps every set reachable from the full set down-closed,
    // so the recursion only ever visits sets that are real prefixes reversed.
    //
    // Cannot overflow: the result is at most K! <= 20! < 2^64.
    std::vector<uint64_t> Count(1ULL << K, 0);
    Count[0] = 1;
    for (uint64_t S = 1; S != (1ULL << K); ++S) {
      uint64_t Total = 0;
      for (unsigned V = 0; V != K; ++V) {
        if (!(S & (1ULL << V)))
          continue;
        if (SuccMask[V] & S)
          continue; // not maximal in S
        Total += Count[S ^ (1ULL << V)];
      }
      Count[S] = Total;
    }
    W.Count = Count[(1ULL << K) - 1];
    W.Exact = true;
    W.LogLower = W.LogUpper = W.Count ? std::log10((double)W.Count) : 0.0;
    return W;
  }

  W.Exact = false;

  // Upper bound. Partition the nodes into chains; a linear extension is then
  // no more than an interleaving of totally ordered sequences, so there are at
  // most K! / prod(len!) of them. Any partition into chains gives a valid
  // bound, so a greedy longest-chain peel is enough -- a minimum cover would
  // only make the bound tighter, not more correct.
  {
    BitVector Left(K, true);
    unsigned Remaining = K;
    W.LogUpper = logFactorial(K);
    SmallVector<unsigned, 32> Len(K, 0);
    while (Remaining) {
      // Longest chain through the still-unassigned nodes.
      SmallVector<unsigned, 32> Best(K, 0), From(K, ~0u);
      unsigned BestEnd = ~0u, BestLen = 0;
      for (unsigned V = 0; V != K; ++V) {
        if (!Left[V])
          continue;
        unsigned B = 1, F = ~0u;
        for (unsigned U : Preds[V]) {
          if (!Left[U] || Best[U] + 1 <= B)
            continue;
          B = Best[U] + 1;
          F = U;
        }
        Best[V] = B;
        From[V] = F;
        if (B > BestLen) {
          BestLen = B;
          BestEnd = V;
        }
      }
      for (unsigned V = BestEnd; V != ~0u; V = From[V]) {
        Left[V] = false;
        --Remaining;
      }
      W.LogUpper -= logFactorial(BestLen);
      (void)Len;
    }
  }

  // Lower bound. Nodes at the same longest-path depth are mutually
  // unordered, so the largest such level is an antichain and may be permuted
  // freely.
  {
    SmallVector<unsigned, 32> Depth(K, 0);
    // Nodes are in region topological order, and Nodes is sorted, so one
    // forward sweep is enough.
    for (unsigned V = 0; V != K; ++V)
      for (unsigned U : Preds[V])
        Depth[V] = std::max(Depth[V], Depth[U] + 1);
    SmallVector<unsigned, 32> Level(K + 1, 0);
    unsigned Widest = 0;
    for (unsigned V = 0; V != K; ++V)
      Widest = std::max(Widest, ++Level[Depth[V]]);
    W.LogLower = logFactorial(Widest);
  }

  if (W.LogLower > W.LogUpper)
    W.LogLower = W.LogUpper;
  return W;
}

//===----------------------------------------------------------------------===//
// Building the decomposition
//===----------------------------------------------------------------------===//

static cl::opt<unsigned> SlotExactLimit(
    "amdgpu-optsched-slot-exact-limit", cl::init(16), cl::Hidden,
    cl::desc("Count a slot's orders exactly when it holds at most this many "
             "instructions; bound it above that"));

Decomposition Decomposition::build(const Problem &P, ArrayRef<unsigned> Order) {
  Decomposition D;
  const unsigned N = P.size();
  D.ChainIndex.assign(N, ~0u);
  D.Slots.assign(N, Slotting());
  D.Cycle.assign(N, 0);
  if (!N)
    return D;

  // --- The spine -------------------------------------------------------
  //
  // Longest path by latency, over real dependences only. Depth is the longest
  // path into a node, Height the longest out of it, so Depth+Height is the
  // length of the longest path through it and is equal for every node on a
  // longest path.
  SmallVector<unsigned, 0> Depth(N, 0), Height(N, 0);
  auto IsReal = [&](const Edge &E) { return !E.Synthetic; };

  for (unsigned V = 0; V != N; ++V)
    for (unsigned EI : P.predEdgesOf(V)) {
      const Edge &E = P.edges()[EI];
      if (IsReal(E))
        Depth[V] = std::max(Depth[V], Depth[E.Pred] + E.Latency);
    }
  for (unsigned V = N; V-- > 0;) {
    Height[V] = P.latencyOf(V);
    for (unsigned EI : P.succEdgesOf(V)) {
      const Edge &E = P.edges()[EI];
      if (IsReal(E))
        Height[V] = std::max(Height[V], E.Latency + Height[E.Succ]);
    }
  }

  unsigned Peak = 0;
  for (unsigned V = 0; V != N; ++V)
    if (Depth[V] + Height[V] > Peak)
      Peak = Depth[V] + Height[V];
  unsigned Seed = 0;
  for (unsigned V = 0; V != N; ++V)
    if (Depth[V] + Height[V] == Peak) {
      Seed = V;
      break;
    }
  D.Length = Peak;

  // Walk out from the seed in both directions, always along an edge that is
  // tight -- one that realises the longest path. Every node reached this way
  // is forced to follow the previous one in any legal schedule.
  SmallVector<unsigned, 0> Back;
  for (unsigned V = Seed;;) {
    Back.push_back(V);
    unsigned Prev = ~0u;
    for (unsigned EI : P.predEdgesOf(V)) {
      const Edge &E = P.edges()[EI];
      if (IsReal(E) && Depth[V] == Depth[E.Pred] + E.Latency) {
        Prev = E.Pred;
        break;
      }
    }
    if (Prev == ~0u)
      break;
    V = Prev;
  }
  D.Chain.assign(Back.rbegin(), Back.rend());
  for (unsigned V = Seed;;) {
    unsigned Next = ~0u;
    for (unsigned EI : P.succEdgesOf(V)) {
      const Edge &E = P.edges()[EI];
      if (IsReal(E) && Height[V] == E.Latency + Height[E.Succ]) {
        Next = E.Succ;
        break;
      }
    }
    if (Next == ~0u)
      break;
    D.Chain.push_back(Next);
    V = Next;
  }

  const unsigned K = D.Chain.size();
  for (unsigned I = 0; I != K; ++I) {
    assert(D.ChainIndex[D.Chain[I]] == ~0u && "chain visits a node twice");
    D.ChainIndex[D.Chain[I]] = I;
  }

  // --- Legal slot intervals --------------------------------------------
  //
  // Over *all* edges, synthetic ones included: this is legality, and a pinned
  // instruction really does have nowhere to go. After[v] is the index of the
  // last chain instruction that reaches v, Before[v] the first one v reaches.
  // One sweep each, because node numbers are a topological order and the
  // relation composes.
  SmallVector<unsigned, 0> After(N, ~0u), Before(N, K);
  for (unsigned V = 0; V != N; ++V)
    for (unsigned EI : P.predEdgesOf(V)) {
      unsigned U = P.edges()[EI].Pred;
      unsigned From = D.isChain(U) ? D.ChainIndex[U] : After[U];
      if (From != ~0u && (After[V] == ~0u || From > After[V]))
        After[V] = From;
    }
  for (unsigned V = N; V-- > 0;)
    for (unsigned EI : P.succEdgesOf(V)) {
      unsigned W = P.edges()[EI].Succ;
      unsigned To = D.isChain(W) ? D.ChainIndex[W] : Before[W];
      Before[V] = std::min(Before[V], To);
    }

  for (unsigned V = 0; V != N; ++V) {
    Slotting &S = D.Slots[V];
    if (D.isChain(V)) {
      S.Lo = S.Hi = S.Slot = D.ChainIndex[V];
      continue;
    }
    S.Lo = After[V] == ~0u ? 0 : After[V] + 1;
    S.Hi = Before[V];
    assert(S.Lo <= S.Hi && "legal slot set is empty: the region has a cycle");
  }

  // --- Where the given order puts everything ---------------------------
  D.Members.assign(K + 1, {});
  unsigned Seen = 0;
  for (unsigned V : Order) {
    if (D.isChain(V)) {
      assert(D.ChainIndex[V] == Seen && "chain out of order in the schedule");
      ++Seen;
      continue;
    }
    D.Slots[V].Slot = Seen;
    assert(D.Slots[V].Lo <= Seen && Seen <= D.Slots[V].Hi &&
           "the order we were given puts an instruction outside its own legal "
           "interval; the interval computation is wrong");
    D.Members[Seen].push_back(V);
  }

  // --- The slot timeline ------------------------------------------------
  D.SlotStart.assign(K + 1, 0);
  D.SlotLast.assign(K + 1, 0);
  SmallVector<unsigned, 0> ChainCycle(K, 0);
  unsigned Cursor = 0;
  for (unsigned S = 0; S <= K; ++S) {
    D.SlotStart[S] = Cursor + 1;
    for (unsigned V : D.Members[S])
      D.Cycle[V] = ++Cursor;
    if (S == K) {
      // Nothing closes the last slot, so an instruction appended there issues
      // right after whatever is already in it.
      D.SlotLast[S] = Cursor + 1;
      break;
    }
    unsigned Ready = Cursor + 1;
    if (S)
      for (unsigned EI : P.predEdgesOf(D.Chain[S])) {
        const Edge &E = P.edges()[EI];
        if (E.Pred == D.Chain[S - 1])
          Ready = std::max(Ready, ChainCycle[S - 1] + E.Latency);
      }
    ChainCycle[S] = Ready;
    D.Cycle[D.Chain[S]] = Ready;
    // The last cycle still inside the slot: the chain instruction owns Ready.
    D.SlotLast[S] = Ready - 1;
    Cursor = Ready;
  }

  // --- Widths -----------------------------------------------------------
  D.Widths.assign(K + 1, Width());
  for (unsigned S = 0; S <= K; ++S)
    D.Widths[S] = countOrders(P, D.Members[S], SlotExactLimit);

  return D;
}

//===----------------------------------------------------------------------===//
// Costing a placement
//===----------------------------------------------------------------------===//

Stall Decomposition::stallOf(const Problem &P, unsigned N, unsigned S) const {
  Stall Worst;
  // Best case within the slot, so this stays a lower bound: an incoming
  // latency gets the whole slot to hide in, an outgoing one gets none of it.
  unsigned Latest = SlotLast[S];
  unsigned Earliest = SlotStart[S];

  // The earliest and latest a neighbour could issue. A chain instruction
  // cannot move at all; a free one can go anywhere in its own interval, and
  // assuming it does is what keeps this a bound rather than a guess about
  // the order we happen to be looking at.
  auto Soonest = [&](unsigned V) {
    return isChain(V) ? Cycle[V] : SlotStart[Slots[V].Lo];
  };
  auto Latest2 = [&](unsigned V) {
    return isChain(V) ? Cycle[V] : SlotLast[Slots[V].Hi];
  };

  for (unsigned EI : P.predEdgesOf(N)) {
    const Edge &E = P.edges()[EI];
    if (Soonest(E.Pred) + E.Latency > Latest) {
      // The wave waits here, in this slot, for a value that cannot be ready
      // in time no matter where its producer goes.
      unsigned Cycles = Soonest(E.Pred) + E.Latency - Latest;
      if (Cycles > Worst.Cycles) {
        Worst.Cycles = Cycles;
        Worst.Where = S;
      }
    } else if (!isChain(E.Pred) && Cycle[E.Pred] + E.Latency > Latest) {
      Worst.NeedsCompanion = true;
    }
  }
  for (unsigned EI : P.succEdgesOf(N)) {
    const Edge &E = P.edges()[EI];
    if (Earliest + E.Latency > Latest2(E.Succ)) {
      // The wave waits at the consumer, which is the interesting half: this
      // is the "putting the ds_read in slot 5 stalls slot 6" case.
      unsigned Cycles = Earliest + E.Latency - Latest2(E.Succ);
      if (Cycles > Worst.Cycles) {
        Worst.Cycles = Cycles;
        Worst.Where = Slots[E.Succ].Slot;
      }
    } else if (!isChain(E.Succ) && Earliest + E.Latency > Cycle[E.Succ]) {
      Worst.NeedsCompanion = true;
    }
  }
  return Worst;
}

double Decomposition::logPlacementFreedom() const {
  double L = 0.0;
  for (unsigned V = 0, N = Slots.size(); V != N; ++V)
    if (!isChain(V))
      L += std::log10((double)Slots[V].freedom());
  return L;
}

double Decomposition::logOrderingFreedom() const {
  double L = 0.0;
  for (const Width &W : Widths)
    L += W.LogUpper;
  return L;
}

//===----------------------------------------------------------------------===//
// Dumping
//===----------------------------------------------------------------------===//

void Decomposition::print(raw_ostream &OS, const Problem &P,
                          function_ref<StringRef(unsigned)> NodeName) const {
  const unsigned K = Chain.size();
  OS << "optsched-chain: " << P.Name << " n=" << P.size() << " chain=" << K
     << " length=" << Length << " slots=" << numSlots() << "\n";

  OS << "optsched-chain:   chain";
  for (unsigned I = 0; I != K; ++I)
    OS << (I ? " -> " : " ") << Chain[I] << " " << NodeName(Chain[I]);
  OS << "\n";

  for (unsigned S = 0; S <= K; ++S) {
    const Width &W = Widths[S];
    // Room is the slot's capacity in issue cycles; free is what is left of it.
    // Free cycles are the whole point of the spine: they are latency the
    // chain is already paying for, so work moved into them costs nothing.
    unsigned Room =
        SlotLast[S] >= SlotStart[S] ? SlotLast[S] - SlotStart[S] + 1 : 0;
    unsigned Used = Members[S].size();
    OS << "optsched-chain:   slot " << S << ": starts " << SlotStart[S]
       << ", room " << Room << " (" << (Room > Used ? Room - Used : 0)
       << " free), " << Used << " instr" << (Used == 1 ? "" : "s")
       << ", width ";
    if (W.Exact)
      OS << W.Count << " exact";
    else
      OS << "10^" << format("%.1f", W.LogLower) << "..10^"
         << format("%.1f", W.LogUpper);
    if (S < K)
      OS << ", closed by " << Chain[S] << " at " << Cycle[Chain[S]];
    OS << "\n";
  }

  for (unsigned V = 0, N = P.size(); V != N; ++V) {
    if (isChain(V))
      continue;
    const Slotting &Sl = Slots[V];
    OS << "optsched-chain:   node " << V << " " << NodeName(V) << ": slots ["
       << Sl.Lo << "," << Sl.Hi << "] now " << Sl.Slot;
    if (Sl.Lo == Sl.Hi) {
      OS << " pinned\n";
      continue;
    }
    OS << ", stalls";
    for (unsigned S = Sl.Lo; S <= Sl.Hi; ++S) {
      Stall St = stallOf(P, V, S);
      OS << " s" << S << "=";
      if (St.Cycles)
        OS << "+" << St.Cycles << "@s" << St.Where;
      else
        OS << "0";
      // Cheap only if a neighbour moves too, so a search may not take this
      // number as the cost of a single placement.
      if (St.NeedsCompanion)
        OS << "*";
    }
    OS << "\n";
  }

  OS << "optsched-chain:   freedom: placement 10^"
     << format("%.1f", logPlacementFreedom()) << ", ordering 10^"
     << format("%.1f", logOrderingFreedom()) << "\n";

  // The line worth reading first. On a real loop body most slots turn out to
  // be empty or tiny, and nearly all the difficulty sits in a handful of wide
  // ones -- which is the whole argument for decomposing at all, and tells a
  // search exactly where it has to be clever rather than exhaustive.
  unsigned Empty = 0, Exact = 0, Widest = 0;
  for (unsigned S = 0; S <= K; ++S) {
    unsigned Used = Members[S].size();
    Empty += Used == 0;
    Exact += Widths[S].Exact;
    Widest = std::max(Widest, Used);
  }
  // The slot timeline enforces the chain and one instruction per cycle, and
  // nothing else -- an off-chain latency that does not fit is simply not in
  // it. So this span is a lower bound on what the order costs, not its
  // makespan, and the gap from the chain is the room a search has to work in.
  OS << "optsched-chain:   summary: " << numSlots() << " slots, " << Empty
     << " empty, " << Exact << " countable exactly, widest " << Widest
     << " instrs; chain " << Length << " of " << SlotLast[K]
     << " cycles, both lower bounds\n";
}
