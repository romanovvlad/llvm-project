//===- AMDGPUOptSched.h - Exact AMDGPU instruction scheduler ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Skeleton of a scheduler that aims to be *provably* optimal rather than
/// merely good: for a region it either returns a schedule it can prove is the
/// shortest legal one, or it says it could not and labels its answer as an
/// upper bound.
///
/// The pieces, and why they are separate:
///
///   Problem   the region, copied out of the DAG into a plain graph. Nothing
///             downstream sees SUnits, so a solver cannot accidentally depend
///             on scheduler state that changes under it.
///   Solver    proposes an order. Interchangeable; today a list scheduler and
///             an exhaustive search, tomorrow something that scales.
///   Verifier  re-derives everything and refuses answers that do not hold up.
///             It shares no code with any solver: a solver proposes, the
///             verifier disposes.
///
/// A solver is allowed to be wrong. The verifier is the thing that must not
/// be, which is why it is small, and why the driver drops a schedule it
/// cannot certify instead of trying to repair it.
///
/// Register allocation is not here yet. It is the reason the Problem carries
/// more than a dependence graph -- see the register budget plumbing in the
/// driver -- but allocation itself is a later milestone.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUOPTSCHED_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUOPTSCHED_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/Support/Error.h"
#include <array>
#include <memory>
#include <string>
#include <tuple>

namespace llvm {

class MachineSchedContext;
class ScheduleDAGInstrs;
class ScheduleDAGMI;
class ScheduleDAGMILive;
class raw_ostream;

namespace AMDGPUOptSched {

/// A precedence constraint: \p Succ may not issue until \p Latency cycles
/// after \p Pred issues.
///
/// \p Kind is kept only so that a failure can say *which* kind of edge was
/// violated. Anti- and output dependences are the ones an optimiser drifts
/// into breaking, because breaking them lowers register pressure, so naming
/// them in the message is worth the byte.
struct Edge {
  unsigned Pred;
  unsigned Succ;
  unsigned Latency;
  SDep::Kind Kind;
  /// True for edges this file invented rather than read out of the DAG --
  /// today only the physical-register pinning. They are legality, so the
  /// verifier enforces them, but they are not dependences, so anything that
  /// reasons about the *shape* of the computation must skip them: a pinned
  /// region would otherwise look like one long dependence chain.
  bool Synthetic = false;
};

/// A functional unit an instruction holds, and for how long after it issues.
///
/// Latency and occupancy are different things and only one of them was here
/// before. Latency says when a *dependent* instruction may read the result; it
/// says nothing about an independent instruction that needs the same unit. On
/// this target that gap is the whole game: an MFMA holds the matrix pipe for
/// four cycles per pass, so two back-to-back MFMAs cost the second one's full
/// occupancy even though nothing connects them, and the only way to spend that
/// window is to put unrelated work in it. That is what "co-execution" means
/// here, and a model with no notion of a unit cannot represent it -- it scores
/// a run of MFMAs and the same MFMAs with the rest of the region threaded
/// between them exactly alike.
///
/// \p Pipe indexes the target's own processor resources and \p Cycles is its
/// ReleaseAtCycles, both read out of the scheduling model rather than invented
/// here: these numbers belong to the subtarget and change with it.
///
/// What is still missing, so that its absence is on the record rather than
/// merely unnoticed: the wait states GCNHazardRecognizer inserts. Those are
/// decided on physical registers after allocation, so there is nothing to ask
/// about them here, and guessing at them would put a number in the cost
/// function that nothing can check. The part of the hazard story that *is*
/// visible before allocation -- an MFMA result costing a plain VALU reader
/// more than it costs the next MFMA in the chain -- is already in the edge
/// latencies, because the subtarget puts it there through its ReadAdvance
/// entries and GCNSubtarget::adjustSchedDependency.
struct PipeUse {
  unsigned Pipe;
  unsigned Cycles;
};

/// The register files a schedule competes for.
///
/// Separate because they are allocated separately: a register freed in one is
/// no help at all to a value that needs another, so these counts must never be
/// added together or traded against each other.
enum class RegFile { SGPR, ArchVGPR, AGPR, NumFiles };

/// A count per register file, in 32-bit registers.
struct Pressure {
  std::array<unsigned, unsigned(RegFile::NumFiles)> Files = {};

  unsigned &operator[](RegFile F) { return Files[unsigned(F)]; }
  unsigned operator[](RegFile F) const { return Files[unsigned(F)]; }

  void takeMax(const Pressure &O) {
    for (unsigned I = 0; I != Files.size(); ++I)
      Files[I] = std::max(Files[I], O.Files[I]);
  }

  /// Only ever a tie-break. Summing across files is meaningless as a
  /// *limit* -- see RegFile -- but as a way of preferring the smaller of two
  /// otherwise equal schedules it is fine, and it is the only use it has.
  unsigned sum() const {
    unsigned S = 0;
    for (unsigned C : Files)
      S += C;
    return S;
  }
};

/// A value whose live range the schedule can stretch or shrink.
///
/// One per virtual register the region touches, and one per *register unit* of
/// every allocatable physical register it touches. The unit is the right
/// granularity for the physical half: it is what the allocator contends for,
/// it is what the driver's demandOf measures, and it is what makes a 64-bit
/// physical def cost two where a 32-bit one costs one without any special
/// case.
///
/// The physical half is a coarser approximation than the virtual half, and
/// deliberately so. Outside SSA a unit may be written several times with
/// several disjoint ranges; this collapses them into one range running from
/// the first definition to the last read. That overstates demand, never
/// understates it, which is the safe direction for something whose job is to
/// stop a schedule from asking for registers that are not there.
struct RegValue {
  unsigned Size = 0; ///< In 32-bit registers.
  RegFile File = RegFile::SGPR;
  unsigned Uses = 0;    ///< Reads inside the region. Zero means a dead def.
  bool LiveIn = false;  ///< Alive above the region, so alive from cycle 0.
  bool LiveOut = false; ///< Alive below it, so it never dies inside.
  /// A physical register unit rather than a virtual register. LiveIn and
  /// LiveOut are then real liveness, read off the block, rather than inferred
  /// from where the definitions are.
  bool Phys = false;
  /// The files this value could actually be given a register in, one bit per
  /// RegFile. Usually just \p File, but not always, and the difference is the
  /// whole of shortfallOf: the allocator widens a value's class when it splits
  /// a live range -- getLargestLegalSuperClass turns vreg_64_align2 into
  /// av_64_align2 -- so such a value is only stuck if *both* vector files are
  /// full, while a plain vgpr_32 next to it is stuck as soon as one is.
  unsigned Files = 0;
};

/// One operand's demand on one value: the files that operand's own register
/// class can be satisfied from.
///
/// A value's class is not fixed over its whole range -- splitting widens it,
/// which is how a result defined in a VGPR gets parked in an AGPR for the
/// crowded part of its life -- but at the instruction the operand's class is
/// a hard constraint. Those two facts are what separate a schedule that
/// allocates from one that does not, so they are recorded separately.
struct Constraint {
  unsigned Node;
  unsigned Value;
  unsigned Files; ///< One bit per RegFile.
};

/// What the allocator can hand out per file, in 32-bit registers.
///
/// This is what *this function* may use, not what the architecture has. The
/// two differ by a lot -- an occupancy request can cut a 512-register file to
/// 64 -- and it is the smaller number every question here is really about.
struct Budget {
  std::array<unsigned, unsigned(RegFile::NumFiles)> Files = {};
  /// gfx90a and later spend arch VGPRs and AGPRs out of one pool, so neither
  /// per-file limit is the real one and \p VGPR is.
  bool UnifiedVGPR = false;
  unsigned VGPR = 0;

  unsigned operator[](RegFile F) const { return Files[unsigned(F)]; }

  /// Registers demanded beyond what exists, summed over the files that
  /// overflow. Zero means the region fits, which is the only distinction that
  /// matters; the magnitude is just a direction to search in.
  unsigned excessOf(const Pressure &P) const;
};

/// What a schedule costs, in the order the costs matter.
///
/// Excess first, because a region that does not fit spills, and no number of
/// cycles saved pays for a spill. Makespan second, which is the thing a
/// scheduler is for. Peak last and only as a tie-break: below the budget a
/// register is free, so ranking it under makespan means we never buy one with
/// a cycle -- we only decline to waste one when the cycles come out equal
/// anyway.
struct Cost {
  unsigned Excess = 0;
  unsigned Makespan = 0;
  unsigned Peak = 0;

  bool operator<(const Cost &O) const {
    return std::tie(Excess, Makespan, Peak) <
           std::tie(O.Excess, O.Makespan, O.Peak);
  }
  bool operator==(const Cost &O) const {
    return std::tie(Excess, Makespan, Peak) ==
           std::tie(O.Excess, O.Makespan, O.Peak);
  }
};

/// The region, as a graph. Node numbers are SUnit::NodeNum, so an order is
/// directly applicable to the DAG it came from.
///
/// The translation copies; it does not simplify. Every non-artificial edge
/// the DAG has becomes an edge here. Dropping one would make an illegal
/// schedule look legal to the solver, and the solver is the one component the
/// verifier cannot see inside.
class Problem {
public:
  /// Weak edges (clustering and similar hints) are deliberately absent: they
  /// are preferences, not legality, and the generic scheduler treats them the
  /// same way. A schedule that ignores them is slower, not wrong.
  static Problem build(const ScheduleDAGMILive &DAG);

  unsigned size() const { return NodeLatency.size(); }
  ArrayRef<Edge> edges() const { return Edges; }
  ArrayRef<unsigned> predEdgesOf(unsigned N) const { return PredEdges[N]; }
  ArrayRef<unsigned> succEdgesOf(unsigned N) const { return SuccEdges[N]; }
  unsigned latencyOf(unsigned N) const { return NodeLatency[N]; }

  /// Functional units \p N holds once it issues. Usually one, sometimes none
  /// -- a pseudo-instruction that becomes no code holds nothing.
  ArrayRef<PipeUse> pipesOf(unsigned N) const { return Pipes[N]; }
  /// How many units the subtarget has, so a timing pass can size its state.
  unsigned numPipes() const { return NumPipes; }

  /// Earliest issue cycles for a fixed order.
  ///
  /// Given the order this assignment is optimal -- each instruction issues
  /// the moment its dependences, the single issue port and the functional
  /// units allow -- so searching orders and timing each one searches the whole
  /// space. That is what makes the exhaustive solver exhaustive. (Holding an
  /// instruction back past its earliest cycle can never help: issue is in
  /// order, so a later instruction cannot overtake it, and a unit freed later
  /// is not freed sooner.)
  void timeOrder(ArrayRef<unsigned> Order,
                 SmallVectorImpl<unsigned> &Cycles) const;

  unsigned makespanOf(ArrayRef<unsigned> Cycles) const;

  /// The order the region already has. Always legal, so it is both the
  /// fallback and the thing every answer has to beat to be worth applying.
  void sourceOrder(SmallVectorImpl<unsigned> &Order) const;

  /// Longest remaining latency path from each node to a sink, counting the
  /// node's own latency. Any schedule obeys makespan >= cycle[N] + height[N],
  /// which is what lets the search prune.
  ArrayRef<unsigned> heights() const { return Height; }

  //===--------------------------------------------------------------------===//
  // Register demand
  //===--------------------------------------------------------------------===//

  ArrayRef<RegValue> values() const { return Values; }
  /// Registers held across the whole region by values it never names: defined
  /// above it, read below it. No order can free one, so this is a constant
  /// added to every candidate's pressure -- but it is the constant that
  /// decides whether the region fits, and Cost::Excess is the first term.
  const Pressure &liveThrough() const { return LiveThru; }
  /// Values whose live range starts at \p N. A value redefined inside the
  /// region is listed only at its first definition, which is where its range
  /// begins.
  ArrayRef<unsigned> defsOf(unsigned N) const { return Defs[N]; }
  /// Values \p N reads, each at most once however many operands name it: one
  /// instruction reading a value twice still kills it once.
  ArrayRef<unsigned> usesOf(unsigned N) const { return Uses[N]; }
  const Budget &budget() const { return Bud; }

  /// Peak demand of a fixed order, per file.
  ///
  /// A *model*, not a measurement, and the difference is load-bearing. Every
  /// value lives from its first definition to its last read, which is exact
  /// for virtual registers in SSA and an overestimate for physical ones; it
  /// does not know about subregister lanes; it does not know what a call's
  /// register mask clobbers. It is here to steer the search, which needs an
  /// answer per candidate order and cannot afford the real trackers. What
  /// actually gets applied is still decided by measuring the chosen order with
  /// the backend's own trackers -- a model that disagrees with the allocator
  /// must never be the last word.
  /// \p Trace, if given, receives the pressure at each position of \p Order,
  /// sampled where the peak is: definitions counted, nothing dead yet. It is
  /// how the model gets lined up against the trackers instruction by
  /// instruction instead of only at its peak.
  Pressure pressureOf(ArrayRef<unsigned> Order,
                      SmallVectorImpl<Pressure> *Trace = nullptr) const;

  Cost costOf(ArrayRef<unsigned> Order) const;

  /// How badly a fixed order asks for registers that provably are not there.
  ///
  /// This is a different question from pressureOf, and a sharper one. Pressure
  /// is about how many registers are wanted at once, and wanting too many is
  /// survivable: the allocator spills something and carries on. This is about
  /// a value having nowhere at all to go, which it does not survive.
  ///
  /// A physical range is pre-coloured and cannot be spilled or moved out of
  /// the way, so every register it holds is one a virtual value live at the
  /// same time may not have. If the physical ranges live across some value's
  /// range already fill that value's file, the value has no colour, and no
  /// amount of spilling anything else will give it one. Returns the number of
  /// registers short, summed over the values that are short -- zero meaning
  /// every value has somewhere to go, which is the distinction that matters.
  ///
  /// Counting pressure cannot see this and no refinement of it can, which is
  /// the whole reason this exists: the schedule that found it held 64 physical
  /// VGPRs across the value in one order and 64 *virtual* ones in the other,
  /// identical at every point by any count, allocatable in exactly one of
  /// them. See optsched-physreg-pin.ll.
  unsigned shortfallOf(ArrayRef<unsigned> Order) const;

  /// "fn:bb.3", for logs.
  std::string Name;

private:
  SmallVector<Edge, 0> Edges;
  /// Indices into Edges, not node numbers: the latency lives on the edge.
  SmallVector<SmallVector<unsigned, 4>, 0> PredEdges;
  SmallVector<SmallVector<unsigned, 4>, 0> SuccEdges;
  SmallVector<unsigned, 0> NodeLatency;
  SmallVector<unsigned, 0> Height;
  SmallVector<SmallVector<PipeUse, 1>, 0> Pipes;
  unsigned NumPipes = 0;

  SmallVector<RegValue, 0> Values;
  SmallVector<SmallVector<unsigned, 2>, 0> Defs;
  SmallVector<SmallVector<unsigned, 2>, 0> Uses;
  SmallVector<Constraint, 0> Constraints;
  Budget Bud;
  Pressure LiveThru;

  void computeHeights();
  void buildRegisterModel(const ScheduleDAGMILive &DAG);
};

//===----------------------------------------------------------------------===//
// Chain decomposition
//===----------------------------------------------------------------------===//

/// Where a non-chain instruction is allowed to go.
///
/// \p Lo and \p Hi are always a contiguous interval, never a scattered set,
/// and that is the whole reason the decomposition is worth building. The
/// chain is a *total* order, so "after every chain instruction I depend on"
/// and "before every chain instruction that depends on me" are a lower and an
/// upper bound on one number. Scheduling becomes picking a point in an
/// interval, and a value's live range becomes an interval on the same axis --
/// which is what makes register allocation tractable on top of this later.
struct Slotting {
  unsigned Lo = 0;   ///< First legal slot.
  unsigned Hi = 0;   ///< Last legal slot, inclusive. Never less than Lo.
  unsigned Slot = 0; ///< Where the order we were given puts it. In [Lo, Hi].

  unsigned freedom() const { return Hi - Lo + 1; }
};

/// What a candidate placement costs.
///
/// \p Cycles is a lower bound on the wait and \p Where is the slot the wave
/// actually stops in -- putting a long-latency load early is free at the load
/// and expensive at its consumer, and the consumer is the interesting half.
///
/// The bound is taken over the best placement of the instruction's free
/// neighbours, not their current one, so it isolates the part of the cost the
/// spine forces and that no amount of moving other instructions can remove.
/// \p NeedsCompanion says the difference matters here: this slot is only
/// this cheap if some neighbour moves as well. Without that split a load
/// looks worse the earlier it goes, purely because its address computation
/// was left behind.
struct Stall {
  unsigned Where = 0;
  unsigned Cycles = 0;
  bool NeedsCompanion = false;
};

/// The size of a slot's local search space: how many distinct orders its
/// instructions admit, which is the number of linear extensions of the
/// sub-order they form.
///
/// Reported in log10 because the counts pass 10^18 on real regions. Exact
/// when the slot is small enough to count by dynamic programming; otherwise
/// bracketed, and \p Exact says which -- a bound must never be mistaken for a
/// count.
struct Width {
  double LogLower = 0.0;
  double LogUpper = 0.0;
  uint64_t Count = 1; ///< Meaningful only when Exact.
  bool Exact = true;
};

/// A region viewed as its longest dependence chain plus everything hanging
/// off it.
///
/// The chain is the spine: its instructions depend on each other in sequence,
/// so no schedule reorders them and no schedule is shorter than the chain is
/// long. K chain instructions cut the region into K+1 *slots* -- slot s being
/// the cycles after chain instruction s-1 and before chain instruction s -- and
/// every other instruction is described by which slots it may occupy and what
/// each would cost.
///
/// A useful consequence: each slot's instructions form a self-contained
/// sub-order. If u and v share a slot and u reaches v through w, then w is
/// ordered between them, and a slot is a contiguous span of cycles, so w is in
/// that slot too. No slot's search can be wrong because of something outside
/// it.
class Decomposition {
public:
  /// \p Order is the schedule to measure against -- "the current scheduling".
  /// Slot assignments and stall estimates are relative to it.
  static Decomposition build(const Problem &P, ArrayRef<unsigned> Order);

  /// Chain instructions, in order. Node numbers.
  ArrayRef<unsigned> chain() const { return Chain; }
  /// Latency along the chain, and so a lower bound on the region's makespan.
  unsigned chainLength() const { return Length; }
  unsigned numSlots() const { return Chain.size() + 1; }

  bool isChain(unsigned N) const { return ChainIndex[N] != ~0u; }
  const Slotting &slottingOf(unsigned N) const { return Slots[N]; }
  ArrayRef<unsigned> membersOf(unsigned S) const { return Members[S]; }
  const Width &widthOf(unsigned S) const { return Widths[S]; }

  /// First cycle a slot's instructions may issue, and the last cycle
  /// available to them before the chain instruction that closes the slot.
  unsigned slotStart(unsigned S) const { return SlotStart[S]; }
  unsigned slotLast(unsigned S) const { return SlotLast[S]; }

  /// Estimated issue cycle of a node under the order we were given.
  unsigned cycleOf(unsigned N) const { return Cycle[N]; }

  /// Lower bound on the stall that putting \p N in slot \p S would cost,
  /// holding everything else in place.
  Stall stallOf(const Problem &P, unsigned N, unsigned S) const;

  /// log10 of the number of ways free instructions may be distributed over
  /// slots, ignoring what the distribution costs.
  double logPlacementFreedom() const;
  /// log10 of the number of orders available inside the slots, as currently
  /// filled.
  double logOrderingFreedom() const;

  void print(raw_ostream &OS, const Problem &P,
             function_ref<StringRef(unsigned)> NodeName) const;

private:
  SmallVector<unsigned, 0> Chain;
  unsigned Length = 0;
  SmallVector<unsigned, 0> ChainIndex; ///< ~0u for non-chain nodes.
  SmallVector<Slotting, 0> Slots;
  SmallVector<SmallVector<unsigned, 4>, 0> Members; ///< Free nodes, per slot.
  SmallVector<Width, 0> Widths;
  SmallVector<unsigned, 0> SlotStart;
  SmallVector<unsigned, 0> SlotLast;
  SmallVector<unsigned, 0> Cycle;
};

/// Number of linear extensions of the sub-order \p Nodes induces in \p P.
///
/// Exact by subset dynamic programming when \p Nodes is at most \p ExactLimit
/// long (which cannot overflow: the count is at most 20! < 2^64). Larger
/// slots get a bracket instead: a chain cover gives an upper bound, the
/// largest level set gives a lower one.
Width countOrders(const Problem &P, ArrayRef<unsigned> Nodes,
                  unsigned ExactLimit);

/// How much is actually known about an answer.
///
/// These are not decoration. `Optimal` is a claim that no legal schedule is
/// shorter, and the verifier will not let it be made without a reason
/// attached; a search that ran out of budget produces `UpperBound` even if it
/// happens to have found the optimum.
enum class Status {
  Witness,    ///< A legal schedule. No claim about its quality.
  UpperBound, ///< The best found, search incomplete.
  Optimal     ///< Proved minimal. Requires Proof to be non-empty.
};

StringRef toString(Status S);

struct Answer {
  SmallVector<unsigned, 0> Order;
  /// Issue cycle per node, indexed by node number. Empty if the solver did
  /// not reason about time.
  SmallVector<unsigned, 0> Cycles;
  unsigned Makespan = 0;
  Status St = Status::Witness;
  /// Why Status may say Optimal. The verifier cannot replay a proof; it can
  /// insist one was written down.
  std::string Proof;
  /// Which solver produced this, for logs.
  StringRef Solver;
};

/// Check an answer against the problem it claims to solve.
///
/// Every check here has a failure mode behind it:
///   - a permutation check, because a solver that drops an instruction looks
///     spectacular on every other metric;
///   - *every* edge, not just data edges, for the anti-dependence reason
///     above;
///   - cycles replayed against latencies, when the answer is timed;
///   - the word "optimal", which may only be used with a proof.
Error verify(const Problem &P, const Answer &A);

class Solver {
public:
  virtual ~Solver();
  virtual StringRef getName() const = 0;
  virtual Answer solve(const Problem &P) = 0;
};

/// Critical-path list scheduling. Fast, never optimal except by luck, and
/// labelled accordingly. It is the fallback and the thing to beat.
std::unique_ptr<Solver> createListSolver();

/// Exhaustive search over legal orders, lowest Cost wins.
///
/// Exact when it finishes, which it only does on small regions -- the number
/// of legal orders is exponential in the region's width. \p StepLimit caps
/// the work; hitting it downgrades the answer to UpperBound rather than
/// silently returning "the best of the part I looked at" as a minimum.
///
/// "Optimal" here means optimal for Cost, and the proof string says so. It is
/// a weaker claim than it was when the only term was makespan: Cost's third
/// term is measured with Problem::pressureOf, which is a model.
std::unique_ptr<Solver> createExactSolver(unsigned NodeLimit,
                                          uint64_t StepLimit);

/// Exhaustive inside a window of \p Width consecutive instructions, swept
/// along the region and repeated until nothing moves.
///
/// Why a window at all. The exhaustive solver is the thing we want and it
/// stops being affordable at about sixteen instructions, while real regions
/// run to a thousand. Everything above the limit used to get the list
/// scheduler, which is a different kind of answer: it commits to one
/// instruction at a time and never reconsiders, so on a region whose cost
/// comes from where two instructions sit relative to each other it is not
/// close, it is not even looking.
///
/// What makes a window sound is that any rearrangement of a *contiguous* run
/// of a legal order is still legal as long as it respects the edges inside the
/// run. Every edge that crosses the window's boundary goes from the prefix in
/// or from the window out, and permuting the window moves neither end across
/// the boundary, so no crossing edge can be broken. That means each window can
/// be searched on its own, exactly, with everything outside it held fixed, and
/// the answer spliced straight back in.
///
/// What a window cannot do is move an instruction further than \p Width
/// positions in one step, which is why the sweep repeats: each pass can carry
/// an instruction most of a window's length, and passes run until a pass
/// changes nothing or \p Passes is reached.
///
/// The result is `UpperBound`, never `Optimal`, and the distinction is real
/// rather than cautious. Exhausting every window proves that no single window
/// can be improved on its own; it says nothing about a rearrangement that
/// spans two of them, and such rearrangements exist. A local minimum is not a
/// minimum and must not be labelled as one.
std::unique_ptr<Solver> createWindowSolver(unsigned Width, unsigned Passes,
                                           uint64_t StepLimit,
                                           uint64_t TotalStepLimit);

} // end namespace AMDGPUOptSched

/// The scheduler itself, for MachineSchedRegistry.
ScheduleDAGInstrs *createAMDGPUOptSchedDAG(MachineSchedContext *C);

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUOPTSCHED_H
