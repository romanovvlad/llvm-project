//===-- AMDGPUGlobalSched.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Scheduling taken over the whole function instead of one region at a time.
///
/// Every other scheduler here is handed a region -- a run of instructions
/// inside one block, between two scheduling boundaries -- and may reorder
/// what is in it and nothing else. Which block an instruction lives in is
/// then whatever lowering decided, and is never revisited.
///
/// This asks the other question. The input is a fixed set of instructions and
/// a fixed CFG; a schedule is a map from instruction to (block, index); we
/// want the cheapest schedule that computes the same thing. Under that
/// reading there is no such operation as hoisting or sinking -- those name a
/// difference between two schedules, not a step for getting from one to the
/// other.
///
/// For an instruction with no side effects, computing the same thing asks for
/// exactly one property: the block it lands in is dominated by the definition
/// of every operand and dominates every use. Nothing else is needed, because
/// running a pure instruction more or fewer times is not observable. The set
/// of such blocks is the dominator-tree path between the deepest operand
/// definition and the nearest common dominator of the uses -- an interval,
/// the same shape the per-block code already works with, and the same
/// interval SSA itself is defined by. Keeping to it is also why no PHI ever
/// has to be inserted: a placement is only legal if the definition still
/// dominates every use, so no new merge of reaching definitions can arise.
///
/// Everything that is not pure is pinned to the block lowering gave it. That
/// is not one rule but a list of separate mechanisms, and it is written out
/// in classify() below.
///
/// The block chosen out of the interval is currently the one that runs least
/// often, ties going to the latest. That is a placement rule, not the
/// objective; the objective, and a search that uses it, come later.
///
/// Live intervals are not maintained, so the pass is not yet in the pipeline.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUGlobalSched.h"
#include "AMDGPU.h"
#include "GCNHazardRecognizer.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/MachineCycleAnalysis.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/TargetSchedule.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>

using namespace llvm;

#define DEBUG_TYPE "amdgpu-global-sched"

static cl::opt<unsigned> RefineSpan(
    "amdgpu-global-sched-refine-span", cl::Hidden, cl::init(8),
    cl::desc("Most walls a single refinement step may dissolve at once"));

namespace {

/// One depth of the recursive cut: the regions that exist at that depth, how
/// much of them a cut of their own can anchor, and what is left to order.
struct Tier {
  unsigned Regions = 0; ///< regions at this depth
  unsigned Steps = 0;   ///< instructions the cuts at this depth anchor
  unsigned Insts = 0;   ///< instructions those regions hold
  unsigned Worst = 0;   ///< instructions in the biggest of them
  unsigned Stuck = 0;   ///< regions with no order inside left to cut by
  double Bits = 0;      ///< bits to order every region at this depth
};

/// Where one instruction is allowed to live.
///
/// \p Pin is null when the instruction is free to be placed anywhere on the
/// dominator-tree path from \p Lo to \p Hi, and otherwise names the mechanism
/// that holds it in \p Home. A free instruction always has Lo dominating Home
/// dominating Hi, so the three of them describe one interval and a point
/// inside it.
struct Placement {
  MachineBasicBlock *Home = nullptr;
  MachineBasicBlock *Lo = nullptr;
  MachineBasicBlock *Hi = nullptr;
  const char *Pin = nullptr;

  bool isFree() const { return Pin == nullptr; }
};

/// One end of one ordering constraint: the earlier instruction has to issue,
/// and \p Latency cycles have to pass, before the later one may. \p Other is
/// the instruction at the far end -- the earlier one in a predecessor list,
/// the later one in a successor list.
struct Dep {
  MachineInstr *Other;
  unsigned Latency;
};

/// Everything a schedule has to respect, in one graph. The instructions are
/// in a topological order of it, so a single sweep each way answers when
/// anything may run.
struct DepGraph {
  SmallVector<MachineInstr *, 0> Order;
  DenseMap<const MachineInstr *, SmallVector<Dep, 4>> Preds, Succs;

  void add(MachineInstr *Pred, MachineInstr *Succ, unsigned Latency) {
    if (Pred == Succ)
      return;
    Preds[Succ].push_back({Pred, Latency});
    Succs[Pred].push_back({Succ, Latency});
  }

  ArrayRef<Dep> preds(const MachineInstr *MI) const { return at(Preds, MI); }
  ArrayRef<Dep> succs(const MachineInstr *MI) const { return at(Succs, MI); }

private:
  static ArrayRef<Dep>
  at(const DenseMap<const MachineInstr *, SmallVector<Dep, 4>> &M,
     const MachineInstr *MI) {
    auto It = M.find(MI);
    return It == M.end() ? ArrayRef<Dep>() : ArrayRef<Dep>(It->second);
  }
};

/// When everything could run if the only thing anybody ever waited for was
/// a value. \p Start is the earliest cycle an instruction may issue, \p Slack
/// how much later it may issue without pushing \p Length out, and \p Length
/// the longest chain in the function.
struct Timing {
  DenseMap<const MachineInstr *, unsigned> Start, Slack;
  unsigned Length = 0;
};

/// Where one instruction not on the path may go. \p First and \p Last are
/// the regions dependence allows it into, inclusive, and \p Here is the one
/// it is in now. \p Asap and \p Alap are the earliest and the latest cycle it
/// may issue at without anybody waiting for it or it waiting for anybody.
struct Window {
  MachineInstr *MI;
  unsigned First, Last, Here, Asap, Alap;
};

/// The function cut into regions by a path, and what may go where.
struct Regions {
  SmallVector<unsigned, 0> Count;      ///< how many instructions each allows
  SmallVector<unsigned, 0> Begin, End; ///< the cycles each region spans
  SmallVector<Window, 0> Wins;         ///< one per instruction not on the path

  /// Cycles each pipe is held for by what is in each region now, indexed by
  /// region and then by the scheduling model's resource number.
  SmallVector<SmallVector<unsigned, 0>, 0> Load;

  unsigned size() const { return Count.size(); }

  /// How many instructions region \p R has room for. The machine issues one
  /// instruction a cycle, so a region holds as many as it is long.
  unsigned cap(unsigned R) const { return std::max(End[R] - Begin[R], 1u); }

  /// What putting \p W at slot \p Slot of region \p R costs: the cycles spent
  /// waiting for its operands, plus the cycles its consumers then wait. Zero
  /// when the slot falls inside the instruction's own window.
  unsigned costAt(const Window &W, unsigned R, unsigned Slot) const {
    unsigned At = Begin[R] + Slot;
    return (W.Asap > At ? W.Asap - At : 0) + (At > W.Alap ? At - W.Alap : 0);
  }

  /// The slots of region \p R that cost \p W nothing, first and last. The
  /// range is empty, first past last, when none of them do.
  std::pair<int64_t, int64_t> freeIn(const Window &W, unsigned R) const {
    int64_t Lo = std::max<int64_t>((int64_t)W.Asap - Begin[R], 0);
    int64_t Hi = std::min<int64_t>((int64_t)W.Alap - Begin[R], cap(R) - 1);
    return {Lo, Hi};
  }

  /// The shortest region \p R could be whatever order its contents are put
  /// in: the busiest pipe has to be held for as long as it has to be held.
  /// Returns that length and which pipe asked for it.
  std::pair<unsigned, unsigned> floorOf(unsigned R) const {
    unsigned Need = 0, Pipe = 0;
    for (auto [I, Held] : enumerate(Load[R]))
      if (Held > Need) {
        Need = Held;
        Pipe = I;
      }
    return {Need, Pipe};
  }

  /// The cheapest slot of region \p R for \p W, and what it costs.
  std::pair<unsigned, unsigned> bestIn(const Window &W, unsigned R) const {
    auto [Lo, Hi] = freeIn(W, R);
    int64_t Slot = Lo <= Hi ? Lo : (Lo > (int64_t)cap(R) - 1 ? cap(R) - 1 : 0);
    return {(unsigned)Slot, costAt(W, R, (unsigned)Slot)};
  }
};

/// How much smaller the ordering problem inside one region gets once the
/// instructions no schedule can tell apart stop being told apart, and once the
/// region is cut again by a path of its own.
struct Reduction {
  unsigned Insts = 0; ///< instructions in the region
  unsigned Same = 0;  ///< of them, ones that cost the same as an earlier one
  unsigned Twin = 0;  ///< of those, ones an earlier one can really stand for
  unsigned Cut = 0;   ///< pieces left once there is nothing more to cut by
  unsigned Worst = 0; ///< instructions in the biggest piece
  unsigned Hard = 0;  ///< of those, ones no twin covers
  double Bits = 0;    ///< bits to order it now
  double BySame = 0;  ///< bits once same-cost ones are not reordered
  double ByTwin = 0;  ///< bits once only the really interchangeable are
  double ByCut = 0;   ///< bits to order the pieces, twins folded
  double Pick = 0;    ///< bits to choose a piece, which cutting hides
};

/// Bits to put \p N things in an order: log2(N!).
static double bitsOf(unsigned N) {
  return std::lgamma(N + 1.0) / std::log(2.0);
}

/// What a step of a path is worth.
///
/// Cycles weighs an edge by its latency, so the longest path is the critical
/// path: the floor under any schedule. Steps weighs every edge one, so it is
/// the longest run of instructions that cannot be exchanged. Control weighs a
/// step onto a sequence point far above everything else, so the longest path
/// is the one crossing the most of them, ties going to the longer run -- the
/// path that cuts the function into the most pieces rather than the path that
/// is simply longest.
enum class Weigh { Cycles, Steps, Control };

/// What a cut is judged by: the ordering left over all of it, or the ordering
/// left in the one region that has the most.
enum class Spread { Total, Worst };

/// More than any path can be long, so a path with more control on it beats a
/// longer one whatever else is counted.
static constexpr unsigned ControlWeight = 1u << 20;

/// How large a function may be before who-comes-before-whom stops being worth
/// writing down in full. The table is one bit per pair, so this is 32MB.
static constexpr unsigned ClosureLimit = 16384;

/// How many memory instructions back to ask about aliasing before giving up
/// and ordering everything against one point. The comparison is pairwise, so
/// without a bound one block of loads and stores is quadratic.
static constexpr unsigned MemWindow = 512;

class AMDGPUGlobalSchedImpl {
public:
  AMDGPUGlobalSchedImpl(MachineDominatorTree &DT, MachineCycleInfo &CI)
      : DT(DT), CI(CI) {}

  bool run(MachineFunction &MF);

private:
  MachineDominatorTree &DT;
  MachineCycleInfo &CI;
  const SIInstrInfo *TII = nullptr;
  const SIRegisterInfo *TRI = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  TargetSchedModel SchedModel;

  /// Blocks holding nothing that writes EXEC or MODE. Crossing one of those
  /// changes what an instruction computes, or which lanes see it, without any
  /// dependence edge saying so.
  DenseSet<const MachineBasicBlock *> Clean;

  /// The mechanism pinning \p MI to its own block, or null if there is none.
  const char *classify(const MachineInstr &MI) const;

  /// The dominator-tree interval \p MI may be placed in. Only called when
  /// classify() returned null.
  Placement intervalOf(MachineInstr &MI) const;

  /// Which block of \p P to schedule into.
  MachineBasicBlock *chooseBlock(const Placement &P) const;

  /// Every instruction of \p MF in a topological order of its dependences.
  SmallVector<MachineInstr *, 0> orderOf(MachineFunction &MF) const;

  /// Add the cycles \p MI holds each pipe for to \p Load, which is indexed by
  /// the scheduling model's resource number.
  void pipesOf(const MachineInstr &MI, SmallVectorImpl<unsigned> &Load) const;

  /// Whether \p MI is a sequence point: nothing may be moved across it in
  /// either direction, whatever the values say.
  bool isFence(const MachineInstr &MI) const;

  /// Everything that orders one instruction before another, in one graph.
  void buildDeps(MachineFunction &MF, DepGraph &G) const;

  /// One longest path through \p G under \p How, and its weight in \p Weight.
  SmallVector<MachineInstr *, 0> chainPath(const DepGraph &G, Weigh How,
                                           unsigned &Weight) const;

  /// A path of N instructions cuts the function into N+1 regions: one before
  /// each step and one after the last. Fills \p R with how big each region
  /// is, in instructions and in cycles, and with one Window per instruction
  /// not on \p Path saying where that instruction may go.
  void regionsOf(const DepGraph &G, ArrayRef<MachineInstr *> Path,
                 const Timing &T, Regions &R) const;

  /// The chain that leaves the least ordering to do, and in \p Bits how much
  /// that is. \p How chooses whether to minimize the total ordering or the
  /// worst-case region.
  SmallVector<MachineInstr *, 0> spreadPath(const DepGraph &G, Spread How,
                                            double &Bits) const;

  /// Two ways one region's ordering problem is smaller than its size says.
  /// \p In is everything in that region, in topological order.
  void reduceOf(const DepGraph &G, ArrayRef<MachineInstr *> In, const Timing &T,
                Reduction &Out) const;

  /// The instructions of \p In, which must be in topological order, grouped so
  /// that nothing in one group is ordered against anything in another. Each
  /// group's size and its longest chain land in \p Out, biggest group first.
  void partsOf(const DepGraph &G, ArrayRef<MachineInstr *> In,
               SmallVectorImpl<std::pair<unsigned, unsigned>> &Out) const;

  /// Cut \p In by the chain that best spreads it, cut each region that chain
  /// makes the same way, and keep going until a region has no order inside it
  /// left to cut by. One Tier per depth lands in \p Out.
  /// Every region the recursion stops on has its size appended to \p Leaves:
  /// those are what a search would still have to order outright.
  /// Every instruction a cut at any depth put on a chain lands in \p Anchors.
  void anchorsOf(const DepGraph &G, ArrayRef<MachineInstr *> In,
                 SmallVectorImpl<Tier> &Out, SmallVectorImpl<unsigned> &Leaves,
                 SmallVectorImpl<MachineInstr *> &Anchors) const;

  /// Make \p Anchors finer without making it unsound. A wall an instruction
  /// is not ordered against can be dissolved and a chain of such instructions
  /// put in its place, and every surviving wall stays ordered against every
  /// new one. \p MaxSpan is the most walls dissolved at once. Returns the
  /// rounds that found something.
  unsigned refineOf(const DepGraph &G, const Timing &T, unsigned MaxSpan,
                    SmallVectorImpl<MachineInstr *> &Anchors) const;

  /// What two chains cost in place of one. \p Down and \p Across must each be
  /// ordered within themselves; nothing need order one against the other.
  /// Prints what the frame they make costs against what \p Down costs alone.
  void gridOf(const DepGraph &G, const Timing &T, ArrayRef<MachineInstr *> Down,
              ArrayRef<MachineInstr *> Across) const;

  /// What \p Walls cost as a frame when nothing promises its walls are
  /// ordered against each other. \p Walls must be in topological order.
  void meltOf(const DepGraph &G, const Timing &T,
              ArrayRef<MachineInstr *> Walls) const;

  /// Earliest and latest start time of each instruction, in cycles, as \p G
  /// alone allows. Fills \p Start and \p Slack and returns the length of the
  /// longest chain.
  void chainOf(const DepGraph &G, Timing &T) const;

  /// Put \p MI into \p To, in a position that reads nothing defined after it.
  void place(MachineInstr &MI, MachineBasicBlock &To) const;

  /// Cycles \p MI occupies. An instruction the scheduling model knows nothing
  /// about -- WAVE_BARRIER, SCHED_BARRIER, IMPLICIT_DEF -- is given the
  /// model's default rather than a real number, and the default is large
  /// enough to swamp every real chain, so it is worth nothing instead. None of
  /// them assemble to anything.
  unsigned latencyOf(const MachineInstr &MI) const {
    if (MI.isMetaInstruction())
      return 0;
    return SchedModel.computeInstrLatency(&MI);
  }

  /// Cycles between \p Def issuing and \p Use being able to read \p MO.
  unsigned latencyBetween(const MachineInstr &Def, const MachineInstr &Use,
                          const MachineOperand &MO) const {
    if (Def.isMetaInstruction() || Use.isMetaInstruction())
      return 0;
    unsigned DefIdx = 0, DefNo = 0;
    for (const MachineOperand &DefMO : Def.defs()) {
      if (DefMO.isReg() && DefMO.getReg() == MO.getReg()) {
        DefIdx = Def.getOperandNo(&DefMO);
        break;
      }
      ++DefNo;
    }

    // A pseudo defining more registers than the model has writes for --
    // S_UADDO_PSEUDO defines a value and a carry, the model names one -- makes
    // computeOperandLatency abort on an incomplete model. Its own latency is
    // the answer in that case.
    if (DefNo >= SchedModel.resolveSchedClass(&Def)->NumWriteLatencyEntries)
      return latencyOf(Def);

    return SchedModel.computeOperandLatency(&Def, DefIdx, &Use,
                                            Use.getOperandNo(&MO));
  }

  /// Cycles the chain keeps running after \p MI issues. Only a value anyone
  /// can wait for is worth anything here: S_BARRIER is modelled at 500 cycles
  /// (SISchedule.td, "XXX: Guessed ???"), but it defines nothing, so no edge
  /// leaves it and no instruction is later because of it. Counting it would
  /// make every chain 500 long and say nothing about the code.
  unsigned tailOf(const MachineInstr &MI) const {
    for (const MachineOperand &MO : MI.defs())
      if (MO.isReg() && MO.getReg().isVirtual())
        return latencyOf(MI);
    return 0;
  }

  static unsigned levelOf(const MachineDomTreeNode *N) { return N->getLevel(); }

  unsigned depthOf(const MachineBasicBlock *MBB) const {
    return CI.getCycleDepth(MBB);
  }
};

/// The block a use of a virtual register has to be dominated by. For a PHI
/// the value is read on the incoming edge, not in the PHI's own block, so the
/// predecessor named next to the operand is what counts.
static MachineBasicBlock *useBlockOf(const MachineOperand &MO) {
  const MachineInstr *MI = MO.getParent();
  if (!MI->isPHI())
    return const_cast<MachineBasicBlock *>(MI->getParent());
  unsigned Idx = MI->getOperandNo(&MO);
  return MI->getOperand(Idx + 1).getMBB();
}

} // end anonymous namespace

const char *AMDGPUGlobalSchedImpl::classify(const MachineInstr &MI) const {
  // Stores, calls, PHIs, terminators, ordered memory references, inline asm,
  // anything with effects the model does not carry, and any load that might
  // fault. Equivalence for these is not a question about dominance, so they
  // are not asked it.
  bool SawStore = false;
  if (!MI.isSafeToMove(SawStore))
    return "unsafe";

  // A convergent operation must not be made control-dependent on anything it
  // was not already control-dependent on.
  if (MI.isConvergent())
    return "convergent";

  // The scheduling-boundary set: writing EXEC, the mode register, the
  // priority, or the VGPR indexing mode changes the meaning of instructions
  // around it rather than producing a value.
  if (TII->isSchedulingBoundary(MI, MI.getParent(), *MI.getMF()))
    return "exec";

  // Running these with no lane active is not merely useless, it hangs.
  if (TII->hasUnwantedEffectsWhenEXECEmpty(MI))
    return "exec-empty";

  // Physical registers. A use has no definition for a block to be dominated
  // by, and a definition has no live range anyone is tracking, so neither can
  // be placed by the dominance argument. EXEC and MODE are excused where the
  // target says the use is not a real read -- a per-lane VALU writing a VGPR
  // computes the same thing whatever EXEC holds -- and are handled instead by
  // the Clean filter in intervalOf().
  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isReg() || !MO.getReg().isPhysical())
      continue;
    Register Reg = MO.getReg();
    if (Reg == AMDGPU::SCC)
      return "scc";
    if (Reg == AMDGPU::M0)
      return "m0";
    if (Reg == AMDGPU::VCC || Reg == AMDGPU::VCC_LO || Reg == AMDGPU::VCC_HI)
      return "vcc";
    if (MO.isDef())
      return "physreg";
    if (TRI->isSubRegisterEq(AMDGPU::EXEC, Reg)) {
      if (TII->isIgnorableUse(MI, MI.getOperandNo(&MO)))
        continue;
      return "exec-read";
    }
    if (Reg == AMDGPU::MODE || MRI->isConstantPhysReg(Reg))
      continue;
    return "physreg";
  }

  // One virtual definition, because the interval is defined by that value's
  // uses. More than one and the instruction is describing something the
  // interval does not capture.
  const MachineOperand *Def = nullptr;
  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isReg() || !MO.isDef())
      continue;
    if (Def)
      return "multidef";
    Def = &MO;
  }
  if (!Def)
    return "nodef";
  if (MRI->use_nodbg_empty(Def->getReg()))
    return "dead";

  return nullptr;
}

Placement AMDGPUGlobalSchedImpl::intervalOf(MachineInstr &MI) const {
  Placement P;
  P.Home = MI.getParent();

  // The earliest block is the deepest operand definition: the placement has to
  // be dominated by all of them, and they all dominate Home already, so they
  // lie on one chain and the deepest is the binding one.
  MachineBasicBlock *Lo = &MI.getMF()->front();
  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isReg() || !MO.isUse() || !MO.getReg().isVirtual())
      continue;
    MachineInstr *DefMI = MRI->getVRegDef(MO.getReg());
    assert(DefMI && "virtual register read with no definition");
    MachineBasicBlock *DefMBB = DefMI->getParent();
    if (levelOf(DT.getNode(DefMBB)) > levelOf(DT.getNode(Lo)))
      Lo = DefMBB;
  }

  // The latest block is the nearest common dominator of the uses, since the
  // definition has to dominate every one of them.
  MachineBasicBlock *Hi = nullptr;
  Register Reg;
  for (const MachineOperand &MO : MI.operands())
    if (MO.isReg() && MO.isDef())
      Reg = MO.getReg();
  for (const MachineOperand &MO : MRI->use_nodbg_operands(Reg)) {
    MachineBasicBlock *U = useBlockOf(MO);
    Hi = Hi ? DT.findNearestCommonDominator(Hi, U) : U;
  }
  assert(Hi && "free instruction with no users");

  assert(DT.dominates(Lo, P.Home) && DT.dominates(P.Home, Hi) &&
         "lowering placed this outside its own legal interval");

  // Now cut the interval down to where EXEC and MODE are what they are at
  // Home. A COPY or a REG_SEQUENCE carries no implicit $exec, so nothing in
  // the dependence graph stops it floating past a write of EXEC; inside a
  // region that never mattered because a region never spans one. Taking the
  // whole function means saying it here instead.
  if (!Clean.count(P.Home)) {
    P.Lo = P.Hi = P.Home;
    return P;
  }

  P.Lo = P.Home;
  for (MachineDomTreeNode *N = DT.getNode(P.Home)->getIDom();
       N && levelOf(N) >= levelOf(DT.getNode(Lo)); N = N->getIDom()) {
    if (!Clean.count(N->getBlock()))
      break;
    P.Lo = N->getBlock();
  }

  // Downwards the path has to be read off backwards, from Hi up to Home.
  SmallVector<MachineBasicBlock *, 8> Down;
  for (MachineDomTreeNode *N = DT.getNode(Hi); N && N->getBlock() != P.Home;
       N = N->getIDom())
    Down.push_back(N->getBlock());
  P.Hi = P.Home;
  for (MachineBasicBlock *B : reverse(Down)) {
    // Leaving a cycle that exits divergently makes a uniform value read after
    // the loop differ from the one the lane computed in it.
    if (!Clean.count(B) || !TII->isSafeToSink(MI, B, &CI))
      break;
    P.Hi = B;
  }

  return P;
}

SmallVector<MachineInstr *, 0>
AMDGPUGlobalSchedImpl::orderOf(MachineFunction &MF) const {
  // Reverse post-order over blocks, program order inside one, is a topological
  // order of the SSA dependence graph: a definition's block dominates every
  // use's, and dominance implies precedence in reverse post-order. The one
  // exception is a PHI, whose operands may be defined anywhere -- but every
  // cycle in SSA runs through a PHI, so dropping the edges into PHIs is both
  // what makes the graph acyclic and what makes this order work. What is left
  // is the chain through a single iteration, which is the one paid per trip.
  SmallVector<MachineInstr *, 0> Order;
  for (MachineBasicBlock *MBB :
       ReversePostOrderTraversal<MachineFunction *>(&MF))
    for (MachineInstr &MI : *MBB)
      if (!MI.isDebugInstr())
        Order.push_back(&MI);
  return Order;
}

bool AMDGPUGlobalSchedImpl::isFence(const MachineInstr &MI) const {
  // S_BARRIER and friends arrive here as "side effects the model does not
  // carry", which is exactly what a sequence point is.
  return MI.isTerminator() || MI.isCall() || MI.hasUnmodeledSideEffects() ||
         TII->isSchedulingBoundary(MI, MI.getParent(), *MI.getMF());
}

void AMDGPUGlobalSchedImpl::buildDeps(MachineFunction &MF, DepGraph &G) const {
  G.Order = orderOf(MF);

  // Values. Every read of a virtual register waits for the one definition of
  // it. Edges into a PHI are dropped -- a PHI reads on its incoming edge, not
  // in its own block, and every cycle in SSA runs through one, so this is both
  // what makes the graph acyclic and what makes G.Order topological.
  for (MachineInstr *MI : G.Order) {
    if (MI->isPHI())
      continue;
    for (const MachineOperand &MO : MI->uses()) {
      if (!MO.isReg() || !MO.getReg().isVirtual())
        continue;
      MachineInstr *Def = MRI->getVRegDef(MO.getReg());
      G.add(Def, MI, latencyBetween(*Def, *MI, MO));
    }
  }

  // Everything else is an order between two instructions of one block, and
  // always points forwards in program order, so G.Order stays topological.
  for (MachineBasicBlock &MBB : MF) {
    DenseMap<MCRegUnit, MachineInstr *> LastDef;
    DenseMap<MCRegUnit, SmallVector<MachineInstr *, 4>> ReadsSince;
    SmallVector<MachineInstr *, 0> MemOps;
    MachineInstr *Fence = nullptr;
    SmallVector<MachineInstr *, 0> Pending;

    for (MachineInstr &MI : MBB) {
      if (MI.isDebugInstr())
        continue;

      // A sequence point. Nothing crosses it either way, so everything since
      // the last one runs into it and everything after it runs out of it, and
      // the state of registers and memory on the far side is unknown. The
      // weight is nought: a barrier orders the code without producing a value
      // anybody waits for, and its modelled 500 cycles would otherwise swamp
      // every real chain in the function.
      if (isFence(MI)) {
        for (MachineInstr *P : Pending)
          G.add(P, &MI, 0);
        if (Fence)
          G.add(Fence, &MI, 0);
        Fence = &MI;
        Pending.clear();
        LastDef.clear();
        ReadsSince.clear();
        MemOps.clear();
        continue;
      }
      if (Fence)
        G.add(Fence, &MI, 0);

      // Physical registers. Nothing tracks a live range for these across a
      // block, so the order they are written and read in is the whole of what
      // keeps them meaning anything.
      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isReg() || !MO.getReg().isPhysical())
          continue;
        for (MCRegUnit Unit : TRI->regunits(MO.getReg())) {
          MachineInstr *Def = LastDef.lookup(Unit);
          if (MO.isUse()) {
            if (Def)
              G.add(Def, &MI, latencyOf(*Def));
            ReadsSince[Unit].push_back(&MI);
            continue;
          }
          // A write nobody reads still clobbers, so whatever came before it
          // keeps its edge to it -- but nothing has to come after it, so it
          // need not wait for the write before. Without this every SALU
          // instruction of a block is on one chain through the SCC none of
          // them meant to write.
          if (Def && !MO.isDead())
            G.add(Def, &MI, 1);
          for (MachineInstr *R : ReadsSince[Unit])
            G.add(R, &MI, 1);
          LastDef[Unit] = &MI;
          ReadsSince.erase(Unit);
        }
      }

      // Memory. Two accesses keep their order only if one writes bytes the
      // other can see, which mayAlias answers -- separating the LDS traffic
      // from everything else, and separating two accesses of the same base
      // whose constant offsets cannot overlap. Without the second of those
      // the chain runs through all sixteen writes of an LDS tile, an order
      // the hardware never has to honour.
      if (MI.mayLoadOrStore()) {
        // At the cap, order this against the whole window whether it aliases
        // or not. It then stands in for all of it -- anything later ordered
        // after this one is ordered after them too -- so the window can
        // start again from here without losing a constraint.
        bool Full = MemOps.size() >= MemWindow;
        for (MachineInstr *M : MemOps) {
          if (!Full && !MI.mayAlias(static_cast<AAResults *>(nullptr), *M,
                                    /*UseTBAA=*/true))
            continue;
          G.add(M, &MI, M->mayStore() && MI.mayLoad() ? latencyOf(*M) : 1);
        }
        if (Full)
          MemOps.clear();
        MemOps.push_back(&MI);
      }

      Pending.push_back(&MI);
    }
  }
}

SmallVector<MachineInstr *, 0>
AMDGPUGlobalSchedImpl::chainPath(const DepGraph &G, Weigh How,
                                 unsigned &Weight) const {
  // A longest path, recovered by remembering which predecessor was the one
  // that made each instruction wait. Every step of the result is ordered
  // against the step before it, so no two instructions on it may be exchanged
  // -- which is what distinguishes a path from the set of instructions with
  // no slack. That set is the union of all the longest paths, and is
  // routinely larger than any one of them.
  DenseMap<const MachineInstr *, unsigned> Depth;
  DenseMap<const MachineInstr *, MachineInstr *> Prev;

  MachineInstr *End = nullptr;
  Weight = 0;
  for (MachineInstr *MI : G.Order) {
    unsigned Step = 1;
    if (How == Weigh::Control && isFence(*MI))
      Step = ControlWeight + 1;

    unsigned D = 0;
    MachineInstr *From = nullptr;
    for (const Dep &P : G.preds(MI)) {
      unsigned E =
          Depth.lookup(P.Other) + (How == Weigh::Cycles ? P.Latency : Step);
      if (E >= D) {
        D = E;
        From = P.Other;
      }
    }
    Depth[MI] = D;
    Prev[MI] = From;
    // An instruction nobody waits for still ends a path, but only its own
    // tail counts past the last edge -- see tailOf.
    unsigned Full = D + (How == Weigh::Cycles ? tailOf(*MI) : 0);
    if (Full >= Weight) {
      Weight = Full;
      End = MI;
    }
  }

  SmallVector<MachineInstr *, 0> Path;
  for (MachineInstr *MI = End; MI; MI = Prev.lookup(MI))
    Path.push_back(MI);
  std::reverse(Path.begin(), Path.end());
  return Path;
}

void AMDGPUGlobalSchedImpl::pipesOf(const MachineInstr &MI,
                                    SmallVectorImpl<unsigned> &Load) const {
  // One instruction holds a pipe for as long as the model says, and nothing
  // else may use that pipe meanwhile. There is no ordering in this -- only a
  // total, which is why it bounds a region however the region is ordered.
  const MCSchedClassDesc *SC = SchedModel.resolveSchedClass(&MI);
  if (!SC || !SC->isValid())
    return;
  for (const MCWriteProcResEntry &P :
       make_range(SchedModel.getWriteProcResBegin(SC),
                  SchedModel.getWriteProcResEnd(SC))) {
    unsigned Units = SchedModel.getProcResource(P.ProcResourceIdx)->NumUnits;
    if (!Units)
      continue;
    Load[P.ProcResourceIdx] += (P.ReleaseAtCycle - P.AcquireAtCycle) / Units;
  }
}

void AMDGPUGlobalSchedImpl::regionsOf(const DepGraph &G,
                                      ArrayRef<MachineInstr *> Path,
                                      const Timing &T, Regions &R) const {
  // Number the steps of the path, and the N+1 regions between them, so that
  // region I is the gap immediately before step I and region N is what comes
  // after the last step.
  DenseMap<const MachineInstr *, unsigned> Step;
  for (auto [I, MI] : enumerate(Path))
    Step[MI] = I;
  const int N = Path.size();

  R.Count.assign(N + 1, 0);
  R.Wins.clear();
  R.Load.assign(
      N + 1, SmallVector<unsigned, 0>(SchedModel.getNumProcResourceKinds(), 0));

  // Where the path is in time. Region I is the stretch between the issue of
  // the step before it and the issue of the step after it; the last region
  // runs to the end of the chain.
  R.Begin.assign(N + 1, 0);
  R.End.assign(N + 1, 0);
  for (int I = 0; I <= N; ++I) {
    R.Begin[I] = I == 0 ? 0 : T.Start.lookup(Path[I - 1]);
    R.End[I] = I == N ? T.Length : T.Start.lookup(Path[I]);
  }

  // The last step that has to come before an instruction. If a predecessor is
  // a step, that step does; otherwise whatever had to precede it does.
  DenseMap<const MachineInstr *, int> Lo;
  for (MachineInstr *MI : G.Order) {
    int L = -1;
    for (const Dep &P : G.preds(MI)) {
      auto It = Step.find(P.Other);
      L = std::max(L, It != Step.end() ? (int)It->second : Lo.lookup(P.Other));
    }
    Lo[MI] = L;
  }

  // And the first step that has to come after it. N means none does.
  DenseMap<const MachineInstr *, int> Hi;
  for (MachineInstr *MI : reverse(G.Order)) {
    int H = N;
    for (const Dep &S : G.succs(MI)) {
      auto It = Step.find(S.Other);
      H = std::min(H, It != Step.end() ? (int)It->second : Hi.lookup(S.Other));
    }
    Hi[MI] = H;
  }

  // An instruction may sit anywhere after the last step it needs and before
  // the first step that needs it. That window is a run of whole regions, and
  // it is never empty, since the path is a path. G.Order is the schedule the
  // function has now, so counting the steps already gone past says which
  // region an instruction is in today.
  unsigned Past = 0;
  for (MachineInstr *MI : G.Order) {
    if (Step.count(MI)) {
      ++Past;
      continue;
    }
    unsigned First = Lo.lookup(MI) + 1, Last = Hi.lookup(MI);
    unsigned Asap = T.Start.lookup(MI);
    R.Wins.push_back({MI, First, Last, Past, Asap, Asap + T.Slack.lookup(MI)});
    pipesOf(*MI, R.Load[Past]);
    for (unsigned I = First; I <= Last; ++I)
      ++R.Count[I];
  }
}

/// The chain through a set of instructions that leaves the least ordering to
/// do. \p Up gives each instruction's predecessors as positions in a
/// topological order of the set, \p How chooses what is minimized, and \p Bits
/// comes back with what is left. The answer is positions into that same order.
static SmallVector<unsigned, 0> spreadIn(ArrayRef<SmallVector<unsigned, 0>> Up,
                                         Spread How, double &Bits) {
  const unsigned N = Up.size();

  // Who must come before each instruction, including at a remove. This is a
  // square of bits, so it is only built while that is cheap; past that the
  // edges alone are used and the answer is merely a chain that exists, not
  // the best one.
  const bool Closed = N <= ClosureLimit;
  SmallVector<BitVector, 0> Before;
  if (Closed) {
    Before.assign(N, BitVector(N));
    for (unsigned I = 0; I != N; ++I)
      for (unsigned J : Up[I]) {
        Before[I] |= Before[J];
        Before[I].set(J);
      }
  }

  SmallVector<double, 0> Cost(N, 0);
  SmallVector<int, 0> Prev(N, -1);

  int End = -1;
  Bits = std::numeric_limits<double>::max();
  for (unsigned I = 0; I != N; ++I) {
    // Starting the chain here leaves everything before it in one region.
    double C = bitsOf(I);
    int From = -1;

    auto Hop = [&](unsigned J) {
      double E = How == Spread::Total ? Cost[J] + bitsOf(I - J - 1)
                                      : std::max(Cost[J], bitsOf(I - J - 1));
      if (E < C) {
        C = E;
        From = J;
      }
    };
    if (Closed) {
      for (unsigned J : Before[I].set_bits())
        Hop(J);
    } else {
      for (unsigned J : Up[I])
        Hop(J);
    }

    Cost[I] = C;
    Prev[I] = From;

    // And ending it here leaves everything after it in one region.
    double Full = How == Spread::Total ? C + bitsOf(N - 1 - I)
                                       : std::max(C, bitsOf(N - 1 - I));
    if (Full < Bits) {
      Bits = Full;
      End = I;
    }
  }

  SmallVector<unsigned, 0> Path;
  for (int I = End; I >= 0; I = Prev[I])
    Path.push_back(I);
  std::reverse(Path.begin(), Path.end());
  return Path;
}

SmallVector<MachineInstr *, 0>
AMDGPUGlobalSchedImpl::spreadPath(const DepGraph &G, Spread How,
                                  double &Bits) const {
  // Regions are a coordinate system, not a commitment. Any chain cuts the
  // function into regions, every schedule is still describable in the cut, and
  // so the chain is free to choose. What it is worth choosing for is the
  // ordering that is left: a region holding k instructions costs log2(k!), and
  // that is smallest when the steps are spread evenly, not when there are many
  // of them.
  //
  // Which region an instruction is in today is read off how many steps the
  // order has passed, so a region is exactly a gap between two consecutive
  // steps in that order, and the cost of a chain is a sum over its edges.
  // Finding the best one is then a shortest path.
  //
  // What does not work is growing a chain by the instructions dependence pins
  // to a single region. There are none, and there can be none: such an
  // instruction comes after one step and before the next, which are joined by
  // an edge, so routing the path through it would make the path longer than
  // the longest path.
  //
  // What a step needs is to be ordered against the step before it, and an
  // edge is not the only way to be ordered: a dependence path will do. So a
  // hop may skip, and the chains worth considering are all the chains of the
  // reachability order, not just the ones the edges spell out. That is what
  // lets a cut be made finer where it is coarse -- a step whose two sides are
  // both crowded can be dropped for a longer run of instructions spanning the
  // same stretch, even though no edge joins that run to what bounds it.
  DenseMap<const MachineInstr *, unsigned> Pos;
  for (auto [I, MI] : enumerate(G.Order))
    Pos[MI] = I;

  SmallVector<SmallVector<unsigned, 0>, 0> Up(G.Order.size());
  for (auto [I, MI] : enumerate(G.Order))
    for (const Dep &P : G.preds(MI))
      Up[I].push_back(Pos.lookup(P.Other));

  SmallVector<MachineInstr *, 0> Path;
  for (unsigned I : spreadIn(Up, How, Bits))
    Path.push_back(G.Order[I]);
  return Path;
}

void AMDGPUGlobalSchedImpl::partsOf(
    const DepGraph &G, ArrayRef<MachineInstr *> In,
    SmallVectorImpl<std::pair<unsigned, unsigned>> &Out) const {
  // Peeling the longest chain off a region, then the longest chain of what is
  // left, says nothing about whether the second is ordered against the first,
  // so it cannot answer what the region is made of. The order itself can. Two
  // instructions belong together when one has to come before the other,
  // however far apart; the components of that relation are what a schedule
  // may interleave freely, and no one chain can anchor two of them at once.
  Out.clear();
  const unsigned N = In.size();
  if (!N)
    return;

  DenseMap<const MachineInstr *, unsigned> Idx;
  for (auto [I, MI] : enumerate(In))
    Idx[MI] = I;

  SmallVector<SmallVector<unsigned, 4>, 0> Up(N);
  for (auto [I, MI] : enumerate(In))
    for (const Dep &P : G.preds(MI))
      if (auto It = Idx.find(P.Other); It != Idx.end())
        Up[I].push_back(It->second);

  // Who must come before whom at any remove. \p In is topological, so one
  // forward sweep settles it.
  SmallVector<BitVector, 0> Before(N, BitVector(N));
  for (unsigned I = 0; I != N; ++I)
    for (unsigned J : Up[I]) {
      Before[I] |= Before[J];
      Before[I].set(J);
    }

  // Joined by that relation, not by the edges: two instructions sharing a
  // predecessor are still free to go in either order, so they are not in the
  // same group merely for sharing it.
  SmallVector<unsigned, 0> Root(N);
  for (unsigned I = 0; I != N; ++I)
    Root[I] = I;
  auto Find = [&](unsigned I) {
    while (Root[I] != I)
      I = Root[I] = Root[Root[I]];
    return I;
  };
  for (unsigned I = 0; I != N; ++I)
    for (unsigned J : Before[I].set_bits())
      Root[Find(I)] = Find(J);

  // How big each group is and the longest run inside it.
  SmallVector<unsigned, 0> Deep(N, 1);
  DenseMap<unsigned, std::pair<unsigned, unsigned>> Group;
  for (unsigned I = 0; I != N; ++I) {
    for (unsigned J : Up[I])
      Deep[I] = std::max(Deep[I], Deep[J] + 1);
    std::pair<unsigned, unsigned> &G2 = Group[Find(I)];
    ++G2.first;
    G2.second = std::max(G2.second, Deep[I]);
  }

  for (const auto &KV : Group)
    Out.push_back(KV.second);
  llvm::sort(Out, std::greater<std::pair<unsigned, unsigned>>());
}

void AMDGPUGlobalSchedImpl::anchorsOf(
    const DepGraph &G, ArrayRef<MachineInstr *> In, SmallVectorImpl<Tier> &Out,
    SmallVectorImpl<unsigned> &Leaves,
    SmallVectorImpl<MachineInstr *> &Anchors) const {
  // A cut is one chain, and a chain follows one thread of the order at a
  // time. Whatever it had to leave behind lands in a region, and that region
  // is a smaller instance of the same problem, so it is cut by the chain that
  // best spreads it in turn. Each cut is chosen over the reachability order
  // rather than the edges, so a step may skip: wherever a wall of the level
  // above fell in the wrong place, the level below is free to span it.
  //
  // This is a picture of the function, not a finer cut of it. Nothing says a
  // sub-cut's steps are ordered against the walls around them, and which
  // piece an instruction falls into is read off where it sits today, which is
  // a choice rather than a fact.
  Out.clear();
  Leaves.clear();
  Anchors.clear();
  SmallVector<SmallVector<MachineInstr *, 0>, 0> Work;
  Work.emplace_back(In.begin(), In.end());

  while (!Work.empty()) {
    SmallVector<SmallVector<MachineInstr *, 0>, 0> Next;
    Tier L;
    for (const SmallVector<MachineInstr *, 0> &Set : Work) {
      const unsigned M = Set.size();
      ++L.Regions;
      L.Insts += M;
      L.Worst = std::max(L.Worst, M);
      L.Bits += bitsOf(M);

      DenseMap<const MachineInstr *, unsigned> Idx;
      for (auto [I, MI] : enumerate(Set))
        Idx[MI] = I;

      SmallVector<SmallVector<unsigned, 0>, 0> Up(M);
      for (auto [I, MI] : enumerate(Set))
        for (const Dep &P : G.preds(MI))
          if (auto It = Idx.find(P.Other); It != Idx.end())
            Up[I].push_back(It->second);

      // No order in here at all, so there is nothing to cut by. Cutting by
      // position anyway would only be choosing an interleaving, and the
      // region would never stop being cut.
      if (none_of(Up, [](ArrayRef<unsigned> P) { return !P.empty(); })) {
        ++L.Stuck;
        Leaves.push_back(M);
        continue;
      }

      double Bits = 0;
      SmallVector<unsigned, 0> Steps = spreadIn(Up, Spread::Total, Bits);

      // A single step is not a chain: nothing ordered it against anything, so
      // where it falls is position again rather than dependence. Two is the
      // least that anchors.
      if (Steps.size() < 2) {
        ++L.Stuck;
        Leaves.push_back(M);
        continue;
      }
      L.Steps += Steps.size();
      for (unsigned S : Steps)
        Anchors.push_back(Set[S]);

      BitVector OnPath(M);
      for (unsigned S : Steps)
        OnPath.set(S);

      SmallVector<SmallVector<MachineInstr *, 0>, 0> Pieces(Steps.size() + 1);
      unsigned Past = 0;
      for (unsigned I = 0; I != M; ++I) {
        if (OnPath.test(I)) {
          ++Past;
          continue;
        }
        Pieces[Past].push_back(Set[I]);
      }
      for (SmallVector<MachineInstr *, 0> &P : Pieces)
        if (!P.empty())
          Next.push_back(std::move(P));
    }
    Out.push_back(L);
    Work = std::move(Next);
  }
}

unsigned AMDGPUGlobalSchedImpl::refineOf(
    const DepGraph &G, const Timing &T, unsigned MaxSpan,
    SmallVectorImpl<MachineInstr *> &Anchors) const {
  // A frame is only worth having while every two of its walls are ordered
  // against each other: that is what makes the regions between them a
  // coordinate system, and what makes every schedule still describable once
  // the frame is imposed. Cutting a region by a chain found inside it throws
  // that away, because nothing said the chain was ordered against the walls
  // around it.
  //
  // There is a way to get finer without losing it. An instruction's window
  // names the walls it is *not* ordered against -- it depends on the wall
  // before the window and the wall after it depends on the instruction, and
  // only the walls strictly inside are free. So if every wall inside the
  // window is dissolved, that instruction is ordered against every wall that
  // remains, and may become one itself. Take a chain of such instructions,
  // put it where the dissolved walls were, and the frame is finer and still
  // sound. It pays whenever the chain is longer than the run it replaces.
  const unsigned N = G.Order.size();
  if (N > ClosureLimit)
    return 0;

  DenseMap<const MachineInstr *, unsigned> Seat;
  for (auto [I, MI] : enumerate(G.Order))
    Seat[MI] = I;

  // Who must come before whom. The graph never changes, so this is built once
  // rather than in every round.
  SmallVector<BitVector, 0> Before(N, BitVector(N));
  for (unsigned I = 0; I != N; ++I)
    for (const Dep &P : G.preds(G.Order[I])) {
      unsigned J = Seat.lookup(P.Other);
      Before[I] |= Before[J];
      Before[I].set(J);
    }

  // What makes a frame good is not how many walls it has. Adding a wall only
  // takes something away if the instructions around it are ordered against
  // it, and makes every other window one region wider if they are not. So a
  // step is taken when it lowers the bits it takes to name a schedule in the
  // frame, and not merely when it leaves more walls standing than it
  // dissolved. One step at a time, best first: with a single span in flight
  // there is no way for one swap to dissolve the wall another one leaned on.
  auto BitsOf = [&](ArrayRef<MachineInstr *> Frame, unsigned &Worst) {
    Regions R;
    regionsOf(G, Frame, T, R);
    SmallVector<unsigned, 0> Busy(R.size(), 0);
    double Bits = 0;
    for (const Window &W : R.Wins) {
      ++Busy[W.Here];
      Bits += std::log2(W.Last - W.First + 1);
    }
    Worst = 0;
    for (unsigned B : Busy) {
      Bits += bitsOf(B);
      Worst = std::max(Worst, B);
    }
    return Bits;
  };

  unsigned Rounds = 0;
  for (unsigned Pass = 0; Pass != 64; ++Pass) {
    Regions R;
    regionsOf(G, Anchors, T, R);
    const unsigned Walls = Anchors.size();

    unsigned Worst = 0;
    const double Have = BitsOf(Anchors, Worst);

    double Best = Have;
    unsigned BestWorst = Worst, BestAt = 0, BestSpan = 0, BestLen = 0;
    SmallVector<MachineInstr *, 0> Take;

    for (unsigned Span = 1; Span <= RefineSpan && Span <= Walls; ++Span) {
      for (unsigned A = 0; A + Span <= Walls; ++A) {
        // Everything whose unordered walls all lie inside this span. Wins is
        // built by a sweep over a topological order, so this stays in one.
        SmallVector<MachineInstr *, 0> Cand;
        for (const Window &W : R.Wins)
          if (W.First >= A && W.Last <= A + Span)
            Cand.push_back(W.MI);
        if (Cand.empty())
          continue;

        // The longest run of them that is ordered. Being ordered at a remove
        // is enough: a wall only has to come after the one before it, not
        // immediately after.
        const unsigned M = Cand.size();
        SmallVector<unsigned, 0> Deep(M, 1);
        SmallVector<int, 0> Back(M, -1);
        unsigned Long = 0;
        int End = -1;
        for (unsigned I = 0; I != M; ++I) {
          unsigned Pi = Seat.lookup(Cand[I]);
          for (unsigned J = 0; J != I; ++J)
            if (Before[Pi].test(Seat.lookup(Cand[J])) &&
                Deep[J] + 1 > Deep[I]) {
              Deep[I] = Deep[J] + 1;
              Back[I] = J;
            }
          if (Deep[I] > Long) {
            Long = Deep[I];
            End = I;
          }
        }
        if (End < 0)
          continue;

        SmallVector<MachineInstr *, 0> Try;
        for (unsigned I = 0; I != Walls; ++I)
          if (I < A || I >= A + Span)
            Try.push_back(Anchors[I]);
        for (int I = End; I >= 0; I = Back[I])
          Try.push_back(Cand[I]);

        // A new wall depends on the wall before the span it replaced and is
        // depended on by the wall after it, so it already sits between them
        // in any topological order. Sorting by that order is enough to put
        // the frame back in sequence.
        llvm::sort(Try, [&](const MachineInstr *X, const MachineInstr *Y) {
          return Seat.lookup(X) < Seat.lookup(Y);
        });

        unsigned Bad = 0;
        double Bits = BitsOf(Try, Bad);
        if (Bits >= Best)
          continue;
        Best = Bits;
        BestWorst = Bad;
        BestAt = A;
        BestSpan = Span;
        BestLen = Long;
        Take = std::move(Try);
      }
    }

    if (Take.empty())
      break;

    Anchors.assign(Take.begin(), Take.end());
    ++Rounds;

    LLVM_DEBUG(dbgs() << "amdgpu-global-sched: refine: round " << Rounds
                      << ", wall " << BestAt << " span " << BestSpan
                      << " -> chain " << BestLen << ", " << Walls << " -> "
                      << Anchors.size() << " walls, "
                      << (uint64_t)std::llround(Have) << " -> "
                      << (uint64_t)std::llround(Best)
                      << " bits, largest region " << BestWorst << "\n");
  }

  // The whole point was to stay sound, so say whether it did. Every wall must
  // come before every later wall; a pair that does not is a frame that has
  // stopped being a coordinate system.
  LLVM_DEBUG({
    unsigned Bad = 0;
    for (unsigned I = 0; I != Anchors.size(); ++I)
      for (unsigned J = I + 1; J != Anchors.size(); ++J)
        if (!Before[Seat.lookup(Anchors[J])].test(Seat.lookup(Anchors[I])))
          ++Bad;
    dbgs() << "amdgpu-global-sched: refine: done, " << Rounds << " rounds, "
           << Anchors.size() << " walls, " << Bad << " unordered pairs\n";
  });

  return Rounds;
}

void AMDGPUGlobalSchedImpl::gridOf(const DepGraph &G, const Timing &T,
                                   ArrayRef<MachineInstr *> Down,
                                   ArrayRef<MachineInstr *> Across) const {
  // A single chain can only be cut by instructions ordered against all of it,
  // and a region it cannot cut is exactly one whose contents are ordered
  // against each other and against nothing on the chain. Two chains do not
  // have that problem, because they are never laid in one line: each
  // instruction gets a coordinate against each chain, and the pair names a
  // cell of a grid.
  //
  // That still names a schedule. Walking a schedule from start to end, the
  // count of passed steps of either chain only ever goes up, so the cells it
  // visits climb the grid in one direction and form a chain of their own. What
  // has to be paid for on top is which way they climb -- the interleaving of
  // the two chains, one choice of p+q steps of which p are Down's.
  Regions RD, RA;
  regionsOf(G, Down, T, RD);
  regionsOf(G, Across, T, RA);

  DenseMap<const MachineInstr *, const Window *> Side;
  for (const Window &W : RA.Wins)
    Side[W.MI] = &W;

  // What the one chain costs, over everything it does not itself contain.
  double Line = 0;
  SmallVector<unsigned, 0> Busy(RD.size(), 0);
  for (const Window &W : RD.Wins) {
    ++Busy[W.Here];
    Line += std::log2(W.Last - W.First + 1);
  }
  unsigned LineWorst = 0;
  for (unsigned B : Busy) {
    Line += bitsOf(B);
    LineWorst = std::max(LineWorst, B);
  }

  // And what the grid costs over the same schedules. An instruction on either
  // chain is a wall and is not placed: the chains' own steps are fixed against
  // each other by the interleaving, so they are paid for there and nowhere
  // else.
  double Place = 0;
  DenseMap<uint64_t, unsigned> Cell;
  unsigned Free = 0;
  for (const Window &W : RD.Wins) {
    auto It = Side.find(W.MI);
    if (It == Side.end())
      continue;
    const Window &V = *It->second;
    ++Free;
    Place += std::log2(W.Last - W.First + 1);
    Place += std::log2(V.Last - V.First + 1);
    ++Cell[(uint64_t)W.Here * RA.size() + V.Here];
  }

  double Order = 0;
  unsigned GridWorst = 0;
  for (const auto &C : Cell) {
    Order += bitsOf(C.second);
    GridWorst = std::max(GridWorst, C.second);
  }

  const double P = Down.size(), Q = Across.size();
  const double Mix =
      (std::lgamma(P + Q + 1.0) - std::lgamma(P + 1.0) - std::lgamma(Q + 1.0)) /
      std::log(2.0);

  LLVM_DEBUG(dbgs() << "amdgpu-global-sched: grid: " << Down.size() << " down "
                    << Across.size() << " across, " << Cell.size()
                    << " cells hold " << Free << ", line "
                    << (uint64_t)std::llround(Line) << " bits largest "
                    << LineWorst << ", grid " << (uint64_t)std::llround(Mix)
                    << " to interleave + " << (uint64_t)std::llround(Place)
                    << " to place + " << (uint64_t)std::llround(Order)
                    << " to order = "
                    << (uint64_t)std::llround(Mix + Place + Order)
                    << " bits largest " << GridWorst << "\n");

  // A grid charges every instruction for both coordinates, and the second one
  // is nearly free information for most of them: an instruction ordered
  // against nothing in Across pays the full width of Across to say so. There
  // is a cheaper way to spend the same interleaving. Once it is bought, the
  // order of all p+q walls is settled, and settled walls are a line -- so lay
  // them in one and charge the line, with no second coordinate for anyone.
  //
  // This is also what was missing before. A frame whose walls are not ordered
  // against each other was never unusable, only unpriced: the price is Mix,
  // and nothing else about it changes.
  DenseMap<const MachineInstr *, unsigned> Seat;
  for (auto [I, MI] : enumerate(G.Order))
    Seat[MI] = I;
  SmallVector<MachineInstr *, 0> Merged(Down.begin(), Down.end());
  append_range(Merged, Across);
  llvm::sort(Merged, [&](const MachineInstr *X, const MachineInstr *Y) {
    return Seat.lookup(X) < Seat.lookup(Y);
  });

  Regions RM;
  regionsOf(G, Merged, T, RM);
  double Melt = 0;
  SmallVector<unsigned, 0> Held(RM.size(), 0);
  for (const Window &W : RM.Wins) {
    ++Held[W.Here];
    Melt += std::log2(W.Last - W.First + 1);
  }
  unsigned MeltWorst = 0;
  for (unsigned B : Held) {
    Melt += bitsOf(B);
    MeltWorst = std::max(MeltWorst, B);
  }

  LLVM_DEBUG(dbgs() << "amdgpu-global-sched: melt: " << Merged.size()
                    << " walls in one line, " << RM.Wins.size() << " to place, "
                    << (uint64_t)std::llround(Mix) << " to interleave + "
                    << (uint64_t)std::llround(Melt) << " = "
                    << (uint64_t)std::llround(Mix + Melt)
                    << " bits largest region " << MeltWorst << "\n");
}

void AMDGPUGlobalSchedImpl::meltOf(const DepGraph &G, const Timing &T,
                                   ArrayRef<MachineInstr *> Walls) const {
  // Walls that are not ordered against each other were never unusable, only
  // unpriced. A schedule settles their order, so what a frame of them costs
  // on top of an ordered one is the number of orders they could have been
  // settled into -- and that is bounded by cutting them into chains, because
  // a chain's own order is forced and only the interleaving is free.
  const unsigned M = Walls.size();
  if (M < 2 || G.Order.size() > ClosureLimit || M > 1500)
    return;

  DenseMap<const MachineInstr *, unsigned> Seat;
  for (auto [I, MI] : enumerate(G.Order))
    Seat[MI] = I;

  // Who must come before whom, over the walls alone.
  SmallVector<BitVector, 0> Before(M, BitVector(M));
  {
    DenseMap<const MachineInstr *, unsigned> Spot;
    for (auto [I, MI] : enumerate(Walls))
      Spot[MI] = I;
    SmallVector<BitVector, 0> Full(G.Order.size(), BitVector(G.Order.size()));
    for (unsigned I = 0, N = G.Order.size(); I != N; ++I)
      for (const Dep &P : G.preds(G.Order[I])) {
        unsigned J = Seat.lookup(P.Other);
        Full[I] |= Full[J];
        Full[I].set(J);
      }
    for (unsigned I = 0; I != M; ++I) {
      unsigned Pi = Seat.lookup(Walls[I]);
      for (unsigned J = 0; J != M; ++J)
        if (Full[Pi].test(Seat.lookup(Walls[J])))
          Before[I].set(J);
    }
  }

  // Peel the longest chain again and again. Any cover will do for a bound,
  // and a greedy one keeps the count of chains down, which is what the
  // interleaving is paid per.
  BitVector Left(M, true);
  SmallVector<unsigned, 0> Cover;
  while (Left.any()) {
    SmallVector<unsigned, 0> Deep(M, 0);
    SmallVector<int, 0> Back(M, -1);
    unsigned Best = 0;
    int End = -1;
    for (unsigned I = 0; I != M; ++I) {
      if (!Left.test(I))
        continue;
      Deep[I] = 1;
      for (unsigned J = 0; J != I; ++J)
        if (Left.test(J) && Before[I].test(J) && Deep[J] + 1 > Deep[I]) {
          Deep[I] = Deep[J] + 1;
          Back[I] = J;
        }
      if (Deep[I] > Best) {
        Best = Deep[I];
        End = I;
      }
    }
    for (int I = End; I >= 0; I = Back[I])
      Left.reset(I);
    Cover.push_back(Best);
  }

  double Mix = std::lgamma((double)M + 1.0);
  for (unsigned C : Cover)
    Mix -= std::lgamma((double)C + 1.0);
  Mix /= std::log(2.0);

  Regions R;
  regionsOf(G, Walls, T, R);
  double Bits = 0;
  SmallVector<unsigned, 0> Held(R.size(), 0);
  for (const Window &W : R.Wins) {
    ++Held[W.Here];
    Bits += std::log2(W.Last - W.First + 1);
  }
  unsigned Worst = 0;
  for (unsigned B : Held) {
    Bits += bitsOf(B);
    Worst = std::max(Worst, B);
  }

  LLVM_DEBUG(dbgs() << "amdgpu-global-sched: meltall: " << M << " walls in "
                    << Cover.size() << " chains, longest " << Cover[0] << ", "
                    << R.Wins.size() << " to place, "
                    << (uint64_t)std::llround(Mix) << " to interleave + "
                    << (uint64_t)std::llround(Bits) << " = "
                    << (uint64_t)std::llround(Mix + Bits)
                    << " bits largest region " << Worst << "\n");
}

void AMDGPUGlobalSchedImpl::reduceOf(const DepGraph &G,
                                     ArrayRef<MachineInstr *> In,
                                     const Timing &T, Reduction &Out) const {
  const unsigned N = In.size();
  Out = Reduction();
  Out.Insts = Out.Worst = Out.Hard = N;
  Out.Bits = Out.BySame = Out.ByCut = bitsOf(N);
  if (N < 2)
    return;

  DenseMap<const MachineInstr *, unsigned> Idx;
  for (auto [I, MI] : enumerate(In))
    Idx[MI] = I;

  // The dependences that stay inside the region. None are lost: a path between
  // two instructions of one region never leaves it, since leaving would mean
  // crossing a step of the path that made the region, and then the two of them
  // would not be in the same one.
  SmallVector<SmallVector<unsigned, 4>, 0> Succ(N), Pred(N);
  for (auto [I, MI] : enumerate(In))
    for (const Dep &S : G.succs(MI))
      if (auto It = Idx.find(S.Other); It != Idx.end()) {
        Succ[I].push_back(It->second);
        Pred[It->second].push_back(I);
      }

  // Who has to come after whom, a bit per instruction. \p In is topological,
  // so one backwards sweep settles all of it.
  SmallVector<BitVector, 0> Reach(N, BitVector(N));
  for (unsigned I = N; I-- > 0;)
    for (unsigned J : Succ[I]) {
      Reach[I] |= Reach[J];
      Reach[I].set(J);
    }

  // Who each instruction waits for and who waits for it, as a sorted list, so
  // two of them can be asked whether they stand in exactly the same place in
  // the graph. The whole graph is used, not the region's part of it: an edge
  // leaving the region still constrains.
  auto sigOf = [](ArrayRef<Dep> D) {
    SmallVector<std::pair<const MachineInstr *, unsigned>, 8> S;
    for (const Dep &E : D)
      S.emplace_back(E.Other, E.Latency);
    llvm::sort(S);
    return S;
  };
  SmallVector<SmallVector<std::pair<const MachineInstr *, unsigned>, 8>, 0> Pre(
      N),
      Post(N);
  for (auto [I, MI] : enumerate(In)) {
    Pre[I] = sigOf(G.preds(MI));
    Post[I] = sigOf(G.succs(MI));
  }

  // Instructions that cost the same wherever they go: the same opcode, so the
  // same pipes and the same latency; the same earliest and latest cycle, so
  // the same price at every slot; and unordered against each other. Putting k
  // of those in order makes k! schedules that the clock cannot tell apart.
  //
  // That is not enough to drop k!-1 of them. Costing the same is not the same
  // as being exchangeable: swapping two of them has to leave a *legal*
  // schedule, and it only does when they wait for the same instructions and
  // the same instructions wait for them. Two adds that both become ready at
  // cycle 4 may do so behind different predecessors, and then the order they
  // go in is forced by those predecessors, not free.
  //
  // So both are counted. \p Same is what the cost model cannot distinguish --
  // an upper bound on what symmetry could ever remove. \p Twin is what is
  // really interchangeable, and only that may be removed without putting the
  // best schedule out of reach.
  auto Alike = [&](ArrayRef<unsigned> Set, unsigned &Same, unsigned &Twin,
                   double &TwinBits) {
    DenseMap<std::tuple<unsigned, unsigned, unsigned>, SmallVector<unsigned, 4>>
        ByKind;
    for (unsigned I : Set) {
      unsigned Asap = T.Start.lookup(In[I]);
      ByKind[{In[I]->getOpcode(), Asap, Asap + T.Slack.lookup(In[I])}]
          .push_back(I);
    }

    double Saved = 0;
    for (auto &[Kind, Members] : ByKind) {
      // Peel off one group of mutually unordered ones at a time. Ordering
      // within a group is free; between groups it is not.
      SmallVector<unsigned, 8> Left(Members.begin(), Members.end()), Group,
          Rest;
      while (!Left.empty()) {
        Group.clear();
        Rest.clear();
        for (unsigned I : Left) {
          if (none_of(Group, [&](unsigned J) {
                return Reach[J].test(I) || Reach[I].test(J);
              }))
            Group.push_back(I);
          else
            Rest.push_back(I);
        }
        Saved += bitsOf(Group.size());
        Same += Group.size() - 1;

        // And of that group, the ones that really may change places.
        SmallVector<unsigned, 8> Open(Group);
        while (!Open.empty()) {
          unsigned Head = Open.front();
          auto *Split =
              std::stable_partition(Open.begin(), Open.end(), [&](unsigned I) {
                return Pre[I] == Pre[Head] && Post[I] == Post[Head];
              });
          unsigned Size = Split - Open.begin();
          TwinBits += bitsOf(Size);
          Twin += Size - 1;
          SmallVector<unsigned, 8> Keep(Split, Open.end());
          Open = std::move(Keep);
        }

        Left.assign(Rest.begin(), Rest.end());
      }
    }
    return Saved;
  };

  SmallVector<unsigned, 0> All;
  append_range(All, llvm::seq(0u, N));
  double TwinBits = 0;
  Out.BySame = Out.Bits - Alike(All, Out.Same, Out.Twin, TwinBits);
  Out.ByTwin = Out.Bits - TwinBits;

  // And the region cuts the same way the function did, and its pieces cut
  // again, for as long as there is anything to cut with. A piece's own longest
  // run of instructions that cannot be exchanged is a path, and a path of P
  // steps splits the rest of the piece into P+1 smaller ones, each ordered on
  // its own. It stops when a piece has no dependences left inside it at all:
  // nothing there is ordered against anything, so there is no path to cut by,
  // and only alike can make it smaller. That leftover is the real unit of the
  // problem -- the widest set of instructions that are mutually unordered and
  // all different from each other.
  Out.Worst = Out.Hard = Out.ByCut = 0;
  Out.Cut = 0;

  SmallVector<SmallVector<unsigned, 0>, 0> Work;
  Work.push_back(std::move(All));
  while (!Work.empty()) {
    SmallVector<unsigned, 0> Set = Work.pop_back_val();
    const unsigned M = Set.size();

    // Longest run of mutually ordered instructions inside this piece alone.
    DenseMap<unsigned, unsigned> Depth;
    DenseMap<unsigned, unsigned> Prev;
    DenseSet<unsigned> Here(Set.begin(), Set.end());
    unsigned Deepest = 0, End = M ? Set[0] : 0;
    for (unsigned I : Set) {
      unsigned D = 0, From = I;
      for (unsigned P : Pred[I])
        if (Here.count(P) && Depth.lookup(P) + 1 > D) {
          D = Depth.lookup(P) + 1;
          From = P;
        }
      Depth[I] = D;
      Prev[I] = From;
      if (D >= Deepest) {
        Deepest = D;
        End = I;
      }
    }

    // Nothing in here is ordered, so no cut is possible. This is a leaf, and
    // only the instructions another really can stand for make it smaller.
    if (M < 2 || Deepest == 0) {
      unsigned Loose = 0, Folded = 0;
      double Kept = 0;
      Alike(Set, Loose, Folded, Kept);
      Out.ByCut += bitsOf(M) - Kept;
      Out.Worst = std::max(Out.Worst, M);
      Out.Hard = std::max(Out.Hard, M - Folded);
      Out.Cut += M != 0;
      continue;
    }

    BitVector OnPath(N);
    for (unsigned I = End, P = Prev.lookup(I); !OnPath.test(I);
         I = P, P = Prev.lookup(I))
      OnPath.set(I);

    // Which piece an instruction goes in is read off where it sits today, and
    // that is a choice, not a fact: dependence allows it into every piece
    // after the last step it waits for and before the first step that waits
    // for it. Cutting by position explores one piece out of that run and says
    // nothing about the others, so count what it passed over. Those bits are
    // not saved -- they move from ordering into placing.
    SmallVector<unsigned, 0> Steps;
    for (unsigned I : Set)
      if (OnPath.test(I))
        Steps.push_back(I);

    SmallVector<SmallVector<unsigned, 0>, 0> Piece(OnPath.count() + 1);
    unsigned Past = 0;
    for (unsigned I : Set) {
      if (OnPath.test(I)) {
        ++Past;
        continue;
      }
      unsigned First = 0, Last = Steps.size();
      for (auto [K, S] : enumerate(Steps)) {
        if (Reach[I].test(S)) {
          Last = K;
          break;
        }
        if (Reach[S].test(I))
          First = K + 1;
      }
      Out.Pick += std::log2(Last - First + 1);
      Piece[Past].push_back(I);
    }
    for (auto &P : Piece)
      if (!P.empty())
        Work.push_back(std::move(P));
  }
}

void AMDGPUGlobalSchedImpl::chainOf(const DepGraph &G, Timing &T) const {
  auto &Start = T.Start;
  auto &Slack = T.Slack;
  unsigned &Length = T.Length;
  // Forwards: the earliest an instruction can start is when the last thing it
  // waits for is ready.
  for (MachineInstr *MI : G.Order) {
    unsigned S = 0;
    for (const Dep &P : G.preds(MI))
      S = std::max(S, Start.lookup(P.Other) + P.Latency);
    Start[MI] = S;
    Length = std::max(Length, S + tailOf(*MI));
  }

  // Backwards: the latest an instruction can start without pushing the end of
  // the chain out is when its first consumer needs it. Everything left over
  // is slack -- a window of start times that costs nothing, and the only room
  // there is to schedule into.
  DenseMap<const MachineInstr *, unsigned> Latest;
  for (MachineInstr *MI : reverse(G.Order)) {
    unsigned L = Length - tailOf(*MI);
    if (auto It = Latest.find(MI); It != Latest.end())
      L = It->second;
    Slack[MI] = L - Start[MI];
    for (const Dep &P : G.preds(MI)) {
      unsigned Want = L - P.Latency;
      auto [DI, New] = Latest.try_emplace(P.Other, Want);
      if (!New)
        DI->second = std::min(DI->second, Want);
    }
  }
}

MachineBasicBlock *
AMDGPUGlobalSchedImpl::chooseBlock(const Placement &P) const {
  // Walk the interval from its latest end towards its earliest, and take the
  // block that runs least often. Picking a block inside a loop costs its trip
  // count, which is what makes a loop-invariant value end up outside without
  // anything here knowing the phrase. Ties go to the latest block, since that
  // is the schedule in which the value is live across the fewest blocks.
  MachineBasicBlock *Best = P.Hi;
  unsigned BestDepth = depthOf(P.Hi);
  for (MachineDomTreeNode *N = DT.getNode(P.Hi); N->getBlock() != P.Lo;
       N = N->getIDom()) {
    MachineBasicBlock *B = N->getIDom()->getBlock();
    if (unsigned D = depthOf(B); D < BestDepth) {
      Best = B;
      BestDepth = D;
    }
  }
  return Best;
}

void AMDGPUGlobalSchedImpl::place(MachineInstr &MI,
                                  MachineBasicBlock &To) const {
  Register Reg;
  for (const MachineOperand &MO : MI.operands())
    if (MO.isReg() && MO.isDef())
      Reg = MO.getReg();

  // Ahead of the first reader already in the block, or at the end if there is
  // none. A reader cannot be a PHI, because a PHI reads on its incoming edge,
  // so To would have had to be a predecessor of the PHI's block rather than
  // the block itself.
  MachineBasicBlock::iterator At = To.getFirstTerminator();
  for (MachineInstr &Other : To) {
    if (!Other.readsRegister(Reg, nullptr))
      continue;
    assert(!Other.isPHI() && "a PHI reads on its edge, not in its own block");
    At = Other.getIterator();
    break;
  }

  To.splice(At, MI.getParent(), MI.getIterator());

  // The value now reaches a different set of program points, so whatever the
  // old last reader was is no longer necessarily the last one.
  for (const MachineOperand &MO : MI.operands())
    if (MO.isReg() && MO.isUse() && MO.getReg().isVirtual())
      MRI->clearKillFlags(MO.getReg());
}

bool AMDGPUGlobalSchedImpl::run(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  TII = ST.getInstrInfo();
  TRI = &TII->getRegisterInfo();
  MRI = &MF.getRegInfo();
  SchedModel.init(&ST);

  Clean.clear();
  for (MachineBasicBlock &MBB : MF) {
    bool IsClean = true;
    for (const MachineInstr &MI : MBB) {
      if (MI.modifiesRegister(AMDGPU::EXEC, TRI) ||
          TII->modifiesModeRegister(MI)) {
        IsClean = false;
        break;
      }
    }
    if (IsClean)
      Clean.insert(&MBB);
  }

  unsigned NumInstrs = 0, NumFree = 0, NumPinned = 0;
  SmallVector<std::pair<MachineInstr *, Placement>, 0> Found;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (MI.isDebugInstr())
        continue;
      ++NumInstrs;
      Placement P;
      P.Home = &MBB;
      if ((P.Pin = classify(MI))) {
        P.Lo = P.Hi = &MBB;
        ++NumPinned;
      } else {
        P = intervalOf(MI);
        ++NumFree;
      }
      Found.emplace_back(&MI, P);
    }
  }

  LLVM_DEBUG({
    dbgs() << "amdgpu-global-sched: " << MF.getName() << ": " << MF.size()
           << " blocks, " << NumInstrs << " instrs, " << NumFree << " free, "
           << NumPinned << " pinned\n";
    for (auto &[MI, P] : Found) {
      dbgs() << "amdgpu-global-sched:   " << printMBBReference(*P.Home) << " "
             << TII->getName(MI->getOpcode());
      if (!P.isFree()) {
        dbgs() << " pinned (" << P.Pin << ")\n";
        continue;
      }
      unsigned Width =
          levelOf(DT.getNode(P.Hi)) - levelOf(DT.getNode(P.Lo)) + 1;
      dbgs() << " free " << printMBBReference(*P.Lo) << ".."
             << printMBBReference(*P.Hi) << " width " << Width << "\n";
    }
  });

  DepGraph G;
  buildDeps(MF, G);

  Timing T;
  chainOf(G, T);
  auto &Start = T.Start;
  auto &Slack = T.Slack;
  unsigned Length = T.Length;

  LLVM_DEBUG({
    unsigned OnChain =
        count_if(Found, [&](auto &P) { return Slack.lookup(P.first) == 0; });
    dbgs() << "amdgpu-global-sched: chain: " << NumInstrs << " instrs, "
           << Length << " cycles, " << OnChain << " critical\n";

    // Three different longest paths, and they are rarely the same one.
    unsigned CycleWeight = 0, StepWeight = 0, CtrlWeight = 0;
    auto ByCycles = chainPath(G, Weigh::Cycles, CycleWeight);
    auto BySteps = chainPath(G, Weigh::Steps, StepWeight);
    auto ByControl = chainPath(G, Weigh::Control, CtrlWeight);
    unsigned NumControl =
        count_if(ByControl, [&](MachineInstr *MI) { return isFence(*MI); });
    dbgs() << "amdgpu-global-sched: path by cycles: " << ByCycles.size()
           << " instrs, " << CycleWeight << " cycles\n";
    dbgs() << "amdgpu-global-sched: path by steps: " << BySteps.size()
           << " instrs, " << StepWeight << " steps\n";
    dbgs() << "amdgpu-global-sched: path by control: " << ByControl.size()
           << " instrs, " << NumControl << " control, " << ByControl.size() - 1
           << " steps\n";
    // Is the graph many separate computations, or one wide one? Two
    // instructions can be unordered either because nothing connects them at
    // all, or because they share an ancestor and neither reaches the other.
    // Weakly connected components tell the two apart, and the count of
    // instructions at each step-depth says how wide the thing is.
    DenseMap<const MachineInstr *, const MachineInstr *> Parent;
    std::function<const MachineInstr *(const MachineInstr *)> Find =
        [&](const MachineInstr *MI) {
          while (Parent[MI] != MI)
            MI = Parent[MI] = Parent[Parent[MI]];
          return MI;
        };

    SmallVector<unsigned, 0> Level;
    DenseMap<const MachineInstr *, unsigned> Depth;
    for (MachineInstr *MI : G.Order) {
      Parent.try_emplace(MI, MI);
      unsigned D = 0;
      for (const Dep &P : G.preds(MI)) {
        Parent.try_emplace(P.Other, P.Other);
        Parent[Find(P.Other)] = Find(MI);
        D = std::max(D, Depth.lookup(P.Other) + 1);
      }
      Depth[MI] = D;
      if (D >= Level.size())
        Level.resize(D + 1);
      ++Level[D];
    }

    DenseMap<const MachineInstr *, unsigned> Size;
    for (auto &[MI, P] : Parent)
      ++Size[Find(MI)];
    unsigned Biggest = 0;
    for (auto &[Root, N] : Size)
      Biggest = std::max(Biggest, N);
    dbgs() << "amdgpu-global-sched: shape: " << Size.size()
           << " components, biggest " << Biggest << ", " << Level.size()
           << " levels, widest " << *max_element(Level) << ", "
           << NumInstrs / Level.size() << " per level\n";

    // Where everything else stands with respect to that path. An instruction
    // that reaches a step of the path has to come before that step; one the
    // path reaches has to come after; the rest are ordered against it by
    // nothing at all, and are the only ones a placement is free to choose for.
    DenseSet<const MachineInstr *> OnPath(ByCycles.begin(), ByCycles.end());
    DenseSet<const MachineInstr *> Before, After;

    SmallVector<MachineInstr *, 0> Work(ByCycles);
    while (!Work.empty()) {
      MachineInstr *MI = Work.pop_back_val();
      for (const Dep &P : G.preds(MI))
        if (!OnPath.count(P.Other) && Before.insert(P.Other).second)
          Work.push_back(P.Other);
    }

    Work.assign(ByCycles.begin(), ByCycles.end());
    while (!Work.empty()) {
      MachineInstr *MI = Work.pop_back_val();
      for (const Dep &S : G.succs(MI))
        if (!OnPath.count(S.Other) && After.insert(S.Other).second)
          Work.push_back(S.Other);
    }

    unsigned Both = count_if(After, [&](auto *MI) { return Before.count(MI); });
    dbgs() << "amdgpu-global-sched: around the path: " << Before.size()
           << " before, " << After.size() << " after, " << Both << " both, "
           << NumInstrs - OnPath.size() - Before.size() - After.size() + Both
           << " unordered\n";

    // The control path is the one worth anchoring to. Regions are only a
    // useful cut if instructions are legal in few of them, and a step of pure
    // arithmetic cuts nothing -- nothing else in the function is ordered
    // against it. A sequence point is ordered against everything in its
    // block, so each one on the path is a real wall.
    Regions R;
    regionsOf(G, ByControl, T, R);

    // Three ways of counting the same thing, each narrower than the last.
    // A region slot is one instruction in one region, ignoring time. A cycle
    // slot is one instruction at one issue cycle of one region. And of those,
    // only some cost nothing at all.
    unsigned Slots = 0, Open = 0, Cycles = 0, Free = 0;
    double Place = 0, Narrow = 0;
    unsigned Forced = 0, Nowhere = 0;
    SmallVector<unsigned, 0> Busy(R.size(), 0);
    for (const Window &W : R.Wins) {
      ++Busy[W.Here];
      Slots += W.Last - W.First + 1;
      unsigned Fits = 0, Here = 0;
      for (unsigned I = W.First; I <= W.Last; ++I) {
        Cycles += R.cap(I);
        auto [Lo, Hi] = R.freeIn(W, I);
        if (Lo > Hi)
          continue;
        ++Fits;
        Here += Hi - Lo + 1;
      }
      Open += Fits;
      Free += Here;
      Place += std::log2(W.Last - W.First + 1);
      Narrow += std::log2(std::max(Here, 1u));
      Forced += Fits == 1;
      Nowhere += Fits == 0;
    }

    double Order = 0;
    unsigned Worst = 0;
    for (unsigned B : Busy) {
      Order += bitsOf(B);
      Worst = std::max(Worst, B);
    }

    // What is in each region now, in topological order -- Wins is built by a
    // sweep over G.Order, so filtering it keeps that order.
    SmallVector<SmallVector<MachineInstr *, 0>, 0> Hold(R.size());
    for (const Window &W : R.Wins)
      Hold[W.Here].push_back(W.MI);

    unsigned Num = R.Wins.size();
    dbgs() << "amdgpu-global-sched: regions: " << R.size() << " regions, "
           << Num << " to place, " << Slots << " legal slots, "
           << (Num ? Slots / Num : 0) << " regions each\n";
    // What the pipes say. A region is as long as the path makes it, but no
    // shorter than its busiest pipe has to be held for, and that floor holds
    // whatever order the region is put in.
    uint64_t ByPath = 0, ByPipe = 0;
    unsigned Over = 0;
    double Same = 0, Twins = 0, ByCut = 0, Pick = 0;
    unsigned Alike = 0, Really = 0, Hardest = 0, Pieces = 0;
    for (unsigned I = 0; I < R.size(); ++I) {
      auto [Need, Pipe] = R.floorOf(I);
      ByPath += R.cap(I);
      ByPipe += std::max(R.cap(I), Need);
      Over += Need > R.cap(I);
      dbgs() << "amdgpu-global-sched:   region " << I << ": " << R.Count[I]
             << " legal, " << Busy[I] << " now, " << R.cap(I) << " cycles";
      if (Need)
        dbgs() << ", needs " << Need << " " << SchedModel.getResourceName(Pipe);

      // The same region, asked how much of its order is a real choice.
      Reduction Red;
      reduceOf(G, Hold[I], T, Red);
      Same += Red.BySame;
      Twins += Red.ByTwin;
      ByCut += Red.ByCut;
      Pick += Red.Pick;
      Alike += Red.Same;
      Really += Red.Twin;
      Pieces += Red.Cut;
      Hardest = std::max(Hardest, Red.Hard);
      if (Red.Same || Red.Cut > 1)
        dbgs() << ", " << Red.Same << " alike " << Red.Twin << " really, cut "
               << "into " << Red.Cut << " biggest " << Red.Worst << " hard "
               << Red.Hard;
      dbgs() << "\n";
    }
    {
      GCNHazardRecognizer HR(MF, GCNHazardRecognizer::OperatingMode::PreRA);
      unsigned Waits = 0, Hit = 0;
      for (MachineBasicBlock &MBB : MF) {
        HR.Reset();
        for (MachineInstr &MI : MBB) {
          if (MI.isDebugInstr())
            continue;
          unsigned W = HR.getHazardWaitStates(&MI);
          Waits += W;
          Hit += W > 0;
          HR.EmitInstruction(&MI);
          HR.AdvanceCycle();
        }
      }
      dbgs() << "amdgpu-global-sched: hazards: " << Waits << " wait states, "
             << Hit << " instrs\n";
    }
    dbgs() << "amdgpu-global-sched: resources: " << ByPath
           << " cycles by dependence, " << ByPipe << " by pipe, " << Over
           << " of " << R.size() << " regions short\n";

    // Of the legal places, the ones that cost nothing. Dependence says where
    // an instruction may go; this says where it may go without waiting, and
    // it is a much smaller place.
    dbgs() << "amdgpu-global-sched: no stall: " << Open << " of " << Slots
           << " regions, " << Free << " of " << Cycles << " slots, " << Forced
           << " with one region, " << Nowhere << " with none\n";

    // How big the problem is. Choosing a place for each instruction costs the
    // log of the product of the choices; ordering what lands in a region
    // costs the log of the factorial of what is in it. In bits, so they add,
    // and so the number stays readable at any size.
    dbgs() << "amdgpu-global-sched: search: " << (uint64_t)std::llround(Place)
           << " bits to place, " << (uint64_t)std::llround(Order)
           << " to order, " << (uint64_t)std::llround(Place + Order)
           << " total, largest region " << Worst << ", "
           << (uint64_t)std::llround(Narrow) << " bits without a stall\n";

    // The ordering bits that are a real choice, and what it costs to be sure
    // of that. Folding instructions that merely cost the same is a guess; only
    // folding the ones another can really stand for keeps the best schedule
    // reachable, and it saves far less.
    dbgs() << "amdgpu-global-sched: alike: " << (uint64_t)std::llround(Order)
           << " bits to order, " << Alike << " cost the same leave "
           << (uint64_t)std::llround(Same) << ", " << Really
           << " really swap leave " << (uint64_t)std::llround(Twins) << "\n";

    // And the cut. The pieces are ordered separately, which is where the big
    // number goes -- but only once each instruction's piece is fixed, and
    // fixing it is itself a choice. Those bits are not saved, they move.
    dbgs() << "amdgpu-global-sched: cut: " << Pieces << " pieces, "
           << (uint64_t)std::llround(ByCut) << " bits to order them, "
           << (uint64_t)std::llround(Pick) << " to choose which, largest "
           << Worst << " -> " << Hardest << "\n";

    // Cutting a region by position throws schedules away. Cutting the
    // function by a different chain does not -- a chain is only a coordinate
    // system -- so the chain is worth choosing for the ordering it leaves
    // rather than for its length. Measured the same way as above, so the two
    // numbers can be compared.
    double SpreadBits = 0;
    SmallVector<MachineInstr *, 0> BySpread =
        spreadPath(G, Spread::Total, SpreadBits);

    Regions R2;
    regionsOf(G, BySpread, T, R2);

    unsigned Slots2 = 0;
    double Place2 = 0, Order2 = 0;
    SmallVector<unsigned, 0> Busy2(R2.size(), 0);
    for (const Window &W : R2.Wins) {
      ++Busy2[W.Here];
      Slots2 += W.Last - W.First + 1;
      Place2 += std::log2(W.Last - W.First + 1);
    }
    unsigned Worst2 = 0;
    for (unsigned B : Busy2) {
      Order2 += bitsOf(B);
      Worst2 = std::max(Worst2, B);
    }

    unsigned Num2 = R2.Wins.size();
    dbgs() << "amdgpu-global-sched: frame: " << BySpread.size()
           << " on the path, " << Num2 << " left to place, "
           << (Num2 ? Slots2 / Num2 : 0) << " regions each, "
           << (uint64_t)std::llround(Place2) << " bits to place, "
           << (uint64_t)std::llround(Order2) << " to order, "
           << (uint64_t)std::llround(Place2 + Order2)
           << " total, largest region " << Worst2 << "\n";

    double SpreadBitsWorst = 0;
    SmallVector<MachineInstr *, 0> BySpreadWorst =
        spreadPath(G, Spread::Worst, SpreadBitsWorst);

    Regions R3;
    regionsOf(G, BySpreadWorst, T, R3);

    unsigned Slots3 = 0;
    double Place3 = 0, Order3 = 0;
    SmallVector<unsigned, 0> Busy3(R3.size(), 0);
    for (const Window &W : R3.Wins) {
      ++Busy3[W.Here];
      Slots3 += W.Last - W.First + 1;
      Place3 += std::log2(W.Last - W.First + 1);
    }
    unsigned Worst3 = 0;
    for (unsigned B : Busy3) {
      Order3 += bitsOf(B);
      Worst3 = std::max(Worst3, B);
    }

    unsigned Num3 = R3.Wins.size();
    dbgs() << "amdgpu-global-sched: worst: " << BySpreadWorst.size()
           << " on the path, " << Num3 << " left to place, "
           << (Num3 ? Slots3 / Num3 : 0) << " regions each, "
           << (uint64_t)std::llround(Place3) << " bits to place, "
           << (uint64_t)std::llround(Order3) << " to order, "
           << (uint64_t)std::llround(Place3 + Order3)
           << " total, largest region " << Worst3 << "\n";

    // The biggest region of the best cut is the one that matters, so ask what
    // it is made of before asking what cutting it again would do.
    unsigned Big = 0;
    for (auto [I, B] : enumerate(Busy2))
      if (B > Busy2[Big])
        Big = I;

    SmallVector<MachineInstr *, 0> Inside;
    for (const Window &W : R2.Wins)
      if (W.Here == Big)
        Inside.push_back(W.MI);

    SmallVector<std::pair<unsigned, unsigned>, 0> Parts;
    partsOf(G, Inside, Parts);

    dbgs() << "amdgpu-global-sched: parts: " << Inside.size() << " instrs, "
           << Parts.size() << " groups, sizes";
    for (unsigned I = 0, E = std::min<size_t>(8, Parts.size()); I != E; ++I)
      dbgs() << " " << Parts[I].first;
    dbgs() << ", chains";
    for (unsigned I = 0, E = std::min<size_t>(8, Parts.size()); I != E; ++I)
      dbgs() << " " << Parts[I].second;
    dbgs() << "\n";

    SmallVector<Tier, 0> Depths;
    SmallVector<unsigned, 0> Leaves;
    SmallVector<MachineInstr *, 0> Anchors;
    anchorsOf(G, G.Order, Depths, Leaves, Anchors);
    for (auto [D, L] : enumerate(Depths))
      dbgs() << "amdgpu-global-sched: tree: depth " << D << ", " << L.Regions
             << " regions, " << L.Insts << " instrs, " << L.Steps
             << " anchored, biggest " << L.Worst << ", " << L.Stuck
             << " with no order left, " << (uint64_t)std::llround(L.Bits)
             << " bits to order\n";

    // Where the recursion stopped is where a search would have to start. The
    // depths above count regions still being cut, so they say nothing about
    // what is left when the cutting is over; these do.
    llvm::sort(Leaves, std::greater<unsigned>());
    unsigned LeafInstrs = 0, LeafMany = 0;
    double LeafBits = 0;
    for (unsigned S : Leaves) {
      LeafInstrs += S;
      LeafMany += S >= 8;
      LeafBits += bitsOf(S);
    }
    dbgs() << "amdgpu-global-sched: leaves: " << Leaves.size() << " regions, "
           << LeafInstrs << " instrs, biggest "
           << (Leaves.empty() ? 0 : Leaves[0]) << ", " << LeafMany
           << " of 8 or more, " << (uint64_t)std::llround(LeafBits)
           << " bits to order, sizes";
    for (unsigned I = 0, E = std::min<size_t>(12, Leaves.size()); I != E; ++I)
      dbgs() << " " << Leaves[I];
    dbgs() << "\n";

    // Which leaf an instruction ended up in was read off where it sits today,
    // not off what dependence allows, and each depth of the recursion made
    // one more such choice. Holding every anchor the recursion produced and
    // then asking the question properly says how much of that was forced.
    // The anchors are not a chain -- two of them from different depths need
    // not be ordered at all -- but they are still fixed points in a known
    // order, which is all that bounding a window needs.
    DenseMap<const MachineInstr *, unsigned> Seat;
    for (auto [I, MI] : enumerate(G.Order))
      Seat[MI] = I;
    llvm::sort(Anchors, [&](const MachineInstr *A, const MachineInstr *B) {
      return Seat.lookup(A) < Seat.lookup(B);
    });

    Regions R4;
    regionsOf(G, Anchors, T, R4);
    meltOf(G, T, Anchors);

    unsigned Only = 0, Widest = 0;
    double Place4 = 0;
    for (const Window &W : R4.Wins) {
      unsigned Wide = W.Last - W.First + 1;
      Only += Wide == 1;
      Widest = std::max(Widest, Wide);
      Place4 += std::log2(Wide);
    }
    dbgs() << "amdgpu-global-sched: pinned: " << Anchors.size() << " anchors, "
           << R4.Wins.size() << " to place, " << Only
           << " with nowhere else to go, widest " << Widest << ", "
           << (uint64_t)std::llround(Place4) << " bits to place\n";

    // And the same question asked the way that keeps the frame sound: start
    // from the best single chain and only ever swap walls for walls that are
    // ordered against everything left.
    SmallVector<MachineInstr *, 0> Fine = BySpread;
    refineOf(G, T, RefineSpan, Fine);

    // The second chain goes through whatever the first one left biggest.
    Regions RF;
    regionsOf(G, Fine, T, RF);
    SmallVector<unsigned, 0> Load(RF.size(), 0);
    for (const Window &W : RF.Wins)
      ++Load[W.Here];
    unsigned Big5 = 0;
    for (auto [I, B] : enumerate(Load))
      if (B > Load[Big5])
        Big5 = I;
    SmallVector<MachineInstr *, 0> Inside5;
    for (const Window &W : RF.Wins)
      if (W.Here == Big5)
        Inside5.push_back(W.MI);

    DenseMap<const MachineInstr *, unsigned> Spot;
    for (auto [I, MI] : enumerate(Inside5))
      Spot[MI] = I;
    SmallVector<SmallVector<unsigned, 0>, 0> Climb(Inside5.size());
    for (auto [I, MI] : enumerate(Inside5))
      for (const Dep &P : G.preds(MI))
        if (auto It = Spot.find(P.Other); It != Spot.end())
          Climb[I].push_back(It->second);

    double Spent = 0;
    SmallVector<MachineInstr *, 0> Across;
    for (unsigned S : spreadIn(Climb, Spread::Total, Spent))
      Across.push_back(Inside5[S]);
    if (!Across.empty())
      gridOf(G, T, Fine, Across);

    // And what each choice costs. A slot of a region issues at a known cycle,
    // so putting an instruction there has a price that can be read off before
    // any search begins: how long the machine waits for its operands, plus
    // how long its consumers then wait for it.
    for (const Window &W : R.Wins) {
      dbgs() << "amdgpu-global-sched:   slots "
             << TII->getName(W.MI->getOpcode()) << " ready " << W.Asap
             << " due " << W.Alap << ":";
      for (unsigned I = W.First; I <= W.Last; ++I) {
        auto [Lo, Hi] = R.freeIn(W, I);
        dbgs() << " r" << I;
        if (Lo <= Hi) {
          dbgs() << " " << Lo << ".." << Hi;
          continue;
        }
        dbgs() << " +" << R.bestIn(W, I).second;
      }
      dbgs() << "\n";
    }

    for (MachineInstr *MI : ByCycles)
      dbgs() << "amdgpu-global-sched:   on path "
             << printMBBReference(*MI->getParent()) << " "
             << TII->getName(MI->getOpcode()) << " at " << Start.lookup(MI)
             << "\n";
    for (MachineInstr *MI : BySteps)
      dbgs() << "amdgpu-global-sched:   on steps "
             << printMBBReference(*MI->getParent()) << " "
             << TII->getName(MI->getOpcode()) << "\n";
    for (MachineInstr *MI : ByControl)
      dbgs() << "amdgpu-global-sched:   on control "
             << printMBBReference(*MI->getParent()) << " "
             << TII->getName(MI->getOpcode()) << "\n";
    for (auto &[MI, P] : Found)
      dbgs() << "amdgpu-global-sched:   " << printMBBReference(*P.Home) << " "
             << TII->getName(MI->getOpcode()) << " start " << Start.lookup(MI)
             << " slack " << Slack.lookup(MI) << " lat " << tailOf(*MI) << "\n";
  });

  // Emit back to front. An instruction is placed ahead of its first reader in
  // the block it lands in, so every reader has to have landed already; reverse
  // program order gives that, because a reader never precedes what it reads.
  bool Changed = false;
  for (auto &[MI, P] : reverse(Found)) {
    if (!P.isFree())
      continue;
    MachineBasicBlock *To = chooseBlock(P);
    if (To == P.Home)
      continue;
    LLVM_DEBUG(dbgs() << "amdgpu-global-sched:   place "
                      << TII->getName(MI->getOpcode()) << " "
                      << printMBBReference(*P.Home) << " -> "
                      << printMBBReference(*To) << "\n");
    place(*MI, *To);
    Changed = true;
  }

  return Changed;
}

namespace {

class AMDGPUGlobalSchedLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUGlobalSchedLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU whole-function scheduler";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    AU.addRequired<MachineCycleInfoWrapperPass>();
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

INITIALIZE_PASS_BEGIN(AMDGPUGlobalSchedLegacy, DEBUG_TYPE,
                      "AMDGPU whole-function scheduler", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineCycleInfoWrapperPass)
INITIALIZE_PASS_END(AMDGPUGlobalSchedLegacy, DEBUG_TYPE,
                    "AMDGPU whole-function scheduler", false, false)

char AMDGPUGlobalSchedLegacy::ID = 0;

char &llvm::AMDGPUGlobalSchedID = AMDGPUGlobalSchedLegacy::ID;

bool AMDGPUGlobalSchedLegacy::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;
  MachineDominatorTree &DT =
      getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  MachineCycleInfo &CI =
      getAnalysis<MachineCycleInfoWrapperPass>().getCycleInfo();
  return AMDGPUGlobalSchedImpl(DT, CI).run(MF);
}

PreservedAnalyses
AMDGPUGlobalSchedPass::run(MachineFunction &MF,
                           MachineFunctionAnalysisManager &MFAM) {
  MachineDominatorTree &DT = MFAM.getResult<MachineDominatorTreeAnalysis>(MF);
  MachineCycleInfo &CI = MFAM.getResult<MachineCycleAnalysis>(MF);
  if (!AMDGPUGlobalSchedImpl(DT, CI).run(MF))
    return PreservedAnalyses::all();
  return getMachineFunctionPassPreservedAnalyses().preserveSet<CFGAnalyses>();
}
