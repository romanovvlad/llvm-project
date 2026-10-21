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

namespace {

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

/// More than any path can be long, so a path with more control on it beats a
/// longer one whatever else is counted.
static constexpr unsigned ControlWeight = 1u << 20;

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

  /// Whether \p MI is a sequence point: nothing may be moved across it in
  /// either direction, whatever the values say.
  bool isFence(const MachineInstr &MI) const;

  /// Everything that orders one instruction before another, in one graph.
  void buildDeps(MachineFunction &MF, DepGraph &G) const;

  /// One longest path through \p G under \p How, and its weight in \p Weight.
  SmallVector<MachineInstr *, 0> chainPath(const DepGraph &G, Weigh How,
                                           unsigned &Weight) const;

  /// A path of N instructions cuts the function into N+1 regions: one before
  /// each step and one after the last. \p Count is filled with how many of
  /// the instructions not on \p Path dependence allows into each region, and
  /// \p Width with how many regions each such instruction is allowed into,
  /// and \p Here with the region each one is in now.
  void regionsOf(const DepGraph &G, ArrayRef<MachineInstr *> Path,
                 SmallVectorImpl<unsigned> &Count,
                 SmallVectorImpl<unsigned> &Width,
                 SmallVectorImpl<unsigned> &Here) const;

  /// Earliest and latest start time of each instruction, in cycles, as \p G
  /// alone allows. Fills \p Start and \p Slack and returns the length of the
  /// longest chain.
  unsigned chainOf(const DepGraph &G,
                   DenseMap<const MachineInstr *, unsigned> &Start,
                   DenseMap<const MachineInstr *, unsigned> &Slack) const;

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

void AMDGPUGlobalSchedImpl::regionsOf(const DepGraph &G,
                                      ArrayRef<MachineInstr *> Path,
                                      SmallVectorImpl<unsigned> &Count,
                                      SmallVectorImpl<unsigned> &Width,
                                      SmallVectorImpl<unsigned> &Here) const {
  // Number the steps of the path, and the N+1 regions between them, so that
  // region I is the gap immediately before step I and region N is what comes
  // after the last step.
  DenseMap<const MachineInstr *, unsigned> Step;
  for (auto [I, MI] : enumerate(Path))
    Step[MI] = I;
  const int N = Path.size();

  Count.assign(N + 1, 0);
  Width.clear();
  Here.clear();

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
  // it is never empty, since the path is a path.
  // G.Order is the schedule the function has now, so counting the steps
  // already gone past says which region an instruction is in today.
  unsigned Past = 0;
  for (MachineInstr *MI : G.Order) {
    if (Step.count(MI)) {
      ++Past;
      continue;
    }
    int First = Lo.lookup(MI) + 1, Last = Hi.lookup(MI);
    Width.push_back(Last - First + 1);
    Here.push_back(Past);
    for (int I = First; I <= Last; ++I)
      ++Count[I];
  }
}

unsigned AMDGPUGlobalSchedImpl::chainOf(
    const DepGraph &G, DenseMap<const MachineInstr *, unsigned> &Start,
    DenseMap<const MachineInstr *, unsigned> &Slack) const {
  // Forwards: the earliest an instruction can start is when the last thing it
  // waits for is ready.
  unsigned Length = 0;
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

  return Length;
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

  DenseMap<const MachineInstr *, unsigned> Start, Slack;
  unsigned Length = chainOf(G, Start, Slack);

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
    SmallVector<unsigned, 0> Count, Width, Here;
    regionsOf(G, ByControl, Count, Width, Here);
    unsigned Slots = 0;
    for (unsigned W : Width)
      Slots += W;
    dbgs() << "amdgpu-global-sched: regions: " << Count.size() << " regions, "
           << Width.size() << " to place, " << Slots << " legal slots, "
           << (Width.empty() ? 0 : Slots / Width.size()) << " regions each\n";
    // How big the problem is. Choosing a region for each instruction costs
    // the log of the product of the window widths; ordering what lands in a
    // region costs the log of the factorial of what is in it. In bits, so
    // they add, and so the number stays readable at any size.
    double Place = 0;
    for (unsigned W : Width)
      Place += std::log2(W);
    SmallVector<unsigned, 0> Busy(Count.size(), 0);
    for (unsigned R : Here)
      ++Busy[R];
    double Order = 0;
    unsigned Worst = 0;
    for (unsigned B : Busy) {
      Order += std::lgamma(B + 1.0) / std::log(2.0);
      Worst = std::max(Worst, B);
    }
    for (auto [I, C] : enumerate(Count))
      dbgs() << "amdgpu-global-sched:   region " << I << ": " << C << " legal, "
             << Busy[I] << " now\n";
    dbgs() << "amdgpu-global-sched: search: " << (uint64_t)std::llround(Place)
           << " bits to place, " << (uint64_t)std::llround(Order)
           << " to order, " << (uint64_t)std::llround(Place + Order)
           << " total, largest region " << Worst << "\n";

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
