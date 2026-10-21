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
#include "llvm/CodeGen/MachineCycleAnalysis.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/TargetSchedule.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

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

  /// Earliest and latest start time of each instruction, in cycles, as the
  /// dependence graph alone allows. Fills \p Start and \p Slack and returns
  /// the length of the longest chain.
  unsigned chainOf(MachineFunction &MF,
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
    unsigned DefIdx = 0;
    for (const MachineOperand &DefMO : Def.operands())
      if (DefMO.isReg() && DefMO.isDef() && DefMO.getReg() == MO.getReg()) {
        DefIdx = Def.getOperandNo(&DefMO);
        break;
      }
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

unsigned AMDGPUGlobalSchedImpl::chainOf(
    MachineFunction &MF, DenseMap<const MachineInstr *, unsigned> &Start,
    DenseMap<const MachineInstr *, unsigned> &Slack) const {
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

  // Forwards: the earliest an instruction can start is when its last operand
  // is ready.
  unsigned Length = 0;
  for (MachineInstr *MI : Order) {
    unsigned S = 0;
    if (!MI->isPHI()) {
      for (const MachineOperand &MO : MI->uses()) {
        if (!MO.isReg() || !MO.getReg().isVirtual())
          continue;
        MachineInstr *Def = MRI->getVRegDef(MO.getReg());
        auto It = Start.find(Def);
        if (It == Start.end())
          continue;
        S = std::max(S, It->second + latencyBetween(*Def, *MI, MO));
      }
    }
    Start[MI] = S;
    Length = std::max(Length, S + tailOf(*MI));
  }

  // Backwards: the latest an instruction can start without pushing the end of
  // the chain out is when its first consumer needs it. Everything left over
  // is slack -- a window of start times that costs nothing, and the only room
  // there is to schedule into.
  DenseMap<const MachineInstr *, unsigned> Latest;
  for (MachineInstr *MI : reverse(Order)) {
    unsigned L = Length - tailOf(*MI);
    if (auto It = Latest.find(MI); It != Latest.end())
      L = It->second;
    Slack[MI] = L - Start[MI];
    if (MI->isPHI())
      continue;
    for (const MachineOperand &MO : MI->uses()) {
      if (!MO.isReg() || !MO.getReg().isVirtual())
        continue;
      MachineInstr *Def = MRI->getVRegDef(MO.getReg());
      unsigned Want = L - latencyBetween(*Def, *MI, MO);
      auto [DI, New] = Latest.try_emplace(Def, Want);
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

  DenseMap<const MachineInstr *, unsigned> Start, Slack;
  unsigned Length = chainOf(MF, Start, Slack);

  LLVM_DEBUG({
    unsigned OnChain =
        count_if(Found, [&](auto &P) { return Slack.lookup(P.first) == 0; });
    dbgs() << "amdgpu-global-sched: chain: " << NumInstrs << " instrs, "
           << Length << " cycles, " << OnChain << " on the chain\n";
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
