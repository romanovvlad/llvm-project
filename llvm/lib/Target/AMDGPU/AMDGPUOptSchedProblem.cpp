//===- AMDGPUOptSchedProblem.cpp - Problem extraction and checking --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Turning a scheduling region into a Problem, and checking an Answer against
/// it. Deliberately in its own file, with no solver anywhere near it: the
/// verifier is only worth having if it cannot share a bug with the thing it
/// verifies.
///
//===----------------------------------------------------------------------===//

#include "AMDGPUOptSched.h"
#include "GCNSubtarget.h"
#include "SIRegisterInfo.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/CodeGen/LiveRegUnits.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/TargetSchedule.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::AMDGPUOptSched;

static cl::opt<bool> PinPhysRegs(
    "amdgpu-optsched-pin-physregs", cl::init(false), cl::Hidden,
    cl::desc("Forbid moving any instruction that names an allocatable "
             "physical register, instead of modelling its live range"));

static cl::opt<bool> UsePipeModel(
    "amdgpu-optsched-pipe-model", cl::init(true), cl::Hidden,
    cl::desc("Charge a schedule for the functional units it keeps busy, not "
             "only for its dependences. Off is the older, shorter-sounding "
             "model, kept for measuring what the difference is worth"));

StringRef llvm::AMDGPUOptSched::toString(Status S) {
  switch (S) {
  case Status::Witness:
    return "witness";
  case Status::UpperBound:
    return "upper-bound";
  case Status::Optimal:
    return "optimal";
  }
  llvm_unreachable("unhandled Status");
}

static StringRef kindName(SDep::Kind K) {
  switch (K) {
  case SDep::Data:
    return "data";
  case SDep::Anti:
    return "anti";
  case SDep::Output:
    return "output";
  case SDep::Order:
    return "order";
  }
  llvm_unreachable("unhandled SDep::Kind");
}

/// Which register files a class can be allocated out of, as a bit per
/// RegFile. The AV classes answer with two, which is the only reason this is
/// a set and not a single file.
static unsigned filesOf(const SIRegisterInfo &TRI,
                        const TargetRegisterClass *RC) {
  unsigned M = 0;
  if (SIRegisterInfo::hasSGPRs(RC))
    M |= 1u << unsigned(RegFile::SGPR);
  if (SIRegisterInfo::hasVGPRs(RC))
    M |= 1u << unsigned(RegFile::ArchVGPR);
  if (SIRegisterInfo::hasAGPRs(RC))
    M |= 1u << unsigned(RegFile::AGPR);
  return M;
}

//===----------------------------------------------------------------------===//
// Problem
//===----------------------------------------------------------------------===//

Problem Problem::build(const ScheduleDAGMI &DAG) {
  Problem P;
  const unsigned N = DAG.SUnits.size();

  P.NodeLatency.resize(N);
  P.PredEdges.resize(N);
  P.SuccEdges.resize(N);
  P.Pipes.resize(N);

  // Which functional unit each instruction holds, and for how long, taken from
  // the subtarget's own scheduling model. Nothing here is a constant of this
  // file: SISchedule.td says an MFMA holds HWXDL for one cycle per pass and
  // that a plain VALU holds HWVALU for one, and those are the numbers used.
  const TargetSchedModel *SM = DAG.getSchedModel();
  if (UsePipeModel && SM->hasInstrSchedModel())
    P.NumPipes = SM->getNumProcResourceKinds();

  for (unsigned I = 0; I != N; ++I) {
    const SUnit &SU = DAG.SUnits[I];
    assert(SU.NodeNum == I && "SUnits are expected to be numbered by index");
    // SUnit::Latency is the instruction's own completion latency, which is
    // what a makespan needs; edge latencies are the def-to-use delays and are
    // read separately below.
    P.NodeLatency[I] = SU.Latency;

    if (!P.NumPipes)
      continue;
    const MCSchedClassDesc *SC = SM->resolveSchedClass(SU.getInstr());
    if (!SC || !SC->isValid())
      continue;
    for (const MCWriteProcResEntry &W :
         make_range(SM->getWriteProcResBegin(SC), SM->getWriteProcResEnd(SC))) {
      // A unit with several copies does not serialise anything until all of
      // them are busy, which needs a count per cycle rather than a free-at
      // time. AMDGPU has no such unit, so rather than write a model with no
      // test behind it, say nothing about those.
      if (SM->getProcResource(W.ProcResourceIdx)->NumUnits != 1)
        continue;
      // One cycle is the default every instruction carries and it constrains
      // nothing beyond the issue port, which is already modelled. Keeping
      // those would make the per-order timing walk the whole resource table
      // for no change in the answer.
      if (W.ReleaseAtCycle <= 1)
        continue;
      // One entry per unit. A sched class may name the same one twice -- an
      // instruction with two defs has two writes -- and it is still one unit
      // held once, for the longer of the two.
      auto *Same = llvm::find_if(P.Pipes[I], [&](const PipeUse &U) {
        return U.Pipe == W.ProcResourceIdx;
      });
      if (Same != P.Pipes[I].end())
        Same->Cycles = std::max(Same->Cycles, unsigned(W.ReleaseAtCycle));
      else
        P.Pipes[I].push_back({W.ProcResourceIdx, W.ReleaseAtCycle});
    }
  }

  for (unsigned I = 0; I != N; ++I) {
    for (const SDep &D : DAG.SUnits[I].Succs) {
      // Barriers and the exit node are not part of the order we emit.
      if (D.getSUnit()->isBoundaryNode())
        continue;
      // Weak edges are hints. Treating them as legality would be safe but
      // would needlessly shrink the search space; see the header.
      if (D.isWeak())
        continue;
      unsigned Succ = D.getSUnit()->NodeNum;
      // An order edge moves no result, but two instructions still cannot
      // issue in the same cycle on an in-order wave, so the gap is at least 1.
      unsigned Lat = std::max(1u, D.getLatency());
      P.Edges.push_back({I, Succ, Lat, D.getKind()});
      P.SuccEdges[I].push_back(P.Edges.size() - 1);
      P.PredEdges[Succ].push_back(P.Edges.size() - 1);
    }
  }

  // Instructions whose effect on the register files we cannot model are pinned
  // where they are, because a schedule we cannot price is a schedule we cannot
  // safely prefer.
  //
  // A register mask is the standing case: a call clobbers a list of physical
  // registers that appears nowhere in its operands as a def, so
  // buildRegisterModel sees none of it and would happily hoist a long live
  // range across the call. There is no cheap way to count a mask per candidate
  // order, so these stay put.
  //
  // -amdgpu-optsched-pin-physregs widens that to anything naming an allocatable
  // physical register, which is what this used to do unconditionally and is
  // kept as a way to tell a physical-liveness bug from a scheduling one. The
  // case that argued for it is a-v-flat-atomicrmw.ll, whose region holds
  //
  //     %13:vreg_1024 = COPY $vgpr0_..._vgpr31
  //     %14:vreg_1024 = COPY $vgpr32_..._vgpr63
  //     %11:vgpr_32   = FLAT_ATOMIC_SWAP_RTN ...
  //
  // and where hoisting the atomic above the copies is shorter and has *lower*
  // virtual pressure -- and is unallocatable, because %11 then lives across 64
  // occupied physical VGPRs.
  //
  // Counting those 64 registers does not catch it and no amount of counting
  // can: in the order that works the same 64 are held by %13 and %14, which
  // are virtual, and every count is identical. What separates them is that a
  // virtual neighbour can be evicted and a physical one cannot, which is
  // colourability, not pressure. That is what shortfallOf asks and what the
  // driver gates on, so this stays limited to register masks.
  //
  // Reserved registers ($exec, $flat_scr, $mode) are exempt: they are on
  // nearly every instruction and the allocator never has to find room for
  // them, so pinning on those would pin the whole region.
  for (unsigned I = 0; I != N; ++I) {
    const MachineInstr *MI = DAG.SUnits[I].getInstr();
    bool Pinned = false;
    for (const MachineOperand &MO : MI->operands()) {
      if (MO.isRegMask()) {
        Pinned = true;
        break;
      }
      if (!PinPhysRegs || !MO.isReg() || !MO.getReg().isPhysical())
        continue;
      if (DAG.MRI.isReserved(MO.getReg()))
        continue;
      Pinned = true;
      break;
    }
    if (!Pinned)
      continue;

    // Node numbers are source order, so this fixes the pinned instruction's
    // position relative to every other one.
    for (unsigned J = 0; J != N; ++J) {
      if (J == I)
        continue;
      unsigned Pred = std::min(I, J), Succ = std::max(I, J);
      P.Edges.push_back({Pred, Succ, 1, SDep::Order, /*Synthetic=*/true});
      P.SuccEdges[Pred].push_back(P.Edges.size() - 1);
      P.PredEdges[Succ].push_back(P.Edges.size() - 1);
    }
  }

  if (N) {
    const MachineBasicBlock *MBB = DAG.SUnits[0].getInstr()->getParent();
    P.Name = (MBB->getParent()->getName() + ":" + MBB->getName()).str();
  }

  P.computeHeights();
  P.buildRegisterModel(DAG);
  return P;
}

void Problem::buildRegisterModel(const ScheduleDAGMI &DAG) {
  const unsigned N = size();
  Defs.resize(N);
  Uses.resize(N);
  if (!N)
    return;

  const MachineRegisterInfo &MRI = DAG.MRI;
  const auto &ST = DAG.MF.getSubtarget<GCNSubtarget>();
  const auto &TRI = *static_cast<const SIRegisterInfo *>(DAG.TRI);

  // What this function may use, which is what the allocator will actually
  // hand out: SIRegisterInfo::getReservedRegs reserves every vector register
  // above these two numbers, so they are limits in the strong sense and not
  // targets. The architectural counts would be a far looser and, for the
  // question shortfallOf asks, a useless bound -- a function held to 64 VGPRs
  // runs out of them long before gfx90a's 512 are gone.
  auto [MaxVGPRs, MaxAGPRs] = ST.getMaxNumVectorRegs(DAG.MF.getFunction());
  Bud.Files[unsigned(RegFile::SGPR)] = ST.getMaxNumSGPRs(DAG.MF);
  Bud.Files[unsigned(RegFile::ArchVGPR)] = MaxVGPRs;
  Bud.Files[unsigned(RegFile::AGPR)] = MaxAGPRs;
  Bud.UnifiedVGPR = ST.hasGFX90AInsts();
  Bud.VGPR = ST.getMaxNumVGPRs(DAG.MF);

  // Which instructions are ours, so "used below the region" can be told from
  // "used later in the region". A value read below is live to the end no
  // matter where its last in-region read lands, so the schedule cannot shorten
  // it and it must not look like it can.
  SmallPtrSet<const MachineInstr *, 32> InRegion;
  for (const SUnit &SU : DAG.SUnits)
    InRegion.insert(SU.getInstr());

  // Physical liveness at the two ends of the region, by register unit.
  //
  // Both are reached by stepping *backward* from the end of the block, which
  // is the direction that needs no kill flags to be right -- the same walk the
  // driver's demandOf does, for the same reason. Above is simply Below carried
  // on through the region itself.
  MachineBasicBlock *MBB = DAG.SUnits[0].getInstr()->getParent();
  // Debug instructions carry operands but no liveness, and stepBackward says
  // so with an assertion.
  auto StepOver = [](LiveRegUnits &LRU, MachineBasicBlock::iterator First,
                     MachineBasicBlock::iterator Last) {
    while (Last != First) {
      const MachineInstr &MI = *--Last;
      if (!MI.isDebugInstr())
        LRU.stepBackward(MI);
    }
  };
  LiveRegUnits Below(TRI);
  Below.addLiveOuts(*MBB);
  StepOver(Below, DAG.end(), MBB->end());
  LiveRegUnits Above = Below;
  StepOver(Above, DAG.begin(), DAG.end());

  // A unit gets one value per definition, not one value in total.
  //
  // Collapsing a unit's definitions into a single range is the obvious thing
  // and it is useless. $vgpr0 here is written by an inline asm, copied out,
  // written again and read again; one range covers the whole region and makes
  // every order look equally crowded, which is the one thing a model asked to
  // compare orders may not do. Segments cost a map lookup and tell the two
  // orders apart.
  //
  // The segments are cut in source order, and stay cut that way whatever
  // order is scored. That is sound because the DAG has output and anti
  // dependences on physical registers, so no legal order moves a read of one
  // segment past the definition that starts the next.
  DenseMap<unsigned, unsigned> CurUnit;  ///< Unit -> its latest value.
  DenseMap<unsigned, unsigned> UnitHead; ///< Unit -> its first value.
  DenseMap<unsigned, unsigned> UnitDef;  ///< Unit -> node that opened it.
  auto UnitValueOf = [&](MCRegUnit U, RegFile F, bool IsDef,
                         unsigned At) -> unsigned {
    auto It = CurUnit.find(unsigned(U));
    // One new segment per instruction, however many of its operands cover the
    // unit: a tuple def and an implicit def of the same register are one write.
    bool OpensSegment = IsDef && UnitDef.lookup(unsigned(U)) != At + 1;
    if (It != CurUnit.end() && !OpensSegment)
      return It->second;
    if (IsDef)
      UnitDef[unsigned(U)] = At + 1;

    RegValue V;
    V.Size = 1; ///< A unit is 32 bits on every AMDGPU file.
    V.File = F;
    V.Files = 1u << unsigned(F);
    V.Phys = true;
    // Only the first segment can start above the region. Giving it the
    // incoming liveness merges the two into one range rather than
    // overstating: the incoming range ends where this one begins.
    if (!UnitHead.contains(unsigned(U)))
      V.LiveIn = Above.getBitVector().test(unsigned(U));
    Values.push_back(V);
    unsigned I = Values.size() - 1;
    CurUnit[unsigned(U)] = I;
    UnitHead.try_emplace(unsigned(U), I);
    return I;
  };

  DenseMap<Register, unsigned> Index;
  auto ValueOf = [&](Register Reg) -> unsigned {
    auto It = Index.find(Reg);
    if (It != Index.end())
      return It->second;

    const TargetRegisterClass *RC = MRI.getRegClass(Reg);
    RegValue V;
    V.Size = TRI.getRegSizeInBits(*RC) / 32;
    V.File = TRI.isSGPRClass(RC)   ? RegFile::SGPR
             : TRI.isAGPRClass(RC) ? RegFile::AGPR
                                   : RegFile::ArchVGPR;
    // Where it may end up *between* its instructions, which is wider than its
    // class says: the allocator asks getLargestLegalSuperClass before it
    // splits a live range, so a vreg_32 becomes an av_32 and gains the AGPR
    // file. At the instructions themselves the operand's own class still
    // binds, and that is recorded separately, per operand -- the difference
    // between the two is exactly how a value with a VGPR-only definition
    // survives a stretch of code where no VGPR is free.
    //
    // Pressure is still charged to V.File, because that is where
    // GCNRegPressure charges it and so what the driver's demand gate is
    // comparing against.
    if (const TargetRegisterClass *Widest =
            TRI.getLargestLegalSuperClass(RC, DAG.MF))
      RC = Widest;
    V.Files = filesOf(TRI, RC);
    for (const MachineInstr &User : MRI.use_nodbg_instructions(Reg)) {
      if (!InRegion.contains(&User)) {
        V.LiveOut = true;
        break;
      }
    }
    Values.push_back(V);
    unsigned I = Values.size() - 1;
    Index[Reg] = I;
    return I;
  };

  // Node numbers are source order, which is a topological order, so the first
  // definition seen is the one that opens the range.
  SmallVector<bool, 0> HasDef;
  auto Record = [&](unsigned I, unsigned V, bool IsDef) {
    if (HasDef.size() <= V)
      HasDef.resize(V + 1, false);
    if (IsDef) {
      if (!HasDef[V]) {
        HasDef[V] = true;
        Defs[I].push_back(V);
      }
      return;
    }
    // Dedupe within the instruction: two operands naming one value is one
    // read as far as liveness is concerned.
    if (!llvm::is_contained(Uses[I], V)) {
      Uses[I].push_back(V);
      ++Values[V].Uses;
    }
  };

  // Uses before definitions, within each instruction. An instruction that
  // reads and writes the same physical register -- a tied operand, an
  // accumulator -- must have its read attributed to the segment that ends
  // there and not to the one that starts there, and MachineInstr lists
  // definitions first.
  for (unsigned I = 0; I != N; ++I) {
    const MachineInstr *MI = DAG.SUnits[I].getInstr();
    for (bool DefPass : {false, true}) {
      for (const MachineOperand &MO : MI->operands()) {
        if (!MO.isReg() || !MO.getReg() || MO.isDef() != DefPass)
          continue;
        Register Reg = MO.getReg();
        if (Reg.isVirtual()) {
          unsigned V = ValueOf(Reg);
          Record(I, V, DefPass);
          // What this one operand will take, which may be narrower than what
          // the value as a whole may take. Operands with no class of their
          // own -- COPY, REG_SEQUENCE, inline asm -- say nothing and are
          // skipped, leaving the value looking freer than it is. Freer means
          // fewer values look stuck, which is the way a gate should err.
          if (const TargetRegisterClass *OpRC =
                  DAG.TII->getRegClass(MI->getDesc(), MO.getOperandNo()))
            Constraints.push_back({I, V, filesOf(TRI, OpRC)});
          continue;
        }
        // Reserved registers are not the allocator's to hand out, so how long
        // they stay live costs a schedule nothing.
        if (MRI.isReserved(Reg))
          continue;
        const TargetRegisterClass *RC = TRI.getPhysRegBaseClass(Reg.asMCReg());
        if (!RC)
          continue;
        RegFile F;
        if (TRI.isSGPRClass(RC))
          F = RegFile::SGPR;
        else if (TRI.isAGPRClass(RC))
          F = RegFile::AGPR;
        else if (TRI.isVGPRClass(RC))
          F = RegFile::ArchVGPR;
        else
          continue; // Not a file a schedule competes for.
        for (MCRegUnit U : TRI.regunits(Reg.asMCReg()))
          Record(I, UnitValueOf(U, F, DefPass, I), DefPass);
      }
    }
  }

  // The last segment of each unit is the one that can outlive the region.
  for (auto [U, V] : CurUnit)
    Values[V].LiveOut = Below.getBitVector().test(U);

  for (unsigned V = 0, E = Values.size(); V != E; ++V) {
    bool Defined = V < HasDef.size() && HasDef[V];
    if (!Values[V].Phys) {
      // A virtual register with no definition here came in alive from above.
      Values[V].LiveIn = !Defined;
    } else if (!Defined) {
      // A physical unit already knows whether it was live in, from the block.
      // But if it is read here without being written here then it was, and the
      // operands outrank the flag: believing otherwise would have pressureOf
      // close a range it never opened and wrap the running count.
      Values[V].LiveIn = true;
    }
  }
}

unsigned Budget::excessOf(const Pressure &P) const {
  unsigned Over = 0;
  auto Add = [&](unsigned Demand, unsigned Limit) {
    if (Demand > Limit)
      Over += Demand - Limit;
  };
  Add(P[RegFile::SGPR], (*this)[RegFile::SGPR]);
  if (UnifiedVGPR) {
    // One pool, so the only real limit is on the total; the per-file counts
    // are not limits here and charging them separately would invent an
    // overflow that does not exist.
    Add(P[RegFile::ArchVGPR] + P[RegFile::AGPR], VGPR);
  } else {
    Add(P[RegFile::ArchVGPR], (*this)[RegFile::ArchVGPR]);
    Add(P[RegFile::AGPR], (*this)[RegFile::AGPR]);
  }
  return Over;
}

Pressure Problem::pressureOf(ArrayRef<unsigned> Order) const {
  Pressure Cur;
  SmallVector<unsigned, 0> Left(Values.size());
  for (unsigned V = 0, E = Values.size(); V != E; ++V) {
    Left[V] = Values[V].Uses;
    if (Values[V].LiveIn)
      Cur[Values[V].File] += Values[V].Size;
  }

  Pressure Peak = Cur;
  for (unsigned Node : Order) {
    for (unsigned V : Defs[Node])
      if (!Values[V].LiveIn)
        Cur[Values[V].File] += Values[V].Size;

    // Sample with the definitions counted and before anything dies: a dead
    // def is live nowhere, but the register it writes still has to exist for
    // the cycle that writes it.
    Peak.takeMax(Cur);

    for (unsigned V : Defs[Node])
      if (!Values[V].LiveIn && !Values[V].LiveOut && Values[V].Uses == 0)
        Cur[Values[V].File] -= Values[V].Size;
    for (unsigned V : Uses[Node])
      if (--Left[V] == 0 && !Values[V].LiveOut)
        Cur[Values[V].File] -= Values[V].Size;
  }
  return Peak;
}

unsigned Problem::shortfallOf(ArrayRef<unsigned> Order) const {
  const unsigned N = Order.size();
  if (!N)
    return 0;

  // Where each value's range opens and closes, as positions in this order.
  // Same convention as pressureOf: counted from the instruction that defines
  // it through the one that last reads it, both ends included, so a dead def
  // still occupies its register for the one position that writes it.
  //
  // Both ends included on both ranges means a value that dies exactly where a
  // physical register is written is treated as overlapping it. That is one
  // position more overlap than really exists, and the direction to err in:
  // this only ever causes a schedule to be refused, never applied.
  const unsigned None = ~0u;
  SmallVector<unsigned, 0> Start(Values.size(), None);
  SmallVector<unsigned, 0> End(Values.size(), 0);
  for (unsigned V = 0, E = Values.size(); V != E; ++V) {
    if (Values[V].LiveIn)
      Start[V] = 0;
    if (Values[V].LiveOut)
      End[V] = N - 1;
  }
  for (unsigned P = 0; P != N; ++P) {
    for (unsigned V : Defs[Order[P]]) {
      Start[V] = std::min(Start[V], P);
      End[V] = std::max(End[V], P);
    }
    for (unsigned V : Uses[Order[P]]) {
      Start[V] = std::min(Start[V], P);
      End[V] = std::max(End[V], P);
    }
  }

  // Registers held at each position by something pre-coloured. These are the
  // ones nothing can be done about: a physical range cannot be evicted, and
  // spilling some other value does not free it.
  std::array<SmallVector<unsigned, 0>, unsigned(RegFile::NumFiles)> PhysAt;
  for (auto &A : PhysAt)
    A.assign(N + 1, 0);
  for (unsigned V = 0, E = Values.size(); V != E; ++V) {
    if (!Values[V].Phys || Start[V] == None)
      continue;
    auto &A = PhysAt[unsigned(Values[V].File)];
    A[Start[V]] += Values[V].Size;
    A[End[V] + 1] -= Values[V].Size;
  }
  for (auto &A : PhysAt) {
    unsigned Run = 0;
    for (unsigned &C : A) {
      Run += C;
      C = Run;
    }
  }

  // Colours left over once the pre-coloured ranges that overlap [Lo, Hi] have
  // taken theirs.
  //
  // Summed over the files, which overstates what a tuple can reach: it cannot
  // be split across two files, so one register free in each is not the same as
  // two free in one. Overstating availability understates the shortfall, and
  // that is the right way round -- a wrong zero here costs a schedule we could
  // have taken, a wrong non-zero costs a schedule we had no reason to refuse.
  auto AvailOver = [&](unsigned Files, unsigned Lo, unsigned Hi) {
    unsigned Avail = 0;
    for (unsigned F = 0; F != unsigned(RegFile::NumFiles); ++F) {
      if (!(Files & (1u << F)))
        continue;
      unsigned Worst = 0;
      for (unsigned P = Lo; P <= Hi; ++P)
        Worst = std::max(Worst, PhysAt[F][P]);
      if (Worst < Bud.Files[F])
        Avail += Bud.Files[F] - Worst;
    }
    return Avail;
  };

  // How short each value is, at its worst moment. Per value and not per
  // moment, because one value with nowhere to go is one failed allocation
  // however many instructions agree about it.
  SmallVector<unsigned, 0> ShortOf(Values.size(), 0);
  auto Note = [&](unsigned V, unsigned Avail) {
    if (Avail < Values[V].Size)
      ShortOf[V] = std::max(ShortOf[V], Values[V].Size - Avail);
  };

  for (unsigned V = 0, E = Values.size(); V != E; ++V) {
    const RegValue &RV = Values[V];
    if (RV.Phys || !RV.Size || Start[V] == None)
      continue;
    // Across the range, where the allocator may widen the value's class and
    // move it between files as often as it likes.
    Note(V, AvailOver(RV.Files, Start[V], End[V]));
  }

  // At each instruction, where it may not: the operand's class is what the
  // encoding can name. This is the half that catches a result defined where
  // its own file is full -- splitting can rescue the rest of a live range,
  // but nothing rescues the definition itself.
  SmallVector<unsigned, 0> Pos(size(), 0);
  for (unsigned P = 0; P != N; ++P)
    Pos[Order[P]] = P;
  for (const Constraint &C : Constraints)
    Note(C.Value, AvailOver(C.Files, Pos[C.Node], Pos[C.Node]));

  unsigned Short = 0;
  for (unsigned V = 0, E = Values.size(); V != E; ++V) {
    if (!ShortOf[V])
      continue;
    DEBUG_WITH_TYPE("amdgpu-optsched-regs", {
      dbgs() << "  value " << V << " size " << Values[V].Size << " files 0x"
             << Twine::utohexstr(Values[V].Files) << " live [" << Start[V]
             << "," << End[V] << "] short " << ShortOf[V] << "\n";
    });
    Short += ShortOf[V];
  }
  return Short;
}

Cost Problem::costOf(ArrayRef<unsigned> Order) const {
  SmallVector<unsigned, 0> Cycles;
  timeOrder(Order, Cycles);
  Pressure Peak = pressureOf(Order);
  return Cost{Bud.excessOf(Peak), makespanOf(Cycles), Peak.sum()};
}

void Problem::computeHeights() {
  const unsigned N = size();
  Height.assign(N, 0);
  // Node numbers in a ScheduleDAG are a topological order, so one backward
  // sweep suffices. Asserting it here rather than trusting it: a renumbering
  // upstream would otherwise silently weaken the search's pruning bound into
  // an unsound one.
  for (const Edge &E : Edges)
    assert(E.Pred < E.Succ && "SUnit numbering is not topological");

  for (unsigned I = N; I-- > 0;) {
    unsigned H = NodeLatency[I];
    for (unsigned EI : SuccEdges[I])
      H = std::max(H, Edges[EI].Latency + Height[Edges[EI].Succ]);
    Height[I] = H;
  }
}

void Problem::timeOrder(ArrayRef<unsigned> Order,
                        SmallVectorImpl<unsigned> &Cycles) const {
  Cycles.assign(size(), 0);
  // When each functional unit is free again. Indexed by the target's own
  // resource numbering, so this is a handful of entries and most stay zero.
  SmallVector<unsigned, 16> PipeFree(NumPipes, 0);
  unsigned Last = 0;
  for (unsigned Node : Order) {
    unsigned T = Last + 1; // one instruction per cycle
    for (unsigned EI : predEdgesOf(Node)) {
      const Edge &E = edges()[EI];
      T = std::max(T, Cycles[E.Pred] + E.Latency);
    }
    // And the unit it needs has to be free, which is a constraint between
    // instructions that have no dependence at all. This is the only place a
    // run of independent MFMAs costs more than a run of independent adds, and
    // the only reason threading other work between them is worth anything.
    for (const PipeUse &U : Pipes[Node])
      T = std::max(T, PipeFree[U.Pipe]);
    for (const PipeUse &U : Pipes[Node])
      PipeFree[U.Pipe] = T + U.Cycles;
    Cycles[Node] = T;
    Last = T;
  }
}

unsigned Problem::makespanOf(ArrayRef<unsigned> Cycles) const {
  unsigned M = 0;
  for (unsigned I = 0, N = size(); I != N; ++I)
    M = std::max(M, Cycles[I] + latencyOf(I));
  return M;
}

void Problem::sourceOrder(SmallVectorImpl<unsigned> &Order) const {
  Order.resize(size());
  for (unsigned I = 0, N = size(); I != N; ++I)
    Order[I] = I;
}

//===----------------------------------------------------------------------===//
// Verifier
//===----------------------------------------------------------------------===//

static Error fail(const Twine &Msg) {
  return make_error<StringError>(Msg.str(), inconvertibleErrorCode());
}

Error llvm::AMDGPUOptSched::verify(const Problem &P, const Answer &A) {
  const unsigned N = P.size();

  if (A.St == Status::Optimal && A.Proof.empty())
    return fail("claimed optimal without a proof: a minimum needs a reason it "
                "cannot be beaten. Use UpperBound for a search that stopped "
                "early");

  if (A.Order.size() != N)
    return fail("order has " + Twine(A.Order.size()) +
                " entries, the region "
                "has " +
                Twine(N));

  SmallVector<unsigned, 0> Pos(N, ~0u);
  for (unsigned I = 0; I != N; ++I) {
    unsigned Node = A.Order[I];
    if (Node >= N)
      return fail("order contains node " + Twine(Node) + ", out of range");
    if (Pos[Node] != ~0u)
      return fail("node " + Twine(Node) + " appears twice in the order");
    Pos[Node] = I;
  }

  for (const Edge &E : P.edges())
    if (Pos[E.Pred] >= Pos[E.Succ])
      return fail("edge " + Twine(E.Pred) + "->" + Twine(E.Succ) + " (" +
                  kindName(E.Kind) + ") violated: predecessor at position " +
                  Twine(Pos[E.Pred]) + ", successor at " + Twine(Pos[E.Succ]));

  if (A.Cycles.empty()) {
    if (A.Makespan)
      return fail("makespan claimed without cycles to replay it from");
    return Error::success();
  }

  if (A.Cycles.size() != N)
    return fail("cycles given for " + Twine(A.Cycles.size()) + " of " +
                Twine(N) + " nodes");

  // Latency before single-issue on purpose: when a solver collapses a
  // dependent pair into one cycle both rules break, and "latency" is the
  // message that points at the real mistake.
  for (const Edge &E : P.edges())
    if (A.Cycles[E.Succ] < A.Cycles[E.Pred] + E.Latency)
      return fail("latency violated on " + Twine(E.Pred) + "->" +
                  Twine(E.Succ) + " (" + kindName(E.Kind) + "): issues at " +
                  Twine(A.Cycles[E.Pred]) + " with latency " +
                  Twine(E.Latency) + ", but successor issues at " +
                  Twine(A.Cycles[E.Succ]));

  for (unsigned I = 1; I < N; ++I)
    if (A.Cycles[A.Order[I]] <= A.Cycles[A.Order[I - 1]])
      return fail("nodes " + Twine(A.Order[I - 1]) + " and " +
                  Twine(A.Order[I]) +
                  " do not issue in increasing cycles. A "
                  "wave issues one instruction per cycle; a stall is a gap, "
                  "not co-issue");

  // One instruction per functional unit at a time. This is the rule a solver
  // breaks by accident rather than by bug: a cost function that counts only
  // dependences sees no reason not to put every MFMA in the region back to
  // back, and the schedule it then reports is not short, it is mistimed.
  {
    SmallVector<unsigned, 16> PipeFree(P.numPipes(), 0);
    for (unsigned Node : A.Order) {
      for (const PipeUse &U : P.pipesOf(Node)) {
        if (A.Cycles[Node] < PipeFree[U.Pipe])
          return fail("node " + Twine(Node) + " issues at cycle " +
                      Twine(A.Cycles[Node]) +
                      " but the unit it needs is busy until " +
                      Twine(PipeFree[U.Pipe]));
        PipeFree[U.Pipe] = A.Cycles[Node] + U.Cycles;
      }
    }
  }

  unsigned Measured = P.makespanOf(A.Cycles);
  if (A.Makespan != Measured)
    return fail("makespan claimed " + Twine(A.Makespan) +
                ", replaying the cycles gives " + Twine(Measured));

  return Error::success();
}

Solver::~Solver() = default;
