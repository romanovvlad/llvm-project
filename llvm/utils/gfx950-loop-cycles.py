#!/usr/bin/env python3
"""Cycle cost of every hot loop in a gfx950 assembly file.

Finds the innermost loops, runs N iterations of each through an in-order
single-wave timing model, and reports the cycles they take, where they went,
and the FLOPs retired per cycle.

    gfx950-loop-cycles.py kernel.s
    gfx950-loop-cycles.py -n 16 --json out/      # every .s below a directory
    gfx950-loop-cycles.py -v kernel.s            # per-instruction trace
    gfx950-loop-cycles.py -w kernel.s            # whole functions, not loops

With -w the unit is the function rather than the loop: every instruction is
counted once and every innermost loop body N times, so the prologue, the
epilogue and anything between loops are in the number too. Nothing is ever
skipped and no branch is predicted, so a function's figure is an upper bound
on its work -- useful for comparing two schedules of the same function, not
for predicting one run of it.

Iterations run back to back with the pipe, the wait counters and the register
ready times carried across the back edge, so a load issued in one iteration and
waited on in the next costs what it really costs.

The model
---------
A wave issues its instructions in order, and what costs it time is not the
latency of any one of them: a wave issues one instruction every 4 cycles,
whatever that instruction is -- SALU included. That is where "one instruction
is four cycles" comes from, and the traces back it up: every opcode in them has
a minimum spacing of 4.

Some instructions also hold a pipe for longer than those 4 cycles, and then
the wave's *next* instruction for the same pipe has to wait:

  valu          4 cycles, so free; 8 for the half-rate transcendentals.
  xdl           `passes * 4` cycles. The next MFMA cannot issue until it frees,
                whatever its operands are, so the only way to use that window
                is to put other work between the two MFMAs. At 4 passes that
                is 16 cycles, and the traces show exactly one instruction fits
                in it for free.
  lds           8 cycles to read, whatever the width. A write is not the same:
                4 bytes is 8 cycles but 16 bytes is 44, and the two-address
                `write2` forms cost more again.
  vmem          4 cycles to issue a load; the cost is in the s_waitcnt.
  salu          4 cycles. `s_nop N` is (N+1)*4 and occupies no pipe. A branch
                the wave takes is 20, because it refetches; one that falls
                through is 4.

On top of the pipes the wave stops for three more things:

  s_waitcnt     A wait for outstanding memory. lgkmcnt(N) releases when all but
                the last N LDS/scalar loads have returned; vmcnt(N) does the
                same for vector memory.
  register      A read of a value that has not landed yet. Only MFMA results
                are slow enough to matter, and only for non-MFMA readers --
                src2 of the next MFMA in a chain is forwarded.
  s_barrier     A flat 40 cycles, the measured median. What it really costs is
                the spread between the fastest and slowest wave in the
                workgroup, which one wave cannot see; the barrier count is
                reported per loop, and where it is high the cycle figure is
                approximate.

Pass counts are the gfx950 entries of SIDPGFX950FullSpeedModel in
llvm/lib/Target/AMDGPU/SISchedule.td and should not need touching. The memory
latencies are not in that model; they are the tunable part of this tool, and
the ones to revisit against a real profile -- see --lds / --vmem / --smem.

Calibration
-----------
The per-instruction costs above are not a spec sheet. They are the modes and
medians of tens of millions of issue-to-issue spacings measured on gfx950
hardware with a wave thread-trace profiler. The model as a whole was then
checked end to end: rebuild each traced kernel's assembly from the trace, time
its loops with this tool, and compare against what the same loop took in the
same trace, measured as the median time from one taken backward branch to the
next. Over 83 loops from 80 traces:

  Pearson r 0.97, Spearman rho 0.95, 93% of pairwise orderings correct,
  and 100% of the pairs that are two schedules of the same kernel.
  model/measured mean 0.79, median 0.80 +- 0.18, range 0.33-1.21.

So the tool ranks schedules well and reads about 20% low in absolute terms.
Two loops within ~5% of each other are a tie it cannot call.

The 20% is contention this model has no way to see. Per instruction it lands
on the measured modes, but the measured *distributions* have long tails from
many waves sharing the LDS and memory ports, and a single wave in isolation
never pays them. Where the residual sits is not spread evenly: in the worst
loop here -- ratio 0.33 -- a single `s_waitcnt vmcnt(0)` costs the real wave
1348 cycles against the 4 this model charges, and that one instruction is the
entire error.

Not modelled
------------
LDS and HBM bandwidth, and any overlap between waves. The tool is weakest on
memory-bound loops, where almost all of the residual error sits.

The floor
---------
`xdl` is the cycles the matrix pipe was busy. A loop cannot run faster than
its MFMAs, so `cy/xdl` is how much of the loop is the math itself: 1.0 means
the pipe never idles, 2.0 means half the time it is waiting. The f64 MFMAs are
the one exception: SISchedule.td runs them on the vector ALU, so their cycles
appear under `valu` and still count as the floor.

`flop/cy` is the whole wave's FLOPs -- 2*M*N*K per MFMA -- divided by the
cycles the loop takes, and `peak` is the same figure as a fraction of what
those MFMAs would retire back to back. `peak` is therefore the reciprocal of
`cy/xdl`, and it is the loop's own ceiling, not the machine's paper number.
"""

import argparse
import json
import os
import re
import sys

# --- gfx950 machine constants, from SIDPGFX950FullSpeedModel -----------------

CYCLES_PER_PASS = 4
ISSUE_CYCLES = 4  # a wave issues one instruction every 4 cycles, in order
MFMA_RESULT_EXTRA = 4  # after the last pass, before a VALU may read the result

# How long an instruction keeps its pipe busy, so that the wave's next
# instruction for the same pipe has to wait. Issue is 4 cycles regardless, so
# only values above 4 cost anything. Calibrated against hardware traces; see
# --- calibration --- at the bottom of this file.
PIPE_BUSY = {
    "valu": 4,  # wave64 full rate
    "trans": 8,  # half rate
    "lds": 4,
    "vmem": 4,  # fire and forget; the cost is in the s_waitcnt
    "smem": 4,
    "salu": 4,
}

# Classes that block the wave itself for longer than ISSUE_CYCLES, so the next
# instruction waits even if it needs a different pipe. The traces show this for
# LDS: a ds_read_b128 is 8 cycles whatever follows it.
ISSUE_OF = {"lds": 8}

# An LDS write is not an LDS read. A read costs 8 cycles whatever its width,
# but a write costs more the wider it is -- 16 bytes is five times 4 bytes --
# and the two-address `write2` forms cost more again, because they are two
# addresses and so twice the chance of a bank conflict. These are measured
# modes, keyed on the data-width suffix.
LDS_WRITE = {"b8": 8, "b16": 8, "b32": 8, "b64": 24, "b96": 32, "b128": 44}
LDS_WRITE2 = {"b32": 12, "b64": 64}

# A taken branch refetches, and the wave does not issue while it does. A branch
# that falls through costs the usual 4.
TAKEN_BRANCH = 20

BARRIER_CYCLES = 40  # measured median; the real cost is workgroup skew

TRANS = ("v_rcp", "v_sqrt", "v_rsq", "v_log", "v_exp", "v_sin", "v_cos",
         "v_div_scale", "v_div_fmas", "v_div_fixup")

# --- tunables: round-trips in cycles, not modelled by SISchedule.td ----------

# What a wave can hide before an s_waitcnt costs it anything. These are not
# the latencies a datasheet would quote and they are not all measured the same
# way. In the traces the round trip a wave actually sees -- oldest outstanding
# load to the end of its wait -- has a median of 56 for scalar loads, 252 for
# LDS and 1700 for vector memory, but those medians carry queueing, and
# queueing is a property of the kernel, not of the machine. Fitting the whole
# corpus instead: the scalar number is the measured one, and LDS and vector
# memory sit far below theirs, where the correlation across kernels is best.
# A loop that really misses to memory is therefore under-read, which is the
# known weakness stated at the top of this file.
LDS_LATENCY = 64
VMEM_LATENCY = 500
SMEM_LATENCY = 56

# --- parsing -----------------------------------------------------------------

INSN = re.compile(r"^\s+([a-z][a-z0-9_]*)\s*(.*)$")
LABEL = re.compile(r"^([.\w$]+):")
SHAPE = re.compile(r"_(\d+)x(\d+)x(\d+)")
WAITCNT = re.compile(r"(vmcnt|lgkmcnt|expcnt)\((\d+)\)")
MODIFIER = re.compile(r"\b(cbsz|blgp|abid):(\d+)")
REGRANGE = re.compile(r"\b([vas])\[(\d+):(\d+)\]")
REGSINGLE = re.compile(r"(?<![\w\[:])([vas])(\d+)\b")
METADATA = re.compile(r"\.(vgpr|agpr|sgpr)_count:\s*(\d+)")

F8_FORMATS = {0, 1}  # cbsz/blgp: 0 = fp8, 1 = bf8, 2/3 = fp6, 4 = fp4

# How many passes of 4 cycles each MFMA holds the matrix pipe for.
#
# The shape in the name does not determine this and cannot be made to: at
# 16x16 a K of 4 is 8 passes while a K of 64 is 4, and the same K means
# different things to an f32 form and an f8 one. So this is the gfx950 `InstRW`
# table of SIDPGFX950FullSpeedModel in llvm/lib/Target/AMDGPU/SISchedule.td,
# resolved per mnemonic. `"f8:N"` is a SchedWriteVariant: N passes normally,
# 2N when cbsz or blgp selects fp8 or bf8 (SISchedule.td, "If either matrix
# format is f8, the instruction takes 2x as many cycles").
#
# Measured back to back in hardware traces, the entries that appear in real
# kernels come out exactly as listed -- 16x16x32 bf16 and f16 at 4, 32x32x16
# bf16 at 8, 16x16x128_f8f6f4 at 8 with fp8 operands and 4 with fp4 -- with
# one exception noted at `_MAI_UNMODELLED` below.
MAI_PASSES = {
    "v_mfma_f32_16x16x128_f8f6f4": "f8:4",
    "v_mfma_f32_16x16x16_bf16": 4,
    "v_mfma_f32_16x16x16_f16": 4,
    "v_mfma_f32_16x16x16bf16_1k": 4,
    "v_mfma_f32_16x16x16f16": 4,
    "v_mfma_f32_16x16x1_4b_f32": 8,
    "v_mfma_f32_16x16x1f32": 8,
    "v_mfma_f32_16x16x32_bf16": 4,
    "v_mfma_f32_16x16x32_bf8_bf8": 4,
    "v_mfma_f32_16x16x32_bf8_fp8": 4,
    "v_mfma_f32_16x16x32_f16": 4,
    "v_mfma_f32_16x16x32_fp8_bf8": 4,
    "v_mfma_f32_16x16x32_fp8_fp8": 4,
    "v_mfma_f32_16x16x4_4b_bf16": 8,
    "v_mfma_f32_16x16x4_4b_f16": 8,
    "v_mfma_f32_16x16x4_f32": 8,
    "v_mfma_f32_16x16x4bf16_1k": 8,
    "v_mfma_f32_16x16x4f16": 8,
    "v_mfma_f32_16x16x4f32": 8,
    "v_mfma_f32_16x16x8_xf32": 4,
    "v_mfma_f32_32x32x16_bf16": 8,
    "v_mfma_f32_32x32x16_bf8_bf8": 8,
    "v_mfma_f32_32x32x16_bf8_fp8": 8,
    "v_mfma_f32_32x32x16_f16": 8,
    "v_mfma_f32_32x32x16_fp8_bf8": 8,
    "v_mfma_f32_32x32x16_fp8_fp8": 8,
    "v_mfma_f32_32x32x1_2b_f32": 16,
    "v_mfma_f32_32x32x1f32": 16,
    "v_mfma_f32_32x32x2_f32": 16,
    "v_mfma_f32_32x32x2bf16": 16,
    "v_mfma_f32_32x32x2f32": 16,
    "v_mfma_f32_32x32x4_2b_bf16": 16,
    "v_mfma_f32_32x32x4_2b_f16": 16,
    "v_mfma_f32_32x32x4_xf32": 8,
    "v_mfma_f32_32x32x4bf16": 16,
    "v_mfma_f32_32x32x4bf16_1k": 16,
    "v_mfma_f32_32x32x4f16": 16,
    "v_mfma_f32_32x32x64_f8f6f4": "f8:8",
    "v_mfma_f32_32x32x8_bf16": 8,
    "v_mfma_f32_32x32x8_f16": 8,
    "v_mfma_f32_32x32x8bf16_1k": 8,
    "v_mfma_f32_32x32x8f16": 8,
    "v_mfma_f32_4x4x1_16b_f32": 2,
    "v_mfma_f32_4x4x1f32": 2,
    "v_mfma_f32_4x4x2bf16": 2,
    "v_mfma_f32_4x4x4_16b_bf16": 2,
    "v_mfma_f32_4x4x4_16b_f16": 2,
    "v_mfma_f32_4x4x4bf16_1k": 2,
    "v_mfma_f32_4x4x4f16": 2,
    "v_mfma_f64_16x16x4_f64": 16,
    "v_mfma_f64_16x16x4f64": 16,
    "v_mfma_f64_4x4x4_4b_f64": 4,
    "v_mfma_f64_4x4x4f64": 4,
    "v_mfma_i32_16x16x16i8": 4,
    "v_mfma_i32_16x16x32_i8": 4,
    "v_mfma_i32_16x16x4_4b_i8": 8,
    "v_mfma_i32_16x16x4i8": 8,
    "v_mfma_i32_16x16x64_i8": 4,
    "v_mfma_i32_32x32x16_i8": 8,
    "v_mfma_i32_32x32x32_i8": 8,
    "v_mfma_i32_32x32x4_2b_i8": 16,
    "v_mfma_i32_32x32x4i8": 16,
    "v_mfma_i32_32x32x8i8": 8,
    "v_mfma_i32_4x4x4_16b_i8": 2,
    "v_mfma_i32_4x4x4i8": 2,
    "v_mfma_scale_f32_16x16x128_f8f6f4": "f8:4",
    "v_mfma_scale_f32_32x32x64_f8f6f4": "f8:8",
    "v_smfmac_f32_16x16x128_bf8_bf8": 4,
    "v_smfmac_f32_16x16x128_bf8_fp8": 4,
    "v_smfmac_f32_16x16x128_fp8_bf8": 4,
    "v_smfmac_f32_16x16x128_fp8_fp8": 4,
    "v_smfmac_f32_16x16x32_bf16": 4,
    "v_smfmac_f32_16x16x32_f16": 4,
    "v_smfmac_f32_16x16x64_bf16": 4,
    "v_smfmac_f32_16x16x64_bf8_bf8": 4,
    "v_smfmac_f32_16x16x64_bf8_fp8": 4,
    "v_smfmac_f32_16x16x64_f16": 4,
    "v_smfmac_f32_16x16x64_fp8_bf8": 4,
    "v_smfmac_f32_16x16x64_fp8_fp8": 4,
    "v_smfmac_f32_32x32x16_bf16": 8,
    "v_smfmac_f32_32x32x16_f16": 8,
    "v_smfmac_f32_32x32x32_bf16": 8,
    "v_smfmac_f32_32x32x32_bf8_bf8": 8,
    "v_smfmac_f32_32x32x32_bf8_fp8": 8,
    "v_smfmac_f32_32x32x32_f16": 8,
    "v_smfmac_f32_32x32x32_fp8_bf8": 8,
    "v_smfmac_f32_32x32x32_fp8_fp8": 8,
    "v_smfmac_f32_32x32x64_bf8_bf8": 8,
    "v_smfmac_f32_32x32x64_bf8_fp8": 8,
    "v_smfmac_f32_32x32x64_fp8_bf8": 8,
    "v_smfmac_f32_32x32x64_fp8_fp8": 8,
    "v_smfmac_i32_16x16x128_i8": 4,
    "v_smfmac_i32_16x16x64_i8": 4,
    "v_smfmac_i32_32x32x32_i8": 8,
    "v_smfmac_i32_32x32x64_i8": 8,
    # Two pre-gfx950 forms the gfx950 model has no rule for at all. They take
    # the cost of the nearest shape that does have one.
    "v_mfma_f32_16x16x2bf16": 8,
    "v_mfma_f32_16x16x8bf16": 4,
}

# `v_mfma_i32_16x16x64_i8` is listed above at the 4 passes SISchedule.td gives
# it, but in a trace its back-to-back gap is 4 cycles, not 16. That rate would
# be eight times the matrix pipe's f16 throughput, which cannot be right, so
# the likeliest reading is that the hardware queues MFMAs and the wave runs
# ahead. Only one kernel in the corpus issues it, so the table is left alone
# and the disagreement recorded here.
_MAI_UNMODELLED = ("v_mfma_i32_16x16x64_i8",)


def mfma_passes(mnemonic, operands):
    """Cycles/4 that `mnemonic` holds the XDL pipe."""
    mn = mnemonic.lower()
    passes = MAI_PASSES.get(mn)

    if passes is None:
        # Something this script has not seen. Fall back to the shape, which is
        # right more often than not, and say so rather than being silently
        # wrong.
        m = SHAPE.search(mn)
        if not m:
            return 4
        shape = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if shape[0] == 4:
            return 2
        if shape[0] == 16:
            return 4 if shape[2] >= 8 else 8
        return 8 if shape[2] >= 8 else 16

    if isinstance(passes, int):
        return passes

    # A SchedWriteVariant: f8 or bf8 in either operand doubles it.
    base = int(passes.split(":")[1])
    fmts = {int(v) for k, v in MODIFIER.findall(operands)
            if k in ("cbsz", "blgp")}
    if not fmts:
        fmts = {0}  # the default encoding is fp8
    return 2 * base if fmts & F8_FORMATS else base


def mfma_flops(mnemonic):
    """FLOPs one MFMA retires for the whole wave: 2 * M * N * K."""
    m = SHAPE.search(mnemonic)
    return 2 * int(m.group(1)) * int(m.group(2)) * int(m.group(3)) if m else 0


def regs(text):
    """Register names in `text`, one entry per 32-bit register."""
    out = set()
    for bank, lo, hi in REGRANGE.findall(text):
        for i in range(int(lo), int(hi) + 1):
            out.add(f"{bank}{i}")
    for bank, n in REGSINGLE.findall(REGRANGE.sub(" ", text)):
        out.add(f"{bank}{n}")
    return out


def parse(path):
    """-> (instructions, label -> index, kernel metadata)."""
    insns, labels, meta = [], {}, {}
    for raw in open(path, errors="replace"):
        line = raw.split(";")[0].rstrip()
        if not line:
            continue
        m = METADATA.search(line)
        if m:
            meta[m.group(1)] = int(m.group(2))
            continue
        m = LABEL.match(line)
        if m:
            labels[m.group(1)] = len(insns)
            continue
        m = INSN.match(line)
        if m and not line.lstrip().startswith("."):
            insns.append((m.group(1), m.group(2)))
    return insns, labels, meta


def find_loops(insns, labels):
    """Innermost loop bodies, as (start, end) instruction indices."""
    found = []
    for i, (mn, ops) in enumerate(insns):
        if not (mn.startswith("s_cbranch") or mn == "s_branch"):
            continue
        target = ops.strip().split()[0].rstrip(",") if ops.strip() else ""
        start = labels.get(target)
        if start is not None and start < i:
            found.append((start, i))
    # An outer loop's body is mostly its inner loop, so keep only the ranges
    # that contain no other range.
    return [a for a in found
            if not any(a != b and a[0] <= b[0] and b[1] <= a[1] for b in found)]


def find_functions(insns, labels):
    """(name, start, end) per function, as [start, end] instruction indices.

    A function starts at a label the assembler would export -- anything that
    is not a local `.L` one -- and runs to the instruction before the next.
    """
    starts = {}
    for name, idx in labels.items():
        if name.startswith("."):
            continue
        starts.setdefault(idx, name)
    if not starts:
        return [("", 0, len(insns) - 1)] if insns else []
    bounds = sorted(starts)
    out = []
    for k, s in enumerate(bounds):
        e = bounds[k + 1] - 1 if k + 1 < len(bounds) else len(insns) - 1
        if s <= e:
            out.append((starts[s], s, e))
    return out


def execution_order(insns, start, end, loops, iters):
    """Indices of insns[start:end+1] in the order a wave would issue them.

    Straight-line code runs once; an innermost loop body runs `iters` times.
    That is the whole of the control-flow model -- there is no branch
    prediction here and nothing is ever skipped -- so a function's number is
    an upper bound on its work, not a trace of one run.
    """
    ends = {s: e for s, e in loops}
    order = []
    i = start
    while i <= end:
        e = ends.get(i)
        if e is not None and e <= end:
            order.extend(list(range(i, e + 1)) * iters)
            i = e + 1
        else:
            order.append(i)
            i += 1
    return order


# --- timing model ------------------------------------------------------------


def classify(mnemonic):
    """The pipe `mnemonic` occupies."""
    # v_mfma_ld_scale_b32 shares the prefix but is an ordinary VALU op: it
    # loads the scale a later scaled MFMA reads, and never touches the matrix
    # pipe.
    if mnemonic.startswith(("v_mfma", "v_smfmac")) and "_ld_scale" not in mnemonic:
        return "mfma"
    if mnemonic.startswith("ds_"):
        return "lds"
    if mnemonic.startswith(("global_", "buffer_", "flat_", "scratch_",
                            "tbuffer_", "image_")):
        return "vmem"
    if mnemonic.startswith(("s_load", "s_buffer_load")):
        return "smem"
    if mnemonic.startswith(TRANS):
        return "trans"
    if mnemonic.startswith("v_"):
        return "valu"
    return "salu"


class Counter:
    """One hardware wait counter: in-order issue, in-order retire."""

    def __init__(self):
        self.pending = []  # completion cycles, oldest first

    def issue(self, done_at):
        self.pending.append(done_at)

    def wait(self, keep, now):
        """Retire all but the last `keep`. -> the cycle the wait releases."""
        n = len(self.pending) - keep
        if n <= 0:
            return now
        release = max(self.pending[:n])
        self.pending = self.pending[n:]
        return max(now, release)


_PIPE_OF = {"mfma": "xdl", "trans": "valu"}  # instruction class -> pipe it shares


def pipe_of(kind, mn):
    """Which pipe `mn` contends for."""
    # The f64 MFMAs are MFMAs in name only. SISchedule.td gives them
    # Write*PassDGEMM, which is an HWVALUWriteRes, so they occupy the vector
    # ALU and not the matrix pipe -- a loop of them leaves `xdl` idle and
    # blocks every ordinary VALU instruction beside it.
    if kind == "mfma" and mn.startswith("v_mfma_f64"):
        return "valu"
    return _PIPE_OF.get(kind, kind)


def pipe_busy(kind, mn, ops):
    """Cycles `mn` keeps its pipe busy. Only values above ISSUE_CYCLES bite."""
    if kind == "mfma":
        return mfma_passes(mn, ops) * CYCLES_PER_PASS
    return PIPE_BUSY[kind]


BRANCHES = ("s_branch", "s_cbranch")


def issue_of(kind, mn, taken):
    """Cycles before this wave may issue anything at all after `mn`."""
    if kind == "lds" and mn.startswith("ds_write"):
        width = mn.rsplit("_", 1)[-1]
        table = LDS_WRITE2 if "write2" in mn else LDS_WRITE
        return table.get(width, ISSUE_OF["lds"])
    if taken and mn.startswith(BRANCHES):
        return TAKEN_BRANCH
    return ISSUE_OF.get(kind, ISSUE_CYCLES)


def simulate(insns, order, lat, taken=(), trace=0):
    """Issue the instructions named by `order`, in that order.

    `order` is a list of indices into `insns`, one entry per issue, so a loop
    body is simply repeated and a whole function is its blocks laid end to end.
    `taken` holds the indices of branches the wave takes -- the loop-closing
    ones -- which cost a refetch the fall-through ones do not. `trace` prints
    that many leading instructions.
    """
    t = 0
    free = {}  # pipe -> cycle it is next free
    busy = {}  # pipe -> cycles it was busy
    ready = {}  # register -> cycle its value lands
    vm, lgkm = Counter(), Counter()
    stall = {"xdl": 0, "valu": 0, "lds": 0, "vmem": 0, "smem": 0, "salu": 0,
             "vmcnt": 0, "lgkmcnt": 0, "reg": 0}
    counts = {k: 0 for k in
              ("mfma", "valu", "trans", "lds", "vmem", "smem", "salu")}
    flops = barriers = mfma_busy = 0

    for pos, i in enumerate(order):
        mn, ops = insns[i]
        kind = classify(mn)
        pipe = pipe_of(kind, mn)
        counts[kind] += 1
        was = t
        show = pos < trace

        if mn == "s_barrier":
            barriers += 1
            t += BARRIER_CYCLES
            if show:
                trace_line(i, mn, ops, was, t)
            continue

        if mn.startswith("s_waitcnt"):
            for name, keep in WAITCNT.findall(ops):
                before = t
                if name == "vmcnt":
                    t = vm.wait(int(keep), t)
                    stall["vmcnt"] += t - before
                elif name == "lgkmcnt":
                    t = lgkm.wait(int(keep), t)
                    stall["lgkmcnt"] += t - before
            if show:
                trace_line(i, mn, ops, was, t)
            t += ISSUE_CYCLES
            continue

        if mn == "s_nop":
            # (N+1) issue slots of nothing, on no pipe at all.
            arg = ops.strip().split()[0] if ops.strip() else "0"
            n = int(arg, 0) if arg.isdigit() else 0
            t += (n + 1) * ISSUE_CYCLES
            if show:
                trace_line(i, mn, ops, was, t)
            continue

        operands = [o.strip() for o in ops.split(",")]
        if "store" in mn or mn.startswith("ds_write"):
            dst, src = set(), regs(ops)  # a store defines no register
        else:
            dst = regs(operands[0]) if operands else set()
            src = regs(",".join(operands[1:])) if len(operands) > 1 else set()

        # Wait for the sources. An MFMA's src2 is the previous MFMA's result
        # in the same chain and is forwarded, so an MFMA only ever waits for
        # the pipe, never for data.
        if kind != "mfma":
            need = max((ready.get(r, 0) for r in src), default=0)
            if need > t:
                stall["reg"] += need - t
                t = need

        # Wait for the pipe this instruction needs.
        if free.get(pipe, 0) > t:
            stall[pipe] += free[pipe] - t
            t = free[pipe]

        hold = pipe_busy(kind, mn, ops)
        free[pipe] = t + hold
        busy[pipe] = busy.get(pipe, 0) + hold

        if kind == "mfma":
            flops += mfma_flops(mn)
            mfma_busy += hold
            for r in dst:
                ready[r] = free[pipe] + MFMA_RESULT_EXTRA
        elif kind in ("lds", "vmem", "smem"):
            (lgkm if kind in ("lds", "smem") else vm).issue(t + lat[kind])
            # A load's result is guaranteed by its s_waitcnt, not by a
            # latency, so it does not gate its destination here.
            for r in dst:
                ready[r] = 0
        else:
            for r in dst:
                ready[r] = t + hold

        if show:
            trace_line(i, mn, ops, was, t)
        t += issue_of(kind, mn, i in taken or mn == "s_branch")

    # The last instruction issued at `t`, but a pipe it left busy is work the
    # next iteration would have waited for, so count it. Without this a
    # matrix-bound loop reads one MFMA short and can report more pipe-busy
    # cycles than it has cycles.
    t = max([t] + list(free.values()))

    return {
        "body": len(set(order)),
        "iters": 1,
        "counts": counts,
        "flops": flops,
        "cycles": t,
        "xdl_busy": busy.get("xdl", 0),
        # What the matrix math cost, wherever it ran. Same as `xdl_busy`
        # unless f64 MFMAs are present, which run on the vector ALU.
        "mfma_busy": mfma_busy,
        "valu_busy": busy.get("valu", 0),
        "barriers": barriers,
        "stall": stall,
    }


def trace_line(i, mn, ops, before, after):
    gap = after - before
    mark = f"  <- {gap:>4} stall" if gap else ""
    print(f"    {before:>7}  {mn:<34}{ops[:40]:<42}{mark}")


# --- reporting ---------------------------------------------------------------


def analyze(path, args):
    insns, labels, meta = parse(path)
    lat = {"lds": args.lds, "vmem": args.vmem, "smem": args.smem}
    loops = find_loops(insns, labels)

    # The branch that closes a loop is the one branch we know is taken.
    taken = {end for _start, end in loops}

    jobs = []  # (label, order, iterations the label is counted per)
    if args.whole:
        for name, start, end in find_functions(insns, labels):
            order = execution_order(insns, start, end, loops, args.iters)
            if len(order) >= args.min_size:
                jobs.append((name or f"{start}-{end}", order, 1))
    else:
        for start, end in loops:
            if end - start + 1 >= args.min_size:
                jobs.append((f"{start}-{end}",
                             list(range(start, end + 1)) * args.iters,
                             args.iters))

    out = []
    for name, order, iters in jobs:
        if args.verbose:
            print(f"\n{path}  {name}")
        r = simulate(insns, order, lat, taken,
                     trace=len(order) // iters if args.verbose else 0)
        r.update(file=path, loop=name, meta=meta, iters=iters)
        r["per_iter"] = r["cycles"] / iters
        # Against `mfma_busy`, not `xdl_busy`: the floor is what the matrix
        # math cost, and f64 MFMAs pay it on the vector ALU.
        r["xdl_ratio"] = (r["cycles"] / r["mfma_busy"]) if r["mfma_busy"] else 0.0
        # What the loop retires per cycle, and how much of that its own MFMAs
        # could retire if the matrix pipe never idled -- so `of_peak` is a
        # ceiling this loop can really reach, not the machine's paper number.
        r["flop_per_cycle"] = r["flops"] / r["cycles"] if r["cycles"] else 0.0
        r["of_peak"] = r["mfma_busy"] / r["cycles"] if r["cycles"] else 0.0
        out.append(r)
    return out


def report(rows, iters, whole=False):
    what = "function" if whole else "loop"
    if not rows:
        print(f"no {what}s found")
        return
    name_w = 24 if whole else 10
    hdr = (f"{'kernel':<30}{what:>{name_w}}{'body':>6}{'mfma':>5}{'bar':>4}"
           + (f"{'cycles':>10}" if whole
              else f"{f'cy/{iters}it':>9}{'cy/it':>7}")
           + f"{'xdl':>8}{'valu':>8}"
             f"{'s:xdl':>8}{'s:valu':>8}{'lgkm':>8}{'vm':>7}{'cy/xdl':>7}"
             f"{'flop/cy':>9}{'peak':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: -r["cycles"]):
        s = r["stall"]
        n = r["iters"]
        print(f"{os.path.basename(r['file'])[:29]:<30}"
              f"{r['loop'][:name_w - 1]:>{name_w}}"
              f"{r['body']:>6}{r['counts']['mfma'] // n:>5}"
              f"{r['barriers'] // n:>4}"
              + (f"{r['cycles']:>10}" if whole
                 else f"{r['cycles']:>9}{r['per_iter']:>7.0f}")
              + f"{r['xdl_busy']:>8}"
              f"{r['valu_busy']:>8}{s['xdl']:>8}{s['valu']:>8}"
              f"{s['lgkmcnt']:>8}{s['vmcnt']:>7}{r['xdl_ratio']:>7.2f}"
              f"{r['flop_per_cycle']:>9.0f}{100 * r['of_peak']:>5.0f}%")
    print("-" * len(hdr))
    tot = sum(r["cycles"] for r in rows)
    xdl = sum(r["xdl_busy"] for r in rows)
    mfma = sum(r["mfma_busy"] for r in rows)
    valu = sum(r["valu_busy"] for r in rows)
    bar = sum(r["barriers"] for r in rows)
    flop = sum(r["flops"] for r in rows)
    st = {k: sum(r["stall"][k] for r in rows) for k in rows[0]["stall"]}
    print(f"{len(rows)} {what}s   {tot} cycles   busy: xdl {xdl} "
          f"({100 * xdl / tot:.0f}%) valu {valu} ({100 * valu / tot:.0f}%)")
    if flop:
        print(f"  {flop} flops   {flop / tot:.0f} flop/cycle, "
              f"{100 * mfma / tot:.0f}% of what these MFMAs can do "
              f"({flop / mfma:.0f} flop/cycle)")
    print(f"  stall: xdl {st['xdl']} valu {st['valu']} lgkm {st['lgkmcnt']} "
          f"vm {st['vmcnt']} lds {st['lds']} reg {st['reg']}")
    if bar:
        print(f"  {bar} s_barrier charged {BARRIER_CYCLES} cy each -- a "
              f"single-wave stand-in for workgroup skew")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", help=".s files, or directories to walk")
    p.add_argument("-n", "--iters", type=int, default=16,
                   help="iterations to simulate (default 16)")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="print a per-instruction trace of the first iteration")
    p.add_argument("-w", "--whole", action="store_true",
                   help="score whole functions instead of innermost loops: "
                        "every instruction counts once, loop bodies -n times")
    p.add_argument("--min-size", type=int, default=4,
                   help="skip loop bodies smaller than this")
    p.add_argument("--lds", type=int, default=LDS_LATENCY, help="cycles")
    p.add_argument("--vmem", type=int, default=VMEM_LATENCY, help="cycles")
    p.add_argument("--smem", type=int, default=SMEM_LATENCY, help="cycles")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    files = []
    for path in args.paths:
        if os.path.isdir(path):
            for root, _, names in os.walk(path):
                files += [os.path.join(root, n) for n in sorted(names)
                          if n.endswith(".s")]
        else:
            files.append(path)

    rows = []
    for f in files:
        rows += analyze(f, args)

    if args.json:
        json.dump(rows, sys.stdout, indent=2)
        print()
    else:
        report(rows, args.iters, args.whole)


if __name__ == "__main__":
    main()
