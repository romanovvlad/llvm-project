#!/usr/bin/env python3
"""Cycle cost of every hot loop in a gfx950 assembly file.

Finds the innermost loops, runs N iterations of each through an in-order
single-wave timing model, and reports the cycles they take, where they went,
and the FLOPs retired per cycle.

    gfx950-loop-cycles.py kernel.s
    gfx950-loop-cycles.py -n 16 --json out/      # every .s below a directory
    gfx950-loop-cycles.py -v kernel.s            # per-instruction trace

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

  valu          4 cycles, so free; 16 for the quarter-rate transcendentals.
  xdl           `passes * 4` cycles. The next MFMA cannot issue until it frees,
                whatever its operands are, so the only way to use that window
                is to put other work between the two MFMAs. At 4 passes that
                is 16 cycles, and the traces show exactly one instruction fits
                in it for free.
  lds           8 cycles to read, 52 to write.
  vmem          4 cycles to issue a load, 40 for a store.
  salu          4 cycles. `s_nop N` is (N+1)*4 and occupies no pipe.

On top of the pipes the wave stops for two more things:

  s_waitcnt     A wait for outstanding memory. lgkmcnt(N) releases when all but
                the last N LDS/scalar loads have returned; vmcnt(N) does the
                same for vector memory.
  register      A read of a value that has not landed yet. Only MFMA results
                are slow enough to matter, and only for non-MFMA readers --
                src2 of the next MFMA in a chain is forwarded.

Pass counts are the gfx950 entries of SIDPGFX950FullSpeedModel in
llvm/lib/Target/AMDGPU/SISchedule.td and should not need touching. The memory
latencies are not in that model; they are the tunable part of this tool, and
the ones to revisit against a real profile -- see --lds / --vmem / --smem.

Calibration
-----------
The per-instruction costs above are not a spec sheet. They are the medians of
several million issue-to-issue spacings measured on gfx950 hardware with a
wave thread-trace profiler, and the model as a whole was checked against the
measured time of a dozen hot loops from the same traces:

  Pearson r 0.99, Spearman rho 0.94, 92% of pairwise orderings correct.
  model/measured 0.92 +- 0.03, range 0.85-0.96.

So the tool is reliable for comparing two schedules and reads about 8% low in
absolute terms. Two loops within ~5% of each other are a tie it cannot call.

The 8% is contention this model has no way to see: per instruction it lands
within 0.5% of the measured medians, but the measured *distributions* have
long tails -- buffer_load is a median of 4 cycles and a mean of 8.7 -- from
many waves sharing the LDS and memory ports. s_barrier has the same problem
and is charged a flat 16 cycles, a single-wave stand-in for what is really the
spread between the fastest and slowest wave in the workgroup. The barrier
count is reported per loop; where it is high, read the cycle figure as
approximate.

Not modelled
------------
LDS and HBM bandwidth, and any overlap between waves. The tool is weakest on
memory-bound loops, where almost all of the residual error sits.

The floor
---------
`xdl` is the cycles the matrix pipe was busy. A loop cannot run faster than
that, so `cy/xdl` is how much of the loop is the math itself: 1.0 means the
matrix pipe never idles, 2.0 means half the time it is waiting.

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
    "trans": 16,  # quarter rate
    "lds": 4,
    "vmem": 4,  # fire and forget; the cost is in the s_waitcnt
    "smem": 4,
    "salu": 4,
}

# Classes that block the wave itself for longer than ISSUE_CYCLES, so the next
# instruction waits even if it needs a different pipe. The traces show this for
# LDS: a ds_read_b128 is 8 cycles whatever follows it.
ISSUE_OF = {"lds": 8}

BARRIER_CYCLES = 16  # measured median; the real cost is workgroup skew

TRANS = ("v_rcp", "v_sqrt", "v_rsq", "v_log", "v_exp", "v_sin", "v_cos",
         "v_div_scale", "v_div_fmas", "v_div_fixup")

# --- tunables: round-trips in cycles, not modelled by SISchedule.td ----------

LDS_LATENCY = 64
VMEM_LATENCY = 500
SMEM_LATENCY = 300

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


def mfma_passes(mnemonic, operands):
    """Cycles/4 that `mnemonic` holds the XDL pipe."""
    u = mnemonic.upper()
    m = SHAPE.search(u)
    if not m:
        return 4
    shape = (int(m.group(1)), int(m.group(2)), int(m.group(3)))

    if "F8F6F4" in u:
        # WriteMFMAScale_*_F8F6F4: f8 in either operand costs twice as much.
        fmts = {int(v) for k, v in MODIFIER.findall(operands)
                if k in ("cbsz", "blgp")}
        if not fmts:
            fmts = {0}  # the default encoding is fp8
        f8 = bool(fmts & F8_FORMATS)
        return (8 if f8 else 4) if shape[0] == 16 else (16 if f8 else 8)

    if "SMFMAC" in u:
        return 4 if shape[0] == 16 else 8

    if "_F64_" in u:
        return 4 if shape[0] == 4 else 16  # DGEMM pipe

    if shape[0] == 4:
        return 2
    if shape[0] == 16:
        return 4 if shape[2] >= 8 else 8
    # 32x32: K >= 8 is 8 passes, the small-K forms are 16.
    return 8 if shape[2] >= 8 else 16


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


# --- timing model ------------------------------------------------------------


def classify(mnemonic):
    """The pipe `mnemonic` occupies."""
    if mnemonic.startswith(("v_mfma", "v_smfmac")):
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


PIPE_OF = {"mfma": "xdl", "trans": "valu"}  # instruction class -> pipe it shares


def pipe_busy(kind, mn, ops):
    """Cycles `mn` keeps its pipe busy. Only values above ISSUE_CYCLES bite."""
    if kind == "mfma":
        return mfma_passes(mn, ops) * CYCLES_PER_PASS
    return PIPE_BUSY[kind]


def simulate(insns, start, end, iters, lat, trace=None):
    """`iters` back-to-back iterations of insns[start:end+1]."""
    t = 0
    free = {}  # pipe -> cycle it is next free
    busy = {}  # pipe -> cycles it was busy
    ready = {}  # register -> cycle its value lands
    vm, lgkm = Counter(), Counter()
    stall = {"xdl": 0, "valu": 0, "lds": 0, "vmem": 0, "smem": 0, "salu": 0,
             "vmcnt": 0, "lgkmcnt": 0, "reg": 0}
    counts = {k: 0 for k in
              ("mfma", "valu", "trans", "lds", "vmem", "smem", "salu")}
    flops = barriers = 0

    for it in range(iters):
        for i in range(start, end + 1):
            mn, ops = insns[i]
            kind = classify(mn)
            pipe = PIPE_OF.get(kind, kind)
            counts[kind] += 1
            was = t

            if mn == "s_barrier":
                barriers += 1
                t += BARRIER_CYCLES
                if trace is not None and it == trace:
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
                if trace is not None and it == trace:
                    trace_line(i, mn, ops, was, t)
                t += ISSUE_CYCLES
                continue

            if mn == "s_nop":
                # (N+1) issue slots of nothing, on no pipe at all.
                arg = ops.strip().split()[0] if ops.strip() else "0"
                n = int(arg, 0) if arg.isdigit() else 0
                t += (n + 1) * ISSUE_CYCLES
                if trace is not None and it == trace:
                    trace_line(i, mn, ops, was, t)
                continue

            operands = [o.strip() for o in ops.split(",")]
            if "store" in mn or mn.startswith("ds_write"):
                dst, src = set(), regs(ops)  # a store defines no register
            else:
                dst = regs(operands[0]) if operands else set()
                src = regs(",".join(operands[1:])) if len(operands) > 1 else set()

            # Wait for the sources. An MFMA's src2 is the previous MFMA's
            # result in the same chain and is forwarded, so an MFMA only ever
            # waits for the pipe, never for data.
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

            if trace is not None and it == trace:
                trace_line(i, mn, ops, was, t)
            t += ISSUE_OF.get(kind, ISSUE_CYCLES)

    return {
        "body": end - start + 1,
        "iters": iters,
        "counts": counts,
        "flops": flops,
        "cycles": t,
        "xdl_busy": busy.get("xdl", 0),
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
    out = []
    for start, end in find_loops(insns, labels):
        if end - start + 1 < args.min_size:
            continue
        if args.verbose:
            print(f"\n{path}  loop {start}-{end}")
        r = simulate(insns, start, end, args.iters, lat,
                     trace=0 if args.verbose else None)
        r.update(file=path, loop=f"{start}-{end}", meta=meta)
        r["per_iter"] = r["cycles"] / r["iters"]
        r["xdl_ratio"] = r["cycles"] / r["xdl_busy"] if r["xdl_busy"] else 0.0
        # What the loop retires per cycle, and how much of that its own MFMAs
        # could retire if the matrix pipe never idled -- so `of_peak` is a
        # ceiling this loop can really reach, not the machine's paper number.
        r["flop_per_cycle"] = r["flops"] / r["cycles"] if r["cycles"] else 0.0
        r["of_peak"] = r["xdl_busy"] / r["cycles"] if r["cycles"] else 0.0
        out.append(r)
    return out


def report(rows, iters):
    if not rows:
        print("no loops found")
        return
    hdr = (f"{'kernel':<30}{'loop':>10}{'body':>6}{'mfma':>5}{'bar':>4}"
           f"{f'cy/{iters}it':>9}{'cy/it':>7}{'xdl':>8}{'valu':>8}"
           f"{'s:xdl':>8}{'s:valu':>8}{'lgkm':>8}{'vm':>7}{'cy/xdl':>7}"
           f"{'flop/cy':>9}{'peak':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: -r["cycles"]):
        s = r["stall"]
        print(f"{os.path.basename(r['file'])[:29]:<30}{r['loop']:>10}"
              f"{r['body']:>6}{r['counts']['mfma'] // iters:>5}"
              f"{r['barriers'] // iters:>4}"
              f"{r['cycles']:>9}{r['per_iter']:>7.0f}{r['xdl_busy']:>8}"
              f"{r['valu_busy']:>8}{s['xdl']:>8}{s['valu']:>8}"
              f"{s['lgkmcnt']:>8}{s['vmcnt']:>7}{r['xdl_ratio']:>7.2f}"
              f"{r['flop_per_cycle']:>9.0f}{100 * r['of_peak']:>5.0f}%")
    print("-" * len(hdr))
    tot = sum(r["cycles"] for r in rows)
    xdl = sum(r["xdl_busy"] for r in rows)
    valu = sum(r["valu_busy"] for r in rows)
    bar = sum(r["barriers"] for r in rows)
    flop = sum(r["flops"] for r in rows)
    st = {k: sum(r["stall"][k] for r in rows) for k in rows[0]["stall"]}
    print(f"{len(rows)} loops   {tot} cycles   busy: xdl {xdl} "
          f"({100 * xdl / tot:.0f}%) valu {valu} ({100 * valu / tot:.0f}%)")
    if flop:
        print(f"  {flop} flops   {flop / tot:.0f} flop/cycle, "
              f"{100 * xdl / tot:.0f}% of what these MFMAs can do "
              f"({flop / xdl:.0f} flop/cycle)")
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
        report(rows, args.iters)


if __name__ == "__main__":
    main()
