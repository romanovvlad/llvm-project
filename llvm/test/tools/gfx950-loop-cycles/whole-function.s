// RUN: %python %S/../../../utils/gfx950-loop-cycles.py -n 2 %s \
// RUN:   | FileCheck %s --check-prefix=LOOP
// RUN: %python %S/../../../utils/gfx950-loop-cycles.py -n 2 -w %s \
// RUN:   | FileCheck %s --check-prefix=WHOLE

// Loop mode scores the loop; whole-function mode scores the function. The
// difference here is the prologue, which loads the operands and then waits
// 44 cycles for them, plus the trailing store -- none of which a loop-only
// number can see.
//
// Both modes agree on everything the loop does: two iterations, 2 MFMAs, 64
// cycles of matrix pipe. Whole-function mode spreads that same work over the
// longer run, so the flop rate falls by more than half.

// LOOP: kernel {{.*}}cy/xdl flop/cy peak
// LOOP-NEXT: ---
// LOOP-NEXT: whole-function.s 7-10 4 1 0 64 32 64 0 0 0 0 0 1.00 512 100%
// LOOP-NEXT: ---
// LOOP-NEXT: 1 loops 64 cycles busy: xdl 64 (100%) valu 0 (0%)

// WHOLE: kernel {{.*}}cy/xdl flop/cy peak
// WHOLE-NEXT: ---
// WHOLE-NEXT: whole-function.s main 13 2 0 148 64 8 0 0 44 0 2.31 221 43%
// WHOLE-NEXT: ---
// WHOLE-NEXT: 1 functions 148 cycles busy: xdl 64 (43%) valu 8 (5%)

main:
	s_load_dwordx2 s[0:1], s[4:5], 0x0
	s_load_dwordx4 s[8:11], s[4:5], 0x8
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v1, 0
	s_waitcnt lgkmcnt(0)
	s_mov_b32 s2, 0
	s_mov_b32 s3, 16
.LBB0_1:
	v_mfma_f32_32x32x8_f16 a[0:15], v[0:1], v[2:3], a[0:15]
	s_add_u32 s2, s2, 1
	s_cmp_lt_u32 s2, s3
	s_cbranch_scc1 .LBB0_1
	global_store_dwordx4 v[0:1], a[0:3], s[0:1]
	s_endpgm
