// RUN: %python %S/../../../utils/gfx950-loop-cycles.py -n 4 %s | FileCheck %s

// Two loops that retire the same FLOPs, so the only thing separating them is
// how long the matrix pipe sits idle.
//
// `dense` issues its MFMAs back to back. A 32x32x8 f16 MFMA is 4 passes, so it
// holds the matrix pipe for 16 cycles, and the three scalar instructions beside
// it fit inside that window for free. The loop costs exactly what the math
// costs, which is the most it can do: 2*32*32*8 flops in 16 cycles, 1024
// flop/cycle, 100% of its own ceiling.
//
// `stalled` is the same MFMA and the same flops, with ten more VALU
// instructions beside it. Issue alone is then 56 cycles an iteration, 40 of
// which the matrix pipe cannot use, and the flop rate falls to under a third.
// Nothing here waits on anything -- the loop is slower purely because a wave
// issues one instruction every 4 cycles whatever it is.

// CHECK: kernel {{.*}}cy/xdl flop/cy  peak
// CHECK-NEXT: ---
// CHECK-NEXT: loops.s 5-18 14 1 0 224 56 64 160 0 0 0 0 3.50 293 29%
// CHECK-NEXT: loops.s 0-3 4 1 0 64 16 64 0 0 0 0 0 1.00 1024 100%
// CHECK-NEXT: ---
// CHECK-NEXT: 2 loops 288 cycles busy: xdl 128 (44%) valu 160 (56%)
// CHECK-NEXT: 131072 flops 455 flop/cycle, 44% of what these MFMAs can do (1024 flop/cycle)

dense:
.LBB0_1:
	v_mfma_f32_32x32x8_f16 a[0:15], v[0:1], v[2:3], a[0:15]
	s_add_u32 s0, s0, 1
	s_cmp_lt_u32 s0, s1
	s_cbranch_scc1 .LBB0_1
	s_endpgm

stalled:
.LBB1_1:
	v_mfma_f32_32x32x8_f16 a[16:31], v[4:5], v[6:7], a[16:31]
	v_add_u32_e32 v4, 1, v4
	v_add_u32_e32 v5, 1, v5
	v_add_u32_e32 v6, 1, v6
	v_add_u32_e32 v7, 1, v7
	v_add_u32_e32 v8, 1, v8
	v_add_u32_e32 v9, 1, v9
	v_add_u32_e32 v10, 1, v10
	v_add_u32_e32 v11, 1, v11
	v_add_u32_e32 v12, 1, v12
	v_add_u32_e32 v13, 1, v13
	s_add_u32 s2, s2, 1
	s_cmp_lt_u32 s2, s3
	s_cbranch_scc1 .LBB1_1
	s_endpgm
