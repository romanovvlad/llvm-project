// RUN: %python %S/../../../utils/gfx950-loop-cycles.py -n 1 --min-size 1 %s \
// RUN:   | FileCheck %s

// How long each MFMA holds a pipe. Every loop here is one MFMA and a backward
// branch, so the busy columns are that one MFMA's occupancy.
//
// The shape in the name does not tell you the answer, which is the point of
// the table this checks. Four of these are 16x16 and they hold a pipe for 64,
// 32, 32 and 16 cycles: a bigger K is cheaper, and for the f8f6f4 form the
// operand width in cbsz/blgp decides it rather than anything in the name.
//
// `dense_f64` also checks which pipe. The f64 MFMAs are given
// Write16PassDGEMM, an HWVALUWriteRes, so their 64 cycles land in `valu` and
// the matrix pipe stays idle -- the one MFMA here that does not use it.

// CHECK: kernel {{.*}}cy/xdl flop/cy peak
// CHECK-NEXT: ---
// CHECK-NEXT: mfma-passes.s 0-1 2 1 0 64 64 0 64 0 0 0 0 1.00 32 100%
// CHECK-NEXT: mfma-passes.s 3-4 2 1 0 32 32 32 0 0 0 0 0 1.00 64 100%
// CHECK-NEXT: mfma-passes.s 12-13 2 1 0 32 32 32 0 0 0 0 0 1.00 2048 100%
// CHECK-NEXT: mfma-passes.s 6-7 2 1 0 24 24 16 0 0 0 0 0 1.50 683 67%
// CHECK-NEXT: mfma-passes.s 9-10 2 1 0 24 24 8 0 0 0 0 0 3.00 5 33%
// CHECK-NEXT: mfma-passes.s 15-16 2 1 0 24 24 16 0 0 0 0 0 1.50 2731 67%
// CHECK-NEXT: ---
// CHECK-NEXT: 6 loops 200 cycles busy: xdl 104 (52%) valu 64 (32%)

// 16 passes, and on the vector ALU rather than the matrix pipe.
dense_f64:
.LBB0_1:
	v_mfma_f64_16x16x4_f64 a[0:7], v[0:1], v[2:3], a[0:7]
	s_cbranch_scc1 .LBB0_1
	s_endpgm

// 8 passes: same 16x16 tile, a quarter of the K, twice the time.
small_k:
.LBB1_1:
	v_mfma_f32_16x16x4f32 a[0:3], v0, v1, a[0:3]
	s_cbranch_scc1 .LBB1_1
	s_endpgm

// 4 passes.
big_k:
.LBB2_1:
	v_mfma_f32_16x16x32_bf16 a[0:3], v[0:3], v[4:7], a[0:3]
	s_cbranch_scc1 .LBB2_1
	s_endpgm

// 2 passes, the cheapest MFMA there is.
tiny:
.LBB3_1:
	v_mfma_f32_4x4x4f16 a[0:3], v[0:1], v[2:3], a[0:3]
	s_cbranch_scc1 .LBB3_1
	s_endpgm

// One instruction, two costs: fp8 in either operand is 8 passes, fp4 in both
// is 4, and only cbsz/blgp say which.
f8f6f4_fp8:
.LBB4_1:
	v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[0:3] cbsz:0 blgp:0
	s_cbranch_scc1 .LBB4_1
	s_endpgm

f8f6f4_fp4:
.LBB5_1:
	v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:7], v[8:11], v[0:3] cbsz:4 blgp:4
	s_cbranch_scc1 .LBB5_1
	s_endpgm
