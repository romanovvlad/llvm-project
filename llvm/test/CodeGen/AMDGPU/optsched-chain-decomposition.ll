; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -misched=amdgpu-optsched \
; RUN:     -debug-only=amdgpu-optsched-chain -o /dev/null %s 2>&1 \
; RUN:   | FileCheck %s
;
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -misched=amdgpu-optsched \
; RUN:     -debug-only=amdgpu-optsched-chain -o /dev/null %s 2>&1 \
; RUN:   | FileCheck %s --check-prefix=WIDE
;
; REQUIRES: asserts

; The region is decomposed around its longest dependence chain. The chain is a
; total order, so it is a coordinate system: K chain instructions cut the
; region into K+1 slots, and every other instruction gets the interval of
; slots its dependences allow, plus what each of those slots would cost.
;
; Here the fmul chain is the spine and the LDS load is free: it may sit
; anywhere from the top of the region down to the fadd that consumes it, and
; parking it late costs the wave its latency.

; CHECK: optsched-chain: {{.*}} n={{[0-9]+}} chain={{[0-9]+}} length={{[0-9]+}} slots={{[0-9]+}}
;
; The spine is the fmul chain, so it is named after V_MUL_F32.
; CHECK: optsched-chain:   chain {{.*}}V_MUL_F32{{.*}} -> {{.*}}V_MUL_F32
;
; Every slot reports its width: the number of orders its instructions admit.
; Small slots get an exact count, which is the point of cutting the region up.
; CHECK: optsched-chain:   slot 0: starts {{[0-9]+}}, room {{[0-9]+}} ({{[0-9]+}} free), {{[0-9]+}} instr{{[a-z]*}}, width {{[0-9]+}} exact
;
; The load is off the spine, so it spans several slots, and parking it in the
; last one it is allowed to occupy stalls its consumer.
; CHECK: optsched-chain:   node {{[0-9]+}} DS_READ{{[^:]*}}: slots [{{[0-9]+}},{{[0-9]+}}] now {{[0-9]+}}{{.*}}+{{[0-9]+}}@s
;
; And the two freedoms the decomposition exposes, which together are the
; search space it has to cover.
; CHECK: optsched-chain:   freedom: placement {{.*}} ordering
;
; And the line that says where a search would have to do real work.
; CHECK: optsched-chain:   summary: {{[0-9]+}} slots, {{[0-9]+}} empty, {{[0-9]+}} countable exactly, widest {{[0-9]+}} instrs; chain {{[0-9]+}} of {{[0-9]+}} cycles, both lower bounds

; A slot too wide to count exactly is bracketed instead, and says so. The
; distinction is the point: a bound must never be read as a count.
; WIDE: optsched-chain:   slot {{[0-9]+}}: {{.*}}, width 10^{{[0-9.]+}}..10^{{[0-9.]+}}

define amdgpu_kernel void @chain_with_a_free_load(ptr addrspace(1) %out,
                                                  ptr addrspace(3) %lds,
                                                  float %a) {
entry:
  %x0 = fmul float %a, %a
  %x1 = fmul float %x0, %x0
  %x2 = fmul float %x1, %x1
  %x3 = fmul float %x2, %x2
  %x4 = fmul float %x3, %x3
  %x5 = fmul float %x4, %x4
  %x6 = fmul float %x5, %x5
  %x7 = fmul float %x6, %x6
  %x8 = fmul float %x7, %x7
  %x9 = fmul float %x8, %x8
  %x10 = fmul float %x9, %x9
  %x11 = fmul float %x10, %x10
  %x12 = fmul float %x11, %x11
  %x13 = fmul float %x12, %x12
  %x14 = fmul float %x13, %x13
  %x15 = fmul float %x14, %x14
  %ld = load float, ptr addrspace(3) %lds
  %sum = fadd float %x15, %ld
  store float %sum, ptr addrspace(1) %out
  ret void
}

; A region whose slots are wide rather than deep: 20 mutually
; independent values, so one slot holds an antichain and the exact count
; is out of reach. Here the bracket is tight -- an antichain of k admits
; exactly k! orders -- which is a useful check that it is not nonsense.
define amdgpu_kernel void @wide_slot(ptr addrspace(1) %out, float %a, float %b) {
entry:
  %p0 = fadd float %a, 1.0
  %p1 = fadd float %a, 2.0
  %p2 = fadd float %a, 3.0
  %p3 = fadd float %a, 4.0
  %p4 = fadd float %a, 5.0
  %p5 = fadd float %a, 6.0
  %p6 = fadd float %a, 7.0
  %p7 = fadd float %a, 8.0
  %p8 = fadd float %a, 9.0
  %p9 = fadd float %a, 10.0
  %p10 = fadd float %a, 11.0
  %p11 = fadd float %a, 12.0
  %p12 = fadd float %a, 13.0
  %p13 = fadd float %a, 14.0
  %p14 = fadd float %a, 15.0
  %p15 = fadd float %a, 16.0
  %p16 = fadd float %a, 17.0
  %p17 = fadd float %a, 18.0
  %p18 = fadd float %a, 19.0
  %p19 = fadd float %a, 20.0
  %c0 = fmul float %b, %b
  %c1 = fmul float %c0, %c0
  %c2 = fmul float %c1, %c1
  %c3 = fmul float %c2, %c2
  %c4 = fmul float %c3, %c3
  %c5 = fmul float %c4, %c4
  %s0 = fadd float %c5, %p0
  %s1 = fadd float %s0, %p1
  %s2 = fadd float %s1, %p2
  %s3 = fadd float %s2, %p3
  %s4 = fadd float %s3, %p4
  %s5 = fadd float %s4, %p5
  %s6 = fadd float %s5, %p6
  %s7 = fadd float %s6, %p7
  %s8 = fadd float %s7, %p8
  %s9 = fadd float %s8, %p9
  %s10 = fadd float %s9, %p10
  %s11 = fadd float %s10, %p11
  %s12 = fadd float %s11, %p12
  %s13 = fadd float %s12, %p13
  %s14 = fadd float %s13, %p14
  %s15 = fadd float %s14, %p15
  %s16 = fadd float %s15, %p16
  %s17 = fadd float %s16, %p17
  %s18 = fadd float %s17, %p18
  %s19 = fadd float %s18, %p19
  store float %s19, ptr addrspace(1) %out
  ret void
}
