; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -misched=amdgpu-optsched \
; RUN:     -o - %s 2>&1 | FileCheck %s
;
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -misched=amdgpu-optsched \
; RUN:     -debug-only=amdgpu-optsched -o /dev/null %s 2>&1 \
; RUN:   | FileCheck %s --check-prefix=LOG
;
; REQUIRES: asserts

; An order can be shorter, hold fewer registers, and still be impossible.
;
; v0-v63 are occupied by inline asm. Hoisting the atomic above the copies out
; of them is three cycles shorter and lower on every pressure count there is,
; and leaves the atomic's result with nowhere to live: its definition needs a
; VGPR, and at that point all 64 this function may use are held by a physical
; range, which cannot be evicted and does not spill.
;
; The order already there puts the result's definition before those 64 are
; written, so only the *rest* of its live range crosses them -- and the rest
; can be widened to av_32 and parked in an AGPR. Both orders hold 64 physical
; VGPRs across the result; no count tells them apart. What tells them apart is
; that one of them asks for a colour at a point where none exists, which is
; what shortfallOf measures and what the driver gates on.

; CHECK-NOT: ran out of registers
; CHECK: flat_atomic_swap

; LOG: optsched: budget {{.*}} shortfall 0 -> {{[1-9]}}
; LOG-NEXT: optsched: shortfall 0 -> {{[1-9]}}
; LOG-NOT: optsched: applying

define void @atomic_result_must_not_cross_physreg_copies(ptr %ptr) #0 {
  %gep.0 = getelementptr inbounds [512 x i32], ptr %ptr, i64 0, i64 10
  %data = call i32 asm "; def $0", "=^VA"()
  %vgpr.def = call { <32 x i32>, <32 x i32> } asm sideeffect "; def $0", "=${v[0:31]},=${v[32:63]}"()
  %vgpr.0 = extractvalue { <32 x i32>, <32 x i32> } %vgpr.def, 0
  %vgpr.1 = extractvalue { <32 x i32>, <32 x i32> } %vgpr.def, 1
  %result = atomicrmw xchg ptr %gep.0, i32 %data seq_cst
  call void asm sideeffect "; use $0", "{v[0:31]},{v[32:63]}"(<32 x i32> %vgpr.0, <32 x i32> %vgpr.1)
  call void asm "; use $0", "^VA"(i32 %result)
  ret void
}

attributes #0 = { nounwind "amdgpu-waves-per-eu"="10,10" }
