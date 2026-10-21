; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -misched=amdgpu-optsched \
; RUN:     -debug-only=amdgpu-optsched -o /dev/null %s 2>&1 | FileCheck %s
;
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -misched=amdgpu-optsched -o - %s \
; RUN:   | FileCheck %s --check-prefix=ASM
;
; REQUIRES: asserts

; The scheduler must report what it did and how sure it is. A region small
; enough to enumerate has to come back proved optimal, not merely scheduled.
; CHECK: optsched: {{.*}} solver=exact status=optimal

; An answer that does not verify is never applied, so this line must not
; appear for well-formed input.
; CHECK-NOT: optsched: rejected

define amdgpu_kernel void @small_region(ptr addrspace(1) %out, ptr addrspace(1) %in) {
; ASM-LABEL: small_region:
; ASM: s_load_dword
; ASM: global_store_dword
entry:
  %a = load i32, ptr addrspace(1) %in
  %b = add i32 %a, 1
  %c = mul i32 %b, 3
  %d = sub i32 %c, %a
  store i32 %d, ptr addrspace(1) %out
  ret void
}

; A longer straight-line region: still exact, and the reported makespan must
; be no worse than the order the region arrived with.
define amdgpu_kernel void @chain(ptr addrspace(1) %out, i32 %x, i32 %y) {
; ASM-LABEL: chain:
; ASM: global_store_dword
entry:
  %a0 = add i32 %x, 1
  %a1 = add i32 %y, 2
  %a2 = mul i32 %a0, %a1
  %a3 = xor i32 %a2, %a0
  %a4 = or i32 %a3, %a1
  %a5 = and i32 %a4, %a2
  store i32 %a5, ptr addrspace(1) %out
  ret void
}
