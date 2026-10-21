; RUN: llc -mtriple=amdgpu9.00-amd-amdhsa -misched=amdgpu-optsched \
; RUN:     -amdgpu-optsched-pin-physregs=1 -o - %s \
; RUN:   | FileCheck %s --check-prefix=STRICT
;
; RUN: llc -mtriple=amdgpu9.00-amd-amdhsa -misched=amdgpu-optsched \
; RUN:     -amdgpu-optsched-pin-physregs=1 -amdgpu-optsched-allow-ties=1 -o - %s \
; RUN:   | FileCheck %s --check-prefix=TIES

; A schedule that merely ties the one already there must be refused.
;
; -amdgpu-optsched-pin-physregs=1 is here to keep the tie a tie. Without it
; the search can also move the instructions that name physical registers, and
; it then finds an order that is four cycles shorter rather than equal -- a
; different rule decides that one, and this test is about this rule.
;
; A tie buys nothing and bets the region on our model being complete, which it
; is not: every later pass reads the order too, and some read things this model
; has no name for. On materialize-frame-index-sgpr.ll a tie moved an SCC def
; down three slots, which shortened $scc's live range, which made frame-index
; lowering take its scalar path, which asked for an SGPR that was not there.
; No cycle count and no pressure count can see that coming -- so ties lose.
;
; Here the region is 115 cycles either way and the solver has a second order of
; the same length, which sinks the scratch-base setup past the lane shift.
; -amdgpu-optsched-allow-ties=1 takes it; the default does not.

; STRICT: s_add_u32 s0, s0, s17
; STRICT: v_lshlrev_b32_e32 v3, 6, v0

; TIES: v_lshlrev_b32_e32 v3, 6, v0
; TIES: s_add_u32 s0, s0, s17

define amdgpu_kernel void @insertelement_with_call(ptr addrspace(1) %ptr, i32 %idx) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %ptr.tid = getelementptr inbounds <16 x i32>, ptr addrspace(1) %ptr, i32 %tid
  %vec = load <16 x i32>, ptr addrspace(1) %ptr.tid
  %i6 = insertelement <16 x i32> %vec, i32 8, i32 %idx
  call void @foo()
  store <16 x i32> %i6, ptr addrspace(1) null
  ret void
}

declare void @foo()

attributes #0 = { nounwind }
