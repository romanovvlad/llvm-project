; RUN: llc -mtriple=x86_64-unknown-linux-gnu -misched=amdgpu-optsched -o - %s \
; RUN:   | FileCheck %s
;
; RUN: llc -mtriple=r600 -mcpu=redwood -misched=amdgpu-optsched -o - %s \
; RUN:   | FileCheck %s --check-prefix=R600
;
; REQUIRES: x86-registered-target

; The strategy lives in the global MachineSchedRegistry, so -misched can select
; it for any target -- and will, for any .ll file that gets its triple from the
; RUN line rather than from the module. Everything it does is AMDGPU-specific,
; most sharply the pressure measurement, which reads GCN register classes; fed
; x86 ones it used to abort inside GCNUpwardRPTracker. So on a foreign target
; it must stand down and leave the order alone.
;
; r600 is foreign too, which is less obvious because it is the same backend.
; It has no GCNSubtarget, so the register budget cannot be asked for at all:
; AMDGPU::IsaInfo::getMaxNumSGPRs asserts rather than answer.

; CHECK-LABEL: foreign:
; CHECK: retq

; R600-LABEL: foreign:
; R600: CF_END
define <10 x i16> @foreign(<10 x i16> %a) {
entry:
  br label %next

next:
  ret <10 x i16> %a
}
