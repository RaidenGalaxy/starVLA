# Read-only audit. This file deliberately contains no timing exceptions.
proc simd_reset_truth {net {visited {}}} {
 if {$net eq "NULL"} {error "Disconnected reset cone"}
 set name [$net getName]
 if {$name in $visited} {error "Loop in reset cone"}
 lappend visited $name
 set ports {}
 foreach pin [$net getBTerms] {
  if {[$pin getIoType] eq "INPUT"} {lappend ports [$pin getName]}
 }
 set drivers {}
 foreach pin [$net getITerms] {
  if {[[$pin getMTerm] getIoType] eq "OUTPUT"} {lappend drivers $pin}
 }
 if {[llength $ports]==1 && ![llength $drivers]} {
  switch -- [lindex $ports 0] {
   io_coreReset {return 12}
   io_memoryReset {return 10}
   default {error "Unexpected external reset source: $ports"}
  }
 }
 if {[llength $ports] || [llength $drivers]!=1} {error "Ambiguous reset cone: $name"}
 set inst [[lindex $drivers 0] getInst]
 set master [[$inst getMaster] getName]
 if {[string match TIEHI* $master]} {return 15}
 if {[string match TIELO* $master]} {return 0}
 set inputs {}
 foreach pin [$inst getITerms] {
  set term [$pin getMTerm]
  if {[$term getIoType] eq "INPUT" && [$term getSigType] ni {POWER GROUND}} {
   lappend inputs [simd_reset_truth [$pin getNet] $visited]
  }
 }
 if {[llength $inputs]==1} {
  set value [lindex $inputs 0]
  if {[string match BUF* $master]} {return $value}
  if {[regexp {^(INV|CKINV)} $master]} {return [expr {$value ^ 15}]}
 }
 if {[llength $inputs]==2} {
  lassign $inputs a b
  if {[string match OR2x* $master]} {return [expr {$a | $b}]}
  if {[string match NOR2x* $master]} {return [expr {($a | $b) ^ 15}]}
  if {[string match AND2x* $master]} {return [expr {$a & $b}]}
  if {[string match NAND2x* $master]} {return [expr {($a & $b) ^ 15}]}
 }
 error "Unsupported gate in raw reset cone: [$inst getName] $master"
}

proc simd_audit_reset_pair {prefix clock stream} {
 set block [ord::get_db_block]
 set flops {}
 foreach inst [$block getInsts] {
  set canonical [string map {\\ ""} [$inst getName]]
  if {[string first "$prefix/" $canonical]==0 &&
      [string match DFFASRHQN* [[$inst getMaster] getName]]} {lappend flops $inst}
 }
 if {[llength $flops]!=2} {error "Expected exactly two asynchronous reset stages: $prefix"}
 set first NULL
 foreach inst $flops {
  set root [simd_reset_logic_root [[$inst findITerm D] getNet]]
  if {[string match PIN:*/L [lindex $root 0]] && [lindex $root 1]==0} {
   if {[simd_reset_truth [[$inst findITerm D] getNet]]!=0} {error "First reset D is not constant zero"}
   set first $inst
  }
 }
 if {$first eq "NULL"} {error "Missing first reset stage: $prefix"}
 foreach inst $flops {
  if {$inst ne $first} {set second $inst}
  if {[simd_clock_root [[$inst findITerm CLK] getNet]] ne [list $clock 0]} {
   error "Reset stage changed clock domain/polarity: [$inst getName]"
  }
  # Liberty QN=!D; logical Q=!QN. SETN must assert logical one.
  if {[simd_reset_truth [[$inst findITerm SETN] getNet]]!=1} {
   error "Async preset pin is not the negated OR of both external resets"
  }
  if {[simd_reset_truth [[$inst findITerm RESETN] getNet]]!=15} {
   error "Unexpected active logical-clear input"
  }
 }
 if {[simd_reset_logic_root [[$second findITerm D] getNet]] ne [list "PIN:[$first getName]/QN" 1]} {
  error "Second reset stage no longer samples first stage"
 }
 puts $stream "$prefix\t$clock\t[$first getName]\t[$second getName]\t15\t1"
 return [list [$first getName] [$second getName]]
}
