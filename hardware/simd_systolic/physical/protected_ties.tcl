# Tie replication may reconnect asynchronous inactive reset/set inputs of
# protected synchronizers. Permit only that bounded operation, and verify
# every other connection and the logical value of each constant afterwards.
proc simd_protected_signal_state {instances} {
 set state [dict create]
 set constants [dict create]
 foreach inst $instances {
  foreach pin [$inst getITerms] {
   if {[[$pin getMTerm] getSigType] in {POWER GROUND}} {continue}
   set net [$pin getNet]
   set value NULL
   if {$net ne "NULL"} {
    set name [$net getName]
    if {![dict exists $constants $name]} {
     set drivers {}
     foreach other [$net getITerms] {
      if {[[$other getMTerm] getIoType] ne "OUTPUT"} {continue}
      lappend drivers [list [[[$other getInst] getMaster] getName] [[$other getMTerm] getName]]
     }
     set logical "NET:$name"
     if {[llength $drivers] == 1} {
      set driver [lindex $drivers 0]
      if {$driver eq {TIELOx1_ASAP7_75t_R L}} {set logical CONST0}
      if {$driver eq {TIEHIx1_ASAP7_75t_R H}} {set logical CONST1}
     }
     dict set constants $name $logical
    }
    set value [dict get $constants $name]
   }
   dict set state "[$inst getName]/[[$pin getMTerm] getName]" $value
  }
 }
 return $state
}
proc simd_repair_protected_ties {} {
 set instances {}
 foreach inst [[ord::get_db_block] getInsts] {
  if {[$inst isDoNotTouch]} {lappend instances $inst}
 }
 set before [simd_protected_signal_state $instances]
 foreach inst $instances {$inst setDoNotTouch false}
 try {
  repair_tie_fanout -separation 0 TIELOx1_ASAP7_75t_R/L
  repair_tie_fanout -separation 0 TIEHIx1_ASAP7_75t_R/H
 } finally {
  foreach inst $instances {$inst setDoNotTouch true}
 }
 set after [simd_protected_signal_state $instances]
 if {$before ne $after} {error "Tie repair changed a protected signal or constant value"}
 puts "SIMD_PROTECTED_TIE_LOGIC_VERIFIED [llength $instances]"
}
