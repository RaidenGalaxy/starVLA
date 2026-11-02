# CTS must replace a synchronizer's clock net while its data path and cells
# remain protected. Release only around CTS and audit the clock roots/phase,
# cell masters and all non-clock signal connections before restoring timing.
proc simd_clock_root {net {polarity 0} {visited {}}} {
 if {$net eq "NULL"} {error "Disconnected protected clock input"}
 set name [$net getName]
 if {$name in $visited} {error "Loop in protected clock fanin"}
 lappend visited $name
 set roots {}
 foreach pin [$net getBTerms] {
  if {[$pin getIoType] eq "INPUT"} {lappend roots [$pin getName]}
 }
 set drivers {}
 foreach pin [$net getITerms] {
  if {[[$pin getMTerm] getIoType] eq "OUTPUT"} {lappend drivers $pin}
 }
 if {[llength $roots] == 1 && ![llength $drivers]} {return [list [lindex $roots 0] $polarity]}
 if {[llength $roots] || [llength $drivers] != 1} {error "Protected clock has ambiguous driver: $name"}
 set inst [[lindex $drivers 0] getInst]
 set master [[$inst getMaster] getName]
 if {[string match BUF* $master]} {set inversion 0} elseif {[regexp {^(CKINV|INV)} $master]} {
  set inversion 1
 } else {error "Unexpected gate in protected clock fanin: $master"}
 set input [$inst findITerm A]
 if {$input eq "NULL"} {error "Clock buffer has no A input: $master"}
 return [simd_clock_root [$input getNet] [expr {$polarity ^ $inversion}] $visited]
}
proc simd_protected_clock_state {names} {
 set block [ord::get_db_block]
 set state [dict create]
 set clocks [dict create]
 foreach name $names {
  set inst [$block findInst $name]
  if {$inst eq "NULL"} {error "Protected instance removed by CTS: $name"}
  dict set state "$name/master" [[$inst getMaster] getName]
  foreach pin [$inst getITerms] {
   set term [$pin getMTerm]
   if {[$term getSigType] in {POWER GROUND}} {continue}
   set net [$pin getNet]
   set value NULL
   if {[$term getIoType] eq "INPUT" && ([$term getSigType] eq "CLOCK" || [$term getName] in {CLK clk CK clock})} {
    set key [$net getName]
    if {![dict exists $clocks $key]} {dict set clocks $key [simd_clock_root $net]}
    set value [dict get $clocks $key]
   } elseif {$net ne "NULL"} {set value [$net getName]}
   dict set state "$name/[$term getName]" $value
  }
 }
 return $state
}
proc simd_protected_cts {args} {
 set block [ord::get_db_block]
 set names {}
 foreach inst [$block getInsts] {
  if {[$inst isDoNotTouch]} {lappend names [$inst getName]}
 }
 set before [simd_protected_clock_state $names]
 foreach name $names {[$block findInst $name] setDoNotTouch false}
 try {
  clock_tree_synthesis {*}$args
 } finally {
  foreach name $names {
   set inst [$block findInst $name]
   if {$inst eq "NULL"} {error "Protected instance removed during CTS: $name"}
   $inst setDoNotTouch true
  }
 }
 set after [simd_protected_clock_state $names]
 if {$before ne $after} {error "CTS changed a protected cell, signal, clock root or clock phase"}
 puts "SIMD_PROTECTED_CTS_CLOCKS_AND_SIGNALS_VERIFIED [llength $names]"
}
