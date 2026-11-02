# Explicit net candidates come from a source-bound mapped netlist census.
# OpenDB checks the actual loads before the pinned resizer touches each net.
# reset_repair.tcl supplies the buffer/inverter logic-root and protected-state
# guards. Clocks and constant ties are excluded; no timing exception is added.
proc simd_repair_control_fanout {candidates minimum limit max_length_um report {checkpoint {}}} {
 set block [ord::get_db_block]
 set stream [open $report w]
 puts $stream "net\tinput_sinks\tlogical_source_and_polarity"
 flush $stream
 set repaired {}
 try {
  foreach name $candidates {
   if {[llength $repaired] >= $limit} {break}
   set net [$block findNet $name]
   if {$net eq "NULL"} {error "Census candidate is absent from the checkpoint: $name"}
   if {[$net getSigType] in {POWER GROUND CLOCK}} {continue}
   set sinks {}
   set clock_net false
   foreach pin [$net getITerms] {
    set term [$pin getMTerm]
    if {[$term getIoType] ne "INPUT" || [$term getSigType] in {POWER GROUND}} {continue}
    if {[$term getSigType] eq "CLOCK" || [$term getName] in {CLK clk CK clock}} {set clock_net true}
    lappend sinks $pin
   }
   if {$clock_net || [llength $sinks] < $minimum} {continue}
   set original [simd_reset_logic_root $net]
   set root [lindex $original 0]
   if {[string match PIN:* $root]} {
    set driver [$block findInst [string range $root 4 [expr {[string last / $root]-1}]]]
    if {$driver ne "NULL" && [string match TIE* [[$driver getMaster] getName]]} {continue}
   }
   set protected [dict create]
   set root_pins {}
   foreach pin $sinks {
    set inst [$pin getInst]
    lappend root_pins "[$inst getName]/[[$pin getMTerm] getName]"
    if {[$inst isDoNotTouch]} {dict set protected [$inst getName] 1}
   }
   set names [dict keys $protected]
   set before [simd_reset_protected_state $names $root_pins]
   puts $stream "$name\t[llength $sinks]\t$original"
   flush $stream
   puts "SIMD_CONTROL_REPAIR_BEGIN $name [llength $sinks]"
   flush stdout
   foreach n $names {[$block findInst $n] setDoNotTouch false}
   try {
    rsz::repair_net_cmd [lindex [get_nets [list $name]] 0] [sta::distance_ui_sta $max_length_um] 20 20
   } finally {
    foreach n $names {[$block findInst $n] setDoNotTouch true}
   }
   if {$before ne [simd_reset_protected_state $names $root_pins]} {
    error "Control repair changed a protected cell, data, clock or input function"
   }
   set after_cache [dict create]
   foreach pin $sinks {
    if {[simd_reset_logic_root_cached [$pin getNet] {} after_cache] ne $original} {error "Control repair changed a sink's logic or polarity"}
   }
   lappend repaired $name
   puts "SIMD_CONTROL_REPAIR_END $name"
   flush stdout
   if {$checkpoint ne ""} {
    # Keep the last fully checked net even if later timing/reporting times
    # out. This is a connectivity checkpoint; legalization and PG follow.
    write_db "${checkpoint}.pending"
    file rename -force "${checkpoint}.pending" $checkpoint
    puts "SIMD_CONTROL_CHECKPOINT_WRITTEN $name"
    flush stdout
   }
  }
 } finally {close $stream}
 puts "SIMD_CONTROL_NETS_REPAIRED [llength $repaired]"
 return $repaired
}
