# Pinned OpenROAD exposes the same electrical repair engine per net. This
# avoids sweeping every arithmetic net to fix a small number of reset trees.
proc simd_reset_logic_root {net {polarity 0} {visited {}}} {
 set cache [dict create]
 set result [simd_reset_logic_root_cached $net $visited cache]
 return [list [lindex $result 0] [expr {[lindex $result 1] ^ $polarity}]]
}
# The caller owns the cache for one immutable connectivity snapshot. Never
# reuse it across a resizer command: that would conceal a changed function.
proc simd_reset_logic_root_cached {net visited cache_name} {
 upvar 1 $cache_name cache
 if {$net eq "NULL"} {error "Disconnected reset input"}
 set name [$net getName]
 if {$name in $visited} {error "Loop in reset fanin"}
 if {[dict exists $cache $name]} {return [dict get $cache $name]}
 lappend visited $name
 set roots {}
 foreach pin [$net getBTerms] {if {[$pin getIoType] eq "INPUT"} {lappend roots [$pin getName]}}
 set drivers {}
 foreach pin [$net getITerms] {if {[[$pin getMTerm] getIoType] eq "OUTPUT"} {lappend drivers $pin}}
 if {[llength $roots] == 1 && ![llength $drivers]} {
  set result [list "PORT:[lindex $roots 0]" 0]
  dict set cache $name $result
  return $result
 }
 if {[llength $roots] || [llength $drivers] != 1} {error "Ambiguous reset driver: $name"}
 set pin [lindex $drivers 0]
 set inst [$pin getInst]
 set master [[$inst getMaster] getName]
 if {[string match BUF* $master]} {set inversion 0} elseif {[regexp {^(INV|CKINV)} $master]} {
  set inversion 1
 } else {
  set result [list "PIN:[$inst getName]/[[$pin getMTerm] getName]" 0]
  dict set cache $name $result
  return $result
 }
 set input [$inst findITerm A]
 if {$input eq "NULL"} {error "Reset buffer has no A input"}
 set parent [simd_reset_logic_root_cached [$input getNet] $visited cache]
 set result [list [lindex $parent 0] [expr {[lindex $parent 1] ^ $inversion}]]
 dict set cache $name $result
 return $result
}
proc simd_reset_protected_state {names {root_pins {}}} {
 set block [ord::get_db_block]
 set state [dict create]
 set cache [dict create]
 foreach name $names {
  set inst [$block findInst $name]
  if {$inst eq "NULL"} {error "Protected reset sink removed: $name"}
  dict set state "$name/master" [[$inst getMaster] getName]
  foreach pin [$inst getITerms] {
   set term [$pin getMTerm]
   if {[$term getSigType] in {POWER GROUND}} {continue}
   set net [$pin getNet]
   set value NULL
   if {$net ne "NULL"} {
    set value [$net getName]
    if {[$term getName] in {RESETN SETN} || "$name/[$term getName]" in $root_pins} {
     set value [simd_reset_logic_root_cached $net {} cache]
    }
   }
   dict set state "$name/[$term getName]" $value
  }
 }
 return $state
}
proc simd_repair_reset_fanout {minimum max_length_um report} {
 set block [ord::get_db_block]
 set masters [dict create]
 set candidates [dict create]
 foreach inst [$block getInsts] {
  set master [$inst getMaster]
  set key [$master getName]
  if {![dict exists $masters $key]} {
   set resets {}
   foreach term [$master getMTerms] {
    if {[$term getName] in {RESETN SETN} && [$term getIoType] eq "INPUT"} {lappend resets [$term getName]}
   }
   dict set masters $key $resets
  }
  foreach name [dict get $masters $key] {
   set pin [$inst findITerm $name]
   set net [$pin getNet]
   if {$net ne "NULL"} {dict lappend candidates [$net getName] $pin}
  }
 }
 set selected [dict create]
 set loads [dict create]
 foreach name [dict keys $candidates] {
  set pins [dict get $candidates $name]
  set net [$block findNet $name]
  set sinks {}
  set clock_net false
  foreach pin [$net getITerms] {
   set term [$pin getMTerm]
   if {[$term getIoType] ne "INPUT" || [$term getSigType] in {POWER GROUND}} {continue}
   if {[$term getSigType] eq "CLOCK" || [$term getName] in {CLK clk CK clock}} {set clock_net true}
   lappend sinks $pin
  }
  if {$clock_net || [llength $sinks] < $minimum} {continue}
  set root [lindex [simd_reset_logic_root $net] 0]
  if {[string match PIN:* $root]} {
   # Hierarchical names contain '/'; remove only the final pin component.
   set source_name [string range $root 4 [expr {[string last / $root]-1}]]
   set driver [$block findInst $source_name]
   if {$driver ne "NULL" && [string match TIE* [[$driver getMaster] getName]]} {continue}
  }
  dict set selected $name $pins
  dict set loads $name $sinks
 }
 set stream [open $report w]
 puts $stream "net\treset_sinks\ttotal_input_sinks\tlogical_source_and_polarity"
 set targets {}
 foreach name [dict keys $selected] {
  set pins [dict get $selected $name]
  set sinks [dict get $loads $name]
  puts $stream "$name\t[llength $pins]\t[llength $sinks]\t[simd_reset_logic_root [$block findNet $name]]"
  foreach pin $pins {lappend targets "[[$pin getInst] getName]/[[$pin getMTerm] getName]"}
  set protected {}
  set root_pins {}
  foreach pin $sinks {
   set inst [$pin getInst]
   lappend root_pins "[$inst getName]/[[$pin getMTerm] getName]"
   if {[$inst isDoNotTouch]} {dict set protected [$inst getName] 1}
  }
  set names [dict keys $protected]
  set before [simd_reset_protected_state $names $root_pins]
  set original [simd_reset_logic_root [$block findNet $name]]
  foreach n $names {[$block findInst $n] setDoNotTouch false}
  try {
   rsz::repair_net_cmd [lindex [get_nets [list $name]] 0] [sta::distance_ui_sta $max_length_um] 20 20
  } finally {
   foreach n $names {[$block findInst $n] setDoNotTouch true}
  }
  if {$before ne [simd_reset_protected_state $names $root_pins]} {error "Reset repair changed a protected cell, data, clock or reset function"}
  set after_cache [dict create]
  foreach pin $sinks {
   if {[simd_reset_logic_root_cached [$pin getNet] {} after_cache] ne $original} {error "Reset repair changed a sink's logic or polarity"}
  }
 }
 close $stream
 puts "SIMD_RESET_NETS_REPAIRED [dict size $selected]"
 return $targets
}
