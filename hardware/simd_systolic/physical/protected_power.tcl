# OpenDB global_connect skips dont_touch instances, including synchronizers.
# Connect only their POWER/GROUND pins in this single-supply design. Protection
# is released solely around those database edits and restored before returning;
# signal connections, placement, timing and optimization are never changed.
proc simd_connect_protected_supplies {} {
 set block [ord::get_db_block]
 set connected 0
 foreach inst [$block getInsts] {
  if {![$inst isDoNotTouch]} {continue}
  set pending {}
  foreach iterm [$inst getITerms] {
   set kind [[$iterm getMTerm] getSigType]
   if {$kind eq "POWER"} {set supply VDD} elseif {$kind eq "GROUND"} {set supply VSS} else {continue}
   set target [$block findNet $supply]
   if {$target eq "NULL"} {error "Missing $supply net before protected supply connection"}
   set prior [$iterm getNet]
   if {$prior ne "NULL" && $prior ne $target} {
    error "Protected supply pin already belongs to another power net: [$inst getName]/[[$iterm getMTerm] getName]"
   }
   if {$prior eq "NULL"} {lappend pending [list $iterm $target]}
  }
  if {![llength $pending]} {continue}
  $inst setDoNotTouch false
  try {
   foreach pair $pending {
    lassign $pair pin net
    $pin connect $net
    # Match global_connect's special-pin classification. Otherwise DEF also
    # emits these supply connections as ordinary signal NETS.
    $pin setSpecial
    incr connected
   }
  } finally {
   $inst setDoNotTouch true
  }
 }
 puts "SIMD_PROTECTED_SUPPLY_PINS_CONNECTED $connected"
 return $connected
}
