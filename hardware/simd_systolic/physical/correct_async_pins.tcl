# Correct the legacy mapper's logical-Q reset/preset reversal. No resizer,
# placement, clock, routing or timing-exception command is used here.
proc simd_async_inactive {net cache_name} {
 upvar 1 $cache_name cache
 lassign [simd_reset_logic_root_cached $net {} cache] root inversion
 if {![string match PIN:* $root]} {return false}
 set split [string last / $root]
 set inst [[ord::get_db_block] findInst [string range $root 4 [expr {$split-1}]]]
 if {$inst eq "NULL"} {return false}
 set master [[$inst getMaster] getName]
 if {[string match TIEHI* $master]} {return [expr {!$inversion}]}
 if {[string match TIELO* $master]} {return [expr {$inversion}]}
 return false
}
proc simd_async_geometry {path} {
 set stream [open $path w]
 foreach inst [[ord::get_db_block] getInsts] {
  puts $stream [list [$inst getName] [[$inst getMaster] getName] [$inst getLocation] [$inst getOrient] [$inst getPlacementStatus] [$inst isDoNotTouch]]
 }
 close $stream
}
proc simd_correct_async_pins {expected report} {
 set block [ord::get_db_block]
 set candidates {}
 set cache [dict create]
 foreach inst [$block getInsts] {
  set master [[$inst getMaster] getName]
  if {![string match DFFAS* $master]} {continue}
  if {$master ne "DFFASRHQNx1_ASAP7_75t_R"} {error "Unsupported async master: $master"}
  set reset [[$inst findITerm RESETN] getNet]
  set preset [[$inst findITerm SETN] getNet]
  if {![simd_async_inactive $reset cache] && ![simd_async_inactive $preset cache]} {
   error "Both async inputs may be active; correction requires a single-reset flop: [$inst getName]"
  }
  lappend candidates [list $inst $reset $preset]
 }
 if {[llength $candidates]!=$expected || !$expected} {error "Actual async flop inventory differs"}
 set stream [open $report w]
 puts $stream "instance\told_RESETN\told_SETN\tnew_RESETN\tnew_SETN"
 foreach entry $candidates {
  lassign $entry inst reset preset
  set protected [$inst isDoNotTouch]
  if {$protected} {$inst setDoNotTouch false}
  try {
   [$inst findITerm RESETN] connect $preset
   [$inst findITerm SETN] connect $reset
  } finally {
   if {$protected} {$inst setDoNotTouch true}
  }
  if {[[$inst findITerm RESETN] getNet] ne $preset || [[$inst findITerm SETN] getNet] ne $reset} {
   error "Async pin swap did not take effect"
  }
  puts $stream "[$inst getName]\t[$reset getName]\t[$preset getName]\t[$preset getName]\t[$reset getName]"
 }
 close $stream
 puts "SIMD_ASYNC_PINS_CORRECTED [llength $candidates]"
}
