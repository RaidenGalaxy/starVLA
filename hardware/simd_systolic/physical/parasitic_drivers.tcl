# Prove zero fanout using the actual routed database. No timing exception.
proc simd_audit_parasitic_driver {name} {
 set block [ord::get_db_block]
 set slash [string last / $name]
 if {$slash<0} {
  set pin [$block findBTerm $name]
  if {$pin eq "NULL" || [$pin getIoType] ne "INPUT"} {return [list unknown_port -1 -1]}
 } else {
  set inst [$block findInst [string range $name 0 [expr {$slash-1}]]]
  if {$inst eq "NULL"} {return [list unknown_cell -1 -1]}
  set pin [$inst findITerm [string range $name [expr {$slash+1}] end]]
  if {$pin eq "NULL" || [[$pin getMTerm] getIoType] ne "OUTPUT"} {return [list unknown_output -1 -1]}
 }
 set net [$pin getNet]
 if {$net eq "NULL"} {return [list unconnected_output 0 0]}
 set sinks 0
 foreach term [$net getITerms] {
  if {[[$term getMTerm] getIoType] in {INPUT INOUT} && [[$term getMTerm] getSigType] ni {POWER GROUND}} {incr sinks}
 }
 foreach term [$net getBTerms] {if {[$term getIoType] in {OUTPUT INOUT}} {incr sinks}}
 set routed [expr {[$net getWire] ne "NULL" || [llength [$net getSWires]]>0}]
 if {!$sinks && !$routed} {return [list zero_fanout_no_routing 0 0]}
 return [list connected_or_routed $sinks $routed]
}
