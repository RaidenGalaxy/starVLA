# Net labels can shadow a different top-level bus bit after buffer_ports.
# Keep every BTerm name and all connections; rename only those shadowing nets.
proc simd_export_port_aliases {out} {
 set block [ord::get_db_block]
 set targets {}
 set index 0
 foreach net [$block getNets] {
  set port [$block findBTerm [$net getName]]
  if {$port eq "NULL" || [$port getNet] eq $net} {continue}
  if {[$net isSpecial] || [$net getSigType] in {POWER GROUND}} {error "Protected supply shadows a port"}
  set new simd_port_alias_export_$index
  if {[$block findNet $new] ne "NULL" || [$block findBTerm $new] ne "NULL"} {error "Port export prefix collision"}
  lappend targets [list $net [$net getName] $new]
  incr index
 }
 write_def $out/before.def
 simd_export_connectivity $out/before_connectivity.tsv
 set stream [open $out/port_net_names.tsv w]
 puts $stream "original_name\tunique_net_name"
 try {
  foreach target $targets {
   lassign $target net old new
   if {![$net rename $new]} {error "Port-shadowing net rename failed"}
   puts $stream "$old\t$new"
  }
  close $stream
  simd_export_connectivity $out/export_connectivity.tsv
  write_db $out/repaired.odb
  write_verilog $out/repaired.v
 } finally {
  foreach target $targets {
   lassign $target net old new
   if {[$net getName] ne $old && ![$net rename $old]} {error "Port net label restoration failed"}
  }
 }
 write_def $out/restored.def
 simd_export_connectivity $out/restored_connectivity.tsv
 puts "SIMD_PORT_SHADOWING_NETS_EXPORTED $index"
}
