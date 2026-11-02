# Change internal net labels only. Keep top-level ports, cells and routing.
# The caller compares full DEFs after restoring the original labels, and
# connectivity snapshots taken on both sides of the canonical export.
proc simd_export_connectivity {path} {
 set block [ord::get_db_block]
 set stream [open $path w]
 foreach inst [$block getInsts] {
  puts $stream [list INST $inst [[$inst getMaster] getName] [$inst getLocation] [$inst getOrient] [$inst getPlacementStatus] [$inst isDoNotTouch]]
 }
 foreach net [$block getNets] {
  puts $stream [list NET $net [$net getWire] [$net getSigType] [$net isSpecial] [$net isDoNotTouch]]
  foreach pin [$net getITerms] {puts $stream [list ITERM $pin [$pin getInst] [[$pin getMTerm] getName] [$pin getNet]]}
  foreach pin [$net getBTerms] {puts $stream [list BTERM $pin [$pin getName] [$pin getNet]]}
 }
 close $stream
}

proc simd_export_canonical_net_names {out} {
 set block [ord::get_db_block]
 set prefix simd_export_internal_
 foreach net [$block getNets] {
  if {[string match "$prefix*" [$net getName]]} {error "Canonical naming prefix already exists"}
 }
 write_def $out/before_rename.def
 simd_export_connectivity $out/before_connectivity.tcllist
 set map [open $out/net_names.tsv w]
 puts $map "index\tcanonical_name\toriginal_utf8_hex"
 set restore {}
 set index 0
 try {
  foreach net [$block getNets] {
   if {[$net getSigType] in {POWER GROUND} || [$net isSpecial] || [llength [$net getBTerms]]} {continue}
   set old [$net getName]
   set new "$prefix$index"
   if {![$net rename $new]} {error "Internal net rename failed"}
   lappend restore [list $net $old]
   puts $map "$index\t$new\t[binary encode hex [encoding convertto utf-8 $old]]"
   incr index
  }
  close $map
  simd_export_connectivity $out/canonical_connectivity.tcllist
  write_db $out/canonical.odb
  write_verilog $out/routed.v
  puts "SIMD_CANONICAL_NETS_EXPORTED $index"
 } finally {
  foreach pair $restore {
   lassign $pair net old
   if {![$net rename $old]} {error "Original net label restoration failed"}
  }
 }
 write_def $out/restored_names.def
 puts SIMD_CANONICAL_EXPORT_ROUNDTRIP_COMPLETE
}
