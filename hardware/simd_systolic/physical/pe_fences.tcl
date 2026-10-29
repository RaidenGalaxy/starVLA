proc simd_apply_pe_fences {} {
# Partition the actual standard-cell PE instances into neighboring 2x2 groups.
# Regions and groups are written into the ODB/DEF used by placement and routing.
set simd_block [ord::get_db_block]
set simd_core [$simd_block getCoreArea]
set simd_rows $::env(SIMD_ARRAY_ROWS)
set simd_cols $::env(SIMD_ARRAY_COLS)
set simd_nr [expr {($simd_rows+1)/2}]
set simd_nc [expr {($simd_cols+1)/2}]
if {$simd_nr*$simd_nc==1} {
 # A 2x2 design is already one physical cluster. Keep its collector and
 # peripheral controls in that cluster's core, rather than forcing them
 # into an artificial perimeter outside a single fenced PE group.
 set report [open $::env(SIMD_OUT)/placement_regions.txt w]
 puts $report "single_2x2_cluster whole_core [llength [$simd_block getInsts]] cells"
 close $report
 puts "SIMD_SINGLE_2X2_CLUSTER: whole core"
 return
}
set simd_width [expr {[$simd_core xMax]-[$simd_core xMin]}]
set simd_height [expr {[$simd_core yMax]-[$simd_core yMin]}]
set simd_site [[lindex [$simd_block getRows] 0] getSite]
set simd_pitchx [$simd_site getWidth]
set simd_pitchy [$simd_site getHeight]
array set simd_groups {}
array set simd_counts {}
for {set r 0} {$r<$simd_nr} {incr r} {
 for {set c 0} {$c<$simd_nc} {incr c} {
  set region [odb::dbRegion_create $simd_block simd_2x2_${r}_${c}]
  $region setRegionType FENCE
  set x0 [expr {int([$simd_core xMin]+($c+0.025)*$simd_width/$simd_nc)}]
  set x1 [expr {int([$simd_core xMin]+($c+0.975)*$simd_width/$simd_nc)}]
  set y0 [expr {int([$simd_core yMin]+($r+0.025)*$simd_height/$simd_nr)}]
  set y1 [expr {int([$simd_core yMin]+($r+0.975)*$simd_height/$simd_nr)}]
  set x0 [expr {[$simd_core xMin]+int(ceil(double($x0-[$simd_core xMin])/$simd_pitchx))*$simd_pitchx}]
  set x1 [expr {[$simd_core xMin]+int(floor(double($x1-[$simd_core xMin])/$simd_pitchx))*$simd_pitchx}]
  set y0 [expr {[$simd_core yMin]+int(ceil(double($y0-[$simd_core yMin])/$simd_pitchy))*$simd_pitchy}]
  set y1 [expr {[$simd_core yMin]+int(floor(double($y1-[$simd_core yMin])/$simd_pitchy))*$simd_pitchy}]
  odb::dbBox_create $region $x0 $y0 $x1 $y1
  set simd_groups($r,$c) [odb::dbGroup_create $region simd_2x2_${r}_${c}]
  # This pinned GPL iterates only groups registered with dbPowerDomain.
  # These objects provide placement partitions, with no power-switch, supply
  # network or UPF insertion. Omitting them silently excludes grouped cells
  # from global placement (Replace::initNesterovPlace in the pinned source).
  set domain [odb::dbPowerDomain_create $simd_block simd_placement_${r}_${c}]
  $domain setGroup $simd_groups($r,$c)
  set simd_counts($r,$c) 0
 }
}
foreach inst [$simd_block getInsts] {
 if {[regexp {^pes_([0-9]+)_([0-9]+)/} [$inst getName] -> r c]} {
  set rr [expr {$r/2}];set cc [expr {$c/2}]
  $simd_groups($rr,$cc) addInst $inst
  incr simd_counts($rr,$cc)
 }
}
set report [open $::env(SIMD_OUT)/placement_regions.txt w]
foreach key [lsort [array names simd_groups]] {
 if {$simd_counts($key)==0} {error "PE hierarchy missing from region $key"}
 puts $report "$key $simd_counts($key)"
 puts "SIMD_2X2_REGION $key $simd_counts($key) cells"
}
close $report

}
simd_apply_pe_fences
