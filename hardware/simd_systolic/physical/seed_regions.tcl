# Deterministic, nonoverlapping initial positions inside each existing fence.
# Large arrays must not start every PE at the whole-chip origin outside its
# own region. GPL still optimizes these positions; this is not a fixed layout.
proc simd_seed_box {instances x0 y0 x1 y1 pitchy label} {
 set width [expr {$x1-$x0}]
 set length 0
 set widest 0
 foreach inst $instances {
  set w [[$inst getMaster] getWidth]
  set h [[$inst getMaster] getHeight]
  if {$h!=$pitchy} {error "regional seed requires single-height cells: [$inst getName]"}
  incr length $w
  if {$w>$widest} {set widest $w}
 }
 if {$length==0} {return}
 if {$width<=$widest} {error "regional seed box too narrow: $label"}
 # Reserve one maximum cell width per row for packing fragmentation.
 set needed [expr {int(ceil(double($length)/($width-$widest)))}]
 set available [expr {int(floor(double($y1-$y0)/$pitchy))}]
 if {$needed>$available} {error "regional seed capacity exceeded: $label"}
 set step [expr {max(1,int($available/$needed))*$pitchy}]
 set x $x0
 set y $y0
 foreach inst $instances {
  set w [[$inst getMaster] getWidth]
  if {$x+$w>$x1} {set x $x0;incr y $step}
  if {$y+$pitchy>$y1} {error "regional seed packing exceeded box: $label"}
  $inst setLocation $x $y
  $inst setPlacementStatus PLACED
  incr x $w
 }
 puts "SIMD_REGION_SEEDED $label [llength $instances] cells"
}
set simd_seed_block [ord::get_db_block]
set simd_seed_core [$simd_seed_block getCoreArea]
set simd_seed_site [[lindex [$simd_seed_block getRows] 0] getSite]
set simd_seed_pitchy [$simd_seed_site getHeight]
foreach domain [$simd_seed_block getPowerDomains] {
 set group [$domain getGroup]
 if {$group=="NULL"} {continue}
 set boxes [[$group getRegion] getBoundaries]
 if {[llength $boxes]!=1} {error "regional seed requires one rectangle per PE fence"}
 set box [lindex $boxes 0]
 simd_seed_box [$group getInsts] [$box xMin] [$box yMin] [$box xMax] [$box yMax] $simd_seed_pitchy [$group getName]
}
set simd_seed_others {}
foreach inst [$simd_seed_block getInsts] {
 if {[$inst getGroup]=="NULL"} {lappend simd_seed_others $inst}
}
# The narrow left perimeter is outside every fenced PE region. GPL may move
# peripheral cells through the other free corridors during optimization.
set simd_seed_x0 [$simd_seed_core xMin]
set simd_seed_width [expr {[$simd_seed_core xMax]-$simd_seed_x0}]
set simd_seed_nc [expr {($::env(SIMD_ARRAY_COLS)+1)/2}]
set simd_seed_pitchx [$simd_seed_site getWidth]
set simd_seed_strip [expr {int(floor(0.025*$simd_seed_width/$simd_seed_nc/$simd_seed_pitchx))*$simd_seed_pitchx}]
simd_seed_box $simd_seed_others $simd_seed_x0 [$simd_seed_core yMin] \
 [expr {$simd_seed_x0+$simd_seed_strip}] [$simd_seed_core yMax] $simd_seed_pitchy periphery
write_db $::env(SIMD_OUT)/seeded_placement.odb
