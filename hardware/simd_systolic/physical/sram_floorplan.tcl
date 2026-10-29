# Place the sixteen 8KiB macros on a checked 4x4 grid before cell placement.
set block [ord::get_db_block]
set core [$block getCoreArea]
set macros {}
set sram_master "srambank_256x4x64_6t122"
if {[info exists ::env(SIMD_SRAM_MASTER)]} {set sram_master $::env(SIMD_SRAM_MASTER)}
foreach inst [$block getInsts] {
 if {[[$inst getMaster] getName] eq $sram_master} {lappend macros $inst}
}
if {[llength $macros] != 16} {error "Expected exactly sixteen SRAM hard macros"}
set dbu [$block getDbUnitsPerMicron]
set halo [expr {int(2.0*$dbu)}]
set pitchx [expr {([$core xMax]-[$core xMin])/4}]
set pitchy [expr {([$core yMax]-[$core yMin])/4}]
set index 0
foreach inst $macros {
 set master [$inst getMaster]
 if {[$master getWidth]+2*$halo > $pitchx || [$master getHeight]+2*$halo > $pitchy} {
  error "SRAM grid cannot fit macros and 2um halos; enlarge the floorplan"
 }
 set x [expr {[$core xMin]+($index%4)*$pitchx+$halo}]
 set y [expr {[$core yMin]+($index/4)*$pitchy+$halo}]
 $inst setOrient R0
 $inst setLocation $x $y
 $inst setPlacementStatus FIRM
 incr index
}
cut_rows -halo_width_x 2 -halo_width_y 2
add_global_connection -net VDD -pin_pattern {^(VDD|vdd)$} -power
add_global_connection -net VSS -pin_pattern {^(VSS|vss)$} -ground
global_connect
source [file join [file dirname [info script]] protected_power.tcl]
simd_connect_protected_supplies
source $::env(SIMD_PDN_SCRIPT)
pdngen -failed_via_report $::env(SIMD_OUT)/pdn_failed_vias.rpt
# A global net name does not prove that macro supply shapes connect to the
# grid. In particular, generated research macros may expose M3 pins while
# the platform's reference macro strategy expects M4. Retain the actual
# disconnected-pin report and fail here until its access/strap geometry is
# implemented and checked; do not continue to a misleading system result.
check_power_grid -net VDD -floorplanning -error_file $::env(SIMD_OUT)/pdn_VDD.rpt
check_power_grid -net VSS -floorplanning -error_file $::env(SIMD_OUT)/pdn_VSS.rpt
