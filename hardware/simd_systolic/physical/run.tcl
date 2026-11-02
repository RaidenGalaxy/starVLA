# Post-route reference flow. Inputs and paths are supplied by run.py.
set_thread_count 8
read_lef $::env(SIMD_TECH_LEF)
read_lef $::env(SIMD_CELL_LEF)
read_liberty $::env(SIMD_LIB)
if {[info exists ::env(SIMD_MACRO_LEF)]} {
 read_lef $::env(SIMD_MACRO_LEF)
 read_liberty $::env(SIMD_MACRO_LIB)
}
set_cmd_units -time ns -capacitance fF
report_units
# Pinned ASAP7 platform routing policy, applied on fresh and resumed DBs.
proc simd_routing_layers {} {
 set_dont_use [get_lib_cells {*/*x1p*_ASAP7* */*xp*_ASAP7* */SDF* */ICG*}]
 set_routing_layers -signal M2-M7 -clock M4-M7
 set_global_routing_layer_adjustment M2-M3 0.5
 set_global_routing_layer_adjustment M4-M7 0.25
}
proc simd_connect_system_supplies {} {
 if {![info exists ::env(SIMD_SYSTEM)]} {return}
 if {![llength [info commands simd_connect_protected_supplies]]} {
  source [file join [file dirname $::env(SIMD_MACRO_FLOORPLAN)] protected_power.tcl]
 }
 global_connect
 simd_connect_protected_supplies
}
if {[info exists ::env(SIMD_RESUME_CTS)] || [info exists ::env(SIMD_RESUME_INITIAL_HOLD)] || [info exists ::env(SIMD_RESUME_GLOBAL_ROUTE)] || [info exists ::env(SIMD_RESUME_DETAILED_ROUTE)]} {
 if {[info exists ::env(SIMD_RESUME_DETAILED_ROUTE)]} {
  read_db $::env(SIMD_OUT)/post_detailed_route.odb
 } elseif {[info exists ::env(SIMD_RESUME_GLOBAL_ROUTE)]} {
  read_db $::env(SIMD_OUT)/post_global_route.odb
 } elseif {[info exists ::env(SIMD_RESUME_INITIAL_HOLD)]} {
  read_db $::env(SIMD_OUT)/post_initial_hold.odb
 } else {read_db $::env(SIMD_OUT)/post_cts.odb}
 read_sdc $::env(SIMD_SDC)
source $::env(SIMD_SET_RC)
 set_wire_rc -signal -layer M3
 set_wire_rc -clock -layer M5
 simd_routing_layers
} else {
read_verilog $::env(SIMD_NETLIST)
link_design $::env(SIMD_TOP)
read_sdc $::env(SIMD_SDC)
source $::env(SIMD_SET_RC)
initialize_floorplan -utilization 30 -aspect_ratio 1 -core_space 3 \
    -site asap7sc7p5t
if {[info exists ::env(SIMD_FENCE_SCRIPT)]} {source $::env(SIMD_FENCE_SCRIPT)}
source $::env(SIMD_PROTECTED_TIES)
simd_repair_protected_ties
source $::env(SIMD_TRACKS)
if {[info exists ::env(SIMD_SYSTEM)]} {source $::env(SIMD_MACRO_FLOORPLAN)}
set_wire_rc -signal -layer M3
set_wire_rc -clock -layer M5
simd_routing_layers
buffer_ports
place_pins -hor_layers M4 -ver_layers M5
set placement_grid_args {}
if {[info exists ::env(SIMD_PLACE_GRID)]} {
 lappend placement_grid_args -bin_grid_count $::env(SIMD_PLACE_GRID)
}
if {[info exists ::env(SIMD_REGION_SEED_SCRIPT)]} {
 source $::env(SIMD_REGION_SEED_SCRIPT)
 global_placement -density 0.35 -skip_initial_place -routability_driven {*}$placement_grid_args
} elseif {[info exists ::env(SIMD_GEOMETRIC_PLACEMENT)]} {
 # Large full-array placement can spend hours in virtual resizing. This
 # profile changes placement search only; all clocks/paths remain timed
 # during subsequent repair, routing, and extracted signoff.
 global_placement -density 0.35 -initial_place_max_iter 20 -routability_driven {*}$placement_grid_args
} else {
 global_placement -density 0.35 -initial_place_max_iter 20 -timing_driven -routability_driven {*}$placement_grid_args
}
write_db $::env(SIMD_OUT)/post_global.odb
detailed_placement
estimate_parasitics -placement
repair_design -cap_margin 20 -slew_margin 20
detailed_placement
source $::env(SIMD_PROTECTED_CLOCKS)
if {[info exists ::env(SIMD_CTS_CLUSTER_SIZE)]} {
 simd_protected_cts -root_buf $::env(SIMD_CTS_ROOT) -buf_list $::env(SIMD_CTS_BUFFERS) \
  -sink_clustering_enable -sink_clustering_size $::env(SIMD_CTS_CLUSTER_SIZE) \
  -sink_clustering_max_diameter $::env(SIMD_CTS_DIAMETER) -balance_levels
} elseif {[info exists ::env(SIMD_CTS_ROOT)]} {
 simd_protected_cts -root_buf $::env(SIMD_CTS_ROOT) -buf_list $::env(SIMD_CTS_BUFFERS)
} else {
 simd_protected_cts -root_buf BUFx8_ASAP7_75t_R -buf_list {BUFx2_ASAP7_75t_R BUFx4_ASAP7_75t_R BUFx8_ASAP7_75t_R}
}
}
set_propagated_clock [all_clocks]
set setup_repair_margin 0.050
if {[info exists ::env(SIMD_SETUP_MARGIN)]} {set setup_repair_margin $::env(SIMD_SETUP_MARGIN)}
estimate_parasitics -placement
simd_connect_system_supplies
if {![info exists ::env(SIMD_RESUME_INITIAL_HOLD)] && ![info exists ::env(SIMD_RESUME_GLOBAL_ROUTE)] && ![info exists ::env(SIMD_RESUME_DETAILED_ROUTE)]} {
write_db $::env(SIMD_OUT)/post_cts.odb
report_checks -path_delay max -group_count 10 -format full_clock_expanded -digits 9 \
    > $::env(SIMD_OUT)/preopt_setup.rpt
report_checks -path_delay min -group_count 10 -format full_clock_expanded -digits 9 \
    > $::env(SIMD_OUT)/preopt_hold.rpt
puts "SIMD_CTS_SETUP"
report_worst_slack -max -digits 9
puts "SIMD_CTS_HOLD"
report_worst_slack -min -digits 9
report_design_area
if {[info exists ::env(SIMD_CTS_ONLY)]} { exit }
set initial_slack [sta::format_time [sta::worst_slack_cmd max] 6]
if {$initial_slack > -0.5} {
 repair_timing -setup -setup_margin $setup_repair_margin -max_passes 10 -repair_tns 100
} else {
 puts "SIMD_SETUP_REQUIRES_ARITHMETIC_CHANGE: pre-repair slack $initial_slack ns"
}
# Hold repair must preserve setup. Retain unresolved endpoints in the final
# report rather than consuming setup slack without limit to hide hold errors.
repair_timing -hold -max_passes 12 -hold_margin 0.050 -max_buffer_percent 40 -verbose
detailed_placement
simd_connect_system_supplies
write_db $::env(SIMD_OUT)/post_initial_hold.odb
} else {
 if {[info exists ::env(SIMD_RESUME_DETAILED_ROUTE)]} {puts SIMD_RESUMED_DETAILED_ROUTE_CHECKPOINT} elseif {[info exists ::env(SIMD_RESUME_GLOBAL_ROUTE)]} {puts SIMD_RESUMED_GLOBAL_ROUTE_CHECKPOINT} else {puts SIMD_RESUMED_INITIAL_HOLD_CHECKPOINT}
 report_clock_properties [all_clocks]
}
if {![info exists ::env(SIMD_RESUME_DETAILED_ROUTE)]} {
if {![info exists ::env(SIMD_RESUME_GLOBAL_ROUTE)]} {
global_route -guide_file $::env(SIMD_OUT)/route.guide -congestion_iterations 50
estimate_parasitics -global_routing
global_route -start_incremental
# Electrical repair can resize delay buffers. Run it before setup/hold
# timing repair, so it cannot undo the last hold fix immediately afterwards.
repair_design -cap_margin 20 -slew_margin 20
set routed_estimate_slack [sta::format_time [sta::worst_slack_cmd max] 6]
if {$routed_estimate_slack > -0.5} {
 repair_timing -setup -setup_margin $setup_repair_margin -max_passes 10 -repair_tns 100
}
# Save the router-estimated paths before hold repair: a bulk guard band can
# create more buffers than this design can route. Earlier 180ps attempts hit
# the buffer limit/congestion. Start with 50ps, then use the extracted failing
# endpoints for further ECOs; acceptance remains strict extracted STA.
write_db $::env(SIMD_OUT)/pre_global_hold.odb
report_checks -path_delay max -group_count 20 -format full_clock_expanded -digits 9 > $::env(SIMD_OUT)/pre_global_hold_setup.rpt
report_checks -path_delay min -group_count 20 -format full_clock_expanded -digits 9 > $::env(SIMD_OUT)/pre_global_hold_min.rpt
set routed_hold_margin 0.050
if {[info exists ::env(SIMD_HOLD_MARGIN)]} {set routed_hold_margin $::env(SIMD_HOLD_MARGIN)}
repair_timing -hold -max_passes 30 -hold_margin $routed_hold_margin -max_buffer_percent 40 -verbose
detailed_placement
simd_connect_system_supplies
global_route -end_incremental
puts "SIMD_GLOBAL_ROUTE_SETUP"
report_worst_slack -max -digits 9
puts "SIMD_GLOBAL_ROUTE_HOLD"
report_worst_slack -min -digits 9
write_guides $::env(SIMD_OUT)/post_global_route.guide
write_db $::env(SIMD_OUT)/post_global_route.odb
report_checks -path_delay max -group_count 10 -format full_clock_expanded -digits 9 > $::env(SIMD_OUT)/post_global_route_setup.rpt
report_checks -path_delay min -group_count 20 -format full_clock_expanded -digits 9 > $::env(SIMD_OUT)/post_global_route_hold.rpt
} else {
 read_guides $::env(SIMD_OUT)/post_global_route.guide
}
detailed_route -output_drc $::env(SIMD_OUT)/route.drc
# Save routed geometry before the independently expensive RC extraction.
# A bounded extraction timeout must not discard a completed detailed route.
write_db $::env(SIMD_OUT)/post_detailed_route.odb.pending
file rename -force $::env(SIMD_OUT)/post_detailed_route.odb.pending $::env(SIMD_OUT)/post_detailed_route.odb
write_verilog $::env(SIMD_OUT)/post_detailed_route.v
puts SIMD_DETAILED_ROUTE_CHECKPOINT_WRITTEN
if {[info exists ::env(SIMD_DETAILED_ROUTE_ONLY)]} {exit}
}
if {[info exists ::env(SIMD_SYSTEM)]} {
 # Final checks include placed standard cells, synchronizers and CTS/ECO
 # buffers. The earlier floorplanning-only macro/grid probe is insufficient.
 check_power_grid -net VDD -error_file $::env(SIMD_OUT)/final_pdn_VDD.rpt
 check_power_grid -net VSS -error_file $::env(SIMD_OUT)/final_pdn_VSS.rpt
 puts "SIMD_FINAL_POWER_GRID_VERIFIED"
}
define_process_corner -ext_model_index 0 X
extract_parasitics -ext_model_file $::env(SIMD_RCX)
write_spef $::env(SIMD_OUT)/design.spef
puts SIMD_SPEF_READ_BEGIN
read_spef $::env(SIMD_OUT)/design.spef
report_parasitic_annotation -report_unannotated
puts SIMD_SPEF_READ_END
write_def $::env(SIMD_OUT)/design.def
write_verilog $::env(SIMD_OUT)/routed.v
write_db $::env(SIMD_OUT)/design.odb
puts "SIMD_FINAL_SETUP"
report_worst_slack -max -digits 9
puts "SIMD_FINAL_HOLD"
report_worst_slack -min -digits 9
puts "SIMD_FINAL_TNS"
report_tns -digits 9
report_design_area
report_checks -path_delay max -group_count 20 -format full_clock_expanded -fields {slew capacitance input_pin net} -digits 9 > $::env(SIMD_OUT)/setup.rpt
report_checks -path_delay min -group_count 20 -format full_clock_expanded -fields {slew capacitance input_pin net} -digits 9 > $::env(SIMD_OUT)/hold.rpt
report_check_types -max_slew -max_capacitance -max_fanout -min_period -min_pulse_width -violators -digits 9 > $::env(SIMD_OUT)/electrical.rpt
check_setup -verbose > $::env(SIMD_OUT)/coverage.rpt
