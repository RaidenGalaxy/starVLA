# Run from a routed database and the parasitics that identified its violations.
set_thread_count 8
read_lef $::env(SIMD_TECH_LEF)
read_lef $::env(SIMD_CELL_LEF)
read_liberty $::env(SIMD_LIB)
read_db $::env(SIMD_OUT)/seed.odb
read_sdc $::env(SIMD_OUT)/design.sdc
set_propagated_clock [all_clocks]
set_wire_rc -signal -layer M3
set_wire_rc -clock -layer M5
set_dont_use [get_lib_cells {*/*x1p*_ASAP7* */*xp*_ASAP7* */SDF* */ICG*}]
set_routing_layers -signal M2-M7 -clock M4-M7
set_global_routing_layer_adjustment M2-M3 0.5
set_global_routing_layer_adjustment M4-M7 0.25
# Rebuild signal routes instead of counting existing detailed wires as fixed
# routing obstructions while adding ECO cells. Connectivity and placed/CTS
# cells are preserved; the final design is routed and extracted again.
set block [ord::get_db_block]
set removed_wires 0
foreach net [$block getNets] {
 set wire [$net getWire]
 if {$wire != "NULL"} {odb::dbWire_destroy $wire;incr removed_wires}
}
puts "SIMD_ECO_REMOVED_DETAILED_WIRES $removed_wires"
# Rebuild live router state before entering the incremental repair bracket.
set initial_route_options {}
if {[info exists ::env(SIMD_ECO_ALLOW_INITIAL_CONGESTION)]} {lappend initial_route_options -allow_congestion}
global_route -guide_file $::env(SIMD_OUT)/route.guide -congestion_iterations 50 {*}$initial_route_options
# Initialize the resizer's incremental-routing mode BEFORE restoring SPEF.
# In pinned EstimateWireParasitics.cc, updateParasitics updates only invalid
# nets once this mode is set. A subsequent whole-design estimate would erase
# the measured clock/data RC that identified the actual failing endpoints.
estimate_parasitics -global_routing
read_spef $::env(SIMD_OUT)/seed.spef
puts "SIMD_ECO_EXTRACTED_INPUT_SETUP"
report_worst_slack -max -digits 9
puts "SIMD_ECO_EXTRACTED_INPUT_HOLD"
report_worst_slack -min -digits 9
# Retain extracted RC on unchanged nets, including the clock tree. Modified
# nets receive incremental global-route estimates. The final reroute and new
# extraction, rather than this mixed optimization model, certify the result.
report_checks -path_delay min -group_count 20 -format full_clock_expanded -digits 9 > $::env(SIMD_OUT)/eco_input_hold.rpt
report_checks -path_delay max -group_count 20 -format full_clock_expanded -digits 9 > $::env(SIMD_OUT)/eco_input_setup.rpt
global_route -start_incremental {*}$initial_route_options
repair_design -cap_margin 20 -slew_margin 20
set setup_repair_margin 0.050
if {[info exists ::env(SIMD_SETUP_MARGIN)]} {set setup_repair_margin $::env(SIMD_SETUP_MARGIN)}
repair_timing -setup -setup_margin $setup_repair_margin -max_passes 10 -repair_tns 100
repair_timing -hold -hold_margin $::env(SIMD_HOLD_MARGIN) -max_passes 10
detailed_placement
# Preserve repaired placement even when incremental global routing fails.
# These mixed-parasitic estimates are optimization inputs, not signoff.
write_db $::env(SIMD_OUT)/post_repair.odb
write_verilog $::env(SIMD_OUT)/post_repair.v
write_spef $::env(SIMD_OUT)/post_repair_mixed.spef
report_checks -path_delay max -group_count 20 -format full_clock_expanded -digits 9 > $::env(SIMD_OUT)/post_repair_setup.rpt
report_checks -path_delay min -group_count 20 -format full_clock_expanded -digits 9 > $::env(SIMD_OUT)/post_repair_hold.rpt
puts "SIMD_ECO_REPAIR_CHECKPOINT"
if {[info exists ::env(SIMD_ECO_REPAIR_ONLY)]} {exit}
global_route -end_incremental {*}$initial_route_options
estimate_parasitics -global_routing
puts "SIMD_ECO_GLOBAL_ROUTE_SETUP"
report_worst_slack -max -digits 9
puts "SIMD_ECO_GLOBAL_ROUTE_HOLD"
report_worst_slack -min -digits 9
write_db $::env(SIMD_OUT)/post_global_route.odb
# SIMD_ECO_ROUTE
