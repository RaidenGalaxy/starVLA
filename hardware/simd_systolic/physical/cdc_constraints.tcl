# Source after clocks and the mapped system have been read. Pass the exact
# mapped pin collections for each FIFO and preserve both synchronizer stages.
# Do not apply asynchronous clock-group false paths over these max-delay paths.
# This file supplies constraints, not evidence that a routed design meets them.
proc simd_constrain_gray {source_q first_d second_d source_period_ns} {
  set width [llength $source_q]
  if {$width == 0 || $width != [llength $first_d] || $width != [llength $second_d]} {
    error "Gray bus pin collections are empty or have inconsistent widths"
  }
  # Bounding every nonnegative data delay below 0.8 * source period also
  # bounds the inter-bit arrival spread. The 20% allowance covers launch
  # clock-to-Q and source-clock variation; check it against the actual library.
  set budget [expr {0.8 * $source_period_ns}]
  set launch_cells [get_cells -of_objects $source_q]
  if {[llength $launch_cells] != $width} {
    error "Each Gray bit must resolve to a distinct launch register"
  }
  # OpenSTA implements this data-path bound with -ignore_clock_latency.
  # Keep the library setup check in the budget; clock insertion delay must
  # not introduce a synchronous relationship between the two domains.
  # Register instances are the real timing startpoints in OpenSTA. Using
  # their Q pins creates additional internal-pin paths and leaves the real
  # clock-launched paths outside this exception.
  set_max_delay -ignore_clock_latency -from $launch_cells -to $first_d $budget
  # The first asynchronous stage can be metastable; it has no synchronous
  # hold relationship to the source clock. This exception leaves max-delay
  # checking intact. Stage 1 -> stage 2 remains an ordinary timed path.
  set_false_path -hold -from $launch_cells -to $first_d
  set_dont_touch [get_cells -of_objects $first_d]
  set_dont_touch [get_cells -of_objects $second_d]
}
