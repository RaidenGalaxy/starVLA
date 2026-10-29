package simd

import circt.stage.ChiselStage
import java.nio.file.{Files,Paths}

object Generate extends App {
  val kind=args.headOption.getOrElse("alu")
  val stages=args.lift(1).map(_.toInt).getOrElse(Architecture.stages)
  require(stages==2,"only F2 is supported")
  val rows=args.lift(2).map(_.toInt).getOrElse(8)
  val cols=args.lift(3).map(_.toInt).getOrElse(rows)
  val out=args.lift(4).getOrElse(s".build/${kind}_${stages}_${rows}x$cols")
  val sramLatency=args.lift(5).map(_.toInt).getOrElse(Architecture.sramReadLatency)
  val cfg=Config(stages,rows,cols,hop=if(stages==2)1 else 2,sramReadLatency=sramLatency)
  val gen=()=>kind match {
    case "write_dma" => new WriteDma
    case "dma" => new ReadDma
    case "alu" => new SimdAlu(cfg)
    case "pe" => new SimdPE(cfg)
    case "array" => new SystolicArray(cfg)
    case "system" => new TileSystem(cfg)
    case "gemm" => new GemmSystem(cfg)
    case _ => throw new IllegalArgumentException(s"unknown target $kind")
  }
  Files.createDirectories(Paths.get(out))
  // Chisel 7 extracts verification layers into separate bind files by
  // default. Enable assertions in the emitted modules so every simulator,
  // including hierarchical Verilator builds, actually executes them.
  // Synthesis explicitly ignores assertions and maps the functional logic.
  ChiselStage.emitSystemVerilogFile(gen(),Array("--target-dir",out,"--full-stacktrace"),
    Array("--disable-all-randomization","-strip-debug-info",
      "--enable-layers=Verification","--enable-layers=Verification.Assert"))
  val manifest=s"""{"target":"$kind","grouped_fp4_fp8":${kind=="gemm"},"grouped_int4":${kind=="gemm"},"group_auxiliary_lanes":${if(kind=="gemm")2 else 1},"scale_cache_lines":${if(kind=="gemm")2 else 1},"native_w4a8":${kind=="gemm"},"w4a16_dequant":${kind=="gemm"},"grouped_int8":${kind=="gemm"},"auxiliary_input_latency":${cfg.auxiliaryInputStages},"sram_read_latency":${cfg.sramReadLatency},"profile":"${if(stages==2)"F2"else s"HF$stages"}","stages":$stages,"contexts":16,"counter_settle_cycles":${if(stages==2)2 else 3},"response_credits":${if(stages==2)0 else stages+2},"synchronous_arithmetic_reset":true,"onchip_counters":${Set("system","gemm").contains(kind)},"utilization_counters":${Set("system","gemm").contains(kind)},"autonomous_gemm":${kind=="gemm"},"gemm_source_stride_alignment":8,"gemm_initial_c_dma":${kind=="gemm"},"arithmetic_latency":$stages,"pe_commit_latency":${cfg.commitStages},"pe_state_read_latency":${cfg.stateReadStages},"pe_response_credits":${cfg.peResponseCredits},"pe_read_latency":${cfg.peReadLatency},"array_read_latency":${cfg.arrayReadLatency},"feedback_issue_interval":${cfg.feedbackInterval},"context_state_bits_per_pe":8192,"aggregate_bytes":65536,"wave_fifo_entries":256,"hop_latency":${cfg.hop},"local_wave_registers":${stages!=2 && Set("array","system","gemm").contains(kind)},"rows":$rows,"cols":$cols,"k_max":65536,"k_tile":8,"a_bytes":16384,"b_bytes":32768,"c_bytes":16384,"formats":["I4","I8","I16","I32","I64","F4","F8","F16","BF16","F32","F64"],"timing_status":"target_only"}
"""
  val phaseNames=if(kind=="gemm")GemmPerformance.names.map(x=>"\""+x+"\"").mkString("[",",","]")else"[]"
  val formatNames=Seq("I4","I8","I16","I32","I64","F4","F8","F16","BF16","F32","F64")
  val formatDetails=Formats.widths.zip(Formats.accs).zipWithIndex.map {case((width,acc),index)=>
    s"""{"format":"${formatNames(index)}","input_bits":$width,"lanes":${64/width},"accumulator_bits":$acc,"result_bits":${64/width*acc},"full_integer_product_bits":${if(index<5)(2*width).toString else"null"}}"""
  }.mkString("[",",","]")
  Files.writeString(Paths.get(out,"capabilities.json"),manifest.trim.dropRight(1)+
    ",\"process\":\"asap7\",\"core_period_ps\":"+Architecture.number("core_period_ps")+
    ",\"memory_period_ps\":"+Architecture.number("memory_period_ps")+
    ",\"dram_period_ps\":"+Architecture.number("dram_period_ps")+
    ",\"storage_implementation\":\"asap7_sram_macro\",\"sram_physical_certified\":false"+
    ",\"sram_macro\":\""+Architecture.value("sram_macro")+"\""+
    ",\"architecture_sha256\":\""+java.security.MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(Paths.get("configs/architecture.properties"))).map(b=>f"${b & 0xff}%02x").mkString+"\""+",\"sequencer_cycle_categories\":"+phaseNames+
    ",\"gather_word_fastpath\":"+(kind=="gemm")+",\"scratchpad_access\":"+(kind=="gemm")+
    ",\"autonomous_prefetch\":"+(kind=="gemm")+
    ",\"output_epilogue\":"+(kind=="gemm")+
    ",\"output_push_override\":"+Set("system","gemm").contains(kind)+
    ",\"output_epilogue_arithmetic\":\"FP32_scale_then_RNE_cast\""+
    ",\"converted_initial_c\":false"+
    ",\"native_prefetch_record_prefix\":\"N\""+
    ",\"prefetch_across_group_boundaries\":false"+
    ",\"aggregate_inspection\":"+Set("system","gemm").contains(kind)+
    ",\"fixed_state_commit\":"+(kind=="pe"&&stages>2)+
    ",\"registered_response_heads\":"+(stages>2&&Set("alu","pe","system","gemm").contains(kind))+
    ",\"local_response_dequeue\":"+(stages>2&&Set("alu","pe","system","gemm").contains(kind))+
    ",\"registered_packed_pe_sources\":"+(stages>2&&Set("pe","array","system","gemm").contains(kind))+
    ",\"registered_preparation_controls\":"+(stages>2&&Set("pe","array","system","gemm").contains(kind))+
    ",\"local_encoded_context_reads\":"+(stages>2&&Set("pe","array","system","gemm").contains(kind))+
    ",\"registered_context_initialization_snapshot\":"+(stages>2&&Set("pe","array","system","gemm").contains(kind))+
    ",\"local_packed_a_sources\":"+(stages>2&&Set("array","system","gemm").contains(kind))+
    ",\"all_wave_stages_tracked\":"+Set("array","system","gemm").contains(kind)+
    ",\"balanced_event_population_count\":"+(stages>2&&Set("array","system","gemm").contains(kind))+
    ",\"overflow_event_gate_after_count\":"+(stages>2&&Set("array","system","gemm").contains(kind))+
    ",\"registered_context_admission\":"+(stages>2&&kind=="pe")+
    ",\"elastic_local_preparation_controls\":"+(stages>2&&kind=="pe")+
    ",\"distributed_request_capture\":"+(stages>2&&kind=="pe")+
    ",\"registered_ready_lookahead\":"+(stages>2&&kind=="pe")+
    ",\"local_observation_initialization\":"+(stages>2&&Set("pe","array","system","gemm").contains(kind))+
    ",\"inline_chisel_assertions\":true"+
    ",\"scheduled_streaming_admission\":"+Set("array","system","gemm").contains(kind)+
    ",\"local_streaming_expansion\":"+(stages>2&&Set("array","system","gemm").contains(kind))+
    ",\"state_commit_latency\":"+(stages+cfg.stateReadStages+cfg.commitStages)+
    ",\"seed_write_latency\":"+cfg.seedWriteLatency+",\"array_busy_release_latency\":"+cfg.busyReleaseLatency+
    ",\"operand_word_bits\":64,\"result_container_bits\":512,\"format_capabilities\":"+formatDetails+"}\n")
}
